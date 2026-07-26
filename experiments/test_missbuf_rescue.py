"""CPU tests for the miss-buffer fan-out + INDEX-ONLY retroactive rescue in glint.stream_driver.

No GPU, no torch: the driver's blind indexer (`_blind_index` / `_fanout`) and the q-only known-cell
indexer (`_known_index`) are injected as fakes through the same seams the production code resolves
lazily. Exercises the watchdog relock (`_watchdog`) and the rescue counter directly, so nothing needs
the batched GPU path. Run: `python experiments/test_missbuf_rescue.py` or `pytest`.
"""
import numpy as np
from glint.stream_driver import StreamDriver


# --------------------------------------------------------------------- fixtures ------------------
def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

M_A = _rot(0.5) @ np.diag([1 / 79.0, 1 / 79.0, 1 / 38.0])      # primary (locked) lyso cell, 79/79/38
M_B = _rot(0.9) @ np.diag([1 / 95.0, 1 / 95.0, 1 / 45.0])      # a DIFFERENT tetragonal cell, 95/95/45
Mi_B = np.linalg.inv(M_B.T)                                    # q -> hkl index matrix for cell B (q @ Mi_B = hkl)

_GOOD = np.full((10, 3), 7.0)                                  # sentinel q for a "good" missed frame (votes B)

N = 32
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
              cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5), coffset=0.0,
              min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]


def _cell(rng):                                               # a valid but random reciprocal basis (junk vote)
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R @ np.diag(1.0 / rng.uniform(50, 110, 3))


def _driver(rescue_buffer=0, fanout=None, B=16):
    return StreamDriver(M_A, PANELS, 0.1, 1.3, (N, N), dtype=np.uint16, B=B,
                        dmin=3.0, use_gpu=False, adaptive_relock=True,
                        rescue_buffer=rescue_buffer, fanout=fanout)


def _fake_blind(q, k, rng):
    """index_blind_nbest stand-in: the sentinel frame surfaces cell B; everything else scatters."""
    if q is not None and q.shape == _GOOD.shape and np.allclose(q, 7.0):
        return [(M_B.copy(), 1.0)] + [(_cell(rng), 0.3) for _ in range(k - 1)]
    return [(_cell(rng), 0.5) for _ in range(k)]


def _lattice_q(rng, n=18):
    """q-vectors that lie exactly on cell B's lattice -> _inliers(q, Mi_B) == len(q) (all integer)."""
    hkl = rng.integers(-6, 7, size=(n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0]
    return hkl @ M_B.T


def _offlattice_q(rng, n=18):
    """On B's geometry but half-integer-shifted -> q @ Mi_B == hkl + 0.5 -> ZERO inliers (deterministic)."""
    hkl = rng.integers(-6, 7, size=(n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0]
    return (hkl + 0.5) @ M_B.T


def _fake_known(qs, Mn, B):
    """index_fused stand-in: proposes cell B's orientation for any frame with enough peaks, else fails.

    Deliberately q-agnostic (it does NOT know which frames recur) -- the REAL _inliers gate in the
    driver is what must separate the recurring lattice-B frames from off-lattice / too-sparse ones."""
    return [Mi_B.copy() if (q is not None and len(q) >= 6) else None for q in qs]


def _axes(Mc):
    return np.sort(1.0 / np.linalg.norm(np.asarray(Mc), axis=0))


# ----------------------------------------------------------------------- tests -------------------
def test_watchdog_serial_default_locks():
    """The default (serial) fan-out relocks the new cell B from 5 recurring + 3 junk misses."""
    rng = np.random.default_rng(11)
    drv = _driver()                                          # fanout=None -> serial default _fanout
    drv._blind_index = lambda q, k: _fake_blind(q, k, rng)
    missed = list(range(8))
    for j, i in enumerate(missed):
        drv._q[i] = _GOOD.copy() if j < 5 else rng.normal(size=(10, 3))
    drv._watchdog(missed)
    assert len(drv.extra) == 1 and drv.n_relock == 1, (len(drv.extra), drv.n_relock)
    assert np.allclose(_axes(drv.extra[0]["Mc"]), [45, 95, 95], rtol=0.05), _axes(drv.extra[0]["Mc"]).tolist()


def test_watchdog_fanout_matches_serial():
    """An out-of-order fan-out yields the SAME relock as the serial loop (consensus is a histogram)."""
    def build(reverse):
        rng = np.random.default_rng(12)
        blind = lambda q, k: _fake_blind(q, k, rng)
        if reverse:
            def fan(Q, k):                                   # workers finish in reverse, order-preserving return
                out = [None] * len(Q)
                for j in reversed(range(len(Q))):
                    out[j] = blind(Q[j], k)
                return out
            drv = _driver(fanout=fan)
        else:
            drv = _driver(fanout=None)
        drv._blind_index = blind
        missed = list(range(8))
        for j, i in enumerate(missed):
            drv._q[i] = _GOOD.copy() if j < 5 else rng.normal(size=(10, 3))
        drv._watchdog(missed)
        return drv
    ser, rev = build(False), build(True)
    assert ser.n_relock == rev.n_relock == 1, (ser.n_relock, rev.n_relock)
    assert np.allclose(_axes(ser.extra[0]["Mc"]), _axes(rev.extra[0]["Mc"])), "same cell regardless of order"
    assert np.allclose(_axes(ser.extra[0]["Mc"]), [45, 95, 95], rtol=0.05)


def test_rescue_counts_recurring():
    """On relock, the buffered pre-lock misses are re-indexed: recurring lattice-B frames are rescued,
    off-lattice and too-sparse frames are refused by the real _inliers gate."""
    rng = np.random.default_rng(10)
    drv = _driver(rescue_buffer=64)
    drv._known_index = _fake_known
    drv._fanout = lambda Q, k: [[(M_B.copy(), 1.0)] for _ in Q]   # every missed frame votes B -> locks
    n_recur = 4
    for _ in range(n_recur):
        drv._missbuf.append((drv.n_pushed, _lattice_q(rng)))       # rescuable: on B's lattice
    for _ in range(2):
        drv._missbuf.append((0, _offlattice_q(rng)))               # indexer returns M, but 0 inliers -> gate refuses
    drv._missbuf.append((0, rng.normal(size=(3, 3))))              # too few peaks -> indexer returns None
    missed = list(range(5))
    for i in missed:
        drv._q[i] = _lattice_q(rng)
    drv._watchdog(missed)
    assert len(drv.extra) == 1 and drv.n_relock == 1, (len(drv.extra), drv.n_relock)
    assert drv.n_rescued == n_recur, (drv.n_rescued, n_recur)      # only the recurring lattice-B frames
    assert len(drv._missbuf) == 0, "buffer cleared after the rescue pass"
    assert drv.stats()["n_rescued"] == n_recur


def test_default_off_is_inert():
    """rescue_buffer=0: no buffer is allocated, relock still works, and nothing is rescued."""
    drv = _driver(rescue_buffer=0, fanout=None)
    assert drv._missbuf is None
    drv._fanout = lambda Q, k: [[(M_B.copy(), 1.0)] for _ in Q]
    missed = list(range(5))
    for i in missed:
        drv._q[i] = _GOOD.copy()
    drv._watchdog(missed)
    assert len(drv.extra) == 1 and drv.n_relock == 1              # adaptive relock unchanged
    assert drv.n_rescued == 0 and drv.stats()["n_rescued"] == 0   # no buffer -> no rescue


if __name__ == "__main__":
    tests = (test_watchdog_serial_default_locks, test_watchdog_fanout_matches_serial,
             test_rescue_counts_recurring, test_default_off_is_inert)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"{ok}/{len(tests)} passed")
