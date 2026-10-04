"""StreamDriver(null_floor=...): the live gate's opt-in chance floor.

WHAT IS PINNED. The live gate (_fits / _gate_count: >= min_inliers AND >= min_inlier_frac of the frame's
peaks) accepts 211 of 480 azimuth-scrambled cxidb-17 frames at the shipped depth: the known-cell search keeps
the best of ~10^4 orientations, and on a sparse frame that best chance count clears 10 inliers. null_floor adds
a third bar, n_inl >= a*n + b + c*sqrt(n) (experiments/live_gate_null.py, RESULTS_live_gate_null.md). This file
checks that
  * the default is None and the gate is then exactly the historical two-bar rule, over a grid of counts;
  * bad floors raise, and (a, b) / (a, b, c) / NULL_FLOOR_CXIDB17 are accepted;
  * through flush(), a sparse frame at the measured chance level (48 peaks, 12 on the lattice) is ACCEPTED by
    the default gate and REFUSED with the floor, while the NEGATIVE CONTROLS -- a sparse frame at the level of
    the frames the floor keeps (55 peaks, 24 on the lattice) and a dense frame (300 peaks, 60 on it) -- are
    accepted either way; the event log and stats() say which bar refused it;
  * the floor reaches the other gate sites through _gate_count (the warm-up rescue, best-fit assignment);
  * REAL FRAMES (torch half): on the committed 120 cxidb-17 frames, one fixed azimuth-scrambled copy per frame
    and three more, searched with the lysozyme cell by the driver's own index_fused on CPU torch -- the live
    gate accepts >= 30% of the scrambled copies (measured 52%: the defect is present), the floor accepts <= 2.5%
    (measured 0.8%), and the floor keeps >= 90% of the strict-gate frames (measured 77 of 80). A floor that
    stopped applying fails the second bound; one that refused real frames fails the third.

The numpy half runs in the torch-free CI job with the fit-oracle seams of test_cell_registry; the real-frame half
SKIPS, exit 0, without torch or below pyproject's floor (>= 1.12), and runs in the CPU-torch job.

  PYTHONPATH=. python experiments/test_live_gate_floor.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import glint.stream_driver as sd                                             # noqa: E402
from glint.stream_driver import NULL_FLOOR_CXIDB17, StreamDriver              # noqa: E402
from test_cell_registry import A, _Oracle, _driver, _terminal                  # noqa: E402

SEED = 20260928


def _frame(M, n_on, n_off, rng):
    """n_on peaks exactly on lattice M and n_off peaks whose hkl sit at half-integers (residual 0.5 > HKL_TOL),
    so the driver's count under M is exactly n_on."""
    hkl = rng.integers(-6, 7, size=(4 * (n_on + n_off), 3)).astype(float)
    hkl = np.unique(hkl[np.abs(hkl).sum(1) > 0], axis=0)
    hkl = hkl[rng.permutation(len(hkl))]
    on = hkl[:n_on]
    off = hkl[n_on:n_on + n_off] + np.array([0.5, 0.5, 0.5])
    q = np.vstack([on, off]) @ np.linalg.inv(M)
    assert StreamDriver._inliers(None, q, M) == n_on, "fixture: off-lattice peaks landed inside the window"
    return q


# (n_peaks, n_on): the measured chance level of the refused sparse frames, and the controls
CHANCE = (48, 12)          # the lost strict frames: n 38-60, 10-16 inliers; floor at 48 is 14.8
SPARSE_REAL = (55, 24)     # the sparse strict frames the floor keeps: median n 55, 24 inliers; floor 15.5
DENSE_REAL = (300, 60)     # 20%: passes the fraction bar; floor 33.0


def _floor(n, fl=NULL_FLOOR_CXIDB17):
    return fl[0] * n + fl[1] + fl[2] * np.sqrt(n)


def test_fixture_sits_where_the_measurement_says():
    """The three cases are what they claim to be under the SHIPPED constants, checked before any driver runs:
    every one passes the historical two-bar gate, and only CHANCE falls below the floor."""
    for n, m in (CHANCE, SPARSE_REAL, DENSE_REAL):
        assert m >= 10 and m >= 0.15 * n, (n, m)
    assert CHANCE[1] < _floor(CHANCE[0]), (CHANCE, _floor(CHANCE[0]))
    assert SPARSE_REAL[1] >= _floor(SPARSE_REAL[0]) and DENSE_REAL[1] >= _floor(DENSE_REAL[0])


def test_signature_default_and_validation():
    import inspect
    sig = inspect.signature(StreamDriver.__init__).parameters
    assert list(sig)[-1] == "null_floor" and sig["null_floor"].default is None, list(sig)[-3:]
    for bad in ((0.1,), (0.1, 1.0, 2.0, 3.0), (float("nan"), 1.0), (0.1, float("inf")), "0.1,5", (True, 1.0),
                {"a": 1, "b": 2}, 3.0, (0.1, "5"), {0.1, 5.0}, frozenset((0.02, 5.0, 1.2)),
                (v for v in (0.1, 5.0)), np.array([[0.1, 5.0]])):
        try:
            _driver(null_floor=bad)
        except ValueError as exc:
            assert "null_floor" in str(exc), exc
        else:
            raise AssertionError(f"null_floor={bad!r} accepted")
    assert _driver(null_floor=(0.05, 10))[0].null_floor == (0.05, 10.0, 0.0)
    assert _driver(null_floor=np.array([0.05, 10, 1.5]))[0].null_floor == (0.05, 10.0, 1.5)
    assert _driver(null_floor=NULL_FLOOR_CXIDB17)[0].null_floor == tuple(NULL_FLOOR_CXIDB17)


def test_default_gate_is_the_historical_rule():
    """null_floor=None: _gate_count equals the two-bar rule it replaced, on every (count, peaks) pair of a grid
    and at four gate settings -- the default path's decisions are unchanged, not merely similar. The NaN fraction
    is the edge: the old `n >= nan` refused everything, and `n < nan` would accept everything (Copilot review)."""
    for kw in (dict(), dict(min_inliers=10), dict(min_inliers=10, min_inlier_frac=0.0),
               dict(min_inliers=10, min_inlier_frac=float("nan"))):
        drv, _ = _driver(**kw)
        for npk in range(1, 400, 7):
            for n in range(0, npk + 1):
                old = n >= drv.min_inliers and (not drv.min_inlier_frac or n >= drv.min_inlier_frac * npk)
                assert drv._gate_count(n, npk) == old, (kw, n, npk)
        assert drv.n_null_floor_refused == 0
        st = drv.stats()
        assert "null_floor" not in st and "n_null_floor_refused" not in st, sorted(st)


def _run(floor):
    """Three frames through the driver's own flush(): _Oracle(default=A) registers each under A, so the live
    gate sees exactly the fixture's counts."""
    rng = np.random.default_rng(SEED)
    frames = [_frame(A, m, n - m, rng) for n, m in (CHANCE, SPARSE_REAL, DENSE_REAL)]
    drv, _ = _driver(oracle=_Oracle(default=A), events=True, min_inliers=10, null_floor=floor)
    for q in frames:
        drv.push_q(q)
    drv.flush()
    return drv, {ev: e["outcome"] for ev, e in _terminal(drv.events).items()}


def test_floor_refuses_the_chance_level_frame_and_keeps_the_controls():
    drv0, out0 = _run(None)
    assert out0 == {0: "indexed", 1: "indexed", 2: "indexed"}, out0      # the defect: chance level accepted
    assert drv0.n_indexed == 3 and drv0.n_gate_rejected == 0
    drv1, out1 = _run(NULL_FLOOR_CXIDB17)
    assert out1 == {0: "gate_rejected", 1: "indexed", 2: "indexed"}, out1  # refused; both controls kept
    assert drv1.n_indexed == 2 and drv1.n_gate_rejected == 1
    st = drv1.stats()
    assert st["null_floor"] == list(NULL_FLOOR_CXIDB17) and st["n_null_floor_refused"] == 1, st
    # a straight-line floor (c = 0) is a valid setting too, and the arithmetic is the documented one
    drv2, out2 = _run((0.0, 13.0))
    assert out2 == {0: "gate_rejected", 1: "indexed", 2: "indexed"}, out2


def test_floor_reaches_the_other_gate_sites():
    """_fits and the warm-up rescue (which re-index retained frames at lock) ask the same _gate_count, and so
    does best-fit assignment; a floor that only the batch path applied would let a rescue re-admit the frame."""
    rng = np.random.default_rng(SEED + 1)
    qc = _frame(A, CHANCE[1], CHANCE[0] - CHANCE[1], rng)
    qr = _frame(A, SPARSE_REAL[1], SPARSE_REAL[0] - SPARSE_REAL[1], rng)
    for floor, want in ((None, 2), (NULL_FLOOR_CXIDB17, 1)):
        drv, _ = _driver(oracle=_Oracle(default=A), warmup_rescue=True, min_inliers=10, null_floor=floor)
        assert drv._fits(qc, A) is (floor is None) and drv._fits(qr, A)
        drv._warmup_buf = [qc, qr]                     # retained warm-up frames, re-indexed when the cell locks
        drv._lock(A)
        assert drv.n_warmup_rescued == want, (floor, drv.n_warmup_rescued)
    # best-fit assignment with two active cells: the winner must clear the floor too
    drv, oracle = _driver(oracle=_Oracle(default=A), assign="best", adaptive_relock=True, min_inliers=10,
                          null_floor=NULL_FLOOR_CXIDB17)
    drv.extra.append(dict(Mc=A.copy(), grid=drv.grid, acc=drv.acc, nth=drv.n_theoretical))
    for slot, q in enumerate((qc, qr)):
        drv._q[slot] = q; drv._idx[slot] = slot
    drv._n = 2; drv.n_pushed = 2
    drv._watchdog = lambda missed, cached=None: None   # the watchdog is not under test here
    missed = drv._index_best_fit([0, 1])
    assert missed == [0] and drv.n_indexed == 1, (missed, drv.n_indexed)


# ------------------------------------------------------------------------------------ real frames ------
def _torch_ok():
    try:
        import torch
    except ImportError:
        return "no torch: glint.replica_gpu_batch cannot import here"
    ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if ver < (1, 12):
        return f"torch {torch.__version__} is below pyproject's floor (>= 1.12)"
    return None


def test_real_frames_null_and_floor():
    why = _torch_ok()
    if why:
        print(f"SKIP test_real_frames_null_and_floor -- {why}")
        return
    import glint.replica_gpu_batch as rgb
    from glint.glint_fast import GATE_FRAC, GATE_MIN, LYSO, load, matched_strict
    from glint.multilattice import scramble_azimuth
    from glint.multishot import same_lattice
    real = [np.asarray(q, float) for q in load(os.path.join(HERE, "frames_cxidb_clean.txt"))]
    n = np.array([len(q) for q in real])
    K = 4

    def counts(qs):
        Ms = rgb.index_fused(qs, LYSO, B=20)
        out = []
        for q, M in zip(qs, Ms):
            M = None if M is None else np.asarray(M, float)
            out.append((0, None) if M is None or abs(np.linalg.det(M)) < 1.0 else (StreamDriver._inliers(None, q, M), M))
        return out

    r = counts(real)
    ir = np.array([c for c, _ in r])
    strict = np.array([M is not None and same_lattice(M, LYSO) and matched_strict(M, q) >= GATE_MIN
                       and matched_strict(M, q) / len(q) >= GATE_FRAC for q, (_, M) in zip(real, r)])
    null = np.array([[c for c, _ in counts([scramble_azimuth(q, np.random.default_rng([SEED, i, k]))
                                            for i, q in enumerate(real)])] for k in range(K)]).T
    # decided by the DRIVER's gate, the 480 arm's live setting with and without the floor -- not re-derived here
    d_live, _ = _driver(min_inliers=10)
    d_floor, _ = _driver(min_inliers=10, null_floor=NULL_FLOOR_CXIDB17)
    gate = lambda drv, m, nn: np.array([drv._gate_count(int(a), int(b)) for a, b in zip(m, nn)])   # noqa: E731
    nn = np.repeat(n, K)
    live_null = gate(d_live, null.ravel(), nn).mean()
    floor_null = gate(d_floor, null.ravel(), nn).mean()
    kept = (gate(d_floor, ir, n) & strict).sum()
    print(f"    120 real frames, {K} scrambled copies each: live gate accepts {live_null:.3f} of copies, "
          f"with the floor {floor_null:.4f}; strict frames kept {kept}/{strict.sum()}")
    assert live_null >= 0.30, f"live gate accepted only {live_null:.3f} of scrambled copies: control failed"
    assert floor_null <= 0.025, f"floor accepted {floor_null:.4f} of scrambled copies (> 2.5%)"
    assert strict.sum() >= 70 and kept >= 0.90 * strict.sum(), (kept, strict.sum())


TESTS = (test_fixture_sits_where_the_measurement_says, test_signature_default_and_validation,
         test_default_gate_is_the_historical_rule, test_floor_refuses_the_chance_level_frame_and_keeps_the_controls,
         test_floor_reaches_the_other_gate_sites, test_real_frames_null_and_floor)

if __name__ == "__main__":
    saved = sd.rgb
    ok = 0
    for t in TESTS:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:                                                          # noqa: BLE001
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
        finally:
            sd.rgb = saved                                   # _driver installs its oracle as the module global
    print(f"{ok}/{len(TESTS)} passed")
    raise SystemExit(0 if ok == len(TESTS) else 1)
