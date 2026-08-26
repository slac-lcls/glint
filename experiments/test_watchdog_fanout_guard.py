"""A permanently failing blind fan-out must degrade to the miss path, not crash flush() (glint#147).

WHY THIS EXISTS. `_watchdog`'s fan-out invocation (via `_watchdog_nbest`) had no exception guard,
so a fanout that raises persistently -- a dead GPU worker, a poisoned queue -- propagated out of
`flush()` and took the streaming driver down, even though every affected frame already has a
defined miss path (miss buffer when armed, `n_gate_rejected` otherwise). Pre-existing `main`
behaviour, noted during PR #145's review round and deliberately left out of that PR's scope so its
flag-off sha256 proof stayed untouched. The guard here changes behaviour ONLY on the paths that
used to crash or silently drop frames.

What is pinned, in the order it would hurt if broken:
  * flush() SURVIVES a fanout that always raises: no exception, the ring resets, the frames land
    in the armed miss buffer, a RuntimeWarning fires, and stats()['n_fanout_errors'] counts it.
  * A dead fan-out contributes NO VOTES: it must never relock a cell out of absence of evidence.
  * The driver stays usable: after two failed flushes (streak visible in the warning text) a
    recovered fanout locks the new cell normally, and the streak resets while the total stands.
  * A SHORT fan-out return is padded, not zipped away: the answered slots still vote (three
    one-vote flushes lock the cell), the unanswered ones stay misses, and short is NOT an error.
  * The retry cascade's fan-out seam counts through the same guard (it already survived; now a
    dead fan-out there is visible instead of silent).
  * A healthy fanout is untouched: no warning, zero count -- the flag-off proof in
    test_streamdriver_vs_offline.py and the relock tests in test_missbuf_rescue.py stay green on
    exactly the code they always ran.

Same CPU seam-injection pattern as test_retry_cascade.py / test_missbuf_rescue.py.
Dual mode: `pytest experiments/test_watchdog_fanout_guard.py`, or `python experiments/...`.
"""
import warnings as _warnings

import numpy as np

import glint.stream_driver as sd
from glint.lattice import cell_to_Ar
from glint.stream_driver import StreamDriver

SEED = 20260826
NPX = 256
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
               cx=-(NPX / 2.0 - 0.5), cy=-(NPX / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=NPX - 1, min_ss=0, max_ss=NPX - 1)]
CLEN, WAVE, DMIN = 0.1, 1.322, 3.0


def _rot(a, b, c):
    def R(t, ax):
        s, k = np.sin(t), np.cos(t)
        m = np.eye(3); i, j = [x for x in range(3) if x != ax]
        m[i, i] = m[j, j] = k; m[i, j] = -s; m[j, i] = s
        return m
    return R(a, 0) @ R(b, 1) @ R(c, 2)


A = _rot(0.31, 0.77, 1.13) @ cell_to_Ar(79.0, 79.0, 38.0, 90, 90, 90)   # the LOCKED cell
B = _rot(1.91, 0.42, 2.30) @ cell_to_Ar(52.0, 61.0, 71.0, 90, 90, 90)   # the cell the misses carry
JUNK = _rot(2.71, 1.34, 0.19) @ cell_to_Ar(53.7, 61.3, 71.9, 90, 90, 90)   # fits nothing


def frame_on(M, rng, n=40):
    """q-vectors lying exactly on lattice M, so q @ M == hkl and _fits(q, M) is total."""
    hkl = rng.integers(-5, 6, size=(4 * n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0][:n]
    return hkl @ np.linalg.inv(M)


class _FakeRGB:
    """Batched known-cell indexer stand-in: registers every frame as JUNK, so every frame fails
    the live gate and reaches the watchdog -- the exact population the fan-out runs on."""

    def index_fused(self, qs, Mc, B=1):
        return [JUNK.copy() for _ in qs]


def _driver(rescue_buffer=0, retry_cascade=False):
    drv = StreamDriver(A, PANELS, CLEN, WAVE, (NPX, NPX), dtype=np.uint16, B=8, dmin=DMIN,
                       use_gpu=False, adaptive_relock=not retry_cascade,
                       rescue_buffer=rescue_buffer, retry_cascade=retry_cascade)
    drv._blind_index = lambda q, k: []         # per-frame fallback: solved, found nothing
    drv._known_perframe = None
    return drv


def _load(drv, frames):
    sd.rgb = _FakeRGB()
    for slot, q in enumerate(frames):
        drv._q[slot] = q
        drv._idx[slot] = slot
    drv._n = len(frames)
    drv.n_pushed += len(frames)


def _boom(Q, k):
    raise RuntimeError("worker pool is dead")


def _flush_catching(drv):
    """flush() under a warning trap; returns the RuntimeWarnings raised."""
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        drv.flush()
    return [x for x in w if issubclass(x.category, RuntimeWarning)]


# ---------------------------------------------------------------------- tests --------------------
def test_dead_fanout_does_not_crash_flush():
    """THE DEFECT: a fanout that always raises used to propagate out of flush()."""
    rng = np.random.default_rng(SEED)
    drv = _driver(rescue_buffer=64)
    drv._fanout = _boom
    frames = [frame_on(B, rng) for _ in range(5)]
    _load(drv, frames)
    ws = _flush_catching(drv)                                  # pre-fix: RuntimeError out of flush()
    assert drv._n == 0, "flush() must complete and reset the ring"
    assert drv.n_fanout_errors == 1, drv.n_fanout_errors
    assert drv.stats()["n_fanout_errors"] == 1
    assert len(drv._missbuf) == len(frames), (len(drv._missbuf), len(frames))
    assert len(ws) == 1 and "fan-out" in str(ws[0].message), [str(x.message) for x in ws]
    assert drv.n_relock == 0 and not drv.extra, "no votes may come from a dead fan-out"


def test_streak_counts_and_recovery_relocks():
    """Two failed flushes show the streak; a recovered fanout then locks the cell normally."""
    rng = np.random.default_rng(SEED + 1)
    drv = _driver(rescue_buffer=0)
    drv._fanout = _boom
    for _ in range(2):
        _load(drv, [frame_on(B, rng) for _ in range(5)])
        ws = _flush_catching(drv)
    assert drv.n_fanout_errors == 2 and drv._fanout_fail_streak == 2
    assert "2 consecutive" in str(ws[0].message), str(ws[0].message)
    assert drv.n_relock == 0, "still no votes after two dead flushes"

    drv._fanout = lambda Q, k: [[(B.copy(), 1.0)] for _ in Q]   # workers came back
    _load(drv, [frame_on(B, rng) for _ in range(5)])
    ws = _flush_catching(drv)
    assert not ws, "a healthy fan-out must not warn"
    assert drv.n_relock == 1 and len(drv.extra) == 1, "recovered fan-out relocks normally"
    assert drv._fanout_fail_streak == 0, "success resets the streak"
    assert drv.n_fanout_errors == 2, "...but the total stands"


def test_short_return_is_padded_not_dropped():
    """A fanout answering only the first frame: answered slots vote, the tail is not zipped away,
    and short is NOT counted as an error. One vote per flush -> the third flush locks."""
    rng = np.random.default_rng(SEED + 2)
    drv = _driver(rescue_buffer=64)
    drv._fanout = lambda Q, k: [[(B.copy(), 1.0)]]              # len 1, whatever len(Q) is
    for n_flush in range(1, 4):
        _load(drv, [frame_on(B, rng) for _ in range(4)])
        ws = _flush_catching(drv)
        assert not ws, "short is a degraded answer, not an error"
        if n_flush < 3:                                         # every miss buffered, none zipped away
            assert len(drv._missbuf) == 4 * n_flush, (n_flush, len(drv._missbuf))
    assert drv.n_fanout_errors == 0
    assert drv.n_relock == 1, "the answered slot's votes accumulated across flushes"
    assert len(drv._missbuf) == 0, "the relock's retroactive rescue pass consumed the buffer"


def test_cascade_seam_counts_through_the_same_guard():
    """retry_cascade's fan-out already survived a raise; now it counts and warns too, and the
    per-frame fallback still runs (frames end as gate_rejected, exactly as before)."""
    rng = np.random.default_rng(SEED + 3)
    drv = _driver(retry_cascade=True)
    drv._fanout = _boom
    frames = [frame_on(B, rng) for _ in range(4)]
    _load(drv, frames)
    ws = _flush_catching(drv)
    assert drv.n_fanout_errors == 1 and len(ws) == 1
    assert drv.n_gate_rejected == len(frames), (drv.n_gate_rejected, len(frames))
    assert drv.stats()["n_fanout_errors"] == 1


def test_healthy_fanout_is_untouched():
    """No exception, full-length return: no warning, zero count, relock exactly as always."""
    rng = np.random.default_rng(SEED + 4)
    drv = _driver(rescue_buffer=0)
    drv._fanout = lambda Q, k: [[(B.copy(), 1.0)] for _ in Q]
    _load(drv, [frame_on(B, rng) for _ in range(5)])
    ws = _flush_catching(drv)
    assert not ws and drv.n_fanout_errors == 0
    assert drv.n_relock == 1 and drv.stats()["n_fanout_errors"] == 0


if __name__ == "__main__":
    tests = (test_dead_fanout_does_not_crash_flush, test_streak_counts_and_recovery_relocks,
             test_short_return_is_padded_not_dropped, test_cascade_seam_counts_through_the_same_guard,
             test_healthy_fanout_is_untouched)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
