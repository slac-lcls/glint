"""CPU tests for the opt-in streaming retry CASCADE (glint#75, StreamDriver.retry_cascade).

The scenario every test here builds is the one the issue is about: a frame that the BATCHED
known-cell pass registers badly enough to fail the live accept gate `_fits`, but that a retry ARM can
still recover. With the flag off that frame is a miss; with it on it must be indexed EXACTLY ONCE and
must not also be counted as a miss.

No GPU, no torch. The batched indexer (`glint.stream_driver.rgb.index_fused`), the blind indexer
(`_blind_index`, which the default `_fanout` calls) and the per-frame known-cell indexer
(`_known_perframe`) are all injected through the seams the production code resolves lazily -- the
same pattern test_missbuf_rescue.py uses. Everything else, including `_integrate_one` and the real
`_fits`, is the production path.

Conventions matter here and are easy to get wrong: a GLINT cell matrix M has COLUMNS = real-space
axes in Angstrom, and `q @ M = hkl`. So a frame on lattice A is `hkl @ inv(A)`, and BOTH `_fits` and
`same_lattice(M, A)` are then meaningful against the same matrix. (Older fixtures in this directory
mix the real and reciprocal conventions because they never exercise the two tests together.)

Dual mode: `pytest experiments/test_retry_cascade.py`, or `python experiments/...` for PASS/FAIL.
"""
import inspect
import numpy as np

import glint.stream_driver as sd
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice
from glint.stream_driver import StreamDriver

SEED = 20260825
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
B = _rot(1.91, 0.42, 2.30) @ cell_to_Ar(52.0, 61.0, 71.0, 90, 90, 90)   # a DIFFERENT lattice
JUNK = _rot(2.71, 1.34, 0.19) @ cell_to_Ar(53.7, 61.3, 71.9, 90, 90, 90)   # fits nothing


def frame_on(M, rng, n=40):
    """q-vectors lying exactly on lattice M, so q @ M == hkl and _fits(q, M) is total."""
    hkl = rng.integers(-5, 6, size=(4 * n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0][:n]
    return hkl @ np.linalg.inv(M)


# ----------------------------------------------------------------- driver + seams ----------------
class _FakeRGB:
    """Stand-in for glint.replica_gpu_batch: the BATCHED known-cell indexer. Returns the correct
    registration for frames tagged 'ok' and a junk one (0 inliers -> fails _fits) for the rest --
    which is exactly the situation the cascade exists for."""

    def __init__(self, good):
        self.good = good                       # set of id() of the q arrays it registers correctly

    def index_fused(self, qs, Mc, B=1):
        return [np.asarray(Mc, float).copy() if id(q) in self.good else JUNK.copy() for q in qs]


def _driver(retry_cascade, adaptive_relock=False, **kw):
    drv = StreamDriver(A, PANELS, CLEN, WAVE, (NPX, NPX), dtype=np.uint16, B=8, dmin=DMIN,
                       use_gpu=False, adaptive_relock=adaptive_relock,
                       retry_cascade=retry_cascade, **kw)
    drv._blind_index = lambda q, k: []         # default: blind arm finds nothing
    drv._known_perframe = lambda q, Mc: None   # default: known-cell arm finds nothing
    return drv


def _instrument(drv):
    """Record every _integrate_one the driver performs, as (slot, cell_id), and call through."""
    calls = []
    real = drv._integrate_one

    def wrapped(i, M, grid, acc, cell_id=0, **kw):
        calls.append((i, cell_id))
        return real(i, M, grid, acc, cell_id=cell_id, **kw)
    drv._integrate_one = wrapped
    return calls


def _load(drv, frames, good=()):
    """Put `frames` in the ring and wire the batched indexer to register only `good` correctly."""
    sd.rgb = _FakeRGB({id(frames[k]) for k in good})
    for slot, q in enumerate(frames):
        drv._q[slot] = q
        drv._idx[slot] = slot
    drv._n = len(frames)
    drv.n_pushed = len(frames)


def _nbest_with(cell, position, rng, k=10):
    """An N-best list whose ONLY good candidate sits at `position`. Position > 0 is deliberate: an
    implementation that breaks out of the candidate loop after the first entry (the bug e5ec454
    fixed) silently degrades N-best to top-1 and fails these tests."""
    out = [(_rot(*rng.uniform(0, 3, 3)) @ cell_to_Ar(*rng.uniform(45, 95, 3), 90, 90, 90), 0.5)
           for _ in range(k)]
    out[position] = (np.asarray(cell, float).copy(), 0.9)
    return out


# ---------------------------------------------------------------------- tests --------------------
def test_fixture_is_a_real_gate_failure():
    """The premise: JUNK really does fail the live gate on a lattice-A frame, and A really passes."""
    rng = np.random.default_rng(SEED)
    q = frame_on(A, rng)
    d = _driver(False)
    assert d._fits(q, A), "a frame ON lattice A must clear its own gate"
    assert not d._fits(q, JUNK), f"JUNK must fail the gate, got {d._inliers(q, JUNK)}/{len(q)} inliers"
    assert not same_lattice(B, A) and not same_lattice(JUNK, A), "B/JUNK must be other lattices"


def test_flag_off_frame_is_missed():
    """Default: the gate-failing frame is refused and counted, and nothing retries it."""
    rng = np.random.default_rng(SEED + 1)
    q_ok, q_bad = frame_on(A, rng), frame_on(A, rng)
    drv = _driver(False)
    drv._blind_index = lambda q, k: _nbest_with(A, 2, rng)      # a retry COULD have saved it...
    drv._known_perframe = lambda q, Mc: np.asarray(Mc, float).copy()
    calls = _instrument(drv)
    _load(drv, [q_ok, q_bad], good=(0,))
    drv.flush()
    assert calls == [(0, 0)], f"only the well-registered frame should be integrated, got {calls}"
    assert drv.n_gate_rejected == 1, drv.n_gate_rejected
    assert drv.n_indexed == 1, drv.n_indexed
    s = drv.stats()
    assert "n_cascade_rescued" not in s, "cascade counters must not appear with the flag off"
    assert s["gate_rejected"] == 1


def test_flag_on_frame_indexed_exactly_once():
    """Flag on: the SAME frame is recovered by the blind N-best arm, integrated once, not a miss."""
    rng = np.random.default_rng(SEED + 1)                       # same frames as the test above
    q_ok, q_bad = frame_on(A, rng), frame_on(A, rng)
    drv = _driver(True)
    drv._blind_index = lambda q, k: _nbest_with(A, 2, rng)
    calls = _instrument(drv)
    _load(drv, [q_ok, q_bad], good=(0,))
    drv.flush()
    assert sorted(calls) == [(0, 0), (1, 0)], f"both frames integrated exactly once, got {calls}"
    assert drv.n_indexed == 2, drv.n_indexed
    assert drv.n_gate_rejected == 0, "a rescued frame must NOT also be counted as a miss"
    s = drv.stats()
    assert s["n_cascade_retried"] == 1 and s["n_cascade_rescued"] == 1, s
    assert s["n_cascade_by_arm"] == {"blind_nbest_k10": 1}, s["n_cascade_by_arm"]
    assert s["gate_rejected"] == 0 and s["indexed"] == 2


def test_arms_are_complementary():
    """The finding the issue rests on: the per-frame known-cell arm rescues frames the blind arm
    cannot. A cascade that stopped at the first blind arm would leave this frame missed."""
    rng = np.random.default_rng(SEED + 2)
    q_blind, q_known, q_dead = frame_on(A, rng), frame_on(A, rng), frame_on(A, rng)
    drv = _driver(True)
    drv._blind_index = lambda q, k: (_nbest_with(A, 3, rng) if q is q_blind else _nbest_with(B, 1, rng))
    drv._known_perframe = lambda q, Mc: (np.asarray(Mc, float).copy() if q is q_known else None)
    calls = _instrument(drv)
    _load(drv, [q_blind, q_known, q_dead])                      # NONE registers under the batch pass
    drv.flush()
    assert sorted(calls) == [(0, 0), (1, 0)], f"slot 2 is unrecoverable, got {calls}"
    s = drv.stats()
    assert s["n_cascade_retried"] == 3 and s["n_cascade_rescued"] == 2, s
    assert s["n_cascade_by_arm"] == {"blind_nbest_k10": 1, "known_perframe": 1}, s["n_cascade_by_arm"]
    assert s["gate_rejected"] == 1, "the frame no arm saves still takes the ordinary miss path"


def test_wrong_lattice_candidate_is_refused():
    """A blind candidate can FIT a frame and still be the wrong lattice. Accepting it would add a
    different crystal's reflections to this cell's merge -- corruption, not a recovered frame."""
    rng = np.random.default_rng(SEED + 3)
    q_b = frame_on(B, rng)                                      # a frame that is genuinely lattice B
    drv = _driver(True)
    drv._blind_index = lambda q, k: _nbest_with(B, 0, rng)      # ...and the blind arm nails it
    calls = _instrument(drv)
    _load(drv, [q_b])
    assert drv._fits(q_b, B), "premise: the candidate really does fit the frame"
    drv.flush()
    assert calls == [], "a candidate on a different lattice must not be integrated into cell 0"
    s = drv.stats()
    assert s["n_cascade_retried"] == 1 and s["n_cascade_rescued"] == 0, s
    assert s["gate_rejected"] == 1


def test_unrescued_frame_still_reaches_the_miss_buffer():
    """adaptive_relock path: the cascade runs BEFORE the miss buffer, so a rescued frame is not
    buffered and an unrescued one is -- the existing miss machinery is untouched by either."""
    rng = np.random.default_rng(SEED + 4)
    q_res, q_dead = frame_on(A, rng), frame_on(A, rng)
    drv = _driver(True, adaptive_relock=True, rescue_buffer=64, min_inliers=6)
    drv._blind_index = lambda q, k: (_nbest_with(A, 2, rng) if q is q_res else _nbest_with(JUNK, 0, rng))
    drv._known_index = lambda qs, Mn, B=1: [None] * len(qs)
    calls = _instrument(drv)
    _load(drv, [q_res, q_dead])
    drv.flush()
    assert calls == [(0, 0)], f"only the rescuable frame is integrated, got {calls}"
    buffered = [q for _, q in drv._missbuf]
    assert len(buffered) == 1 and np.array_equal(buffered[0], q_dead), \
        "exactly the unrescued frame reaches the miss buffer"
    assert drv.stats()["n_cascade_rescued"] == 1


def _counting_fanout(drv):
    """Wrap the default serial fan-out and record how many frames each call blind-solves."""
    calls = []

    def fan(Q, k):
        calls.append((len(Q), k))
        return [drv._blind_index(q, k) for q in Q]
    drv._fanout = fan
    return calls


def test_watchdog_reuses_the_cascade_blind_solve():
    """glint#145 review: with adaptive_relock AND retry_cascade on, an unrescued frame used to be
    blind-indexed TWICE -- once by the cascade at retry_nbest, once by the watchdog at warmup_nbest.
    The blind solve is the ~26 ms that dominates the retry, so the frames the cascade cannot help
    were the ones paying double for it. Exactly one solve per frame per flush now."""
    rng = np.random.default_rng(SEED + 5)
    frames = [frame_on(A, rng) for _ in range(4)]
    drv = _driver(True, adaptive_relock=True, min_inliers=6)
    drv._blind_index = lambda q, k: _nbest_with(B, 0, rng, k)   # fits, but wrong lattice -> no rescue
    drv._known_index = lambda qs, Mn, B=1: [None] * len(qs)
    calls = _counting_fanout(drv)
    _load(drv, frames)
    drv.flush()
    assert drv.stats()["n_cascade_rescued"] == 0, "premise: no frame is rescuable here"
    solved = sum(n for n, _ in calls)
    assert solved == len(frames), \
        f"one blind solve per frame per flush, got {solved} for {len(frames)} frames: {calls}"


def test_one_blind_solve_even_when_the_fanout_raises():
    """The failure path must not smuggle the double solve back in. When `_fanout` raises, the
    cascade falls back to solving each frame serially -- and if that result is not written back into
    `nbs` it is invisible to the cache, so `_watchdog_nbest` solves the frame AGAIN. Exactly one
    blind solve per frame, fan-out working or not.

    The fan-out raises only on its FIRST call so the assertion below measures the SOLVE COUNT rather
    than tripping over a second exception: a permanently dead fan-out also takes the watchdog down,
    which is pre-existing behaviour this PR does not change."""
    rng = np.random.default_rng(SEED + 9)
    frames = [frame_on(A, rng) for _ in range(4)]
    drv = _driver(True, adaptive_relock=True, min_inliers=6)
    solves, fan_calls = [], []

    def blind(q, k):
        solves.append(k)                                        # every ACTUAL blind solve, wherever from
        return _nbest_with(B, 0, rng, k)                        # fits, wrong lattice -> never rescued
    drv._blind_index = blind
    drv._known_index = lambda qs, Mn, B=1: [None] * len(qs)

    def flaky_fanout(Q, k):
        fan_calls.append(k)
        if len(fan_calls) == 1:
            raise RuntimeError("fan-out worker died")           # the cascade's call
        return [blind(q, k) for q in Q]                         # a later call would work fine
    drv._fanout = flaky_fanout

    _load(drv, frames)
    drv.flush()
    assert drv.stats()["n_cascade_rescued"] == 0, "premise: nothing is rescuable here"
    assert len(solves) == len(frames), \
        f"one blind solve per frame despite the failed fan-out, got {len(solves)}: {solves}"
    assert all(k == drv.retry_nbest for k in solves), \
        f"the surviving solves should be the cascade's, at retry_nbest: {solves}"
    assert len(fan_calls) == 1, \
        f"the watchdog must not fan out again for frames the cascade already solved: {fan_calls}"


def _scattered(rng, k):
    """N-best of nothing but junk: no two frames agree, so no consensus can form."""
    return [(_rot(*rng.uniform(0, 3, 3)) @ cell_to_Ar(*rng.uniform(45, 95, 3), 90, 90, 90), 0.5)
            for _ in range(k)]


def test_short_fanout_return_is_padded_not_dropped():
    """A fan-out that returns FEWER results than frames must not lose the tail. The original zip
    silently dropped those slots -- neither retried, nor integrated, nor returned to the miss path,
    just gone. The miss buffer is the direct witness: every frame must still arrive there."""
    rng = np.random.default_rng(SEED + 10)
    frames = [frame_on(A, rng) for _ in range(4)]
    drv = _driver(True, adaptive_relock=True, min_inliers=6, rescue_buffer=64)
    drv._blind_index = lambda q, k: _scattered(rng, k)          # junk -> nothing rescues, nothing relocks
    drv._known_index = lambda qs, Mn, B=1: [None] * len(qs)
    drv._fanout = lambda Q, k: [drv._blind_index(Q[0], k)]      # ONE result for four frames
    _load(drv, frames)
    drv.flush()
    s = drv.stats()
    assert s["n_cascade_retried"] == len(frames), \
        f"every frame must go through the cascade, got {s['n_cascade_retried']}"
    assert s["n_cascade_rescued"] == 0 and drv.n_relock == 0, (s, drv.n_relock)
    assert len(drv._missbuf) == len(frames), \
        f"all {len(frames)} frames must reach the miss path, only {len(drv._missbuf)} did"


def test_reused_candidates_still_drive_the_watchdog_relock():
    """The saving must not cost the watchdog its consensus. index_blind_nbest's N only truncates the
    final dedup loop, so top-`warmup_nbest` is a verbatim prefix of top-`retry_nbest` -- the watchdog
    votes on identical candidates and must still discover the new cell."""
    rng = np.random.default_rng(SEED + 6)
    frames = [frame_on(B, rng) for _ in range(5)]               # a recurring SECOND lattice
    drv = _driver(True, adaptive_relock=True, min_inliers=6)
    drv._blind_index = lambda q, k: [(B.copy(), 0.9)][:k]
    drv._known_index = lambda qs, Mn, B_=1: [None] * len(qs)
    calls = _counting_fanout(drv)
    _load(drv, frames)
    drv.flush()
    assert sum(n for n, _ in calls) == len(frames), "still one solve per frame"
    assert drv.n_relock == 1 and len(drv.extra) == 1, \
        f"the watchdog must still relock on the reused candidates ({drv.n_relock} relocks)"
    got = np.sort(np.linalg.norm(drv.extra[0]["Mc"], axis=0))
    assert np.allclose(got, np.sort(np.linalg.norm(B, axis=0)), rtol=0.05), got.tolist()


def test_reuse_is_identical_to_a_fresh_solve():
    """THE equivalence the reuse rests on, asserted directly rather than argued: what the watchdog
    gets from the cache must equal, element for element, what a fresh `_fanout(q, warmup_nbest)`
    would have handed it. That is only true because the cached list is SLICED back to warmup_nbest;
    handing over all retry_nbest candidates would give the consensus more cells per frame to vote on
    than it would ever have seen, which is a changed vote, not a saved solve."""
    rng = np.random.default_rng(SEED + 8)
    drv = _driver(True, adaptive_relock=True, min_inliers=6)
    ranked = {}

    def blind(q, k):
        """Mirrors index_blind_nbest's contract: one ranked list per frame, top-k is a PREFIX."""
        ranked.setdefault(id(q), _nbest_with(B, 0, rng, 10))
        return ranked[id(q)][:k]
    drv._blind_index = blind
    _counting_fanout(drv)
    slots = [0, 1, 2]
    for i in slots:
        drv._q[i] = frame_on(A, rng)
    cache = {i: blind(drv._q[i], drv.retry_nbest) for i in slots}

    reused = drv._watchdog_nbest(slots, cache)
    fresh = drv._watchdog_nbest(slots, None)
    assert len(reused) == len(fresh) == len(slots)
    for j, (r, f) in enumerate(zip(reused, fresh)):
        assert len(r) == drv.warmup_nbest, \
            f"slot {j}: watchdog got {len(r)} candidates, not warmup_nbest={drv.warmup_nbest}"
        assert len(r) == len(f), (len(r), len(f))
        for (cr, sr), (cf, sf) in zip(r, f):
            assert np.array_equal(cr, cf) and sr == sf, f"slot {j}: reused candidate != fresh one"


def test_larger_warmup_nbest_falls_back_rather_than_truncating():
    """warmup_nbest > retry_nbest: the cache is a prefix but too SHORT, and N is the only thing the
    truncation controls -- so 'topping up the delta' means re-running the whole solve anyway. This
    falls back to the ordinary fan-out for those frames instead: no saving in that configuration,
    but the watchdog's candidate set is preserved exactly, which is the property worth keeping."""
    rng = np.random.default_rng(SEED + 7)
    frames = [frame_on(A, rng) for _ in range(3)]
    drv = _driver(True, adaptive_relock=True, min_inliers=6, warmup_nbest=12, retry_nbest=3)
    drv._blind_index = lambda q, k: _nbest_with(B, 0, rng, k)
    drv._known_index = lambda qs, Mn, B=1: [None] * len(qs)
    calls = _counting_fanout(drv)
    _load(drv, frames)
    drv.flush()
    assert [k for _, k in calls] == [3, 12], f"cascade at 3, then a full watchdog solve at 12: {calls}"
    assert sum(n for n, _ in calls) == 2 * len(frames), "deliberately two solves in this regime"


def test_default_off_touches_nothing():
    """Off: no per-frame known-cell indexer is even resolved, and the stats key set is unchanged."""
    off, on = _driver(False), _driver(True)
    assert off.retry_cascade is False and on.retry_cascade is True
    # BY NAME, not by tail position. `__defaults__[-2:]` pinned these to the last two slots of the
    # signature, which breaks the moment anything else is appended -- and appending is the
    # documented policy for this constructor (it is not keyword-only, so inserting mid-signature
    # rebinds positional arguments; see test_lock_gate_wiring). glint#131's `bg_mode` landed after
    # these two and turned this into a false failure. The claim being made is about the DEFAULTS of
    # these two parameters, so say that.
    _sig = inspect.signature(StreamDriver.__init__).parameters
    assert _sig["retry_cascade"].default is False, "flag must default OFF"
    assert _sig["retry_nbest"].default is None, "retry_nbest must default to 'inherit nbest'"
    plain = StreamDriver(A, PANELS, CLEN, WAVE, (NPX, NPX), dtype=np.uint16, B=8, dmin=DMIN,
                         use_gpu=False)
    assert plain._known_perframe is None, "the per-frame indexer must not be resolved when off"
    sd.rgb = _FakeRGB(set())
    assert set(plain.stats()) == set(_driver(False).stats()), "stats key set changed with the flag off"


TESTS = (test_fixture_is_a_real_gate_failure, test_flag_off_frame_is_missed,
         test_flag_on_frame_indexed_exactly_once, test_arms_are_complementary,
         test_wrong_lattice_candidate_is_refused, test_unrescued_frame_still_reaches_the_miss_buffer,
         test_watchdog_reuses_the_cascade_blind_solve,
         test_one_blind_solve_even_when_the_fanout_raises,
         test_short_fanout_return_is_padded_not_dropped,
         test_reused_candidates_still_drive_the_watchdog_relock,
         test_reuse_is_identical_to_a_fresh_solve,
         test_larger_warmup_nbest_falls_back_rather_than_truncating,
         test_default_off_touches_nothing)

if __name__ == "__main__":
    ok = 0
    for t in TESTS:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"{ok}/{len(TESTS)} passed")
    raise SystemExit(0 if ok == len(TESTS) else 1)
