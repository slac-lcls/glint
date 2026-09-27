"""CPU tests for StreamDriver's hit-only ring (hits_only) and the pixels kept for its rescues (rescue_pixels).

WHY THESE EXIST. The driver keeps a ring of B frames resident on the device, indexes them as one batch and
integrates each against its pixels. Two things in that arrangement cost memory or data:

  * every pushed frame took a slot, including a frame whose peak list is too short to index. flush()
    skips those, but they still fill the ring: at a 10 % hit rate a B=120 ring holds about 12 frames worth
    indexing, the batch runs at small-batch speed, and 90 % of the ring's memory holds frames that need
    nothing more. hits_only=True gives such a frame's slot back at once, so B counts hits.
  * the warm-up rescue and the relock rescue of the miss buffer re-index a frame long after its batch left
    the ring, so they could only COUNT it -- the pixels were gone and the frame never reached the merge.
    rescue_pixels=N keeps up to N of those frames on the device until their rescue runs, and a frame the
    rescue accepts is integrated.

WHAT IS CHECKED, on rendered pixel frames through the real peak finder and integrator (numpy path):
  * hits_only: every index call gets B hits whatever the blanks between them; the per-frame outcomes and
    the merge are identical to the default ring's on the same stream (only the batch boundaries move);
    blanks keep their `blank` record and arrival index; the ring never holds a blank. The default ring
    still gives blanks a slot, so the index-call check fails if hits_only is ignored.
  * rescue_pixels, relock: the frames of a new cell that missed before the relock are integrated into the
    new cell's merge (n_rescued_integrated), each `integrated` record after its `rescued_relock` record;
    with rescue_pixels=0 the same rescue stays index-only. A store smaller than the misses evicts the
    oldest frames, which then stay index-only.
  * rescue_pixels, warm-up: per-frame warm-up (push) and warmup_batch both integrate the rescued warm-up
    frames into the primary merge.
  * a frame the watchdog integrates on the spot has its kept pixels released, so a later relock cannot
    integrate it twice.
  * rescue_pixels is validated, and the defaults build no store and add no stats keys.

No GPU, no torch: the known-cell and blind indexers are a fit oracle injected through the seams
test_cell_registry.py uses. GLINT_TEST_GPU=1 runs the same tests on the device path instead (cupy ring,
peak finder, fused integrator and pixel store; needs cupy and a GPU).

  PYTHONPATH=. python experiments/test_stream_hit_ring.py
  GLINT_TEST_GPU=1 PYTHONPATH=. python experiments/test_stream_hit_ring.py      # on a GPU node
"""
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import glint.stream_driver as sd                                             # noqa: E402
from glint.geom import read_crystfel_peaks                                   # noqa: E402
from glint.multishot import same_lattice                                      # noqa: E402
from test_cell_registry import (A, B, CLEN, DMIN, NPX, PANELS, SEED, WAVE,   # noqa: E402
                                _driver, _pixel_frame, _terminal)

PF = dict(abs_thr=200.0, son_min=5.0, min_pix=2)
GPU = os.environ.get("GLINT_TEST_GPU") == "1"
GRID = {k: sd.HKLGrid(M, DMIN, gpu=False, panels=PANELS, clen_m=CLEN, wavelength_A=WAVE)
        for k, M in (("A", A), ("B", B))}


class _FitOracle:
    """Known-cell and blind indexers that decide from the peaks: a frame is on lattice L when most of its
    q sit near-integer under L (the driver's own HKL_TOL). A pixel frame's q come from the peak finder, so
    they cannot be registered in advance the way test_cell_registry's _Oracle registers q-lists. `refuse`
    holds (call number, position) pairs the known-cell pass returns None for -- a batched miss the blind
    watchdog can still rescue. `calls` logs the batch size of every known-cell call."""

    def __init__(self, lattices=(A, B), refuse=()):
        self.lattices = [np.asarray(L, float) for L in lattices]
        self.refuse = set(refuse)
        self.calls = []

    def _on(self, q):
        q = np.asarray(q, float)
        for L in self.lattices:
            hf = q @ L
            if len(q) and (np.abs(hf - np.round(hf)).max(1) < sd.HKL_TOL).mean() >= 0.5:
                return L
        return None

    def index_fused(self, qs, Mc, B=1):
        c = len(self.calls)
        self.calls.append(len(qs))
        out = []
        for j, q in enumerate(qs):
            L = self._on(q)
            ok = L is not None and same_lattice(L, Mc) and (c, j) not in self.refuse
            out.append(L.copy() if ok else None)
        return out

    def blind(self, q, k):
        L = self._on(q)
        return [(L.copy(), 1.0)] if L is not None else []


def _frame(lat, rng):
    img, _ = _pixel_frame(types.SimpleNamespace(grid=GRID[lat]), A if lat == "A" else B, rng)
    return img


def _blank(rng):
    return np.clip(rng.normal(0.0, 3.0, (NPX, NPX)), 0, None).astype(np.float32)


def _same(s1, s2):
    """Merge stats equal key for key; NaN (completeness without a theoretical count) equals NaN."""
    return s1.keys() == s2.keys() and all(
        s1[k] == s2[k] or (isinstance(s1[k], float) and np.isnan(s1[k]) and np.isnan(s2[k])) for k in s1)


def _drv(Mc=A, oracle=None, **kw):
    oracle = oracle or _FitOracle()
    kw.setdefault("pf_kw", PF)
    kw.setdefault("use_gpu", GPU)
    return _driver(Mc=Mc, oracle=oracle, **kw)


# ------------------------------------------------------------------ hits_only ----------------------
PLAN = "HbbHHbHbbbHHHbHHbH"          # 10 hits, 8 blanks


def _run_plan(hits_only, plan=PLAN, B=4):
    rng = np.random.default_rng(SEED)
    frames = [_frame("A", rng) if c == "H" else _blank(rng) for c in plan]
    drv, oracle = _drv(B=B, events=True, hits_only=hits_only)
    held = []
    for img in frames:
        drv.push(img)
        held.append(drv._n)                         # frames resident in the ring after this push
    drv.close()
    return drv, oracle, held


def test_hits_only_batches_count_hits_and_nothing_else_changes():
    on, o_on, held_on = _run_plan(True)
    off, o_off, held_off = _run_plan(False)
    hits = PLAN.count("H")
    assert o_on.calls == [4, 4, 2], o_on.calls                          # B hits per batch, remainder at close
    per_chunk = [PLAN[i:i + 4].count("H") for i in range(0, len(PLAN), 4)]
    assert o_off.calls == [n for n in per_chunk if n], (o_off.calls, per_chunk)   # default: blanks fill slots
    assert max(o_off.calls) < 4                                         # ...so no default batch is full here
    # the ring holds hits only: after every push, the resident count is the hits since the last flush
    n = 0
    for c, h in zip(PLAN, held_on):
        n = (n + (c == "H")) % 4
        assert h == n, (PLAN, held_on)
    # per-frame outcomes and the merge are the default ring's, frame for frame
    t_on, t_off = _terminal(on.events), _terminal(off.events)
    assert sorted(t_on) == list(range(len(PLAN))) == sorted(t_off)
    assert {k: (e["outcome"], e["cell"]) for k, e in t_on.items()} == \
        {k: (e["outcome"], e["cell"]) for k, e in t_off.items()}
    assert [k for k, e in t_on.items() if e["outcome"] == "blank"] == [k for k, c in enumerate(PLAN) if c == "b"]
    assert on.n_pushed == off.n_pushed == len(PLAN)
    assert on.n_indexed == off.n_indexed == on.n_integrated == off.n_integrated == hits, \
        (on.n_indexed, off.n_indexed, on.n_integrated, off.n_integrated)
    assert _same(on.acc.stats(), off.acc.stats()), (on.acc.stats(), off.acc.stats())
    assert on.acc.stats()["frames"] == hits


def test_hits_only_on_the_peaks_in_path():
    """push_q / push_peaks share the slot bookkeeping (_queue_q): a blank there gives its slot back too."""
    rng = np.random.default_rng(SEED + 1)
    drv, oracle = _drv(B=3, hits_only=True, events=True)
    q_on_a = GRID["A"].predict(A, PANELS, CLEN, WAVE, tol=0.004)
    fs, ss = np.asarray(q_on_a["fs"], float), np.asarray(q_on_a["ss"], float)
    ok = (fs > 0) & (fs < NPX - 1) & (ss > 0) & (ss < NPX - 1)
    fs, ss = fs[ok][:30], ss[ok][:30]
    for c in "HbHbbH" "bHH":
        if c == "H":
            drv.push_peaks(fs + rng.normal(0, 0.02, fs.size), ss + rng.normal(0, 0.02, ss.size))
        else:
            drv.push_q(np.zeros((3, 3)))
    drv.close()
    assert oracle.calls == [3, 2], oracle.calls
    assert drv.n_pushed == 9 and drv.n_indexed == 5 and drv.n_integrated == 0


# ------------------------------------------------------------------ rescue_pixels: relock ----------
def _relock_run(rescue_pixels, n_b=4, B=4):
    """Locked to A; a batch of A, then a batch of B frames that all miss and relock the watchdog to B."""
    rng = np.random.default_rng(SEED + 2)
    kw = dict(adaptive_relock=True, rescue_buffer=64, events=True, B=B)
    if rescue_pixels:
        kw["rescue_pixels"] = rescue_pixels
    drv, oracle = _drv(**kw)
    for _ in range(B):
        drv.push(_frame("A", rng))
    for _ in range(n_b):
        drv.push(_frame("B", rng))
    drv.close()
    return drv


def test_relock_rescue_integrates_the_kept_frames():
    with_px = _relock_run(16)
    without = _relock_run(0)
    for d in (with_px, without):
        assert d.n_relock == 1 and d.n_rescued == 4, (d.n_relock, d.n_rescued)
        assert [same_lattice(e["Mc"], B) for e in d.extra] == [True], [e["Mc"] for e in d.extra]
    s = with_px.stats()
    assert s["n_rescued_integrated"] == 4 and s["pixels_held"] == 0 and s["pixels_evicted"] == 0, s
    assert with_px.n_integrated == without.n_integrated + 4 == 8, (with_px.n_integrated, without.n_integrated)
    assert with_px.n_indexed == without.n_indexed == 4                  # a rescue is counted as a rescue, not re-indexed
    assert with_px.extra[0]["acc"].stats()["frames"] == 4, with_px.extra[0]["acc"].stats()
    assert without.extra[0]["acc"].stats()["frames"] == 0, without.extra[0]["acc"].stats()
    assert _same(with_px.acc.stats(), without.acc.stats())             # the primary merge is untouched
    assert "n_rescued_integrated" not in without.stats()
    # each rescued frame: terminal `miss`, then `rescued_relock`, then its `integrated` marker, in that order
    ev = with_px.events
    for k in range(4, 8):
        rows = [(j, e["outcome"], e["cell"], e.get("slot", "-")) for j, e in enumerate(ev) if e["ev"] == k]
        outs = [r[1] for r in rows]
        assert outs == ["miss", "rescued_relock", "integrated"], rows
        assert rows[2][2] == 1 and rows[2][3] is None, rows            # into cell 1, from no ring slot
    assert sum(e["outcome"] == "integrated" for e in ev) == 8
    assert len(_terminal(ev)) == 8                                      # still one terminal record per frame
    reg = {c["id"]: c["n_frames"] for c in with_px.stats()["cells"]}
    assert reg == {0: 4, 1: 4}, reg                                    # attributed once, not twice


def test_a_small_store_evicts_the_oldest_and_those_stay_index_only():
    d = _relock_run(2)
    s = d.stats()
    assert d.n_rescued == 4 and s["n_rescued_integrated"] == 2, s
    assert s["pixels_evicted"] == 2 and s["pixels_held"] == 0, s
    marks = sorted(e["ev"] for e in d.events if e["outcome"] == "integrated" and e["cell"] == 1)
    assert marks == [6, 7], marks                                      # the two most recent misses were kept


def test_a_watchdog_rescue_releases_the_kept_pixels():
    """The watchdog integrates a same-cell miss on the spot; its kept copy must go, or a relock that
    re-indexed it from the miss buffer would integrate the frame a second time."""
    rng = np.random.default_rng(SEED + 3)
    oracle = _FitOracle(refuse={(0, 1)})                               # the known-cell pass misses frame 1
    drv, _ = _drv(oracle=oracle, adaptive_relock=True, rescue_buffer=64, rescue_pixels=8, events=True, B=4)
    for _ in range(4):
        drv.push(_frame("A", rng))
    drv.flush()
    t = _terminal(drv.events)
    assert t[1]["outcome"] == "rescued_watchdog" and drv.n_integrated == 4, (t[1], drv.n_integrated)
    assert [ev for ev, _ in drv._missbuf] == [1]                        # buffered before the watchdog, as before
    assert len(drv._pix) == 0 and drv._pix.get(1) is None, len(drv._pix)


# ------------------------------------------------------------------ rescue_pixels: warm-up ---------
def _warm_run(rescue_pixels, n=6):
    rng = np.random.default_rng(SEED + 4)
    kw = dict(warmup_rescue=True, events=True, B=4, warmup_nbest=1, lock_support=2, lock_gap=1)
    if rescue_pixels:
        kw["rescue_pixels"] = rescue_pixels
    drv, _ = _drv(Mc=None, **kw)
    for _ in range(n):
        drv.push(_frame("A", rng))
    drv.close()
    return drv


def test_warmup_rescue_integrates_the_kept_frames():
    with_px = _warm_run(8)
    without = _warm_run(0)
    for d in (with_px, without):
        assert d.locked and d.n_warmup >= 2, (d.locked, d.n_warmup)
    nw = with_px.n_warmup
    assert with_px.n_warmup_rescued == without.n_warmup_rescued == nw
    s = with_px.stats()
    assert s["n_warmup_integrated"] == nw and s["pixels_held"] == 0, s
    assert with_px.n_integrated == without.n_integrated + nw, (with_px.n_integrated, without.n_integrated)
    assert with_px.acc.stats()["frames"] == without.acc.stats()["frames"] + nw
    for k in range(nw):                  # the locking frame's rescue record precedes its warmup_lock record
        outs = [e["outcome"] for e in with_px.events if e["ev"] == k]
        assert outs.count("rescued_warmup") == outs.count("integrated") == 1, (k, outs)
        assert outs.index("integrated") == outs.index("rescued_warmup") + 1, (k, outs)
    assert len(_terminal(with_px.events)) == 6


def test_warmup_rescue_keeps_stream_peaks_and_blind_stats_show_the_store():
    rng = np.random.default_rng(SEED + 40)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "warm.stream")
        drv, _ = _drv(Mc=None, warmup_rescue=True, rescue_pixels=8, stream_out=out, stream_peaks="all",
                      events=True, B=4, warmup_nbest=1, lock_support=2, lock_gap=1)
        drv.push(_frame("A", rng))
        s = drv.stats()
        assert s["locked"] is False and s["pixels_held"] == 1 and s["pixels_evicted"] == 0, s
        for _ in range(5):
            drv.push(_frame("A", rng))
        drv.close()
        chunks = {c["event"]: c for c in read_crystfel_peaks(out)}
        for k in range(drv.n_warmup):
            assert len(chunks[k]["peaks"]) > 0, (k, chunks[k])


def test_warmup_batch_keeps_the_picks_pixels():
    rng = np.random.default_rng(SEED + 5)
    stack = np.stack([_frame("A", rng) for _ in range(5)])
    out = {}
    for px in (0, 8):
        kw = dict(warmup_rescue=True, B=4, warmup_nbest=1, lock_support=2, lock_gap=1)
        if px:
            kw["rescue_pixels"] = px
        drv, _ = _drv(Mc=None, **kw)
        assert drv.warmup_batch(stack) is True
        out[px] = drv
    d = out[8]
    assert d.n_warmup_rescued == d.n_warmup == 5 and d.stats()["n_warmup_integrated"] == 5, d.stats()
    assert d.n_integrated == 5 and out[0].n_integrated == 0, (d.n_integrated, out[0].n_integrated)


def test_warmup_batch_keeps_stream_peaks_for_rescued_picks():
    rng = np.random.default_rng(SEED + 41)
    stack = np.stack([_frame("A", rng) for _ in range(5)])
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "warm_batch.stream")
        drv, _ = _drv(Mc=None, warmup_rescue=True, rescue_pixels=8, stream_out=out, stream_peaks="all",
                      B=4, warmup_nbest=1, lock_support=2, lock_gap=1)
        assert drv.warmup_batch(stack) is True
        drv.close()
        chunks = {c["event"]: c for c in read_crystfel_peaks(out)}
        for k in range(drv.n_warmup):
            assert len(chunks[k]["peaks"]) > 0, (k, chunks[k])


# ------------------------------------------------------------------ options -------------------------
def test_options_are_validated_and_off_by_default():
    d, _ = _drv()
    assert d.hits_only is False and d._pix is None
    for k in ("n_warmup_integrated", "n_rescued_integrated", "pixels_held", "pixels_evicted"):
        assert k not in d.stats(), k
    for bad in (-1, 1.5, True, "4"):
        try:
            _drv(adaptive_relock=True, rescue_buffer=8, rescue_pixels=bad)
        except ValueError as exc:
            assert "rescue_pixels" in str(exc)
        else:
            raise AssertionError(f"rescue_pixels={bad!r} accepted")
    for kw in (dict(), dict(adaptive_relock=True), dict(rescue_buffer=8)):   # nothing would read the pixels
        try:
            _drv(rescue_pixels=4, **kw)
        except ValueError as exc:
            assert "nothing would ever read them" in str(exc)
        else:
            raise AssertionError(f"rescue_pixels=4 accepted with {kw}")
    d, _ = _drv(adaptive_relock=True, rescue_buffer=8, rescue_pixels=3)
    assert len(d._pix._buf) == 3 and d._pix._buf[0].shape == (NPX, NPX)   # preallocated at construction


TESTS = [test_hits_only_batches_count_hits_and_nothing_else_changes,
         test_hits_only_on_the_peaks_in_path,
         test_relock_rescue_integrates_the_kept_frames,
         test_a_small_store_evicts_the_oldest_and_those_stay_index_only,
         test_a_watchdog_rescue_releases_the_kept_pixels,
         test_warmup_rescue_integrates_the_kept_frames,
         test_warmup_rescue_keeps_stream_peaks_and_blind_stats_show_the_store,
         test_warmup_batch_keeps_the_picks_pixels,
         test_warmup_batch_keeps_stream_peaks_for_rescued_picks,
         test_options_are_validated_and_off_by_default]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as exc:                                        # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(exc).__name__}({exc})")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed  ({'device path, cupy' if GPU else 'numpy path'})")
    sys.exit(1 if failed else 0)
