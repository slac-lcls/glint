"""CPU tests for the cell REGISTRY, the per-frame EVENT trace and the PEAKS-IN ingest (glint#199).

What the registry adds to StreamDriver: every active cell (the primary and each adaptive-relock extra)
now has a name -- from an optional `roster` of known cells, else cell<k> -- a source (given / warmup /
relock), a lock frame, a frame count, a last-seen index and a share of the recent window. `events=True`
records one terminal outcome per pushed frame plus retroactive rescue annotations; `push_q`/`push_peaks`
feed q-vectors or peak lists with no pixels, so the driver runs index-only (the DRP reducer->indexer
path) and the .stream carries the crystal with zero reflections.

None of this may change what the driver DOES at its defaults: the tests here pin the default stats()
key set, the slot defaults and the constructor order, and the whole existing CPU suite is the other
half of that guarantee.

No GPU, no torch. The batched known-cell indexer (`glint.stream_driver.rgb.index_fused`) and the blind
indexer (`_blind_index`) are injected through the seams the production code resolves lazily -- the
pattern of test_retry_cascade.py -- and `_known_index` is pointed at the same oracle because it is
bound from the module global at construction time. Conventions as in that file: a cell matrix M has
COLUMNS = real-space axes in Angstrom and `q @ M = hkl`.

Dual mode: `pytest experiments/test_cell_registry.py`, or `python experiments/...` for PASS/FAIL.
"""
import contextlib
import inspect
import os
import sys
import tempfile
import types

import numpy as np

import glint.stream_driver as sd
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice
from glint.stream_driver import StreamDriver

SEED = 20260918
NPX = 256
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
               cx=-(NPX / 2.0 - 0.5), cy=-(NPX / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=NPX - 1, min_ss=0, max_ss=NPX - 1)]
CLEN, WAVE, DMIN = 0.1, 1.322, 3.0
CELL_A = (79.0, 79.0, 38.0, 90, 90, 90)          # lysozyme-like, the primary
CELL_B = (52.0, 61.0, 71.0, 90, 90, 90)          # a different lattice, the relock


def _rot(a, b, c):
    def R(t, ax):
        s, k = np.sin(t), np.cos(t)
        m = np.eye(3); i, j = [x for x in range(3) if x != ax]
        m[i, i] = m[j, j] = k; m[i, j] = -s; m[j, i] = s
        return m
    return R(a, 0) @ R(b, 1) @ R(c, 2)


A = _rot(0.31, 0.77, 1.13) @ cell_to_Ar(*CELL_A)
B = _rot(1.91, 0.42, 2.30) @ cell_to_Ar(*CELL_B)
JUNK = _rot(2.71, 1.34, 0.19) @ cell_to_Ar(53.7, 61.3, 71.9, 90, 90, 90)


def frame_on(M, rng, n=40):
    """q-vectors exactly on lattice M (q @ M == hkl), so _fits(q, M) is total."""
    hkl = rng.integers(-5, 6, size=(4 * n, 3)).astype(float)
    hkl = hkl[np.abs(hkl).sum(1) > 0][:n]
    return hkl @ np.linalg.inv(M)


# ----------------------------------------------------------------- seams -------------------------
def _key(q):
    """Content key for a q array: push_q filters/copies its input, so identity cannot be the key."""
    return np.ascontiguousarray(np.round(np.asarray(q, float), 9)).tobytes()


class _Oracle:
    """Known-cell indexer that knows every frame's TRUE lattice: registers a frame correctly when
    asked for the cell it is on, and returns None (no registration -> a miss) for any other cell.
    None rather than a junk matrix, so a chance near-integer fit of an unrelated cell cannot leak a
    frame into the wrong accumulator and make these tests flaky (the live gate's chance floor is
    what test_retry_cascade exercises with its JUNK; here the bookkeeping is the subject)."""

    def __init__(self, default=None):
        self.truth = {}                          # content key -> true M
        self.default = default                   # lattice for frames never add()ed (pixel frames: the
                                                 # finder's centroids cannot be keyed in advance)

    def add(self, q, M):
        self.truth[_key(q)] = M
        return q

    def index_fused(self, qs, Mc, B=1):
        out = []
        for q in qs:
            M = self.truth.get(_key(q), self.default)
            out.append(M.copy() if M is not None and same_lattice(M, Mc) else None)
        return out

    def blind(self, q, k):
        """N-best blind result: the frame's own lattice, top-1."""
        M = self.truth.get(_key(q))
        return [(M.copy(), 1.0)] if M is not None else []


@contextlib.contextmanager
def _no_torch_needed():
    """Stand in for glint.glint_fast so a Mc=None driver's lazy import resolves without torch
    (test_lock_gate_wiring.py does the same)."""
    stub = types.ModuleType("glint.glint_fast")
    stub.index_blind_nbest = lambda q, k: []
    had = "glint.glint_fast" in sys.modules
    prev = sys.modules.get("glint.glint_fast")
    sys.modules["glint.glint_fast"] = stub
    try:
        yield
    finally:
        if had:
            sys.modules["glint.glint_fast"] = prev
        else:
            sys.modules.pop("glint.glint_fast", None)


def _driver(Mc=A, oracle_default=None, **kw):
    """A CPU driver wired to `oracle` for BOTH indexers. The oracle must be installed as sd.rgb
    BEFORE construction, because _known_index is bound from that module global in __init__."""
    oracle = _Oracle(default=oracle_default)
    sd.rgb = oracle
    kw.setdefault("B", 8); kw.setdefault("dmin", DMIN); kw.setdefault("use_gpu", False)
    if Mc is None:
        with _no_torch_needed():
            drv = StreamDriver(None, PANELS, CLEN, WAVE, (NPX, NPX), dtype=np.uint16, **kw)
    else:
        drv = StreamDriver(Mc, PANELS, CLEN, WAVE, (NPX, NPX), dtype=np.uint16, **kw)
    drv._blind_index = oracle.blind
    drv._known_index = oracle.index_fused
    return drv, oracle


ROSTER = {"lyso": CELL_A, "other": CELL_B}
TERMINAL = {"blank", "warmup_vote", "warmup_lock", "indexed", "rescued_watchdog", "rescued_cascade",
            "miss", "gate_rejected"}
RETRO = {"rescued_warmup", "rescued_relock"}


def _terminal(events):
    """ev -> the single terminal event; asserts exactly one per ev."""
    out = {}
    for e in events:
        if e["outcome"] in TERMINAL:
            assert e["ev"] not in out, f"two terminal outcomes for ev {e['ev']}: {out[e['ev']]['outcome']}, {e['outcome']}"
            out[e["ev"]] = e
    return out


# ----------------------------------------------------------------- tests -------------------------
def test_roster_names_primary_relock_and_fallback():
    rng = np.random.default_rng(SEED)
    drv, oracle = _driver(adaptive_relock=True, roster=ROSTER)
    r0 = drv.registry[0]
    assert (r0["name"], r0["source"], r0["locked_at"], r0["lock_generation"], r0["lock_z"]) == ("lyso", "given", 0, 0, None), r0
    # four frames on B -> fit no active cell -> watchdog votes B four times -> relock, named from the roster
    for slot in range(4):
        q = oracle.add(frame_on(B, rng), B)
        drv._q[slot] = q; drv._idx[slot] = slot
    drv._n = 4; drv.n_pushed = 4
    drv.flush()
    assert drv.n_relock == 1 and len(drv.extra) == 1, (drv.n_relock, len(drv.extra))
    assert drv.extra[0]["name"] == "other", drv.extra[0]["name"]
    r1 = drv.registry[1]
    assert (r1["id"], r1["name"], r1["source"], r1["lock_generation"], r1["locked_at"]) == (1, "other", "relock", 1, 4), r1
    cells = drv.stats()["cells"]
    assert [c["name"] for c in cells] == ["lyso", "other"], cells
    assert same_lattice(np.asarray(drv.extra[0]["Mc"]), B)
    # fallback names: a roster that does not know the relocked cell, and no roster at all
    drv2, oracle2 = _driver(adaptive_relock=True, roster={"lyso": CELL_A})
    for slot in range(4):
        drv2._q[slot] = oracle2.add(frame_on(B, rng), B); drv2._idx[slot] = slot
    drv2._n = 4; drv2.n_pushed = 4
    drv2.flush()
    assert drv2.extra[0]["name"] == "cell1", drv2.extra[0]["name"]
    drv3, _ = _driver(adaptive_relock=True)
    assert drv3.registry[0]["name"] == "cell0"


def test_registry_counts_and_recent_share_after_scripted_flush():
    rng = np.random.default_rng(SEED + 1)
    drv, oracle = _driver(adaptive_relock=True, roster=ROSTER, cell_window=4, rescue_buffer=0)
    for slot in range(5):                                   # five frames on the primary, arrival 10..14
        drv._q[slot] = oracle.add(frame_on(A, rng), A); drv._idx[slot] = 10 + slot
    drv._n = 5; drv.n_pushed = 15
    drv.flush()
    r0 = drv.registry[0]
    assert (r0["n_frames"], r0["last_seen"], len(r0["recent"])) == (5, 14, 4), r0
    assert drv.recent_share() == {0: 1.0}, drv.recent_share()
    # relock on B, then three B frames indexed into cell 1 by the cascade over active cells
    for slot in range(4):
        drv._q[slot] = oracle.add(frame_on(B, rng), B); drv._idx[slot] = 20 + slot
    drv._n = 4; drv.n_pushed = 24
    drv.flush()
    assert drv.n_relock == 1
    for slot in range(3):
        drv._q[slot] = oracle.add(frame_on(B, rng), B); drv._idx[slot] = 30 + slot
    drv._n = 3; drv.n_pushed = 33
    drv.flush()
    cells = drv.stats()["cells"]
    assert [c["n_frames"] for c in cells] == [5, 3], cells
    assert [c["last_seen"] for c in cells] == [14, 32], cells
    assert abs(cells[0]["recent_share"] - 0.25) < 1e-12 and abs(cells[1]["recent_share"] - 0.75) < 1e-12, cells
    total = sum(c["n_frames"] for c in cells)
    assert total == drv.n_indexed + drv.n_warmup_rescued + drv.n_rescued, (total, drv.n_indexed, drv.n_rescued)


def test_event_sequence_two_cells_relock_and_rescued_relock():
    rng = np.random.default_rng(SEED + 2)
    drv, oracle = _driver(adaptive_relock=True, rescue_buffer=64, events=True, roster=ROSTER, B=8)
    plan = [A, A, B, B, B, None, A, B]                      # None = a blank (3 rows, below min_peaks)
    for M in plan:
        if M is None:
            drv.push_q(np.zeros((3, 3)))
        else:
            drv.push_q(oracle.add(frame_on(M, rng), M))    # the 8th push fills the ring -> flush
    assert drv._n == 0 and drv.n_pushed == 8
    term = _terminal(drv.events)
    got = {ev: (e["outcome"], e["cell"]) for ev, e in term.items()}
    assert got == {0: ("indexed", 0), 1: ("indexed", 0), 6: ("indexed", 0), 5: ("blank", None),
                   2: ("miss", None), 3: ("miss", None), 4: ("miss", None), 7: ("miss", None)}, got
    relocks = [e for e in drv.events if e["outcome"] == "relock"]
    assert len(relocks) == 1 and (relocks[0]["cell"], relocks[0]["cell_name"], relocks[0]["source"]) == (1, "other", "relock"), relocks
    retro = sorted((e["ev"], e["cell"]) for e in drv.events if e["outcome"] == "rescued_relock")
    assert retro == [(2, 1), (3, 1), (4, 1), (7, 1)], retro
    assert drv.n_rescued == 4 and drv.n_relock == 1
    assert all(e["at"] >= e["ev"] for e in drv.events)
    for e in drv.events:
        if e["outcome"] in ("indexed", "rescued_relock"):
            assert e["M"] is not None and abs(e["frac"] - 1.0) < 1e-12, e
    # a second batch: B frames now index straight into cell 1, A into cell 0
    for M in [B, B, A]:
        drv.push_q(oracle.add(frame_on(M, rng), M))
    drv.flush()
    term = _terminal(drv.events)
    assert sorted(term) == list(range(11)), sorted(term)
    assert [(term[k]["outcome"], term[k]["cell"], term[k]["cell_name"]) for k in (8, 9, 10)] == \
        [("indexed", 1, "other"), ("indexed", 1, "other"), ("indexed", 0, "lyso")]
    cells = drv.stats()["cells"]
    assert [c["n_frames"] for c in cells] == [4, 6], cells           # 3 A + 1 A ; 4 rescued + 2 indexed
    assert drv.n_integrated == 0                                       # nothing had pixels


def test_gate_rejected_and_blank_in_single_cell_mode():
    rng = np.random.default_rng(SEED + 3)
    drv, oracle = _driver(adaptive_relock=False, events=True, B=8)
    for M in [A, B, A, None, B]:
        drv.push_q(np.zeros((2, 3)) if M is None else oracle.add(frame_on(M, rng), M))
    drv.flush()
    term = _terminal(drv.events)
    got = [term[k]["outcome"] for k in range(5)]
    assert got == ["indexed", "gate_rejected", "indexed", "blank", "gate_rejected"], got
    assert drv.n_gate_rejected == 2 and drv.n_indexed == 2
    cells = drv.stats()["cells"]                                       # events on -> the snapshot is reported
    assert len(cells) == 1 and cells[0]["n_frames"] == 2 and cells[0]["name"] == "cell0", cells


def test_push_q_blind_lock_then_index_only_with_zero_reflection_chunks():
    rng = np.random.default_rng(SEED + 4)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.stream")
        drv, oracle = _driver(None, events=True, B=8, stream_out=path)
        assert drv.locked is False and drv.registry == []
        for _ in range(3):                                              # three votes for A -> lock
            drv.push_q(oracle.add(frame_on(A, rng), A))
        assert drv.locked, drv.stats()
        blind = [e for e in drv.events if e["outcome"].startswith("warmup")]
        assert [e["outcome"] for e in blind] == ["warmup_vote", "warmup_vote", "warmup_lock"], blind
        assert all("support" in e and "lead" in e for e in blind)
        assert blind[-1]["cell"] == 0 and blind[-1]["n_active"] == 1
        assert drv.locked_after == 3 and drv.registry[0]["source"] == "warmup"
        for _ in range(8):                                              # index-only frames, one full ring
            drv.push_q(oracle.add(frame_on(A, rng), A))
        drv.close()
        st = drv.stats()
        assert st["indexed"] == 8 and st["integrated"] == 0 and drv._frame_no == 0, st
        assert st["stream_indexed"] == 8 and st["stream_chunks"] == 8, st
        assert st["cells"][0]["n_frames"] == 8 and st["cells"][0]["source"] == "warmup", st["cells"]
        txt = open(path).read()
        assert txt.count("--- Begin crystal") == 8
        assert txt.count("indexed_by = file") == 8 and "indexed_by = none" not in txt
        assert txt.count("num_reflections = 0") == 8, txt.count("num_reflections = 0")
        assert txt.count("glint/cell_id = 0") == 8 and txt.count("glint/lock_generation = 0") == 8
        assert "glint/frame_no" not in txt
        term = _terminal(drv.events)
        assert sorted(term) == list(range(11)) and all(term[k]["outcome"] == "indexed" for k in range(3, 11))
    # push_peaks below min_peaks is a blank; a valid peak list lands in a pixel-less slot
    drv, oracle = _driver(A, events=True, B=8)
    drv.push_peaks(np.array([10.0, 20.0]), np.array([30.0, 40.0]))
    assert drv.events[-1]["outcome"] == "blank" and drv._haspix[0] is False
    fs = np.linspace(20, 200, 12); ss = np.linspace(30, 210, 12)
    drv.push_peaks(fs, ss, intensity=np.ones(12))
    assert drv._haspix[1] is False and drv._q[1] is not None and len(drv._q[1]) == 12


def test_default_config_stats_key_set_unchanged():
    plain, _ = _driver(A)
    st = plain.stats()
    assert "cells" not in st, sorted(st)
    assert plain.events is None and plain._events_on is False and plain._roster is None
    assert plain._haspix == [True] * plain.B
    same, _ = _driver(A, events=False)
    assert set(same.stats()) == set(st)
    for kw in (dict(roster=ROSTER), dict(events=True), dict(adaptive_relock=True), dict(on_event=lambda r: None)):
        d, _ = _driver(A, **kw)
        assert "cells" in d.stats(), kw
        assert d.stats()["cells"][0]["source"] == "given"
    # the registry lives on the extra dicts too, as a name only -- everything else they carried stays
    # Extra-cell naming is covered by test_roster_names_primary_relock_and_fallback().
    # the bare-object stats() path of test_stream_gate_lock.py must not need any new attribute
    d = object.__new__(StreamDriver)
    d._blind = False; d.acc = plain.acc; d.locked_after = 0; d.consensus_support = 3
    d.n_theoretical = plain.n_theoretical; d.laue = plain.laue; d.consensus_members = 3
    d.n_indexed = d.n_integrated = d.n_pushed = d.n_gate_rejected = 0
    d.n_fanout_errors = d.n_fanout_missed = d.n_gate_refused = d.n_gate_deferred = 0
    d.warmup_rescue = d.retry_cascade = d.adaptive_relock = d.double_hit = False
    d.qc_frac_threshold = None; d._writer = None; d._grefiner = None
    assert "cells" not in d.stats()


def test_new_constructor_options_are_appended():
    params = list(inspect.signature(StreamDriver.__init__).parameters)
    assert params[-4:] == ["roster", "events", "on_event", "cell_window"], params[-6:]
    sig = inspect.signature(StreamDriver.__init__).parameters
    assert sig["roster"].default is None and sig["events"].default is False
    assert sig["on_event"].default is None and sig["cell_window"].default == 200


def _pixel_frame(drv, M, rng, n=36, amp=900.0, noise=3.0):
    """A detector frame whose peaks sit at M's predicted spot positions -- the pixel path's input."""
    pred = drv.grid.predict(M, PANELS, CLEN, WAVE, tol=0.004)
    fs, ss = np.asarray(pred["fs"], float), np.asarray(pred["ss"], float)
    exc = np.abs(np.asarray(pred["exc"], float))
    on = (fs > 6) & (fs < NPX - 7) & (ss > 6) & (ss < NPX - 7)
    order = np.argsort(np.where(on, exc, np.inf))[:n]        # the spots nearest the Ewald sphere: a peak's
    order = order[on[order]]                                 # q (on the sphere) then sits near-integer in M
    fs, ss = fs[order], ss[order]
    assert len(fs) >= 12, len(fs)
    img = rng.normal(0.0, noise, (NPX, NPX))
    yy, xx = np.mgrid[0:NPX, 0:NPX]
    for x, y in zip(fs, ss):
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.8 ** 2))
    return np.clip(img, 0, None).astype(np.float32), len(fs)


def test_pixel_path_emits_integrated_marker_with_slot_and_optional_q():
    """push(frame): one `indexed` terminal record per frame, then an `integrated` MARKER for the same ev
    carrying n_pred/n_refl/frame_no -- never a second terminal outcome. Frame events name their ring
    slot; q rides along only when the driver attribute events_keep_q is set (a recorder that never saw
    the pixels scores from it). The peaks come from PeakFinderV4 on the rendered frame, so the oracle
    answers with its default lattice."""
    rng = np.random.default_rng(SEED)
    for keep_q in (False, True):
        drv, _ = _driver(Mc=A, oracle_default=A, B=2, events=True, dmin=DMIN,
                         pf_kw=dict(abs_thr=200.0, son_min=5.0, min_pix=2))
        drv.events_keep_q = keep_q
        n_planted = []
        for _ in range(2):
            img, npl = _pixel_frame(drv, A, rng); n_planted.append(npl)
            drv.push(img)
        drv.close()
        term = _terminal(drv.events)
        assert set(term) == {0, 1} and all(e["outcome"] == "indexed" for e in term.values()), \
            [(e["ev"], e["outcome"]) for e in drv.events]
        for e in term.values():
            assert e["slot"] in (0, 1)
            assert e["n_peaks"] >= 12, e["n_peaks"]
            assert ("q" in e) == keep_q, (keep_q, list(e))
            if keep_q:
                assert len(e["q"]) == e["n_peaks"] and len(e["q"][0]) == 3
        marks = [e for e in drv.events if e["outcome"] == "integrated"]
        assert [m["ev"] for m in marks] == [0, 1], marks
        for m, npl in zip(marks, n_planted):
            assert m["n_refl"] >= 1 and m["n_pred"] >= m["n_refl"], m
            assert m["frame_no"] in (0, 1) and m["cell"] == 0 and m["slot"] in (0, 1)
        # the marker never counts as a frame outcome
        assert sum(1 for e in drv.events if e["outcome"] in TERMINAL) == 2
        assert drv.n_integrated == 2 and drv.stats()["integrated"] == 2
        idx = [i for i, e in enumerate(drv.events) if e["outcome"] == "indexed"]
        mk = [i for i, e in enumerate(drv.events) if e["outcome"] == "integrated"]
        assert all(a < b for a, b in zip(idx, mk)), "integrated must follow its indexed record"


def test_peaks_in_slots_feed_the_geometry_refiner():
    """push_peaks() under geom_refine=True: the observed peaks reach GeomRefiner (which predicts for
    itself), so live geometry refinement works from a peak list; without geom_refine nothing is kept."""
    rng = np.random.default_rng(SEED + 1)
    for on in (False, True):
        drv, oracle = _driver(Mc=A, oracle_default=A, B=2, dmin=DMIN, geom_refine=on,
                              geom_refine_kw=dict(update_every=2, min_frames=1) if on else None)
        pred = drv.grid.predict(A, PANELS, CLEN, WAVE, tol=0.02)
        fs, ss = np.asarray(pred["fs"], float), np.asarray(pred["ss"], float)
        keep = np.isfinite(fs) & np.isfinite(ss)
        fs, ss = fs[keep][:40], ss[keep][:40]
        for _ in range(2):
            drv.push_peaks(fs + rng.normal(0, 0.05, fs.size), ss + rng.normal(0, 0.05, ss.size))
        drv.close()
        assert drv.n_indexed == 2 and drv.n_integrated == 0, (drv.n_indexed, drv.n_integrated)
        if on:
            assert drv._grefiner is not None and drv._grefiner.n_frames == 2, drv._grefiner.n_frames
            assert "geom_correction" in drv.stats()
        else:
            assert drv._grefiner is None and "geom_correction" not in drv.stats()


def test_source_stamps_the_stream_chunk_and_blind_events_carry_q():
    """push(frame, src=(file, event)) / push_peaks(..., src=) name the frame's origin on its .stream chunk --
    `Image filename:` is the source, the Event line is written only when the source has an event (a
    one-image-per-file source gets none, as CrystFEL writes it) -- instead of the stream_image placeholder
    and the arrival index; and with events_keep_q the blind warm-up records carry q too."""
    rng = np.random.default_rng(SEED + 2)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "s.stream")
        drv, _ = _driver(Mc=A, oracle_default=A, B=2, events=True, dmin=DMIN, stream_out=out,
                         pf_kw=dict(abs_thr=200.0, son_min=5.0, min_pix=2))
        img, _ = _pixel_frame(drv, A, rng); drv.push(img, src=("run7/shot_0001.h5", None))
        img, _ = _pixel_frame(drv, A, rng); drv.push(img, src=("stack.cxi", 12))
        drv.close()
        txt = open(out).read()
        chunks = txt.split("----- Begin chunk -----")[1:]
        assert len(chunks) == 2, len(chunks)
        assert "Image filename: run7/shot_0001.h5\n" in chunks[0] and "Event:" not in chunks[0], chunks[0][:200]
        assert "Image filename: stack.cxi\nEvent: //12\n" in chunks[1], chunks[1][:200]
        assert "glint.cxi" not in txt
    # default: the placeholder image and the arrival index, byte-for-byte as before
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "s.stream")
        drv, _ = _driver(Mc=A, oracle_default=A, B=1, dmin=DMIN, stream_out=out,
                         pf_kw=dict(abs_thr=200.0, son_min=5.0, min_pix=2))
        img, _ = _pixel_frame(drv, A, rng); drv.push(img); drv.close()
        assert "Image filename: glint.cxi\nEvent: //0\n" in open(out).read()
    # blind warm-up records: q rides along under the same opt-in
    rng = np.random.default_rng(SEED + 3)
    for keep in (False, True):
        drv, oracle = _driver(Mc=None, B=4, events=True, dmin=DMIN, warmup_nbest=1, lock_support=2, lock_gap=1)
        drv.events_keep_q = keep
        for _ in range(3):
            q = oracle.add(frame_on(A, rng), A); drv.push_q(q)
        blind = [e for e in drv.events if e["outcome"] in ("warmup_vote", "warmup_lock")]
        assert blind, [e["outcome"] for e in drv.events]
        assert all(("q" in e) == keep for e in blind), (keep, [list(e) for e in blind])


TESTS = [test_roster_names_primary_relock_and_fallback,
         test_registry_counts_and_recent_share_after_scripted_flush,
         test_event_sequence_two_cells_relock_and_rescued_relock,
         test_gate_rejected_and_blank_in_single_cell_mode,
         test_push_q_blind_lock_then_index_only_with_zero_reflection_chunks,
         test_default_config_stats_key_set_unchanged,
         test_new_constructor_options_are_appended,
         test_pixel_path_emits_integrated_marker_with_slot_and_optional_q,
         test_peaks_in_slots_feed_the_geometry_refiner,
         test_source_stamps_the_stream_chunk_and_blind_events_carry_q]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as exc:                                       # noqa: BLE001 -- report, keep going
            failed += 1; print(f"  FAIL  {t.__name__}: {exc!r}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
