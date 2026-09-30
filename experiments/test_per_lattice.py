"""Per-lattice scoring in the StreamDriver (per_lattice=True): a double hit scored per lattice.

WHAT IS PINNED. In a double hit -- two crystals of the same cell in one shot -- the live gate's
denominator counts the second crystal's peaks against the first. With per_lattice=True the driver
(a) searches a frame that fails the live gate under every active cell for a second lattice in the peaks
its best rejected registration leaves (the shipped double-hit rule: same cell, >= 15 deg away), and
accepts the frame under the STRONGER of the two lattices when that one passes the live gate on the peaks
the other does not claim (outcome "rescued_per_lattice"); (b) searches an accepted frame whose lattice-1
share is below per_lattice_below and, when a second lattice is found, feeds the per-lattice fraction to
the QC low-confidence flag -- only when the frame's lattice is the stronger one -- and reports both
fractions. The recorder's per-lattice column (experiments/record_stream_replay.py) scores the kept
lattice on the peaks the other lattice does not claim; its half here skips where torch is missing.
Default off: nothing changes for existing callers.

FIXTURE. q-vectors exactly on two orientations of one cell 40 deg apart, plus uniform noise. The fixture
computes, with the driver's own deflation, how many peaks lattice 1 claims and how many of the rest
lattice 2 claims, and ASSERTS the case it is meant to be (whole-frame gate fails, per-lattice gate
passes, misorientation >= 15 deg) before any driver runs -- so a change in the numbers fails loudly
instead of silently testing something else. The known-cell indexer is the _MixedOracle of
test_cell_registry (returns lattice 1 for the double-hit frame); the second-lattice blind solve is a
closure that returns lattice 2 for exactly that residual.

CPU, numpy only (the torch-free CI job runs it).

  PYTHONPATH=. python experiments/test_per_lattice.py
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
from glint.multilattice import deflate_peaks, misorientation_deg              # noqa: E402
from glint.stream_driver import StreamDriver                                  # noqa: E402
from test_cell_registry import (A, B, CLEN, DMIN, NPX, PANELS, SEED, WAVE,  # noqa: E402
                                _MixedOracle, _driver, _key, _no_torch_needed, _terminal, _two_cells, frame_on)

A1 = A


def _rotate(M, deg, axis=(0.3, 0.8, 0.52)):
    ax = np.asarray(axis, float); ax /= np.linalg.norm(ax)
    t = np.radians(deg); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K) @ M


def _inliers(q, M, tol=0.15):
    h = np.asarray(q, float) @ np.asarray(M, float)
    return int((np.abs(h - np.rint(h)).max(1) < tol).sum())


def _double_hit_frame(n1, n2, noise, seed, deg=40.0):
    """q on A1 (n1) and on A1 rotated by `deg` (n2) plus uniform noise, and what the driver will see."""
    rng = np.random.default_rng(seed)
    A2 = _rotate(A1, deg)
    parts = [frame_on(A1, rng, n=n1), frame_on(A2, rng, n=n2)]
    if noise:
        v = rng.normal(size=(noise, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        parts.append(v * (0.14 * rng.random(noise) ** (1 / 3))[:, None])
    q = np.concatenate(parts)
    q = q[rng.permutation(len(q))]
    resid = deflate_peaks(q, A1)
    fx = types.SimpleNamespace(q=q, A2=A2, npk=len(q), n1=len(q) - len(resid), m2=_inliers(resid, A2),
                               misorientation=misorientation_deg(A1, A2), resid_key=_key(resid))
    fx.blind = lambda P, k: [(A2.copy(), 1.0)] if _key(P) == fx.resid_key else []
    return fx


def _live(n, npk, frac=0.15, nmin=10):
    return n >= nmin and n >= frac * npk


def _find(case, seed, tries=64, **kw):
    """The first frame, scanning seeds deterministically from `seed`, for which `case(fx)` holds --
    chance noise hits move the counts by a few peaks, and the tests must run on the case they name."""
    for t in range(tries):
        fx = _double_hit_frame(seed=seed + 1000 * t, **kw)
        if case(fx):
            return fx
    raise AssertionError(f"no seed from {seed} gives the case {case.__name__} for {kw}")


def _is_rescue_case(fx):
    return (fx.misorientation >= 15.0 and not _live(fx.n1, fx.npk)     # whole-frame live gate fails
            and fx.n1 >= fx.m2                                         # the registration in hand is the stronger
            and _live(fx.n1, fx.npk - fx.m2))                          # per-lattice live gate passes


def _is_swap_case(fx):
    return (fx.misorientation >= 15.0 and not _live(fx.n1, fx.npk)     # the registration in hand fails
            and fx.m2 > fx.n1                                          # the residual holds the STRONGER crystal
            and _live(fx.m2, fx.npk - fx.n1))                          # which passes per lattice


def _rescue_case(seed=SEED + 11):
    return _find(_is_rescue_case, seed, n1=20, n2=18, noise=134)


def _same_orientation(M, Mref, tol=0.2):
    U = np.linalg.solve(np.asarray(Mref, float), np.asarray(M, float))
    return bool(np.abs(U - np.rint(U)).max() < tol and abs(round(np.linalg.det(np.rint(U)))) == 1)


def _run_one(fx, pl, lattices=None, before=None, **kw):
    oracle = _MixedOracle(lattices or [A1, fx.A2])
    oracle.add_mixed(fx.q)
    drv, _ = _driver(oracle=oracle, per_lattice=pl, events=True, min_inliers=10, min_inlier_frac=0.15, **kw)
    drv._dh_index = fx.blind
    if before is not None:
        before(drv)
    drv.push_q(fx.q)
    drv.flush()
    return drv, _terminal(drv.events)


# ------------------------------------------------------------------ tests ------------------------
def test_signature_defaults_and_validation():
    import inspect
    params = list(inspect.signature(StreamDriver.__init__).parameters)
    assert params[-6:-4] == ["per_lattice", "per_lattice_below"], params[-7:]   # hits_only, rescue_pixels, effort, null_floor follow
    sig = inspect.signature(StreamDriver.__init__).parameters
    assert sig["per_lattice"].default is False and sig["per_lattice_below"].default == 0.25
    for bad in (-0.1, 1.5):
        try:
            _driver(per_lattice=True, per_lattice_below=bad)
        except ValueError as exc:
            assert "per_lattice_below" in str(exc)
        else:
            raise AssertionError(f"per_lattice_below={bad} accepted")
    drv, _ = _driver()                                   # default: no per-lattice keys, nothing searched
    st = drv.stats()
    assert not any(k.startswith(("n_per_lattice", "n_pl_", "pl_null")) for k in st), sorted(st)


def test_rescue_single_cell_path():
    fx = _rescue_case()
    drv, term = _run_one(fx, pl=False)
    assert term[0]["outcome"] == "gate_rejected" and "second_lattice" not in term[0], term[0]
    assert drv.n_indexed == 0
    drv, term = _run_one(fx, pl=True)
    e = term[0]
    assert e["outcome"] == "rescued_per_lattice", e["outcome"]
    sl = e["second_lattice"]
    assert (sl["kept"], sl["dominant"]) == ("first", "first"), sl
    assert sl["n1"] == fx.n1 and sl["m2"] == fx.m2 and sl["n_peaks"] == fx.npk, sl
    assert sl["frac_first"] < 0.15 <= sl["frac_per_lattice"], sl
    assert _same_orientation(e["M"], A1), "rescued under the registration in hand"
    assert abs(sl["misorientation"] - fx.misorientation) < 0.05 and sl["misorientation"] >= 15, sl
    st = drv.stats()
    assert (st["n_per_lattice_searched"], st["n_per_lattice_found"], st["n_per_lattice_rescued"],
            st["n_per_lattice_swapped"]) == (1, 1, 1, 0), st
    assert st["indexed"] == 1 and st["gate_rejected"] == 0, st
    assert st["n_pl_null"] == 1 and st["pl_null_found_rate"] == 0.0, st   # the 1st search is null-tested


def test_rescue_first_fit_multicell_path_before_the_watchdog():
    fx = _rescue_case(seed=SEED + 12)
    for pl, want, want_calls in ((True, "rescued_per_lattice", []), (False, "miss", [[0]])):
        calls = []

        def spy(drv):                                                  # count what reaches the watchdog
            orig = drv._watchdog
            drv._watchdog = lambda missed, cached=None: (calls.append(list(missed)), orig(missed, cached))[1]
        drv, term = _run_one(fx, pl=pl, before=spy, adaptive_relock=True, rescue_buffer=0)
        assert term[0]["outcome"] == want, (pl, term[0]["outcome"])
        assert calls == want_calls, (pl, calls)                        # rescued BEFORE the watchdog, or handed to it


def test_rescue_best_fit_path():
    fx = _rescue_case(seed=SEED + 13)
    rng = np.random.default_rng(SEED + 14)
    oracle = _MixedOracle([A1, fx.A2, B])
    drv = _two_cells(oracle, rng, assign="best", per_lattice=True)    # cells: lyso (A), other (B)
    oracle.add_mixed(fx.q)
    drv._dh_index = fx.blind
    drv.push_q(fx.q); drv.flush()
    term = _terminal(drv.events)
    ev = max(term)                                                     # the double-hit frame (last pushed)
    assert term[ev]["outcome"] == "rescued_per_lattice" and term[ev]["cell"] == 0, term[ev]


def test_no_second_lattice_no_rescue():
    fx = _rescue_case(seed=SEED + 15)
    fx.blind = lambda P, k: []                                         # the residual indexes to nothing
    drv, term = _run_one(fx, pl=True)
    assert term[0]["outcome"] == "gate_rejected" and "second_lattice" not in term[0], term[0]
    st = drv.stats()
    assert (st["n_per_lattice_searched"], st["n_per_lattice_found"], st["n_per_lattice_rescued"]) == (1, 0, 0), st


def test_mosaic_clone_is_not_a_second_lattice():
    def clone_case(fx):                                                # the rescue path runs, the clone is < 15 deg
        return fx.misorientation < 15.0 and not _live(fx.n1, fx.npk) and fx.m2 >= 10
    fx = _find(clone_case, SEED + 16, n1=10, n2=30, noise=100, deg=3.0)
    drv, term = _run_one(fx, pl=True)
    assert term[0]["outcome"] == "gate_rejected", term[0]["outcome"]
    assert drv.stats()["n_per_lattice_found"] == 0


def test_stronger_second_lattice_is_the_one_kept():
    """The registration in hand is the WEAKER crystal (or a wrong orientation of the right cell); the
    residual holds the frame's real crystal. Removing the stronger crystal's peaks from the weaker one's
    denominator would reward the wrong lattice, so the stronger one is kept and integrated."""
    fx = _find(_is_swap_case, SEED + 17, n1=12, n2=40, noise=90)
    drv, term = _run_one(fx, pl=True)
    e = term[0]
    assert e["outcome"] == "rescued_per_lattice", e["outcome"]
    sl = e["second_lattice"]
    assert (sl["kept"], sl["dominant"]) == ("second", "second"), sl
    assert abs(sl["frac_per_lattice"] - fx.m2 / (fx.npk - fx.n1)) < 1e-12, sl
    assert _same_orientation(e["M"], fx.A2), "integrated under the stronger, second lattice"
    assert _same_orientation(sl["M1"], A1) and _same_orientation(sl["M2"], fx.A2), "both lattices recorded"
    assert drv._pl_v[0] is None, "the slot's cache is cleared by the flush"
    st = drv.stats()
    assert (st["n_per_lattice_rescued"], st["n_per_lattice_swapped"]) == (1, 1), st


def test_double_hit_of_two_weak_crystals_is_not_rescued():
    fx = _find(lambda f: f.misorientation >= 15 and max(f.n1, f.m2) < 10 and min(f.n1, f.m2) >= 6,
               SEED + 21, n1=7, n2=8, noise=40)
    drv, term = _run_one(fx, pl=True)
    e = term[0]
    assert e["outcome"] == "gate_rejected", e["outcome"]
    assert e["second_lattice"]["kept"] is None and e["second_lattice"]["m2"] == fx.m2, e["second_lattice"]
    st = drv.stats()
    assert (st["n_per_lattice_found"], st["n_per_lattice_rescued"]) == (1, 0), st


def test_accepted_frame_qc_reads_the_per_lattice_fraction():
    def qc_case(fx):                                                   # accepted, searched, dominant, QC call flips
        return (fx.misorientation >= 15.0 and _live(fx.n1, fx.npk) and fx.n1 / fx.npk < 0.25
                and fx.n1 >= fx.m2 and fx.n1 / (fx.npk - fx.m2) - fx.n1 / fx.npk > 0.03)
    fx = _find(qc_case, SEED + 18, n1=30, n2=20, noise=80)
    frac, frac_pl = fx.n1 / fx.npk, fx.n1 / (fx.npk - fx.m2)
    thr = round((frac + frac_pl) / 2, 4)                               # between the two fractions
    assert frac < thr <= frac_pl, (frac, thr, frac_pl)
    chunks = {}
    for pl in (False, True):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "o.stream")
        drv, term = _run_one(fx, pl=pl, stream_out=path, qc_frac_threshold=thr)
        assert term[0]["outcome"] == "indexed", term[0]["outcome"]
        drv.close()
        txt = open(path).read()
        chunks[pl] = txt
        low = [ln for ln in txt.splitlines() if ln.startswith("glint/low_confidence")]
        assert len(low) == 1, low
        if pl:
            assert term[0]["second_lattice"]["kept"] == "first" == term[0]["second_lattice"]["dominant"]
            assert low[0].split("=")[1].strip() in ("0", "False", "false"), low       # above thr per lattice: confident
            assert any(ln.startswith("glint/matched_frac_per_lattice") for ln in txt.splitlines())
            assert any(ln.startswith("glint/second_lattice_deg") for ln in txt.splitlines())
            st = drv.stats()
            assert (st["n_per_lattice_searched"], st["n_per_lattice_found"], st["n_per_lattice_rescued"]) == (1, 1, 0), st
        else:
            assert low[0].split("=")[1].strip() in ("1", "True", "true"), low         # below thr whole-frame: flagged
            assert "matched_frac_per_lattice" not in txt and "second_lattice_deg" not in txt


def test_qc_never_credits_the_weaker_lattice():
    """Accepted under the weaker crystal: the per-lattice fraction belongs to the stronger one, so the QC
    flag keeps the whole-frame fraction and the chunk carries no per-lattice fraction."""
    def weak_case(fx):
        return fx.misorientation >= 15.0 and _live(fx.n1, fx.npk) and fx.n1 / fx.npk < 0.25 and fx.m2 > fx.n1
    fx = _find(weak_case, SEED + 22, n1=24, n2=40, noise=60)
    thr = round(fx.n1 / fx.npk + 0.02, 4)                              # just above the whole-frame fraction
    d = tempfile.mkdtemp(); path = os.path.join(d, "o.stream")
    drv, term = _run_one(fx, pl=True, stream_out=path, qc_frac_threshold=thr)
    drv.close()
    sl = term[0]["second_lattice"]
    assert (sl["kept"], sl["dominant"]) == ("first", "second"), sl
    txt = open(path).read()
    low = [ln for ln in txt.splitlines() if ln.startswith("glint/low_confidence")]
    assert low and low[0].split("=")[1].strip() in ("1", "True", "true"), low       # flagged, not upgraded
    assert "matched_frac_per_lattice" not in txt and "second_lattice_deg" in txt


def test_accepted_frame_above_the_bar_is_not_searched():
    fx = _find(lambda f: f.n1 / f.npk >= 0.25, SEED + 19, n1=40, n2=20, noise=20)
    drv, term = _run_one(fx, pl=True)
    assert term[0]["outcome"] == "indexed" and "second_lattice" not in term[0]
    assert drv.stats()["n_per_lattice_searched"] == 0


def test_double_hit_and_per_lattice_share_one_search():
    fx = _find(lambda f: f.misorientation >= 15 and _live(f.n1, f.npk) and f.n1 / f.npk < 0.25 and f.m2 >= 10,
               SEED + 20, n1=24, n2=40, noise=60)
    for pl in (False, True):
        oracle = _MixedOracle([A1, fx.A2]); oracle.add_mixed(fx.q)
        sd.rgb = oracle
        with _no_torch_needed():                                       # double_hit imports the blind indexer
            drv = StreamDriver(A, PANELS, CLEN, WAVE, (NPX, NPX), dtype=np.uint16, B=8, dmin=DMIN, use_gpu=False,
                               double_hit=True, per_lattice=pl, events=True, min_inliers=10, min_inlier_frac=0.15)
        drv._known_index = oracle.index_fused
        calls = []
        drv._dh_index = lambda P, k, _f=fx.blind: calls.append(len(P)) or _f(P, k)
        drv.push_q(fx.q); drv.flush()
        term = _terminal(drv.events)
        st = drv.stats()
        assert st["n_double"] == 1 and st["n_dh_null"] == 1, st         # the double-hit counters as before
        real_calls = [c for c in calls if c == fx.npk - fx.n1]
        assert len(real_calls) == 2, calls                               # one real search + one null draw
        assert ("second_lattice" in term[0]) == pl, term[0]
        if pl:
            assert st["n_per_lattice_searched"] == 1 and st["n_per_lattice_found"] == 1, st



# The class the double-hit gate is handed, per driver configuration (StreamDriver._sl_laue): a class
# the caller or the stream header GAVE reaches the gate, canonicalised; the 4/mmm merge fallback and a
# label attached to explicit ops do not (the gate then infers the symmetry itself).
_LAUE_CASES = [(dict(), None),
               (dict(laue="4/m"), "4/m"),
               (dict(stream_symmetry=dict(lattice_type="tetragonal", centering="P", unique_axis="c")), "4/mmm"),
               (dict(ops=sd.laue_ops("4/m"), laue="4/m"), None)]


def _verdict_spy():
    """Wrap multilattice.second_lattice_verdict (the driver imports it at call time) and record, per
    call, the calling driver method and the `laue` it passed ("MISSING" if the keyword was dropped)."""
    import glint.multilattice as ml
    real, seen = ml.second_lattice_verdict, []

    def spy(*a, **kw):
        seen.append((sys._getframe(1).f_code.co_name, kw.get("laue", "MISSING")))
        return real(*a, **kw)
    return ml, real, spy, seen


def test_every_verdict_call_site_gets_the_drivers_class():
    """All three second_lattice_verdict call sites -- the search itself (_second_lattice), the double-hit
    null (_integrate_one) and the per-lattice rescue's null (_per_lattice_rescue) -- forward the class
    _sl_laue() returns. Checking _sl_laue() alone would pass with a call site that stopped forwarding it
    (glint#207 review): here each site is reached through the driver and its argument is read."""
    dh = _find(lambda f: f.misorientation >= 15 and _live(f.n1, f.npk) and f.n1 / f.npk < 0.25 and f.m2 >= 10,
               SEED + 20, n1=24, n2=40, noise=60)
    rescue = _rescue_case()
    for kw, want in _LAUE_CASES:
        ml, real, spy, seen = _verdict_spy()
        ml.second_lattice_verdict = spy
        try:
            oracle = _MixedOracle([A1, dh.A2]); oracle.add_mixed(dh.q)          # double_hit: search + its null
            sd.rgb = oracle
            with _no_torch_needed():
                drv = StreamDriver(A, PANELS, CLEN, WAVE, (NPX, NPX), dtype=np.uint16, B=8, dmin=DMIN,
                                   use_gpu=False, double_hit=True, min_inliers=10, min_inlier_frac=0.15, **kw)
            drv._known_index = oracle.index_fused
            drv._dh_index = dh.blind
            drv.push_q(dh.q); drv.flush()
            assert drv.stats()["n_dh_null"] == 1, drv.stats()
            sites_dh = sorted({c for c, _ in seen})
            n_dh = len(seen)
            _run_one(rescue, pl=True, **kw)                                      # per_lattice: rescue + its null
            sites_pl = sorted({c for c, _ in seen[n_dh:]})
        finally:
            ml.second_lattice_verdict = real
        assert sites_dh == ["_integrate_one", "_second_lattice"], (kw, seen)
        assert sites_pl == ["_per_lattice_rescue", "_second_lattice"], (kw, seen)
        got = {laue for _, laue in seen}
        assert got == {want}, f"{kw}: call sites passed {sorted(map(str, got))}, expected {want!r} ({seen})"

def _coincident(M1, M2, k=6, tol=0.1):
    """k reciprocal points of M1 that M2 also explains (componentwise within tol), nearest the origin
    first -- the peaks a double hit's two lattices can both claim."""
    r = int(np.ceil(1.0 / DMIN * 79.02)) + 1
    g = np.arange(-r, r + 1)
    H = np.stack(np.meshgrid(g, g, np.arange(-20, 21), indexing="ij"), -1).reshape(-1, 3).astype(float)
    Q = H @ np.linalg.inv(np.asarray(M1, float))
    n = np.linalg.norm(Q, axis=1)
    Q = Q[(n > 0) & (n <= 1.0 / DMIN)]
    h2 = Q @ np.asarray(M2, float)
    C = Q[np.abs(h2 - np.rint(h2)).max(1) < tol]
    C = C[np.argsort(np.linalg.norm(C, axis=1), kind="stable")][:k]
    assert len(C) == k, f"only {len(C)} coincident points"
    return C


def test_recorder_per_lattice_gate_scores_one_peak_set():
    """The recorder's per-lattice column scores the kept lattice with the published gate on the frame's
    peaks minus the OTHER lattice's claim, so its numerator never counts a peak its denominator dropped.
    Peaks both lattices explain belong to lattice 1 (the deflation's order): kept under lattice 2 (a
    swapped rescue) they are out of the score; kept under lattice 1 they stay in it."""
    try:                                           # the guard covers ONLY the optional dependency
        import torch                               # noqa: F401  record_stream_replay imports glint_fast
    except ImportError as exc:
        print(f"  SKIP  recorder half: optional dependency missing ({exc})")
        return
    import record_stream_replay as rsr             # NOT inside the guard: a regression must fail loudly
    from glint.glint_fast import matched_strict
    from glint.multilattice import claimed_mask
    fx = _find(_is_swap_case, SEED + 17, n1=12, n2=40, noise=90)
    C = _coincident(A1, fx.A2)
    q = np.concatenate([fx.q, C])
    c1 = claimed_mask(q, A1)
    c2 = ~c1 & claimed_mask(q, fx.A2)
    assert c1[-len(C):].all() and claimed_mask(C, fx.A2).all(), "the fixture's shared peaks"
    # kept under lattice 2: scored on what lattice 1 leaves -- the shared peaks are lattice 1's
    ok, m, frac, nsc = rsr.strict_gate_per_lattice(fx.A2, q, A1, dict(M1=A1, M2=fx.A2, dominant="second"))
    assert nsc == int((~c1).sum()) == len(deflate_peaks(q, A1)), (nsc, int((~c1).sum()))
    assert m == matched_strict(fx.A2, q[~c1]) and abs(frac - m / nsc) < 1e-12, (m, frac, nsc)
    assert matched_strict(fx.A2, q) >= m + len(C), "the whole-frame count includes the shared peaks"
    # kept under lattice 1: only lattice 2's share of the residual leaves the score
    ok1, m1, frac1, n1s = rsr.strict_gate_per_lattice(A1, q, A1, dict(M1=A1, M2=fx.A2, dominant="first"))
    assert n1s == len(q) - int(c2.sum()) and m1 == matched_strict(A1, q[~c2]), (n1s, m1)
    assert matched_strict(A1, q[~c2]) == matched_strict(A1, q), "no lattice-1 peak leaves its own score"


def test_report_refuses_replays_of_different_frames():
    """per_lattice_report aligns the two runs' records by i, so it must refuse two runs whose i-th frames
    differ -- same input digests, but another schedule or another frame order -- and accept identical ones."""
    import json
    import per_lattice_report as plr

    def run(kw, recs, schedule=None, refs=None, dirty=False):
        h = dict(inputs=[dict(name="lyso", digest="bf3422", kind="q", n=len(recs))], driver_kw=kw,
                 refs=refs or dict(lyso=[79.02, 79.02, 37.98, 90, 90, 90]), roster=[["lyso", [79.02, 79.02, 37.98, 90, 90, 90]]],
                 gate=dict(frac=0.25, min_refl=10), ingest="q", geometry=None,
                 provenance=dict(git="x", git_dirty=dirty), schedule=schedule, primary_cell=[79.02, 79.02, 37.98, 90, 90, 90],
                 totals=dict(strict_ok=0, indexed=len(recs), miss=0), counters=dict(n_watchdog_rescued=0, n_relock=0))
        return dict(header=h, records=recs)

    def rec(i, j):
        return dict(i=i, truth="lyso", src_index=j, o="indexed", ok=False, cell_name="lyso", npk=20, frac=0.1)

    same = [rec(0, 0), rec(1, 1), rec(2, 2)]
    swapped = [rec(0, 1), rec(1, 0), rec(2, 2)]
    with tempfile.TemporaryDirectory() as d:
        def write(name, obj):
            path = os.path.join(d, name); json.dump(obj, open(path, "w")); return path
        base = write("b.json", run(dict(B=20), same))
        args = ["--base", base, "--input", "unused.txt", "--null", "0", "--out", os.path.join(d, "r.json")]
        plr.main(args + ["--pl", write("p.json", run(dict(B=20, per_lattice=True), same, dirty=True))])   # accepted
        rep = json.load(open(os.path.join(d, "r.json")))
        assert rep["git"]["base_dirty"] is False and rep["git"]["pl_dirty"] is True, rep["git"]
        for bad, why in ((run(dict(B=20, per_lattice=True), swapped), "frame order"),
                         (run(dict(B=20, per_lattice=True), same, schedule=dict(seed=7)), "schedule"),
                         (run(dict(B=20, per_lattice=True), same, refs=dict(lyso=[78.0, 78.0, 37.0, 90, 90, 90])),
                          "reference cell")):
            try:
                plr.main(args + ["--pl", write("p.json", bad)])
            except AssertionError:
                continue
            raise AssertionError(f"report accepted replays that differ in {why}")


TESTS = [test_signature_defaults_and_validation,
         test_rescue_single_cell_path,
         test_rescue_first_fit_multicell_path_before_the_watchdog,
         test_rescue_best_fit_path,
         test_no_second_lattice_no_rescue,
         test_mosaic_clone_is_not_a_second_lattice,
         test_stronger_second_lattice_is_the_one_kept,
         test_double_hit_of_two_weak_crystals_is_not_rescued,
         test_accepted_frame_qc_reads_the_per_lattice_fraction,
         test_qc_never_credits_the_weaker_lattice,
         test_accepted_frame_above_the_bar_is_not_searched,
         test_double_hit_and_per_lattice_share_one_search,
         test_every_verdict_call_site_gets_the_drivers_class,
         test_recorder_per_lattice_gate_scores_one_peak_set,
         test_report_refuses_replays_of_different_frames]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t(); print(f"  PASS  {t.__name__}")
        except Exception as exc:                                        # noqa: BLE001 -- report, keep going
            failed += 1; print(f"  FAIL  {t.__name__}: {exc!r}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
