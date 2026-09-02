"""StreamDriver merges under the Laue class it is told, not under 4/mmm regardless (glint#180).

Before #180 `StreamDriver.__init__` set `self.ops = laue_ops_4mmm()` and handed that to every
MergeAccumulator, to theoretical_unique (the completeness denominator) and to the adaptive-relock
extras; `stream_symmetry` only reached the stream HEADER. So for any non-tetragonal cell the live
completeness/CC*/Rsplit were merged under the wrong group while the file said the right one.

CPU only (`use_gpu=False`), no pixels: the merge is fed through `drv.acc.add_frame`, the same seam
the integrate path uses, with synthetic reflection sets built from the cell's own hkl sphere. The
relock tests inject the blind fan-out and the per-frame indexer through the seams test_lock_probe.py
uses. What is asserted:
  * an orthorhombic set merged under laue="mmm" and under the default gives DIFFERENT unique
    (ASU-key) counts, the mmm count equals theoretical_unique(..., laue_ops("mmm")), and the
    default's equals the (wrong) 4/mmm denominator -- the numbers quoted in the PR;
  * the default path is BIT-IDENTICAL to the pre-#180 behaviour: stats() of a driver fed a
    tetragonal set equals, key for key, a MergeAccumulator built with laue_ops_4mmm() directly;
  * laue is derived from stream_symmetry when unset, an explicit conflict warns and does not fail,
    an explicit `ops` list is used verbatim;
  * tetragonal standardization of a blind/relocked cell is applied for 4/m and 4/mmm ONLY, on both
    the _lock path and the watchdog relock path, and the relock extras inherit the ops.

Run: `PYTHONPATH=. python experiments/test_streamdriver_laue.py` (exit 0/1) or `pytest`.
"""
import math
import warnings

import numpy as np

import glint.stream_driver as sd
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice
from glint.predict import _canonical_axes, _hkl_grid, recip_from_M
from glint.stream_driver import (MergeAccumulator, StreamDriver, laue_name, laue_ops, laue_ops_4mmm,
                                 theoretical_unique)

N = 64
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
               cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
CLEN, WAVE, DMIN = 0.1, 1.3, 3.0
SNR = (0.0, 1.0, 2.0, 3.0, 5.0)
ORTHO = cell_to_Ar(30.0, 40.0, 50.0, 90, 90, 90)            # columns = real axes a, b, c (A)
TET = cell_to_Ar(40.0, 40.0, 25.0, 90, 90, 90)


def _driver(Mc, **kw):
    return StreamDriver(Mc, PANELS, CLEN, WAVE, (N, N), dtype=np.uint16, B=8, dmin=DMIN,
                        use_gpu=False, snr_bins=SNR, **kw)


def _sphere(Mc, dmin=DMIN):
    """Every integer hkl with |q| <= 1/dmin for this cell (the set theoretical_unique counts)."""
    R = recip_from_M(Mc)
    g, q = _hkl_grid(R, 1.0 / dmin)
    return g[np.linalg.norm(q, axis=1) <= 1.0 / dmin]


def _stills(Mc, key_fn, rng, n_frames=24, frac=0.35):
    """Synthetic stills: each frame observes a random subset of the sphere with intensities drawn
    around a per-ASU truth (`key_fn` decides which hkl share a truth) plus noise, so CC1/2 is finite
    and R_split meaningful. Positive I everywhere so no row lands in the dropped snr<=0 bucket."""
    hkl = _sphere(Mc)
    truth = {}
    I_true = np.array([truth.setdefault(key_fn(h), 50.0 + 2000.0 * rng.exponential()) for h in hkl])
    out = []
    for _ in range(n_frames):
        pick = rng.random(len(hkl)) < frac
        I = I_true[pick] * (1.0 + 0.08 * rng.standard_normal(int(pick.sum())))
        I = np.maximum(I, 1.0)
        sig = np.sqrt(I) + 2.0
        out.append((hkl[pick], I, sig))
    return out


def _key_mmm(h):
    return tuple(abs(int(x)) for x in h)


def _key_4mmm(h):
    a, b = sorted((abs(int(h[0])), abs(int(h[1]))))
    return (a, b, abs(int(h[2])))


def _feed(acc, frames):
    for i, (hkl, I, sig) in enumerate(frames):
        acc.add_frame(hkl, I, sig, i)


def _same_ops(G1, G2):
    return len(G1) == len(G2) and all(np.array_equal(a, b) for a, b in zip(G1, G2))


def _same_M(a, b):
    """Exact equality of two 3x3 cell matrices (the settings under test are permutations and sign
    flips of one another, so exactness is the right bar -- no tolerance to hide a swap in)."""
    return np.array_equal(np.asarray(a, float), np.asarray(b, float))


def _eq(a, b):
    """Exact equality that treats NaN == NaN (stats() reports NaN for undefined FoMs)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


# ----------------------------------------------------------------------------------- the defect
def test_orthorhombic_set_merges_differently_under_mmm_and_the_default():
    frames = _stills(ORTHO, _key_mmm, np.random.default_rng(0))
    d_mmm, d_def = _driver(ORTHO, laue="mmm"), _driver(ORTHO)
    _feed(d_mmm.acc, frames); _feed(d_def.acc, frames)
    s_mmm, s_def = d_mmm.stats(), d_def.stats()
    n_mmm = theoretical_unique(ORTHO, DMIN, laue_ops("mmm"))
    n_wrong = theoretical_unique(ORTHO, DMIN, laue_ops_4mmm())
    # The point of the issue: the two groups partition the same measurements into a different
    # number of ASU keys. If `laue` were ignored these would be equal.
    assert s_mmm["unique"] != s_def["unique"], (
        f"laue='mmm' ignored: unique {s_mmm['unique']} (denominator {s_mmm['theoretical_unique']}) == "
        f"default {s_def['unique']} (denominator {s_def['theoretical_unique']}); mmm should count "
        f"against {n_mmm}, 4/mmm against {n_wrong}, on {s_mmm['measurements']} measurements")
    assert s_mmm["laue"] == "mmm" and s_def["laue"] == "4/mmm", (s_mmm["laue"], s_def["laue"])
    assert s_mmm["measurements"] == s_def["measurements"]         # same input, different keying
    # and the completeness DENOMINATOR follows the class too
    assert d_mmm.n_theoretical == n_mmm == s_mmm["theoretical_unique"], (d_mmm.n_theoretical, n_mmm)
    assert d_def.n_theoretical == n_wrong == s_def["theoretical_unique"], (d_def.n_theoretical, n_wrong)
    assert n_mmm > n_wrong, (n_mmm, n_wrong)
    # Whole sphere in one frame -> every ASU key observed: the mmm key count IS theoretical_unique
    hkl = _sphere(ORTHO)
    full = _driver(ORTHO, laue="mmm")
    full.acc.add_frame(hkl, np.full(len(hkl), 100.0), np.full(len(hkl), 5.0), 0)
    s = full.stats()
    assert s["unique"] == n_mmm and abs(s["completeness"] - 100.0) < 1e-9, (s["unique"], n_mmm, s["completeness"])
    full_def = _driver(ORTHO)
    full_def.acc.add_frame(hkl, np.full(len(hkl), 100.0), np.full(len(hkl), 5.0), 0)
    assert full_def.stats()["unique"] == n_wrong != n_mmm, (full_def.stats()["unique"], n_wrong, n_mmm)


def test_default_path_is_bit_identical_to_a_direct_4mmm_accumulator():
    """'Before' = what every pre-#180 caller got: MergeAccumulator(snr_bins, laue_ops_4mmm()) and
    theoretical_unique(Mc, dmin, laue_ops_4mmm()). 'After' = a driver that passes neither laue nor
    ops nor stream_symmetry. Same frames, every stats() key equal, at two I/sigma floors."""
    frames = _stills(TET, _key_4mmm, np.random.default_rng(1))
    before = MergeAccumulator(SNR, laue_ops_4mmm())
    nth = theoretical_unique(TET, DMIN, laue_ops_4mmm())
    _feed(before, frames)
    drv = _driver(TET)
    assert drv.laue == "4/mmm" and _same_ops(drv.ops, laue_ops_4mmm()) and _same_ops(drv.acc.ops, laue_ops_4mmm())
    assert drv.n_theoretical == nth, (drv.n_theoretical, nth)
    _feed(drv.acc, frames)
    for thr in (0.0, 2.0):
        sb, sa = before.stats(thr=thr, n_theoretical=nth), drv.stats(thr=thr)
        for k, v in sb.items():
            assert _eq(v, sa[k]), (thr, k, v, sa[k])
        assert sa["theoretical_unique"] == nth
        assert sb["common"] >= 10 and sb["cc_half"] == sb["cc_half"], sb   # the FoMs were actually computed
    # also the device accumulator's constructor still takes the ops positionally (it is a drop-in)
    import inspect
    from glint.device_merge import MergeAccumulatorDevice
    assert list(inspect.signature(MergeAccumulatorDevice.__init__).parameters)[1:3] == ["snr_bins", "ops"]


# -------------------------------------------------------------------- resolution of the class
def test_stream_symmetry_derives_the_class_when_laue_is_unset():
    sym = dict(lattice_type="orthorhombic", centering="P", unique_axis="*")
    d = _driver(ORTHO, stream_symmetry=sym)
    assert d.laue == "mmm" and _same_ops(d.ops, laue_ops("mmm")) and _same_ops(d.acc.ops, laue_ops("mmm"))
    assert d.n_theoretical == theoretical_unique(ORTHO, DMIN, laue_ops("mmm"))
    assert d.stream_symmetry == sym                              # the header is untouched
    assert _driver(TET, stream_symmetry=dict(lattice_type="tetragonal")).laue == "4/mmm"
    mono = cell_to_Ar(40.0, 55.0, 30.0, 90, 90, 105)
    assert _driver(mono, stream_symmetry=dict(lattice_type="monoclinic", unique_axis="c")).laue == "2/m_uac"
    # a symmetry record with no lattice_type (the other two keys only) falls back to the default
    assert _driver(TET, stream_symmetry=dict(centering="P", unique_axis="*")).laue == "4/mmm"


def test_explicit_laue_wins_over_stream_symmetry_with_a_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = _driver(ORTHO, laue="4/mmm", stream_symmetry=dict(lattice_type="orthorhombic"))
    assert d.laue == "4/mmm" and _same_ops(d.ops, laue_ops_4mmm())
    msgs = [str(x.message) for x in w]
    assert any("orthorhombic" in m and "mmm" in m for m in msgs), msgs
    with warnings.catch_warnings(record=True) as w:               # agreement is silent
        warnings.simplefilter("always")
        d = _driver(ORTHO, laue="mmm", stream_symmetry=dict(lattice_type="orthorhombic"))
        d2 = _driver(ORTHO, laue="2/m", stream_symmetry=dict(lattice_type="monoclinic", unique_axis="b"))
    assert d.laue == "mmm" and d2.laue == "2/m_uab" and not w, [str(x.message) for x in w]
    try:
        _driver(ORTHO, laue="P4/mmm")
    except ValueError as e:
        assert "4/mmm" in str(e)
    else:
        raise AssertionError("an unknown class name must not silently mean the default")


def test_explicit_ops_are_used_verbatim():
    ops = laue_ops("-1")
    d = _driver(ORTHO, ops=ops)
    assert d.laue is None and _same_ops(d.ops, ops) and _same_ops(d.acc.ops, ops)
    assert d.n_theoretical == theoretical_unique(ORTHO, DMIN, ops)
    assert d.stats()["laue"] is None
    d2 = _driver(ORTHO, ops=ops, laue="-1")                       # the label rides along when given
    assert d2.laue == "-1" and _same_ops(d2.ops, ops)
    # a lone inversion is the smallest set that still merges Friedel pairs
    hkl = _sphere(ORTHO)
    d.acc.add_frame(hkl, np.full(len(hkl), 100.0), np.full(len(hkl), 5.0), 0)
    assert d.stats()["unique"] == len(hkl) // 2 == d.n_theoretical


# --------------------------------------------------------- WHEN the tetragonal setting is applied
def test_lock_standardizes_the_setting_under_the_driver_class():
    # a tetragonal cell handed over with the 4-fold axis in column a (what Buerger reduction can do)
    tet_perm = cell_to_Ar(25.0, 40.0, 40.0, 90, 90, 90)
    for laue in ("4/mmm", "4/m"):
        d = _driver(TET, laue=laue)
        d._lock(tet_perm, standardize=True)
        assert np.allclose(np.linalg.norm(d.Mc, axis=0), (40.0, 40.0, 25.0)), (laue, np.linalg.norm(d.Mc, axis=0))
        assert d.n_theoretical == theoretical_unique(d.Mc, DMIN, laue_ops(laue))
        d._lock(tet_perm, standardize=False)                     # a known cell is taken as given
        assert np.array_equal(d.Mc, tet_perm)
    # orthorhombic: mmm orders it a <= b <= c -- the three axes are inequivalent, so the setting has
    # to be a total order, and a length order is the only one available without a reference (#181).
    ortho_perm = cell_to_Ar(50.0, 30.0, 32.0, 90, 90, 90)
    d = _driver(ORTHO, laue="mmm")
    d._lock(ortho_perm, standardize=True)
    assert np.allclose(np.linalg.norm(d.Mc, axis=0), (30.0, 32.0, 50.0)), np.linalg.norm(d.Mc, axis=0)
    assert np.linalg.det(d.Mc) > 0, "the permutation must keep a right-handed basis"
    # 6/mmm takes the unique-axis rule (its outlier is c); the classes with no length rule at all --
    # triclinic, monoclinic, rhombohedral, cubic -- are left exactly as handed in.
    d = _driver(ORTHO, laue="6/mmm")
    d._lock(ortho_perm, standardize=True)
    assert np.allclose(np.linalg.norm(d.Mc, axis=0), (30.0, 32.0, 50.0)), np.linalg.norm(d.Mc, axis=0)
    for laue in ("2/m", "-1", "m-3m", "-3m_R"):
        d = _driver(ORTHO, laue=laue)
        d._lock(ortho_perm, standardize=True)
        assert np.array_equal(d.Mc, ortho_perm), (laue, np.linalg.norm(d.Mc, axis=0))
    d = _driver(ORTHO, ops=laue_ops("mmm"))                      # explicit ops: no label, no permutation
    d._lock(ortho_perm, standardize=True)
    assert np.array_equal(d.Mc, ortho_perm)
    # ...and the default on the same cell DOES permute it (the pre-#180 behaviour, kept for 4/mmm)
    d = _driver(ORTHO)
    d._lock(ortho_perm, standardize=True)
    assert not np.array_equal(d.Mc, ortho_perm) and np.allclose(np.linalg.norm(d.Mc, axis=0), (30.0, 32.0, 50.0))


def _relock_driver(Mc_active, Mc_new, **kw):
    """Adaptive-relock driver on CPU whose watchdog will vote `Mc_new` (columns = real axes) for
    every missed frame, through the same seams test_lock_probe.py injects."""
    drv = _driver(Mc_active, adaptive_relock=True, min_inliers=6, **kw)
    drv._known_index = lambda qs, Mn, B: [Mc_new.copy() if (q is not None and len(q) >= 6) else None
                                          for q in qs]
    drv._fanout = lambda Q, k: [[(Mc_new.copy(), 1.0)] for _ in Q]
    return drv


def _fill_lattice_frames(drv, Mc_new, rng, k=5, n=30):
    Rn = recip_from_M(Mc_new)                                    # q = hkl @ Rn  ->  q @ Mc_new = hkl
    for i in range(k):
        hkl = rng.integers(-6, 7, size=(n, 3)).astype(float)
        hkl = hkl[np.abs(hkl).sum(1) > 0]
        drv._q[i] = hkl @ Rn
    return list(range(k))


def test_relock_extras_inherit_the_ops_and_standardize_under_the_driver_class():
    new_cell = cell_to_Ar(50.0, 30.0, 32.0, 90, 90, 90)          # orthorhombic, length-outlier FIRST
    calls = []
    real = sd.standardize_axes
    sd.standardize_axes = lambda M, **kw: (calls.append((np.asarray(M, float), kw.get("laue"))),
                                           real(M, **kw))[1]
    try:
        drv = _relock_driver(TET, new_cell, laue="mmm")
        drv._watchdog(_fill_lattice_frames(drv, new_cell, np.random.default_rng(5)))
        assert drv.n_relock == 1 and len(drv.extra) == 1, (drv.n_relock, len(drv.extra))
        e = drv.extra[0]
        assert same_lattice(e["Mc"], new_cell), np.linalg.norm(e["Mc"], axis=0)
        assert _same_ops(e["acc"].ops, laue_ops("mmm")), "relock extra accumulator not on the driver's ops"
        assert e["nth"] == theoretical_unique(e["Mc"], DMIN, laue_ops("mmm"))
        assert e["nth"] != theoretical_unique(e["Mc"], DMIN, laue_ops_4mmm()), "test cell not discriminating"
        assert calls and all(c[1] == "mmm" for c in calls), (
            "the relocked cell must be standardized under the driver's own class, not 4/mmm")
        st = drv.stats()
        assert st["n_cells"] == 2 and st["extra_cells"][0]["unique"] == 0 and st["laue"] == "mmm"
        # and the default driver on the same relock DOES standardize (unchanged 4/mmm behaviour)
        drv4 = _relock_driver(TET, new_cell)
        drv4._watchdog(_fill_lattice_frames(drv4, new_cell, np.random.default_rng(6)))
        assert drv4.n_relock == 1 and any(c[1] == "4/mmm" for c in calls), (drv4.n_relock, calls)
        assert _same_ops(drv4.extra[0]["acc"].ops, laue_ops_4mmm())
        assert drv4.extra[0]["nth"] == theoretical_unique(drv4.extra[0]["Mc"], DMIN, laue_ops_4mmm())
    finally:
        sd.standardize_axes = real


# ---------------------------------------------------------------- glint#186 review follow-ups


def test_a_lower_class_on_the_same_lattice_is_not_a_contradiction():
    """A Laue class BELOW the header's holohedry is legitimate and is the reason `laue=` exists: the
    stream record names the lattice, never the point group. The compatibility test is therefore
    subgroup, not equality -- an equality test flagged 4/m on tetragonal, 6/m on hexagonal and m-3 on
    cubic, i.e. exactly the cases the API tells the caller to spell out (glint#186 review)."""
    for laue, lt in (("4/m", "tetragonal"), ("6/m", "hexagonal"), ("-3m1", "hexagonal"),
                     ("-3", "hexagonal"), ("m-3", "cubic"), ("2/m_uab", "monoclinic")):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = _driver(ORTHO, laue=laue, stream_symmetry=dict(lattice_type=lt))
        assert d.laue == laue_name(laue), (laue, d.laue)
        assert not [x for x in w if "subgroup" in str(x.message) or "glint#180" in str(x.message)], (
            f"{laue} on {lt} is a subgroup of its holohedry and must not be reported as a conflict: "
            f"{[str(x.message) for x in w]}")
    # ...and a class the holohedry does NOT contain still warns, in both directions.
    for laue, lt in (("4/mmm", "orthorhombic"),     # higher: 16 operators the lattice cannot carry
                     ("2/m_uac", "monoclinic"),     # same class, unique axis the record puts in b
                     ("mmm", "trigonal")):          # another crystal system
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _driver(ORTHO, laue=laue, stream_symmetry=dict(lattice_type=lt))
        assert any("glint#180" in str(x.message) for x in w), (laue, lt,
                                                               [str(x.message) for x in w])


def test_explicit_ops_skip_header_derivation_entirely():
    """Explicit operators win outright, so a header the operators make irrelevant must not be parsed
    at all -- deriving first meant an unknown lattice_type or a malformed monoclinic unique_axis
    raised on a path that never uses the result (glint#186 review)."""
    for sym in (dict(lattice_type="not-a-lattice"),
                dict(lattice_type="monoclinic", unique_axis="c1")):
        d = _driver(ORTHO, ops=laue_ops("-1"), stream_symmetry=sym)   # must not raise
        assert _same_ops(d.ops, laue_ops("-1"))
    # without ops the same records are still rejected -- the validation is not weakened, only skipped
    for sym in (dict(lattice_type="not-a-lattice"),
                dict(lattice_type="monoclinic", unique_axis="c1")):
        try:
            _driver(ORTHO, stream_symmetry=sym)
        except ValueError:
            pass
        else:
            raise AssertionError(f"a malformed record must still be rejected when it is used: {sym}")


def test_the_laue_label_does_not_permute_axes_when_ops_are_explicit():
    """With `ops` supplied the operators are authoritative and the setting is the caller's, so the
    class name must not reach the axis standardizer -- it used to tetragonally permute blind and
    relocked cells the caller's operators never asked to be permuted (glint#186 review).

    A label that does not DESCRIBE those operators is now refused outright, because stats() reports
    it as the class the numbers were merged under."""
    M = cell_to_Ar(25.0, 40.0, 40.0, 90, 90, 90)          # 4-fold in a: a length rule would move it
    d = _driver(TET, ops=laue_ops("4/mmm"), laue="4/mmm")  # label agrees with the operators
    assert d.laue == "4/mmm" and _same_ops(d.ops, laue_ops("4/mmm"))
    assert _same_M(d._standardize(M), M), "explicit ops: the setting is the caller's"
    d._lock(M, standardize=True)
    assert _same_M(d.Mc, M), "a lock must not permute it either"
    # the same class WITHOUT explicit ops does standardize, so the test is not vacuous
    d2 = _driver(TET, laue="4/mmm")
    assert not _same_M(d2._standardize(M), M), "no ops: 4/mmm must still put the 4-fold axis in c"
    # a label that misdescribes the operators is refused rather than silently misreported
    try:
        _driver(TET, ops=laue_ops("-1"), laue="4/mmm")
    except ValueError as e:
        assert "does not describe" in str(e), e
    else:
        raise AssertionError("a laue label that contradicts ops must not reach stats()")
    d3 = _driver(TET, ops=laue_ops("-1"))                 # ops alone: the class is simply unlabelled
    assert d3.laue is None and _same_ops(d3.ops, laue_ops("-1"))


def test_frames_are_canonicalized_in_the_reference_setting():
    """The grid is built from the reference cell as _standardize left it, so every frame must be put
    in THAT setting -- not through an unconditional (long, long, short) sort. On 30/40/50 under mmm
    the sort moved h from the 30 A axis to the 40 A one while the grid's h bound was still sized for
    30 A, clipping valid reflections (glint#186 review). Pins the call site, which the accumulator
    tests do not reach."""
    seen = {}
    for laue, Mc in (("mmm", ORTHO), ("4/mmm", TET), (None, TET)):
        d = _driver(Mc, **({} if laue is None else dict(laue=laue)))
        real = d.grid.predict

        def spy(M, *a, _real=real, **kw):
            seen[id(d)] = np.array(M, float)
            return _real(M, *a, **kw)

        d.grid.predict = spy
        d._q[0] = np.zeros((0, 3))
        d._ring[0] = np.zeros((N, N), np.uint16)
        d._pk[0] = None
        d._integrate_one(0, np.array(d.Mc, float), d.grid, d.acc)
        got = seen[id(d)]
        assert _same_M(got, d._standardize(d.Mc)), (laue, got, d.Mc)
        if laue == "mmm":
            assert _same_M(got, d.Mc), "mmm: a frame already in the reference setting must not move"
            assert not _same_M(got, _canonical_axes(d.Mc)), (
                "vacuous probe: (long, long, short) must differ here for the test to mean anything")


def test_identity_classes_relabel_the_frame_into_the_reference_order():
    """The known-cell indexer anchors on the SHORTEST axis and hands its basis back shortest-first,
    whatever order the reference was written in (replica_gpu._axes_from_cell). For the classes whose
    standardizer is the identity -- triclinic, the monoclinic settings, rhombohedral, cubic -- that
    left the frame in a different labelling from the grid and the operators, so a monoclinic
    reference with its unique axis in b was predicted against a grid that expected it there while the
    frame had it wherever its length put it (glint#186 review)."""
    # a reference deliberately NOT in shortest-first order: lengths (50, 30, 32)
    ref = cell_to_Ar(50.0, 30.0, 32.0, 90, 100.0, 90)
    order = np.argsort(np.linalg.norm(ref, axis=0), kind="stable")     # what the indexer sorts by
    shortest_first = ref[:, order]                                    # what it hands back
    assert not _same_M(shortest_first, ref), "fixture must actually be out of order"

    for laue in ("2/m_uab", "-1", "m-3m", "-3m_R"):
        d = _driver(ref, laue=laue)
        seen = {}
        real = d.grid.predict

        def spy(M, *a, _real=real, **kw):
            seen["M"] = np.array(M, float)
            return _real(M, *a, **kw)

        d.grid.predict = spy
        d._q[0] = np.zeros((0, 3)); d._ring[0] = np.zeros((N, N), np.uint16); d._pk[0] = None
        d._integrate_one(0, shortest_first, d.grid, d.acc, known_cell=True)
        assert _same_M(seen["M"], ref), (laue, np.linalg.norm(seen["M"], axis=0))

    # a BLIND candidate is not relabelled: it comes through primitivize(buerger_reduce(...)) and can
    # differ from the reference by a general integer change of basis, not a permutation, so undoing a
    # length sort would be a guess (glint#186 review, glint#188).
    d_blind = _driver(ref, laue="2/m_uab")
    assert _same_M(d_blind._standardize(shortest_first), shortest_first), "no ref -> no relabel"

    # a reference ALREADY shortest-first is a fixed point -- the relabel is a round trip, not a sort
    ref2 = cell_to_Ar(30.0, 32.0, 50.0, 90, 100.0, 90)
    d = _driver(ref2, laue="2/m_uab")
    assert _same_M(d._standardize(ref2, ref=ref2), ref2)
    # ...and where a canonical setting IS imposed the reference went through the same function, so
    # passing it changes nothing.
    d4 = _driver(TET, laue="4/mmm")
    M = cell_to_Ar(25.0, 40.0, 40.0, 90, 90, 90)
    assert _same_M(d4._standardize(M), d4._standardize(M, ref=d4.Mc))


if __name__ == "__main__":
    tests = (test_orthorhombic_set_merges_differently_under_mmm_and_the_default,
             test_default_path_is_bit_identical_to_a_direct_4mmm_accumulator,
             test_stream_symmetry_derives_the_class_when_laue_is_unset,
             test_explicit_laue_wins_over_stream_symmetry_with_a_warning,
             test_explicit_ops_are_used_verbatim,
             test_lock_standardizes_the_setting_under_the_driver_class,
             test_relock_extras_inherit_the_ops_and_standardize_under_the_driver_class,
             test_a_lower_class_on_the_same_lattice_is_not_a_contradiction,
             test_explicit_ops_skip_header_derivation_entirely,
             test_the_laue_label_does_not_permute_axes_when_ops_are_explicit,
             test_frames_are_canonicalized_in_the_reference_setting,
             test_identity_classes_relabel_the_frame_into_the_reference_order)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
