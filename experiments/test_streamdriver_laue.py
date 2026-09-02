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
from glint.predict import _hkl_grid, recip_from_M
from glint.stream_driver import (MergeAccumulator, StreamDriver, laue_ops, laue_ops_4mmm,
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
def test_lock_standardizes_the_setting_for_tetragonal_classes_only():
    # a tetragonal cell handed over with the 4-fold axis in column a (what Buerger reduction can do)
    tet_perm = cell_to_Ar(25.0, 40.0, 40.0, 90, 90, 90)
    for laue in ("4/mmm", "4/m"):
        d = _driver(TET, laue=laue)
        d._lock(tet_perm, standardize=True)
        assert np.allclose(np.linalg.norm(d.Mc, axis=0), (40.0, 40.0, 25.0)), (laue, np.linalg.norm(d.Mc, axis=0))
        assert d.n_theoretical == theoretical_unique(d.Mc, DMIN, laue_ops(laue))
        d._lock(tet_perm, standardize=False)                     # a known cell is taken as given
        assert np.array_equal(d.Mc, tet_perm)
    # an orthorhombic cell whose length-outlier is NOT last: mmm must leave it exactly alone
    ortho_perm = cell_to_Ar(50.0, 30.0, 32.0, 90, 90, 90)
    for laue in ("mmm", "2/m", "-1", "6/mmm", "m-3m"):
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


def test_relock_extras_inherit_the_ops_and_skip_tetragonal_standardization_for_mmm():
    new_cell = cell_to_Ar(50.0, 30.0, 32.0, 90, 90, 90)          # orthorhombic, length-outlier FIRST
    calls = []
    real = sd._conventional_tetragonal
    sd._conventional_tetragonal = lambda M: (calls.append(np.asarray(M, float)), real(M))[1]
    try:
        drv = _relock_driver(TET, new_cell, laue="mmm")
        drv._watchdog(_fill_lattice_frames(drv, new_cell, np.random.default_rng(5)))
        assert drv.n_relock == 1 and len(drv.extra) == 1, (drv.n_relock, len(drv.extra))
        e = drv.extra[0]
        assert same_lattice(e["Mc"], new_cell), np.linalg.norm(e["Mc"], axis=0)
        assert _same_ops(e["acc"].ops, laue_ops("mmm")), "relock extra accumulator not on the driver's ops"
        assert e["nth"] == theoretical_unique(e["Mc"], DMIN, laue_ops("mmm"))
        assert e["nth"] != theoretical_unique(e["Mc"], DMIN, laue_ops_4mmm()), "test cell not discriminating"
        assert not calls, "mmm driver ran _conventional_tetragonal on the relocked cell"
        st = drv.stats()
        assert st["n_cells"] == 2 and st["extra_cells"][0]["unique"] == 0 and st["laue"] == "mmm"
        # and the default driver on the same relock DOES standardize (unchanged 4/mmm behaviour)
        drv4 = _relock_driver(TET, new_cell)
        drv4._watchdog(_fill_lattice_frames(drv4, new_cell, np.random.default_rng(6)))
        assert drv4.n_relock == 1 and len(calls) >= 1, (drv4.n_relock, len(calls))
        assert _same_ops(drv4.extra[0]["acc"].ops, laue_ops_4mmm())
        assert drv4.extra[0]["nth"] == theoretical_unique(drv4.extra[0]["Mc"], DMIN, laue_ops_4mmm())
    finally:
        sd._conventional_tetragonal = real


if __name__ == "__main__":
    tests = (test_orthorhombic_set_merges_differently_under_mmm_and_the_default,
             test_default_path_is_bit_identical_to_a_direct_4mmm_accumulator,
             test_stream_symmetry_derives_the_class_when_laue_is_unset,
             test_explicit_laue_wins_over_stream_symmetry_with_a_warning,
             test_explicit_ops_are_used_verbatim,
             test_lock_standardizes_the_setting_for_tetragonal_classes_only,
             test_relock_extras_inherit_the_ops_and_skip_tetragonal_standardization_for_mmm)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
