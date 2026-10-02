"""laue_from_symmetry reads the CENTERING: a rhombohedral lattice on hexagonal axes is -3m, not 6/mmm.

Until this fix `laue_from_symmetry` returned `_HOLOHEDRY[lattice_type]` and never read `centering`,
so {lattice_type: hexagonal, centering: H} (CrystFEL's R3/R32 setting on the hexagonal triple cell)
gave 6/mmm. The holohedry of hR is -3m (-3m1 on those axes). Half of the 24 6/mmm operators send
an obverse-allowed reflection (-h+k+l = 3n) onto an absent node, and the predicted grid carries the
absent nodes too (HKLGrid has no centering filter), so a StreamDriver built with that header and no
`laue=` averaged each general reflection with an absent partner in its live merge: the stats()
figures of merit and merged_by_key, not the written .stream.

What is asserted:
  * hexagonal + H or R (any case) gives '-3m' (-> '-3m1'); every other input gives exactly what the
    pre-fix function gave -- value, warning count and exception -- checked against a verbatim copy
    over every lattice_type x centering x unique_axis combination;
  * every operator of the derived class keeps the obverse condition, and the reverse one, on an hkl
    box; 6/mmm breaks it (negative control: the check can fail);
  * no ASU key of the derived class holds both an allowed and an absent hkl, for an 80/80/100/120
    cell; under 6/mmm thousands do (negative control);
  * a StreamDriver built from the header alone gets -3m1, merges synthetic obverse stills to CC ~1
    against the truth, while a 6/mmm accumulator on the same rows does not; a lower class (-3)
    stays silent and a non-subgroup (-31m, 6/mmm) now warns.

Not covered here (separate gaps): per-frame obverse/reverse labelling of indexed frames, and the blind
path, which locks a primitive rhombohedral cell on which hexagonal-axes operators do not apply.

Run: `PYTHONPATH=. python experiments/test_laue_centering.py` (exit 0/1) or `pytest`. numpy only.
"""
import itertools
import warnings

import numpy as np

from glint.lattice import cell_to_Ar
from glint.predict import _hkl_grid, recip_from_M
from glint.stream_driver import (MergeAccumulator, StreamDriver, _asu_key, laue_from_symmetry,
                                 laue_name, laue_ops)

# ---------------------------------------------------------------- verbatim pre-fix copy (149370e)
_OLD_HOLOHEDRY = {
    "triclinic": "-1", "monoclinic": "2/m", "orthorhombic": "mmm", "tetragonal": "4/mmm",
    "trigonal": "-3m", "rhombohedral": "-3m_R", "hexagonal": "6/mmm", "cubic": "m-3m",
}


def _old_laue_from_symmetry(sym):
    """VERBATIM copy of glint/stream_driver.py::laue_from_symmetry as of main@149370e (pre-fix)."""
    sym = dict(sym or {})
    lt = sym.get("lattice_type")
    if lt is None:
        return None
    lt = str(lt).strip().lower()
    if lt not in _OLD_HOLOHEDRY:
        raise ValueError(f"unknown lattice_type {lt!r}; known: {', '.join(_OLD_HOLOHEDRY)}")
    ua = str(sym.get("unique_axis") or "*").strip().lower()
    if lt == "monoclinic":
        if ua in ("a", "b", "c"):
            return f"2/m_ua{ua}"
        if ua != "*":
            raise ValueError(f"unknown monoclinic unique_axis {ua!r}; known: *, a, b, c")
    if lt in ("tetragonal", "trigonal", "hexagonal") and ua not in ("*", "c"):
        warnings.warn(f"lattice_type {lt} with unique_axis {ua!r}: the {_OLD_HOLOHEDRY[lt]} operator set "
                      "assumes the unique axis in c; the live merge will use c (glint#180)")
    return _OLD_HOLOHEDRY[lt]


def _call(fn, sym):
    """(result or exception type, number of warnings) for one record."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn(sym)
        except ValueError as e:
            out = ("ValueError", str(e))
    return out, len(w)


# ----------------------------------------------------------------------------- fixtures
A_HEX, C_HEX = 80.0, 100.0
MC_HEX = cell_to_Ar(A_HEX, A_HEX, C_HEX, 90, 90, 120)        # hR on hexagonal (triple-cell) axes
HEX_H = dict(lattice_type="hexagonal", centering="H", unique_axis="c")
N = 64
PANELS = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=1e4,
               cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
CLEN, WAVE, DMIN = 0.1, 1.3, 4.0


def _obverse(h):
    h = np.asarray(h, int)
    return ((-h[..., 0] + h[..., 1] + h[..., 2]) % 3) == 0


def _reverse(h):
    h = np.asarray(h, int)
    return ((h[..., 0] - h[..., 1] + h[..., 2]) % 3) == 0


def _sphere(Mc, dmin):
    R = recip_from_M(Mc)
    g, q = _hkl_grid(R, 1.0 / dmin)
    g = g[np.linalg.norm(q, axis=1) <= 1.0 / dmin]
    return g[np.any(g != 0, axis=1)]


def _ops_breaking(ops, cond):
    """How many operators send some hkl satisfying `cond` to one that does not."""
    box = np.array(list(itertools.product(range(-6, 7), repeat=3)))
    ok = cond(box)
    return sum(int(np.any(ok & ~cond(box @ np.asarray(op).T))) for op in ops)


def _mixed_keys(hkl, ops):
    """ASU keys holding both an obverse-allowed and an absent hkl."""
    keys = _asu_key(hkl, ops)
    allowed = _obverse(hkl)
    return len(set(keys[allowed].tolist()) & set(keys[~allowed].tolist()))


def _driver(**kw):
    return StreamDriver(MC_HEX, PANELS, CLEN, WAVE, (N, N), dtype=np.uint16, B=8, dmin=DMIN,
                        use_gpu=False, snr_bins=(0.0, 1.0, 2.0, 3.0, 5.0), **kw)


# --------------------------------------------------------------------------------- tests
def test_hexagonal_with_H_or_R_centering_is_minus_3m():
    for cen in ("H", "R", "h", "r", " H "):
        for ua in (None, "*", "c"):
            sym = dict(lattice_type="hexagonal", centering=cen)
            if ua is not None:
                sym["unique_axis"] = ua
            got = laue_from_symmetry(sym)
            assert got == "-3m" and laue_name(got) == "-3m1", (sym, got)
    assert laue_from_symmetry(dict(lattice_type="hexagonal", centering="P")) == "6/mmm"
    assert laue_from_symmetry(dict(lattice_type="hexagonal")) == "6/mmm"   # centering absent = P


def test_every_other_input_is_unchanged():
    """The fix touches (hexagonal, H|R) only: on every other combination the value, the warnings and
    the exception are those of the verbatim pre-fix copy."""
    lattice_types = list(_OLD_HOLOHEDRY) + ["HEXAGONAL", " Trigonal ", "tetragonall"]
    centerings = (None, "", "P", "A", "B", "C", "I", "F", "R", "H", "p", "r", "h")
    unique_axes = (None, "*", "a", "b", "c", "c1")
    n_same = n_changed = 0
    for lt, cen, ua in itertools.product(lattice_types, centerings, unique_axes):
        sym = {"lattice_type": lt}
        if cen is not None:
            sym["centering"] = cen
        if ua is not None:
            sym["unique_axis"] = ua
        new, old = _call(laue_from_symmetry, sym), _call(_old_laue_from_symmetry, sym)
        hR_on_hex = (lt.strip().lower() == "hexagonal" and str(cen or "").strip().upper() in ("H", "R"))
        if hR_on_hex:
            assert old[0] == "6/mmm" and new[0] == "-3m" and new[1] == old[1], (sym, old, new)
            n_changed += 1
        else:
            assert new == old, (sym, old, new)
            n_same += 1
    # the records with no lattice_type are untouched too
    for sym in (None, {}, {"centering": "H"}, {"centering": "R", "unique_axis": "c"}):
        assert laue_from_symmetry(sym) is None
    # changed: 2 spellings of hexagonal x (H, R, h, r) x every unique_axis; the rest is unchanged
    assert n_changed == 2 * 4 * len(unique_axes), n_changed
    assert n_same == len(lattice_types) * len(centerings) * len(unique_axes) - n_changed, n_same


def test_derived_operators_keep_the_obverse_condition():
    ops = laue_ops(laue_from_symmetry(HEX_H))
    assert len(ops) == 12
    assert _ops_breaking(ops, _obverse) == 0
    assert _ops_breaking(ops, _reverse) == 0       # also right for a reverse-indexed crystal
    assert _ops_breaking(laue_ops("6/mmm"), _obverse) == 12     # negative control: the old class


def test_no_asu_key_mixes_allowed_and_absent():
    hkl = _sphere(MC_HEX, 2.5)
    assert _obverse(hkl).any() and (~_obverse(hkl)).any()
    assert _mixed_keys(hkl, laue_ops(laue_from_symmetry(HEX_H))) == 0
    assert _mixed_keys(hkl, laue_ops("6/mmm")) > 1000            # negative control


def test_driver_from_the_header_merges_obverse_stills_correctly():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = _driver(stream_symmetry=HEX_H)
    assert d.laue == "-3m1" and len(d.ops) == 12, d.laue
    assert d._sl_laue() == "-3m1"                    # the double-hit gate gets the same class
    assert not [x for x in w if "glint#180" in str(x.message)], [str(x.message) for x in w]
    # Synthetic stills: every grid node is "predicted and integrated", as HKLGrid does; absent nodes
    # come back near zero. Truth is constant on each -3m1 orbit of allowed reflections.
    rng = np.random.default_rng(3)
    hkl = _sphere(MC_HEX, DMIN)
    allowed = _obverse(hkl)
    keys = _asu_key(hkl, d.ops)
    truth = {}
    I_true = np.array([truth.setdefault(k, 50.0 + 2000.0 * rng.exponential()) if a else 0.0
                       for k, a in zip(keys.tolist(), allowed)])
    acc_6mmm = MergeAccumulator(d.acc.thr, laue_ops("6/mmm"))
    for f in range(24):
        pick = rng.random(len(hkl)) < 0.35
        I = np.where(allowed[pick], I_true[pick] * (1.0 + 0.05 * rng.standard_normal(int(pick.sum()))),
                     rng.uniform(0.5, 3.0, int(pick.sum())))
        sig = np.where(allowed[pick], np.sqrt(np.abs(I)) + 2.0, 2.0)
        d.acc.add_frame(hkl[pick], I, sig, f)
        acc_6mmm.add_frame(hkl[pick], I, sig, f)

    def cc_to_truth(acc, ops):
        m = acc.merged_by_key()
        k_all = _asu_key(hkl[allowed], ops).tolist()
        t = {}
        for k, i in zip(k_all, I_true[allowed]):
            t.setdefault(k, i)
        common = [k for k in t if k in m]
        return float(np.corrcoef([m[k] for k in common], [t[k] for k in common])[0, 1])

    cc_derived, cc_6mmm = cc_to_truth(d.acc, d.ops), cc_to_truth(acc_6mmm, laue_ops("6/mmm"))
    print(f"      CC(merged, truth): derived {d.laue} {cc_derived:.4f}   6/mmm {cc_6mmm:.4f}")
    assert cc_derived > 0.99, cc_derived
    assert cc_6mmm < 0.9, cc_6mmm                    # negative control: the old class mixes orbits


def test_explicit_laue_on_hR_header():
    """A lower class (-3) stays silent; a class the hR holohedry does not contain now warns."""
    for laue, warns in (("-3", False), ("-3m", False), ("-3m1", False),
                        ("-31m", True), ("6/m", True), ("6/mmm", True)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            d = _driver(laue=laue, stream_symmetry=HEX_H)
        assert d.laue == laue_name(laue)
        got = any("glint#180" in str(x.message) for x in w)
        assert got == warns, (laue, [str(x.message) for x in w])
    # hexagonal P keeps 6/mmm, and its lower classes stay silent, as before
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = _driver(stream_symmetry=dict(lattice_type="hexagonal", centering="P"))
        _driver(laue="-31m", stream_symmetry=dict(lattice_type="hexagonal", centering="P"))
    assert d.laue == "6/mmm" and not [x for x in w if "glint#180" in str(x.message)]


if __name__ == "__main__":
    tests = (test_hexagonal_with_H_or_R_centering_is_minus_3m,
             test_every_other_input_is_unchanged,
             test_derived_operators_keep_the_obverse_condition,
             test_no_asu_key_mixes_allowed_and_absent,
             test_driver_from_the_header_merges_obverse_stills_correctly,
             test_explicit_laue_on_hR_header)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
