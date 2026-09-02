"""Properties of the Laue-class operator registry, glint.stream_driver.laue_ops (glint#180).

Until #180 the streaming merge had exactly one operator set, `laue_ops_4mmm()`, and the driver
hard-wired it. The registry adds the other ten Laue classes (plus the settings that matter for the
integer representation: monoclinic unique axis, -3m1 vs -31m, rhombohedral axes) as generator-closed
groups of integer 3x3 matrices acting on hkl. A wrong generator gives a set that is still closed and
still the right SIZE for many mistakes, so size alone is not the test: every class is also checked
to preserve the reciprocal metric of a cell of its own crystal system, which is what "these
reflections are symmetry-equivalent" means, and a negative control shows that check can fail.

The one thing that must not move is the default: `laue_ops("4/mmm")` and `laue_ops_4mmm()` are
pinned element-for-element, in order, against a verbatim copy of the pre-#180 function.

Run: `PYTHONPATH=. python experiments/test_laue_ops.py` (exit 0/1) or `pytest`. numpy only.
"""
import warnings

import numpy as np

from glint.lattice import cell_to_Ar
from glint.stream_driver import (LAUE_CLASSES, TETRAGONAL_LAUE, laue_from_symmetry, laue_name,
                                 laue_ops, laue_ops_4mmm, theoretical_unique)

ORDER = {"-1": 2, "2/m": 4, "mmm": 8, "4/m": 8, "4/mmm": 16, "-3": 6, "-3m": 12,
         "6/m": 12, "6/mmm": 24, "m-3": 24, "m-3m": 48}
# a cell of each class's crystal system, in the setting the registry assumes
TRICLINIC = (31.0, 47.0, 53.0, 81.0, 97.0, 108.0)
MONO_B = (40.0, 30.0, 55.0, 90.0, 105.0, 90.0)           # unique axis b (beta != 90)
ORTHO = (30.0, 40.0, 50.0, 90.0, 90.0, 90.0)
TET = (40.0, 40.0, 25.0, 90.0, 90.0, 90.0)
HEX = (45.0, 45.0, 60.0, 90.0, 90.0, 120.0)
RHOMBO = (50.0, 50.0, 50.0, 80.0, 80.0, 80.0)            # rhombohedral axes (a=b=c, alpha=beta=gamma)
CUBIC = (50.0, 50.0, 50.0, 90.0, 90.0, 90.0)
CELL = {"-1": TRICLINIC, "2/m": MONO_B, "mmm": ORTHO, "4/m": TET, "4/mmm": TET,
        "-3": HEX, "-3m": HEX, "6/m": HEX, "6/mmm": HEX, "m-3": CUBIC, "m-3m": CUBIC,
        "2/m_uab": MONO_B, "2/m_uaa": (30.0, 40.0, 55.0, 105.0, 90.0, 90.0),
        "2/m_uac": (40.0, 55.0, 30.0, 90.0, 90.0, 105.0),
        "-3m1": HEX, "-31m": HEX, "-3_R": RHOMBO, "-3m_R": RHOMBO}


def _frozen_laue_ops_4mmm():
    """VERBATIM copy of glint/stream_driver.py::laue_ops_4mmm as of main@faa4973 (pre-#180)."""
    gens = [np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            -np.eye(3, dtype=int)]
    G = [np.eye(3, dtype=int)]
    ch = True
    while ch:
        ch = False
        for g in list(G):
            for s in gens:
                h = (s @ g).astype(int)
                if not any(np.array_equal(h, x) for x in G):
                    G.append(h); ch = True
    return G


def _contains(G, m):
    return any(np.array_equal(m, x) for x in G)


def _same_set(G1, G2):
    return len(G1) == len(G2) and all(_contains(G2, g) for g in G1)


def _metric_violations(ops, cell, n=300, seed=0):
    """How many (op, hkl) pairs change |q| on this cell -- 0 iff the ops are symmetries of it."""
    R = np.linalg.inv(cell_to_Ar(*cell))                    # rows a*, b*, c*
    hkl = np.random.default_rng(seed).integers(-8, 9, size=(n, 3))
    q0 = np.linalg.norm(hkl @ R, axis=1)
    bad = 0
    for op in ops:
        q1 = np.linalg.norm((hkl @ op.T) @ R, axis=1)
        bad += int((np.abs(q1 - q0) > 1e-9 * np.maximum(q0, 1.0)).sum())
    return bad


def test_each_class_is_a_group_of_the_right_order_with_identity_and_inversion():
    for name, order in ORDER.items():
        G = laue_ops(name)
        assert len(G) == order, (name, len(G), order)
        assert _contains(G, np.eye(3, dtype=int)), name
        assert _contains(G, -np.eye(3, dtype=int)), f"{name}: inversion missing (not a Laue class)"
        for g in G:
            assert g.dtype.kind == "i" and g.shape == (3, 3), (name, g.dtype, g.shape)
            assert abs(round(np.linalg.det(g))) == 1, (name, g)
        assert len({g.tobytes() for g in G}) == order, f"{name}: duplicate elements"
        for a in G:                                                    # closure (=> inverses too)
            for b in G:
                assert _contains(G, a @ b), (name, a, b)


def test_4mmm_is_bit_identical_to_the_frozen_original():
    ref = _frozen_laue_ops_4mmm()
    for got, label in ((laue_ops("4/mmm"), 'laue_ops("4/mmm")'), (laue_ops_4mmm(), "laue_ops_4mmm()")):
        assert len(got) == len(ref) == 16, (label, len(got))
        for i, (a, b) in enumerate(zip(got, ref)):                     # same matrices, SAME ORDER
            assert np.array_equal(a, b) and a.dtype == b.dtype, (label, i, a, b)


def test_ops_preserve_the_reciprocal_metric_of_their_own_system():
    for name, cell in CELL.items():
        bad = _metric_violations(laue_ops(name), cell)
        assert bad == 0, f"{name}: {bad} (op, hkl) pairs change |q| on cell {cell}"
    # THE INSTRUMENT CAN FAIL: 4/mmm applied to an orthorhombic cell swaps a and b, which are not
    # equivalent there -- the exact over-merge #180 is about -- and the metric check sees it.
    assert _metric_violations(laue_ops("4/mmm"), ORTHO) > 0
    # ...and the hexagonal-axes trigonal ops are not symmetries of a rhombohedral-axes cell, nor the
    # reverse: the _R suffix is a different integer representation, not a synonym.
    assert _metric_violations(laue_ops("-3m"), RHOMBO) > 0
    assert _metric_violations(laue_ops("-3m_R"), HEX) > 0


def test_settings_and_aliases():
    assert laue_name("2/m") == "2/m_uab" and _same_set(laue_ops("2/m"), laue_ops("2/m_uab"))
    assert not _same_set(laue_ops("2/m_uaa"), laue_ops("2/m_uab"))
    assert not _same_set(laue_ops("2/m_uab"), laue_ops("2/m_uac"))
    assert laue_name("-3m") == "-3m1" and _same_set(laue_ops("-3m"), laue_ops("-3m1_H"))
    # -3m1 and -31m: both order 12, both hexagonal symmetries, DIFFERENT subgroups of 6/mmm
    a, b, full = laue_ops("-3m1"), laue_ops("-31m"), laue_ops("6/mmm")
    assert len(a) == len(b) == 12 and not _same_set(a, b)
    assert all(_contains(full, g) for g in a + b)
    assert _same_set(laue_ops("m3m"), laue_ops("m-3m")) and _same_set(laue_ops("m3"), laue_ops("m-3"))
    assert len(laue_ops("-3_R")) == 6 and len(laue_ops("-3m_R")) == 12
    for name in LAUE_CLASSES:
        assert laue_name(name) in (name, "2/m_uab", "-3m1")
    assert TETRAGONAL_LAUE == ("4/m", "4/mmm")
    for bad in ("4mmm", "P4/mmm", "", "mmm ", "4/mmm\n"):        # near-misses must not silently resolve
        if bad.strip() in ("mmm", "4/mmm"):
            assert laue_name(bad) == bad.strip()                 # ...but surrounding whitespace is tolerated
            continue
        try:
            laue_ops(bad)
        except ValueError as e:
            assert "4/mmm" in str(e), e
        else:
            raise AssertionError(f"laue_ops({bad!r}) did not raise")


def test_subgroup_chain_orders_the_theoretical_unique_counts():
    """More operators, fewer asymmetric-unit keys: on the orthorhombic cell -1 > 2/m > mmm; and the
    wrong 4/mmm gives FEWER than mmm because it merges (h,k,l) with (k,h,l), which are not
    equivalent there -- the number the driver used to report as its completeness denominator."""
    Mc, dmin = cell_to_Ar(*ORTHO), 3.0
    n = {k: theoretical_unique(Mc, dmin, laue_ops(k)) for k in ("-1", "2/m", "mmm", "4/mmm")}
    assert n["-1"] > n["2/m"] > n["mmm"] > n["4/mmm"] > 0, n
    Mt = cell_to_Ar(*TET)
    assert theoretical_unique(Mt, dmin, laue_ops("4/m")) > theoretical_unique(Mt, dmin, laue_ops("4/mmm"))


def test_laue_from_symmetry_assumes_the_holohedry():
    assert laue_from_symmetry(None) is None and laue_from_symmetry({}) is None
    assert laue_from_symmetry({"centering": "P", "unique_axis": "*"}) is None
    for lt, cls in (("triclinic", "-1"), ("monoclinic", "2/m"), ("orthorhombic", "mmm"),
                    ("tetragonal", "4/mmm"), ("trigonal", "-3m"), ("rhombohedral", "-3m_R"),
                    ("hexagonal", "6/mmm"), ("cubic", "m-3m")):
        assert laue_from_symmetry({"lattice_type": lt, "centering": "P", "unique_axis": "*"}) == cls, lt
        assert laue_from_symmetry({"lattice_type": lt.upper()}) == cls, lt
    for ua in "abc":
        assert laue_from_symmetry({"lattice_type": "monoclinic", "unique_axis": ua}) == f"2/m_ua{ua}"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert laue_from_symmetry({"lattice_type": "tetragonal", "unique_axis": "a"}) == "4/mmm"
        assert laue_from_symmetry({"lattice_type": "orthorhombic", "unique_axis": "a"}) == "mmm"
    assert len(w) == 1 and "unique_axis 'a'" in str(w[0].message), [str(x.message) for x in w]
    try:
        laue_from_symmetry({"lattice_type": "tetragonall"})
    except ValueError as e:
        assert "tetragonal" in str(e)
    else:
        raise AssertionError("an unknown lattice_type must not silently mean 4/mmm")


if __name__ == "__main__":
    tests = (test_each_class_is_a_group_of_the_right_order_with_identity_and_inversion,
             test_4mmm_is_bit_identical_to_the_frozen_original,
             test_ops_preserve_the_reciprocal_metric_of_their_own_system,
             test_settings_and_aliases,
             test_subgroup_chain_orders_the_theoretical_unique_counts,
             test_laue_from_symmetry_assumes_the_holohedry)
    ok = 0
    for t in tests:
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    raise SystemExit(0 if ok == len(tests) else 1)
