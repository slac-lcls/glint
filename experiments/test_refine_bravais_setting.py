"""refine_bravais from a start basis that is NOT in the conventional setting (glint review s4-04).

REGRESSION. ``glint.refine_sym.refine_bravais`` projects the start cell onto the Bravais manifold and
takes the rotation from it, which is right only when the start is already the conventional cell.
``standardize_setting`` permuted the columns and, for a left-handed basis, negated column a, so:
  * hexagonal: a reduced cell with gamma 60 (about half the frames the blind indexer emits) was
    forced to 120 on the wrong pair of axes, and the column-a flip turned a good 120 into 60;
  * rhombohedral (``trigonal``): a mixed-sign reduced cell (75/105/105) was averaged to ~95;
  * every centred lattice (mC, oC, oI, oF, tI, cI, cF): the primitive reduced cell the blind indexer
    emits has no tetragonal/orthorhombic/... metric at all, and was squeezed onto one;
  * any non-reduced basis of any lattice.
The refined basis stayed on the manifold but was a different lattice, fitting a few percent of the
spots, with nothing downstream to notice (at the default sym_refine_tol its own inlier fraction reads
1.000). Now a start that is not in the setting is first taken to a conventional cell
(``conventional_settings``: integer transforms P, |det P| = 1..4, one per candidate symmetry axis; the
refined candidate that fits the spots best is kept) and, for a centred lattice, the refined cell is
mapped back to the start's lattice.

What is checked, for all 14 Bravais lattices plus hR given as ``hexagonal`` (its R-centred triple
cell), each from its conventional cell and from three other bases of the same lattice -- the
Buerger-reduced primitive cell (the blind indexer's form), a non-reduced primitive basis, and a
left-handed relabelled reduced cell -- at random orientations, from a 0.3%-strained start, on
Ewald-filtered spots of the true lattice:
  1. the refined basis is a basis of the start's lattice: T = inv(M_start_true) @ M_out is integral
     and unimodular (for a centred lattice's conventional cell, the lattice that cell spans);
  2. its metric has the requested symmetry exactly: the refined image of the true conventional cell,
     M_out @ inv(T) @ (inv(M_start_true) @ M_conv_true), is on the manifold to 1e-9 (a=b, 90, 120, ...);
  3. it fits the spots: >= 95% within 0.002 1/A;
  4. no refusal warning for these (valid) requests.
Also: a conventional start gives bit-for-bit the result of ``standardize=False`` (the pre-fix path,
untouched); four requests for a symmetry the lattice does not have are refused with a RuntimeWarning
and M comes back unchanged; and ``conventional_setting`` alone finds the true conventional cell for
random exact cells of all 14 types in three settings each.

On the pre-fix code (origin/main 149370e) this test reports 115 FAIL lines: 97 of the 135 refines from
a non-conventional start, the 14 sweep checks (conventional_setting did not exist) and the 4 refusals.
The 38 non-conventional starts that pass there are all of aP, the reduced and left-handed starts of
mP/oP/tP/cP (signed permutations of the conventional cell), and 5 of the 18 hP/hR starts, the ones its
relabelling happened to leave in the conventional setting. All 45 conventional-start refines pass there.

Run: ``PYTHONPATH=. python experiments/test_refine_bravais_setting.py`` (exit 1 on any failure).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.lattice import buerger_reduce, cell_params, cell_to_Ar, random_rotation   # noqa: E402
from glint import refine_sym                                                         # noqa: E402
from glint.refine_sym import refine_bravais                                          # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


# primitive basis = conventional @ Q (columns = primitive translations in conventional coordinates)
Q = {"P": np.eye(3),
     "C": np.array([[.5, -.5, 0], [.5, .5, 0], [0, 0, 1.]]),
     "I": np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]]),
     "F": np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
     "R": np.array([[2., -1, -1], [1, 1, -2], [1, 1, 1]]) / 3.0}    # obverse, hexagonal axes

_AL = 75.0
_AH = 2 * 60.0 * np.sin(np.radians(_AL / 2))                       # the hR 60/75 cell on hexagonal axes
_CH = 60.0 * np.sqrt(3 * (1 + 2 * np.cos(np.radians(_AL))))

LATTICES = [            # name, conventional cell, centring, system
    ("aP", (50, 60, 70, 80, 95, 105), "P", "triclinic"),
    ("mP", (50, 60, 70, 90, 105, 90), "P", "monoclinic"),
    ("mC", (80, 50, 60, 90, 110, 90), "C", "monoclinic"),
    ("oP", (50, 70, 90, 90, 90, 90), "P", "orthorhombic"),
    ("oC", (50, 70, 90, 90, 90, 90), "C", "orthorhombic"),
    ("oI", (50, 70, 90, 90, 90, 90), "I", "orthorhombic"),
    ("oF", (50, 70, 90, 90, 90, 90), "F", "orthorhombic"),
    ("tP", (60, 60, 90, 90, 90, 90), "P", "tetragonal"),
    ("tI", (60, 60, 90, 90, 90, 90), "I", "tetragonal"),
    ("hR", (60, 60, 60, _AL, _AL, _AL), "P", "trigonal"),            # rhombohedral axes
    ("hP", (60, 60, 100, 90, 90, 120), "P", "hexagonal"),
    ("cP", (60, 60, 60, 90, 90, 90), "P", "cubic"),
    ("cI", (60, 60, 60, 90, 90, 90), "I", "cubic"),
    ("cF", (60, 60, 60, 90, 90, 90), "F", "cubic"),
    ("hR-hex", (_AH, _AH, _CH, 90, 90, 120), "R", "hexagonal"),     # the same hR, as 'hexagonal'
]
N_ORI = 3
TOL_ABS = 0.02            # integrate_cxi's default sym_refine_tol
SKEW = np.array([[1., 1, 0], [0, 1, 1], [0, 0, 1]])                 # unimodular, not reduced
LEFT = np.array([[0., 1, 0], [1, 0, 0], [0, 0, 1]])   # swap a,b: reverse handedness (det -1)


def spots(Mp, rng, dmin=2.5, wavelength=1.3, n=80, pos_sigma=1e-4):
    """Ewald-filtered reciprocal nodes of the lattice with primitive basis Mp (as simulate_shot)."""
    C = np.linalg.inv(Mp).T
    hmax = int(np.ceil(np.linalg.norm(Mp, axis=0).max() / dmin)) + 1
    r = np.arange(-hmax, hmax + 1)
    H = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, axis=1)]
    g = H @ C.T
    g = g[np.linalg.norm(g, axis=1) <= 1.0 / dmin]
    k0 = np.array([0.0, 0.0, 1.0 / wavelength])
    eps = np.abs(np.linalg.norm(k0 + g, axis=1) - 1.0 / wavelength)
    g = g[np.argsort(eps)[:n]]
    return g + rng.normal(0, pos_sigma, g.shape)


def manifold_resid(M, system):
    """Largest violation of the system's constraints by cell_params(M) (relative length or degrees)."""
    a, b, c, al, be, ga = cell_params(M)
    r = {"triclinic": [0.0],
         "monoclinic": [al - 90, ga - 90],
         "orthorhombic": [al - 90, be - 90, ga - 90],
         "tetragonal": [(a - b) / a, al - 90, be - 90, ga - 90],
         "hexagonal": [(a - b) / a, al - 90, be - 90, ga - 120],
         "trigonal": [(a - b) / a, (b - c) / a, al - be, be - ga],
         "cubic": [(a - b) / a, (b - c) / a, al - 90, be - 90, ga - 90]}[system]
    return float(np.max(np.abs(r)))


def tight_fit(g, M, tol=0.002):
    C = np.linalg.inv(M).T
    h = np.rint(g @ M)
    return float((np.linalg.norm(g - h @ C.T, axis=1) < tol).mean())


def run_lattice(name, cell, cen, system, seed):
    rng = np.random.default_rng(seed)
    nfail0 = len(FAILS)
    for o in range(N_ORI):
        Mc_true = random_rotation(rng) @ cell_to_Ar(*cell)          # conventional cell
        Mp_true = Mc_true @ Q[cen]                                  # primitive basis of the lattice
        g = spots(Mp_true, rng)
        Mred = buerger_reduce(Mp_true)
        starts = {"conventional": Mc_true, "reduced": Mred, "non-reduced": Mred @ SKEW,
                  "left-handed": Mred @ LEFT}
        for sname, M0 in starts.items():
            M_start = (np.eye(3) + rng.normal(0, 0.003, (3, 3))) @ M0   # an unconstrained fit's error
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                M_out, _, _, _ = refine_bravais(g, M_start, system, TOL_ABS)
            refused = any(issubclass(x.category, RuntimeWarning) for x in w)
            T = np.linalg.solve(M0, M_out)
            Ti = np.rint(T)
            basis = (np.abs(T - Ti).max() < 0.01 and abs(round(np.linalg.det(Ti))) == 1)
            if basis:
                P_in = np.rint(np.linalg.solve(M0, Mc_true))       # start setting -> conventional
                res = manifold_resid(M_out @ np.linalg.inv(Ti) @ P_in, system)
            else:
                res = float("nan")
            fit = tight_fit(g, M_out)
            ok = basis and res < 1e-9 and fit >= 0.95 and not refused
            check(f"{name:6s} {system:12s} {sname:12s} ori {o}: basis of the lattice, on the manifold, fits",
                  ok, f"T integral+unimodular={basis} (max |T-round T|={np.abs(T - Ti).max():.3f}) "
                      f"manifold resid={res:.2e} tight fit={fit:.3f} refused={refused}")
        # a conventional start takes the pre-fix path untouched
        M_start = (np.eye(3) + rng.normal(0, 0.003, (3, 3))) @ Mc_true
        A = refine_bravais(g, M_start, system, TOL_ABS)
        B = refine_bravais(g, M_start, system, TOL_ABS, standardize=False)
        check(f"{name:6s} {system:12s} conventional ori {o}: identical to standardize=False (pre-fix path)",
              all(np.array_equal(x, y) for x, y in zip(A, B)))
    return len(FAILS) - nfail0


def refusals():
    """A system the lattice has no cell of (within the tolerances) is refused, loudly, M unchanged."""
    rng = np.random.default_rng(7)
    cases = [("hP 60/60/100 asked 'trigonal' (= rhombohedral; hP is 'hexagonal')", (60, 60, 100, 90, 90, 120),
              "trigonal"),
             ("oP 50/70/90 asked 'tetragonal'", (50, 70, 90, 90, 90, 90), "tetragonal"),
             ("mP 50/60/70/105 asked 'orthorhombic'", (50, 60, 70, 90, 105, 90), "orthorhombic"),
             ("tP 60/60/90 asked 'hexagonal'", (60, 60, 90, 90, 90, 90), "hexagonal")]
    for label, cell, system in cases:
        M = random_rotation(rng) @ cell_to_Ar(*cell)
        g = spots(M, rng)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            M_out, _, _, inl = refine_bravais(g, M, system, TOL_ABS)
        warned = any(issubclass(x.category, RuntimeWarning) for x in w)
        check(f"refused with a RuntimeWarning, M returned unchanged: {label}",
              warned and np.array_equal(M_out, M) and inl.all(),
              f"warned={warned} unchanged={np.array_equal(M_out, M)}")


def reduction_and_centering():
    """Large integer shears reduce correctly, and a wrong centering cannot mimic cubic symmetry."""
    M = 60.0 * np.array([[1., 50, 0], [0, 1, 0], [0, 0, 1]])
    P = refine_sym.conventional_setting(M, "cubic")
    check("large-shear cubic basis is reduced to its conventional cell",
          P is not None and abs(round(np.linalg.det(P))) == 1
          and manifold_resid(M @ P, "cubic") < 1e-6,
          None if P is None else (np.linalg.det(P), manifold_resid(M @ P, "cubic")))

    M = np.diag([30., 60., 60.])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g = spots(M, np.random.default_rng(7))
        M_out, _, _, _ = refine_bravais(g, M, "cubic", TOL_ABS)
    warned = any(issubclass(x.category, RuntimeWarning) for x in w)
    check("incompatible centering is refused for a cubic request",
          warned and np.array_equal(M_out, M),
          f"warned={warned} unchanged={np.array_equal(M_out, M)}")


def search_sweep(n=8):
    """conventional_setting alone on random EXACT cells (no refine): the true conventional cell, every
    type, from the reduced cell, a non-reduced basis and a relabelled one."""
    rng = np.random.default_rng(20261001)

    def draw(bl):
        U = lambda: rng.uniform(25, 150)                           # noqa: E731
        f, cen = bl[0], (bl[1] if len(bl) > 1 else "P")
        if bl == "aP":
            return (U(), U(), U(), *rng.uniform(70, 110, 3)), "P", "triclinic"
        if f == "m":
            return (U(), U(), U(), 90, rng.uniform(95, 120), 90), cen, "monoclinic"
        if f == "o":
            return (U(), U(), U(), 90, 90, 90), cen, "orthorhombic"
        if f == "t":
            a = U(); return (a, a, U(), 90, 90, 90), cen, "tetragonal"   # noqa: E702
        if bl == "hP":
            a = U(); return (a, a, U(), 90, 90, 120), "P", "hexagonal"   # noqa: E702
        if bl == "hR":
            a = U(); al = rng.choice([rng.uniform(65, 85), rng.uniform(95, 115)])   # noqa: E702
            return (a, a, a, al, al, al), "P", "trigonal"
        a = U(); return (a, a, a, 90, 90, 90), cen, "cubic"              # noqa: E702

    search = getattr(refine_sym, "conventional_setting", None)      # absent before the fix
    for bl in ("aP", "mP", "mC", "oP", "oC", "oI", "oF", "tP", "tI", "hR", "hP", "cP", "cI", "cF"):
        if search is None:
            check(f"conventional_setting finds the {bl} conventional cell", False, "no conventional_setting")
            continue
        bad = []
        for _ in range(n):
            cell, cen, system = draw(bl)
            Mc = random_rotation(rng) @ cell_to_Ar(*cell)
            Mred = buerger_reduce(Mc @ Q[cen])
            for M0 in (Mred, Mred @ SKEW, Mred @ LEFT):
                P = search(M0, system)
                if P is None:
                    bad.append((np.round(cell, 1), "none")); continue
                T = np.rint(np.linalg.solve(Mc, M0 @ P))
                if not (abs(round(np.linalg.det(T))) == 1 and np.allclose(Mc @ T, M0 @ P, atol=1e-6)
                        and manifold_resid(M0 @ P, system) < 1e-6):
                    bad.append((np.round(cell, 1), round(abs(np.linalg.det(P)))))
        check(f"conventional_setting finds the {bl} conventional cell ({3 * n} random exact starts)",
              not bad, bad[:2])


def main():
    print("refine_bravais from conventional and non-conventional starts (14 Bravais lattices + hR as hexagonal)")
    for i, (name, cell, cen, system) in enumerate(LATTICES):
        run_lattice(name, cell, cen, system, 1000 + i)
    print("refusals")
    refusals()
    print("reduction and centering")
    reduction_and_centering()
    print("conventional_setting sweep")
    search_sweep()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
