"""One axis-setting rule for the frame and for the reference cell (glint#181).

REGRESSION. Two "standard setting" helpers decided which axis is the tetragonal 4-fold, and
disagreed. `predict._canonical_axes` sorted the columns of M by length to (long, long, short) --
"c = the unique SHORT axis" -- while `stream_driver._conventional_tetragonal` took the two most
EQUAL lengths as a,b and the outlier as c, which is what `laue_ops_4mmm` (4-fold about c) and
`theoretical_unique` assume. For c < a (lysozyme 79/79/38) the two agree, so the benchmark cell
could never show it. For c > a they do not: (long, long, short) puts the 4-fold of 58/58/130 in b.

Where it bit: `_integrate_one` canonicalised every ACCEPTED frame with `_canonical_axes` and
predicted it on an `HKLGrid` whose candidate hkl box was built from the `_conventional_tetragonal`
reference cell -- a different setting -- so the grid was missing ~19% of the frame's reflections
(58/58/130, measured in the issue); then folded the frame's hkl into a 4/mmm accumulator whose
4-fold is about c while the frame's was about b. `write_fromfile` (the CrystFEL `tPc` handoff) and
`integrate_cxi` labelled the same wrong axis "c".

Both helpers are now thin wrappers over `glint.lattice.standardize_axes`. What is pinned here:
  1. `_canonical_axes(M) == _conventional_tetragonal(M)` -- EXACTLY, not merely up to handedness --
     for 58/58/130, 140/140/150 and lysozyme 79.02/79.02/37.98 at random orientations in random
     incoming settings (column permutation + signs), with and without per-frame-refine-sized jitter;
  2. the unique axis comes out as c (lengths (a, a, c)), det > 0, idempotent, and the output is a
     signed column permutation of the input -- the transformation semantics every caller relies on;
  3. no two lengths equal -> (long, long, short), the historical `_canonical_axes` order, so
     orthorhombic and lower cells (ClCRY4 54.15/87.29/141.33) are unchanged;
  4. the prediction regression from the issue: an `HKLGrid(_conventional_tetragonal(cell), dmin=2.0,
     gpu=False)` predicts EXACTLY the reflections `predict_spots` finds for frames canonicalised with
     `_canonical_axes` (tol 0.002, one 2000-px panel, 0.15 m, 1.322 A) -- 0 missing for all three
     tetragonal cells. On the pre-fix code this step reports the 19% shortfall for 58/58/130.

Run: `PYTHONPATH=. python experiments/test_axis_standardizer.py` (exit 1 on any failure).
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.lattice import cell_to_Ar, random_rotation
from glint.predict import _canonical_axes, predict_spots
from glint.stream_driver import HKLGrid, _conventional_tetragonal

try:                                              # absent on the pre-fix code: reported as a FAIL below,
    from glint.lattice import standardize_axes    # not as an ImportError, so the other checks still run
except ImportError:                               # and the pre-fix numbers are printed
    standardize_axes = None

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


# The cells from the issue: two c > a tetragonal cells that expose the disagreement, the c < a
# benchmark cell (lysozyme) that cannot, and an orthorhombic cell for the fallback contract.
TETRAGONAL = {
    "58/58/130":   (58.0, 58.0, 130.0),
    "140/140/150": (140.0, 140.0, 150.0),
    "lysozyme 79.02/79.02/37.98": (79.02, 79.02, 37.98),
}
ORTHORHOMBIC = {"ClCRY4 54.15/87.29/141.33": (54.15, 87.29, 141.33)}

PERMS = list(itertools.permutations(range(3)))
SIGNS = [np.array(s, float) for s in itertools.product((1.0, -1.0), repeat=3)]


def random_setting(rng, Ar):
    """The cell at a random orientation, in a random incoming setting: any column order, any signs --
    the freedom a per-frame indexer has when it hands back M."""
    perm = PERMS[rng.integers(len(PERMS))]
    sgn = SIGNS[rng.integers(len(SIGNS))]
    return random_rotation(rng) @ Ar[:, list(perm)] * sgn


def jittered(rng, a, b, c, f=0.004, ang=0.15):
    """A per-frame-refine-sized perturbation of the cell (0.4% lengths, 0.15 deg angles): the
    unconstrained triclinic fit never returns a and b EXACTLY equal, and the setting rule must not
    care."""
    return cell_to_Ar(a * (1 + rng.normal(0, f)), b * (1 + rng.normal(0, f)), c * (1 + rng.normal(0, f)),
                      90 + rng.normal(0, ang), 90 + rng.normal(0, ang), 90 + rng.normal(0, ang))


def is_signed_column_permutation(P, M):
    """Every column of P is +/- some column of M, each column of M used once."""
    used = set()
    for i in range(3):
        hit = [j for j in range(3) if j not in used
               and (np.allclose(P[:, i], M[:, j], atol=1e-12) or np.allclose(P[:, i], -M[:, j], atol=1e-12))]
        if not hit:
            return False
        used.add(hit[0])
    return True


def worst(vals):
    return float(max(vals)) if vals else 0.0


# ------------------------------------------------------------------------------ 1 + 2: the contract
print("the shared standardizer")
check("glint.lattice.standardize_axes exists (the ONE rule both wrappers call)", standardize_axes is not None)

rng = np.random.default_rng(181)
N_ORI = 40
for name, (a, b, c) in TETRAGONAL.items():
    Ar = cell_to_Ar(a, b, c, 90, 90, 90)
    disagree, wrong_c, lefthanded, not_idem, not_perm, shared = [], [], [], [], [], []
    for exact in (True, False):
        for _ in range(N_ORI):
            base = Ar if exact else jittered(rng, a, b, c)
            M = random_setting(rng, base)
            P1 = _canonical_axes(M)
            P2 = _conventional_tetragonal(M)
            disagree.append(np.max(np.abs(P1 - P2)))
            L = np.linalg.norm(P1, axis=0)
            # unique axis in c: a,b the closest (equal within 5%) pair, c the outlier -- to the jitter
            wrong_c.append(not (abs(L[0] - L[1]) <= 0.05 * L[0]
                                and abs(L[2] - L[0]) > abs(L[0] - L[1])
                                and abs(L[2] - L[1]) > abs(L[0] - L[1])))
            lefthanded.append(np.linalg.det(P1) <= 0)
            not_idem.append(not np.array_equal(_canonical_axes(P1), P1))
            not_perm.append(not is_signed_column_permutation(P1, M))
            if standardize_axes is not None:
                shared.append(np.max(np.abs(standardize_axes(M) - P1)))
    print(f" {name}")
    check("_canonical_axes == _conventional_tetragonal (exact, all orientations/settings/jitter)",
          worst(disagree) == 0.0, f"max |diff| = {worst(disagree):.3g} A")
    check("unique axis is c: lengths come out (a, a, c) with c the outlier",
          not any(wrong_c), f"{sum(wrong_c)} results with a/b unequal or c not the outlier")
    check("right-handed (det > 0)", not any(lefthanded), f"{sum(lefthanded)} left-handed results")
    check("idempotent (bit-exact on the second pass)", not any(not_idem), f"{sum(not_idem)} changed")
    check("output is a signed column permutation of the input", not any(not_perm), f"{sum(not_perm)} were not")
    if standardize_axes is not None:
        check("both wrappers return standardize_axes(M)", worst(shared) == 0.0, f"max |diff| = {worst(shared):.3g}")

# The exact incoming setting of the issue: (a, a, c) -- the reference cell as cell_to_Ar makes it.
print(" the issue's own measurement: the reference cell itself, no rotation")
for name, (a, b, c) in TETRAGONAL.items():
    Ar = cell_to_Ar(a, b, c, 90, 90, 90)
    L1 = np.round(np.linalg.norm(_canonical_axes(Ar), axis=0), 2)
    L2 = np.round(np.linalg.norm(_conventional_tetragonal(Ar), axis=0), 2)
    check(f"{name}: _canonical_axes -> {tuple(L1)}, _conventional_tetragonal -> {tuple(L2)}",
          np.array_equal(L1, L2) and L1[0] == L1[1] and L1[2] != L1[0], "settings differ")

# ----------------------------------------------------------------- 3: no equal pair -> (long, long, short)
print("\nno two lengths equal: the historical (long, long, short) order is kept")
for name, (a, b, c) in ORTHORHOMBIC.items():
    Ar = cell_to_Ar(a, b, c, 90, 90, 90)
    bad_order, disagree, lefthanded, not_idem = [], [], [], []
    for _ in range(N_ORI):
        M = random_setting(rng, jittered(rng, a, b, c))
        P1 = _canonical_axes(M)
        L = np.linalg.norm(P1, axis=0)
        bad_order.append(not (L[2] < L[0] <= L[1]))                 # (long, long, short), b the longest
        disagree.append(np.max(np.abs(P1 - _conventional_tetragonal(M))))
        lefthanded.append(np.linalg.det(P1) <= 0)
        not_idem.append(not np.array_equal(_canonical_axes(P1), P1))
    print(f" {name}")
    check("(long, long, short): c the shortest, b the longest", not any(bad_order), f"{sum(bad_order)} misordered")
    check("_canonical_axes == _conventional_tetragonal here too", worst(disagree) == 0.0, f"max |diff| = {worst(disagree):.3g}")
    check("right-handed (det > 0)", not any(lefthanded), f"{sum(lefthanded)} left-handed")
    check("idempotent", not any(not_idem), f"{sum(not_idem)} changed")
    # and it is exactly the old formula, up to the sign of column a that keeps det > 0
    M = random_setting(rng, Ar)
    o = np.argsort(np.linalg.norm(M, axis=0)); old = M[:, [o[1], o[2], o[0]]]
    P1 = _canonical_axes(M)
    check("equals the old sort-by-length result up to the handedness sign of column a",
          np.allclose(np.abs(P1), np.abs(old)) and np.allclose(P1[:, 1:], old[:, 1:]))

# ------------------------------------------------------------------- 4: the prediction regression
# The issue's setup: tol 0.002, one 2000-px panel (100 um pixels), 0.15 m, 1.322 A, dmin 2.0 A.
NPIX = 2000
PANELS = [dict(name="p0", fs=np.array([1.0, 0.0, 0.0]), ss=np.array([0.0, 1.0, 0.0]), res=1e4,
               cx=-(NPIX / 2.0 - 0.5), cy=-(NPIX / 2.0 - 0.5), coffset=0.0,
               min_fs=0, max_fs=NPIX - 1, min_ss=0, max_ss=NPIX - 1)]
CLEN, WL, DMIN, TOL, N_FRAMES = 0.15, 1.322, 2.0, 0.002, 20


def hkl_set(pred):
    return set(zip(pred["h"].tolist(), pred["k"].tolist(), pred["l"].tolist()))


print("\nHKLGrid built from the reference cell covers every predict_spots reflection of a canonicalised frame")
rng = np.random.default_rng(1810)
for name, (a, b, c) in TETRAGONAL.items():
    Ar = cell_to_Ar(a, b, c, 90, 90, 90)
    Mc = _conventional_tetragonal(Ar)                       # what _lock(standardize=True) builds the grid from
    grid = HKLGrid(Mc, DMIN, gpu=False)
    n_full = n_grid = n_missing = n_extra = 0
    for _ in range(N_FRAMES):
        Mcan = _canonical_axes(random_setting(rng, Ar))     # what _integrate_one predicts with
        full = hkl_set(predict_spots(Mcan, PANELS, CLEN, WL, dmin=DMIN, tol=TOL))
        got = hkl_set(grid.predict(Mcan, PANELS, CLEN, WL, tol=TOL))
        n_full += len(full); n_grid += len(got)
        n_missing += len(full - got); n_extra += len(got - full)
    pct = 100.0 * n_missing / max(n_full, 1)
    print(f" {name}: grid predicts {n_grid} of {n_full} reflections over {N_FRAMES} frames -> {pct:.1f} % missing")
    check(f"{name}: 0 missing", n_missing == 0, f"{n_missing} missing ({pct:.1f} %)")
    check(f"{name}: no extras either (identical reflection sets)", n_extra == 0, f"{n_extra} extra")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED:\n  " + "\n  ".join(FAILS))
    sys.exit(1)
print("ALL PASS")
