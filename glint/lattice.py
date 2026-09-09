"""Unit-cell / lattice conventions.

Crystallographic convention, no 2*pi factor:

    Ar  = real-space basis, columns (a, b, c)           [Angstrom]
    Br  = reciprocal basis,  columns (a*, b*, c*)        [1/Angstrom]
        = inv(Ar).T          so that  Ar.T @ Br = I  (a*.a = 1, a*.b = 0, ...)

A reciprocal-lattice node h = (h,k,l) sits at  g = Br @ h.
With a crystal orientation R in SO(3) the *observed* node is  g = R @ Br @ h.

Key identity used throughout the indexer
-----------------------------------------
Let  M = R @ Ar  (columns are the rotated *real* lattice vectors R a, R b, R c).
Then the observed reciprocal vectors satisfy

    g = M^{-T} h          (since (R Ar)^{-T} = R Br)
    h = round(M.T @ g)    (recover Miller indices once M is known)

So indexing == finding M, and the FFT of the observed point cloud {g_i}
peaks at the columns of M and their integer combinations (the direct lattice).
"""

from __future__ import annotations

import numpy as np


def cell_to_Ar(a, b, c, alpha, beta, gamma):
    """Cell parameters (lengths in A, angles in degrees) -> real basis Ar (columns a,b,c).

    Standard crystallographic setup: a along x, b in the xy-plane.
    """
    al, be, ga = np.radians([alpha, beta, gamma])
    ca, cb, cg = np.cos([al, be, ga])
    sg = np.sin(ga)
    # volume factor
    vfac = np.sqrt(max(1.0 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg, 1e-12))
    av = np.array([a, 0.0, 0.0])
    bv = np.array([b * cg, b * sg, 0.0])
    cv = np.array([c * cb, c * (ca - cb * cg) / sg, c * vfac / sg])
    return np.column_stack([av, bv, cv])


def Ar_to_Br(Ar):
    """Reciprocal basis (columns a*,b*,c*) from real basis."""
    return np.linalg.inv(Ar).T


def cell_params(Ar):
    """(a,b,c,alpha,beta,gamma) from a real basis matrix (columns a,b,c)."""
    a, b, c = Ar.T
    la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))
    ang = lambda u, v: np.degrees(
        np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    )
    return (la, lb, lc, ang(b, c), ang(a, c), ang(a, b))


# Two axis lengths count as "equal" -- a tetragonal/hexagonal a = b pair -- when they agree to this
# relative tolerance: the rtol same_lattice / consensus_cell (glint.multishot) already use to call two
# edge lengths the same edge, reused here rather than inventing a second notion of equality. A
# per-frame unconstrained refine scatters a and b of a tetragonal crystal by well under 1%, so every
# frame lands on the same branch of standardize_axes as the reference cell. The tolerance is NOT
# taken as proof that the pair is symmetry-equivalent: a genuinely orthorhombic cell whose a and b
# happen to fall inside it (100/103/150) is still standardized by lengths alone, a the shorter of
# the pair, so every frame of it lands in ONE setting too -- which is all the hkl grid, the merge
# and the CrystFEL handoff need from a setting.
AXIS_EQUAL_RTOL = 0.05


# Which axis rule a Laue class implies. The class is what a length rule cannot know, and guessing it
# from lengths alone is where the setting used to go unstable (glint#185 review): a cell whose three
# axes are all within the tolerance has no "equal pair" a tolerance can identify.
UNIQUE_C_LAUE = ("4/m", "4/mmm", "-3", "-3m1", "-31m", "6/m", "6/mmm")   # one unique axis, goes to c
LENGTH_ORDER_LAUE = ("mmm",)                                            # a <= b <= c IS the setting
# Everything else -- triclinic, the monoclinic settings, the rhombohedral settings and the cubic
# classes -- is left exactly as handed in: no length rule locates a monoclinic unique axis or a
# rhombohedral 3-fold, and for cubic every axis is equivalent so any permutation is already standard.


def standardize_axes(M, laue=None, rtol=AXIS_EQUAL_RTOL, centering=None):
    """Bravais-aware standard setting of a real-space basis (columns of ``M`` = a, b, c).

    The ONE axis-setting rule behind ``predict._canonical_axes`` (the per-frame setting the merge and
    the CrystFEL ``--indexing=file`` handoff use) and ``stream_driver._conventional_tetragonal`` (the
    reference cell the ``HKLGrid`` and the merge operators are built on). Those used to be two rules
    -- sort-by-length (long, long, short) versus most-equal-pair -- which agree for c < a (lysozyme
    79/79/38) and disagree for c > a: (long, long, short) puts the 4-fold axis of 58/58/130 in b, so
    a frame in that setting was predicted against a grid built for (58, 58, 130) and lost ~19% of its
    reflections, and the merge folded (h,0,0) with (0,k,0) while keeping (h,0,0) apart from its true
    equivalent (0,0,l) (glint#181).

    ``laue`` names the Laue class the caller is merging under (``stream_driver.laue_name`` keys, e.g.
    "4/mmm", "mmm", "2/m_uab", "-3m1"), and it decides the rule:

      * ``UNIQUE_C_LAUE`` (tetragonal, trigonal, hexagonal) -- the class guarantees exactly one
        unique axis, so the closest-length pair becomes a, b (a the SHORTER of the two) and the
        outlier becomes c, the axis the operators rotate about, whether c is short or long.
      * ``LENGTH_ORDER_LAUE`` (orthorhombic) -- a <= b <= c, with NO tolerance anywhere in the
        decision.
      * anything else, and ``laue=None`` on a caller that knows no class -- see below.

    Passing the class is what makes the setting stable. Deciding it from lengths alone cannot work
    when all three are similar: 100/103/106 has TWO pairs inside a 5% tolerance, so a closest-pair
    rule picked (103, 106) while a refine-sized change to 100/102.8/106 picked (100, 102.8) -- the
    same clearly ordered orthorhombic axes standardizing to different unique axes on consecutive
    frames, which ``mmm`` cannot absorb and which puts the grid and the merge back in disagreement
    (Copilot review of glint#185).

    So without a class this refuses to guess: the equal-pair rule is applied only when EXACTLY ONE
    pair lies within ``rtol`` -- the unambiguous case, 79/79/38 or 58/58/130 or a pseudo-tetragonal
    100/103/150 -- and otherwise the cell falls back to (long, long, short), the order
    ``_canonical_axes`` has always produced. Both cells of the example above take that fallback and
    land in the same setting, which is the property the caller needs.

    In every branch the setting is a function of the three LENGTHS alone, never of the order the
    indexer happened to hand the columns back in (a <= b always inside a chosen pair). For a true
    tetragonal cell a <-> b is a 4/mmm operator, so which of two equal-to-the-jitter axes is called a
    cannot affect the merge or the grid.

    KNOWN LIMIT -- the classes where a <-> b is NOT an operator. 4/m, -3 and 6/m contain only powers
    of the c-axis rotation plus inversion, so the reindexing that relates [a, b, c] to [b, a, -c] (a
    2-fold about [110]) lies outside the group, and two frames of the same crystal whose refined a
    and b differ only by noise can be sorted into settings the merge then treats as inequivalent.
    This is the ordinary merohedral indexing ambiguity of those classes, not something the sort
    introduces: the two settings are indistinguishable from the cell metric, which is all any
    standardizer sees, and resolving them needs the INTENSITIES (a Brehm-Diederichs-style pass over
    the merged data). Sorting by length at least makes the choice deterministic per frame rather than
    inheriting the indexer's column order. Nothing in the merge currently resolves it, so results
    merged under 4/m, -3 or 6/m carry that ambiguity; the tetragonal work in this repo runs under
    4/mmm, where a <-> b is an operator and the question does not arise (Copilot review of glint#185).

    The result is a column permutation of ``M``; when that permutation is odd one column is negated so
    det > 0 (a proper, right-handed basis). WHICH column is the same function of the setting rule in
    every branch: the unique-axis-c rule negates c, the length-order rule negates a. It has to be the
    same on both sides, because ``predict._canonical_axes`` (no class) and ``_conventional_tetragonal``
    (laue="4/mmm") standardize the SAME left-handed cell and must land on the same basis -- when the
    class-free path flipped a while the class-aware path flipped c, the two disagreed by 2|c| on
    58/58/130 and the frames were predicted against a grid in the other setting (glint#185). Either
    flip is a merge operator of the class that selects the rule -- (h,k,l) -> (h,k,-l) for 4/mmm and
    mmm, (h,k,l) -> (-h,k,l) for mmm -- so neither can move a reflection out of its ASU. Idempotent:
    ``standardize_axes(standardize_axes(M, laue), laue)`` equals ``standardize_axes(M, laue)``.
    """
    M = np.asarray(M, float)
    if str(centering or "").upper() in ("A", "B", "C"):
        # A base-centered cell names the centered face by the axes it is written on: permute them and
        # oC becomes oA, so the standardized cell no longer matches its own centering letter, the
        # systematic absences it implies, or the label written into the stream. Refused on EVERY
        # branch, not only the orthorhombic one -- an oC cell with two near-equal axes reaches the
        # unique-axis-c rule down the class-free path as well (glint#185).
        return M.copy()
    L = np.linalg.norm(M, axis=0)
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]

    def _reldiff(i, j):
        den = max(L[i], L[j])
        return abs(L[i] - L[j]) / den if den > 0 else 0.0

    def _sorted_order():
        o = np.argsort(L, kind="stable")
        return [int(o[0]), int(o[1]), int(o[2])]

    def _unique_c_order():
        i, j, k = min(pairs, key=lambda p: _reldiff(p[0], p[1]))
        if L[j] < L[i]:
            i, j = j, i                            # a the shorter of the pair: lengths decide the
        return [i, j, k]                           # order inside it, never the incoming columns

    if laue is not None and laue in UNIQUE_C_LAUE:
        order, unique_c = _unique_c_order(), True
    elif laue is not None and laue in LENGTH_ORDER_LAUE:
        order, unique_c = _sorted_order(), False   # a <= b <= c, no tolerance in the decision
    elif laue is not None:
        return M.copy()                            # triclinic / monoclinic / rhombohedral / cubic
    else:
        near = [p for p in pairs if _reldiff(p[0], p[1]) <= rtol]
        if len(near) == 1:                         # exactly one candidate: unambiguous, use it
            order, unique_c = _unique_c_order(), True
        else:                                      # none, or an ambiguous near-cubic cell
            if L[1] >= L[0] >= L[2]:               # already (middle, long, short), including ties
                order, unique_c = [0, 1, 2], False
            else:
                o = np.argsort(L, kind="stable")   # -> (long, long, short), the historical order
                order, unique_c = [int(o[1]), int(o[2]), int(o[0])], False
    P = M[:, order].copy()
    if (laue in ("-3", "-3m1", "-31m", "6/m", "6/mmm")
            and np.dot(P[:, 0], P[:, 1]) > 0):
        P[:, 1] = -P[:, 1]                     # conventional hexagonal gamma is obtuse (120 degrees)
    if np.linalg.det(P) < 0:
        # The RULE decides the flip, not the caller's class: the class-free path takes the
        # unique-axis-c rule too, and must flip the same column the class-aware path does or the
        # frames and the reference cell land 2|c| apart (glint#185).
        P[:, 2 if unique_c else 0] *= -1        # keep a proper (right-handed) basis
    return P


_BR = np.arange(-3, 4)
_BN = np.array(np.meshgrid(_BR, _BR, _BR, indexing="ij")).reshape(3, -1).T
_BN = _BN[np.any(_BN != 0, axis=1)]


def buerger_reduce(M):
    """Reduced basis = 3 shortest independent lattice vectors (columns).

    Canonicalizes a basis up to point group, so a non-primitive expression of a
    lattice (e.g. 'true c-axis + true a-axis + a face DIAGONAL' that fits a couple
    extra spots) collapses onto the true primitive cell. Returns the reduced 3x3,
    or the input unchanged if the lattice vectors are degenerate (collinear/coplanar
    within the search box)."""
    M = np.asarray(M, float)
    V = (M @ _BN.T).T
    L = np.linalg.norm(V, axis=1)
    o = np.argsort(L); V, L = V[o], L[o]
    v1 = V[0]; v2 = None
    for k in range(1, len(V)):
        if np.linalg.norm(np.cross(V[k], v1)) > 1e-3 * L[k] * L[0]:
            v2 = V[k]; break
    if v2 is None:
        return M
    nrm = np.cross(v1, v2); v3 = None
    for k in range(1, len(V)):
        if abs(V[k] @ nrm) > 1e-3 * L[k] * np.linalg.norm(nrm):
            v3 = V[k]; break
    if v3 is None:
        return M
    return np.column_stack([v1, v2, v3])


def reduced_params(M):
    """(sorted axis lengths, sorted |cos angle|) of the Buerger-reduced cell --
    a rotation- and basis-invariant fingerprint that respects degeneracy and volume."""
    R = buerger_reduce(M); cols = [R[:, i] for i in range(3)]
    lens = np.array([np.linalg.norm(c) for c in cols])
    cs = np.array([abs(cols[0] @ cols[1]) / (lens[0] * lens[1]),
                   abs(cols[0] @ cols[2]) / (lens[0] * lens[2]),
                   abs(cols[1] @ cols[2]) / (lens[1] * lens[2])])
    return np.sort(lens), np.sort(cs)


def random_rotation(rng):
    """Uniform random rotation matrix in SO(3) (Shoemake)."""
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
    )
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


SYMMETRIES = ["cubic", "tetragonal", "orthorhombic", "hexagonal",
              "trigonal", "monoclinic", "triclinic"]


def random_cell(rng, symmetry="random", lo=30.0, hi=120.0):
    """Random cell parameters (a,b,c,alpha,beta,gamma) of a given symmetry class.

    'monoclinic' and 'triclinic' are the sheared cases. 'random' picks a class
    uniformly. Triclinic angles are redrawn until the cell volume is non-degenerate.
    """
    if symmetry == "random":
        symmetry = SYMMETRIES[rng.integers(len(SYMMETRIES))]
    L = lambda: rng.uniform(lo, hi)
    if symmetry == "cubic":
        a = L(); return (a, a, a, 90.0, 90.0, 90.0)
    if symmetry == "tetragonal":
        return (lambda a: (a, a, L(), 90.0, 90.0, 90.0))(L())
    if symmetry == "orthorhombic":
        return (L(), L(), L(), 90.0, 90.0, 90.0)
    if symmetry == "hexagonal":
        return (lambda a: (a, a, L(), 90.0, 90.0, 120.0))(L())
    if symmetry == "trigonal":                       # rhombohedral setting
        a = L(); ang = rng.uniform(70.0, 110.0); return (a, a, a, ang, ang, ang)
    if symmetry == "monoclinic":
        return (L(), L(), L(), 90.0, rng.uniform(95.0, 125.0), 90.0)
    if symmetry == "triclinic":
        while True:
            al, be, ga = rng.uniform(70.0, 110.0, 3)
            ca, cb, cg = np.cos(np.radians([al, be, ga]))
            if 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg > 0.1:
                return (L(), L(), L(), al, be, ga)
    raise ValueError(f"unknown symmetry: {symmetry}")
