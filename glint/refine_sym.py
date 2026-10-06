"""Bravais-symmetry-CONSTRAINED prediction refinement.

GLINT's native ``index.refine`` fits an unconstrained reciprocal basis
``C = (G H^T)(H H^T)^{-1}`` -- all 9 entries free (a triclinic cell).  Under
realistic peak-position noise those extra DOF absorb noise into a spurious
lower-symmetry cell distortion (measured on real lyso: ``a != b`` by ~2.5%,
``gamma`` ~92.9 off 90), which couples into the orientation and drifts the
PREDICTED spot positions off the Bravais manifold -- the deficiency flagged at
``predict.py`` write_solution_file and the reason a native GLINT merge lags the
CrystFEL ``--fromfile`` handoff (whose prediction-refinement imposes lattice
symmetry).

``refine_bravais`` replaces the free 3x3 fit with a low-DOF Gauss-Newton / LM
fit over ``[rotation tangent (3) + the SYMMETRY-FREE cell params (1..6)]``, the
lattice symmetry built INTO the parameterization so ``a=b`` / angle=90/120 hold
to machine precision (a hard constraint, not a soft penalty): the cell is
REBUILT from ``cell_to_Ar`` each iterate, so it can never leave the manifold.

Convention (matches ``index.refine`` / ``predict.recip_from_M`` exactly)::

    M   = R @ Ar          (columns = rotated real axes,     A)
    C   = inv(M).T = R @ B (columns a*,b*,c*,               1/A)
    g_i = C @ h_i          (observed reciprocal node)
    h_i = round(inv(C) @ g_i) = round(M.T @ g_i)

Free params per Bravais system (standard setting; fixed entries substituted)::

    triclinic     (a,b,c,al,be,ga)          n_p = 6   (== unconstrained control)
    monoclinic    (a,b,c,be)  al=ga=90       n_p = 4   (unique axis b)
    orthorhombic  (a,b,c)     al=be=ga=90     n_p = 3
    hexagonal     (a,c)       b=a al=be=90 ga=120  n_p = 2  (unique axis c)
    trigonal      (a,al)      b=c=a be=ga=al   n_p = 2   (rhombohedral setting)
    tetragonal    (a,c)       b=a al=be=ga=90  n_p = 2  (unique axis c)
    cubic         (a)         b=c=a al=be=ga=90  n_p = 1

The parameterisation is of the CONVENTIONAL cell, so ``refine_bravais`` first puts the start
basis in that setting (``standardize_setting``, then ``conventional_settings`` when relabelling
the axes is not enough -- e.g. the Buerger-reduced primitive cells the blind indexer emits for
centred lattices); see its docstring for which setting the result comes back in.

The rotation is retracted on SO(3) by a left exponential map so it stays a
proper rotation exactly; LM damping + a length/volume guard prevent divergence.

Default OFF everywhere it is wired in (``predict.integrate_cxi(sym_refine=...)``),
so nothing regresses if it underperforms.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np

from .lattice import cell_to_Ar, Ar_to_Br, cell_params, buerger_reduce

# free-parameter dimension per system
_NP = {"cubic": 1, "tetragonal": 2, "hexagonal": 2, "trigonal": 2,
       "orthorhombic": 3, "monoclinic": 4, "triclinic": 6}


# ----------------------------------------------------------------------------
# small SO(3) helpers
# ----------------------------------------------------------------------------
def _skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def _expmap(w):
    """Rodrigues: rotation matrix from an axis-angle vector w."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        K = _skew(w)
        return np.eye(3) + K + 0.5 * K @ K
    k = w / th
    K = _skew(k)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _polar(X):
    """Nearest proper rotation to X (polar factor, det>0)."""
    U, _, Vt = np.linalg.svd(X)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] = -U[:, -1]
        R = U @ Vt
    return R


# ----------------------------------------------------------------------------
# cell <-> free-param mapping (symmetry built in)
# ----------------------------------------------------------------------------
def _expand(p, system):
    """Free params p -> (a,b,c,alpha,beta,gamma) for cell_to_Ar."""
    if system == "cubic":
        a = p[0]; return (a, a, a, 90.0, 90.0, 90.0)
    if system == "tetragonal":
        a, c = p; return (a, a, c, 90.0, 90.0, 90.0)
    if system == "hexagonal":
        a, c = p; return (a, a, c, 90.0, 90.0, 120.0)
    if system == "trigonal":
        a, al = p; return (a, a, a, al, al, al)
    if system == "orthorhombic":
        a, b, c = p; return (a, b, c, 90.0, 90.0, 90.0)
    if system == "monoclinic":
        a, b, c, be = p; return (a, b, c, 90.0, be, 90.0)
    if system == "triclinic":
        return tuple(p)
    raise ValueError(f"unknown system {system}")


def _project(cell, system):
    """Project (a,b,c,al,be,ga) onto the Bravais manifold -> free params p0."""
    a, b, c, al, be, ga = cell
    if system == "cubic":
        return np.array([(a + b + c) / 3.0])
    if system == "tetragonal":
        return np.array([(a + b) / 2.0, c])
    if system == "hexagonal":
        return np.array([(a + b) / 2.0, c])
    if system == "trigonal":
        return np.array([(a + b + c) / 3.0, (al + be + ga) / 3.0])
    if system == "orthorhombic":
        return np.array([a, b, c])
    if system == "monoclinic":
        return np.array([a, b, c, be])
    if system == "triclinic":
        return np.array([a, b, c, al, be, ga])
    raise ValueError(f"unknown system {system}")


def _B_cell(p, system):
    """Reciprocal basis B (columns a*,b*,c*) rebuilt from free params -> ON the manifold."""
    return Ar_to_Br(cell_to_Ar(*_expand(p, system)))


# ----------------------------------------------------------------------------
# axis-setting standardization (unique axis where the parameterization expects it)
# ----------------------------------------------------------------------------
def _closest_pair(lens):
    """Indices (i,j) of the two most nearly equal of three lengths."""
    d = [(abs(lens[0] - lens[1]), 0, 1),
         (abs(lens[0] - lens[2]), 0, 2),
         (abs(lens[1] - lens[2]), 1, 2)]
    d.sort()
    return d[0][1], d[0][2]


def standardize_setting(M, system):
    """Reorder/relabel the real-space axes (columns of M) into the standard setting for
    ``system`` so the free-param constraints (a=b, unique axis, angles) act on the right
    axes.  Handedness preserved (proper rotation).  Idempotent when already standard.

    Only column permutations and sign changes that keep every angle a symmetry-allowed value:

      * hexagonal: gamma made obtuse (b negated when a.b > 0), so a reduced cell's 60 degrees
        becomes the 120 the parameterisation fixes -- the rule ``lattice.standardize_axes`` uses;
      * trigonal: a mixed-sign rhombohedral cell (e.g. 75/105/105, as a reducer hands it back) has
        the column shared by its two odd angles negated, so all three agree.  Skipped when any angle
        is within ``_TRIGONAL_SIGN_MIN`` of 90, where the signs can be the noise's;
      * handedness: a left-handed result is negated as a whole (``-M``), which keeps every angle.
        Negating column a alone, as this did before, turned gamma 120 into 60 and broke a
        rhombohedral cell's equal angles.

    A permutation and sign change cannot reach the conventional cell from every basis of the
    lattice -- a Buerger-reduced cell of a centred lattice is not one -- so ``refine_bravais`` checks
    the result with ``in_setting`` and otherwise asks ``conventional_settings`` for the transform."""
    M = np.asarray(M, float)
    lens = np.linalg.norm(M, axis=0)
    if system in ("cubic", "trigonal", "triclinic", "orthorhombic"):
        # cubic/trigonal: all axes equivalent.  triclinic: no constraint.  orthorhombic:
        # a,b,c all independently free (constraint = angles 90 only), so no axis is special
        # and reordering distinct axes would just relabel the orientation frame (not a symmetry).
        order = [0, 1, 2]
    elif system in ("tetragonal", "hexagonal"):
        i, j = _closest_pair(lens)          # a,b = the equal pair; c = odd one out -> unique axis c
        k = 3 - i - j
        order = [i, j, k]
    elif system == "monoclinic":
        cols = [M[:, i] / lens[i] for i in range(3)]
        cs = [abs(cols[0] @ cols[1]), abs(cols[0] @ cols[2]), abs(cols[1] @ cols[2])]
        pair = [(0, 1), (0, 2), (1, 2)][int(np.argmax(cs))]   # the oblique (a,c) pair
        i, k = pair
        j = 3 - i - k                        # the unique axis b
        order = [i, j, k]
    else:
        raise ValueError(f"unknown system {system}")
    Mo = M[:, order].copy()
    return _fix_signs(Mo, system)


def _fix_signs(Mo, system, near90_guard=True):
    """Angle-preserving sign changes of ``standardize_setting`` (in place on ``Mo``, returned).
    A no-op on a right-handed basis already in the conventional setting.  ``near90_guard=False``
    applies the rhombohedral sign rule even when an angle is near 90 (``conventional_setting``,
    whose score already assumed it)."""
    if system == "hexagonal" and Mo[:, 0] @ Mo[:, 1] > 0:
        Mo[:, 1] = -Mo[:, 1]                 # gamma 60 -> 120: the other hexagonal basis of the plane
    elif system == "trigonal":
        cs = _cos3(Mo)                       # cos alpha (b,c), cos beta (a,c), cos gamma (a,b)
        if not near90_guard or np.all(np.abs(cs) > np.sin(np.radians(_TRIGONAL_SIGN_MIN))):
            want = np.sign(np.prod(cs))      # invariant under column sign changes
            odd = np.sign(cs) != want
            if want != 0 and odd.sum() == 2:     # negating one column flips exactly two cosines
                k = int(np.flatnonzero(~odd)[0])  # the column NOT in the one agreeing angle
                Mo[:, k] = -Mo[:, k]
    if np.linalg.det(Mo) < 0:
        Mo = -Mo                             # keep a proper (right-handed) basis; keeps every angle
    return Mo


def _cos3(M):
    """(cos alpha, cos beta, cos gamma) of the basis ``M``."""
    G = M.T @ M
    n = np.sqrt(np.diag(G))
    return np.array([G[1, 2] / (n[1] * n[2]), G[0, 2] / (n[0] * n[2]), G[0, 1] / (n[0] * n[1])])


# ----------------------------------------------------------------------------
# setting check, and the transform from any basis to the conventional cell
# ----------------------------------------------------------------------------
# A basis is "in the setting" when, after ``standardize_setting``, its cell is this close to the point
# ``_project`` puts it on the manifold.  Inside, the refine starts from it as it always did.  The angle
# bound separates settings (a wrong setting is off by tens of degrees: gamma 60 for 120, a reduced
# body-centred cell at 61/61/90 or 109.5, a mixed rhombohedral 75/105/105) from fit noise (real
# lysozyme's unconstrained gamma drifts ~3 degrees).  The length bound is loose on purpose, so a noisy
# conventional start (synth_refine_validate's S2 perturbs each length by 2% rms) keeps the unchanged
# path; it only catches a request for a system the lattice plainly does not have.
SETTING_ANGLE_TOL = 5.0          # degrees
SETTING_LENGTH_RTOL = 0.10       # relative
# ``conventional_setting`` accepts a candidate cell within SETTING_ANGLE_TOL and this length bound.  It
# is tighter than the one above because the search, unlike the check, chooses among thousands of
# integer combinations, and at 10% (a/b up to 1.22) many lattices offer a cell that is "tetragonal" or
# "cubic" only by tolerance.  Noise in a real start is ~1% here (lysozyme's a != b by 2.5% is 1.25%).
SEARCH_LENGTH_RTOL = 0.05        # relative
# standardize_setting's rhombohedral sign rule needs every angle this far from 90: closer, a 1-degree
# fit error can put an angle on the wrong side, and the rule would then make the cell worse
_TRIGONAL_SIGN_MIN = 2.0         # degrees
# Which candidate cells are distinct.  Cells describing the SAME symmetry (tP's P cell and its C cell,
# hR's rhombohedral cell and (a+b, b+c, c+a)) share their symmetry axis -- the tetragonal or hexagonal
# c, the monoclinic b, the rhombohedral a+b+c, the orthorhombic or cubic frame -- and their scores
# differ only by the noise in M; cells describing a different (pseudo-)symmetry have a different axis.
# So the candidates are grouped by axis (within ``_AXIS_TOL`` degrees), best-scoring group first, and
# each group is represented by its shortest cell, then the one of smallest index: P before C, I before
# F, the reduced monoclinic a and c.
_AXIS_TOL = 3.0                  # degrees
# ... and among those, only cells scoring within ``_SCORE_RATIO * best + _SCORE_SLACK`` of the best: the
# equivalent descriptions' scores differ by up to ~2.3x (a tP P cell's length error is its C cell's angle
# error), while a long vector that merely looks perpendicular -- (a+b)/2 + c of a 184/24/106 mC cell is
# 4.4 degrees off b -- shares the axis but not the fit.
_SCORE_RATIO = 3.0
_SCORE_SLACK = 0.05
_MAX_INDEX = 4                   # conventional / primitive volume: P 1, A B C I 2, R (hex axes) 3, F 4
_INDICES = {"monoclinic": (1, 2), "orthorhombic": (1, 2, 4), "tetragonal": (1, 2),
            "hexagonal": (1, 3), "trigonal": (1,), "cubic": (1, 2, 4)}
_CENTERINGS = {
    "monoclinic": {2: (((0.5, 0.5, 0.0),),)},
    "orthorhombic": {2: (((0.5, 0.5, 0.0),), ((0.5, 0.0, 0.5),),
                         ((0.0, 0.5, 0.5),), ((0.5, 0.5, 0.5),)),
                     4: (((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),)},
    "tetragonal": {2: (((0.5, 0.5, 0.5),),)},
    "hexagonal": {3: (((2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)),)},
    "cubic": {2: (((0.5, 0.5, 0.5),),),
              4: (((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),)},
}

# coefficient vectors (in the reduced basis) of the candidate conventional axes, one per +-pair,
# and every triple of them whose index |det| is 1..4 -- fixed, so computed once
_CN = np.array(list(itertools.product(range(-2, 3), repeat=3)), float)
_CN = _CN[[next((x for x in n if x != 0), 0) > 0 for n in _CN]]           # 62 vectors
_TRI = np.array(list(itertools.combinations(range(len(_CN)), 3)))
_TRI_IDX = np.rint(np.abs(np.linalg.det(_CN[_TRI].transpose(0, 2, 1)))).astype(int)
_keep = (_TRI_IDX >= 1) & (_TRI_IDX <= _MAX_INDEX)
_TRI, _TRI_IDX = _TRI[_keep], _TRI_IDX[_keep]                             # 17701 triples
del _keep


def setting_deviation(M, system):
    """(relative length, degrees): how far ``cell_params(M)`` is from the manifold point ``_project``
    maps it to -- the largest relative length change and the largest angle change."""
    cell = cell_params(np.asarray(M, float))
    e = _expand(_project(cell, system), system)
    dl = max(abs(cell[i] - e[i]) / e[i] for i in range(3))
    da = max(abs(cell[3 + i] - e[3 + i]) for i in range(3))
    return float(dl), float(da)


def in_setting(M, system):
    """True when ``M`` (as ``standardize_setting`` returns it) is within the setting tolerances."""
    dl, da = setting_deviation(M, system)
    return dl <= SETTING_LENGTH_RTOL and da <= SETTING_ANGLE_TOL


def _reduced(M):
    """A shortest-vector basis of the lattice of ``M``: LLL (delta 0.75) first, then ``buerger_reduce``
    until it stops shortening. LLL first because Buerger reduction alone stalls on a strongly sheared
    start (60 * [[1, 50, 0], [0, 1, 0], [0, 0, 1]] stayed at 60/60/2821). Every step is an integer
    column operation or a swap, so the lattice is unchanged; checked on 3,000 random cells under
    unimodular shears up to 50 (same lattice to 5e-10, never longer than Buerger alone, first vector
    the shortest). A singular or non-finite ``M`` is returned as it came: nothing downstream can use it,
    and the Gram-Schmidt step would divide by zero."""
    Mr = np.asarray(M, float).copy()
    if not np.isfinite(Mr).all() or abs(np.linalg.det(Mr)) <= 1e-12 * max(1.0, np.abs(Mr).max() ** 3):
        return Mr
    k = 1
    while k < 3:
        Bstar = np.empty_like(Mr)
        mu = np.zeros((3, 3))
        norms = np.zeros(3)
        for i in range(3):
            Bstar[:, i] = Mr[:, i] - sum(mu[i, j] * Bstar[:, j] for j in range(i))
            norms[i] = Bstar[:, i] @ Bstar[:, i]
            for j in range(i + 1, 3):
                mu[j, i] = Mr[:, j] @ Bstar[:, i] / norms[i]
        for j in range(k - 1, -1, -1):
            q = int(np.rint(mu[k, j]))
            if q:
                Mr[:, k] -= q * Mr[:, j]
                mu[k, :j] -= q * mu[j, :j]
                mu[k, j] -= q
                Bstar[:, k] = Mr[:, k] - sum(mu[k, i] * Bstar[:, i] for i in range(k))
                norms[k] = Bstar[:, k] @ Bstar[:, k]
                for i in range(k + 1, 3):
                    mu[i, k] = Mr[:, i] @ Bstar[:, k] / norms[k]
        if norms[k] >= (0.75 - mu[k, k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            Mr[:, [k, k - 1]] = Mr[:, [k - 1, k]]
            k = max(k - 1, 1)

    while True:
        Mn = buerger_reduce(Mr)
        if np.linalg.norm(Mn, axis=0).sum() >= np.linalg.norm(Mr, axis=0).sum() * (1 - 1e-12):
            break
        Un = np.linalg.solve(Mr, Mn)
        Uni = np.rint(Un)
        if np.abs(Un - Uni).max() > 1e-6 or abs(round(np.linalg.det(Uni))) != 1:
            break
        Mr = Mn
    return Mr


def _centering_compatible(P, system, index):
    """Whether ``P``'s index translations are a centering allowed for ``system``."""
    if index == 1:
        return True
    expected = _CENTERINGS[system].get(index)
    if expected is None:
        return False
    translations = set()
    for n in itertools.product(range(index), repeat=3):
        t = np.linalg.solve(P, n)
        t -= np.floor(t + 1e-8)
        translations.add(tuple(np.round(t, 6)))
        if len(translations) == index:
            break
    return any(translations == {(0.0, 0.0, 0.0), *(tuple(np.round(t, 6)) for t in centering)}
               for centering in expected)


def conventional_settings(M, system, max_groups=3):
    """Integer P's with ``M @ P`` a conventional cell of ``system`` in the lattice spanned by ``M``,
    one per candidate symmetry axis, best first (at most ``max_groups``).

    ``M`` may be any basis of the lattice: a Buerger-reduced cell, a hexagonal cell with gamma 60, a
    mixed-sign rhombohedral cell, the primitive cell of a centred lattice.  The conventional cell is
    then a sublattice basis of index ``|det P|``: 1 for a primitive lattice, 2 for A/B/C/I, 3 for an
    R lattice on hexagonal axes, 4 for F.  It is found by enumeration, not from a lattice-character
    table: every triple of lattice vectors whose coefficients in the reduced basis lie in [-2, 2] and
    whose index is 1..4 is scored by how far its metric is from the system's manifold (the same
    distance as ``setting_deviation``, with any column order and signs).  The triples within
    ``SETTING_ANGLE_TOL`` and ``SEARCH_LENGTH_RTOL`` are grouped by symmetry axis, best score first;
    each group gives its shortest cell (then smallest index) among those scoring about as well as the
    group's best (``_AXIS_TOL``, ``_SCORE_RATIO``, ``_SCORE_SLACK``).  Each cell is returned in the
    setting ``_project`` expects (unique axis c for tetragonal/hexagonal, b for monoclinic; gamma 120;
    equal rhombohedral angles; right-handed).

    A lattice that is close to a higher or different symmetry than it has (pseudo-symmetry) can
    produce a candidate that fits as well as the true one, within the noise of ``M``; then either
    may be returned (``experiments/test_refine_bravais_setting.py`` checks all 14 Bravais types).

    Only the indices a conventional cell of the system can have are searched (``_INDICES``: e.g. 1 or
    2 for monoclinic, P or C/I).  The list is empty when no triple qualifies -- the lattice does not
    have the requested symmetry within the tolerances, or ``M`` is (nearly) singular.  The first
    entry is the metric's best guess; ``refine_bravais`` refines each and keeps the one that fits the
    spots best, because on a noisy ``M`` a lattice can offer another axis that fits the METRIC as well
    (a pseudo-symmetry; common on random skewed monoclinic cells with 0.5% noise in M), but a cell 1
    degree off the lattice misses a 0.3/A spot by ~0.005/A once refined.

    The lattice's symmetry may be higher than ``system``: a cP lattice asked for ``tetragonal`` gives
    its P cell, an hR lattice asked for ``hexagonal`` its R-centred hexagonal cell (index 3).  An hP
    lattice asked for ``trigonal`` gives nothing: ``trigonal`` is searched as the rhombohedral
    PRIMITIVE cell only (index 1)."""
    M = np.asarray(M, float)
    if system not in _NP:
        raise ValueError(f"unknown system {system}")
    if system == "triclinic":
        return [np.eye(3)]
    L0 = np.linalg.norm(M, axis=0)
    if not np.all(L0 > 0) or abs(np.linalg.det(M)) < 1e-6 * np.prod(L0):
        return []
    Mr = _reduced(M)
    V = Mr @ _CN.T                                     # (3, K) candidate axes
    L = np.linalg.norm(V, axis=0)
    Cs = (V.T @ V) / np.outer(L, L)
    i, j, k = _TRI[:, 0], _TRI[:, 1], _TRI[:, 2]
    l3 = np.stack([L[i], L[j], L[k]], 1)
    c3 = np.stack([Cs[j, k], Cs[i, k], Cs[i, j]], 1)   # opposite pair: col0 = (j,k), col1 = (i,k), col2 = (i,j)
    phi = np.degrees(np.arccos(np.clip(np.abs(c3), 0.0, 1.0)))   # angle with signs free, in [0, 90]
    off90 = 90.0 - phi

    # per system: deviation (dl, da) for each choice of the special axis s (0, 1, 2 = i, j, k)
    def pair_dl(s):                                    # |l_p - l_q| / (l_p + l_q) of the other two
        p, q = [x for x in range(3) if x != s]
        return np.abs(l3[:, p] - l3[:, q]) / (l3[:, p] + l3[:, q])

    def mean_dl():
        m = l3.mean(1, keepdims=True)
        return np.abs(l3 - m).max(1) / m[:, 0]

    nT = len(_TRI)
    zeros = np.zeros(nT)
    if system in ("orthorhombic", "cubic"):
        da = off90.max(1)
        dl = mean_dl() if system == "cubic" else zeros
        dls, das = [dl], [da]
    elif system == "tetragonal":
        da = off90.max(1)
        dls, das = [pair_dl(s) for s in range(3)], [da] * 3
    elif system == "hexagonal":
        # the angle opposite axis s is the in-plane one (60 or 120 with signs free); the other two 90
        dls = [pair_dl(s) for s in range(3)]
        das = [np.maximum(np.abs(phi[:, s] - 60.0),
                          np.delete(off90, s, axis=1).max(1)) for s in range(3)]
    elif system == "trigonal":
        sgn = np.sign(np.prod(c3, axis=1))[:, None]    # all acute (+) or all obtuse (-) after signs
        th = np.where(sgn >= 0, phi, 180.0 - phi)
        da = np.abs(th - th.mean(1, keepdims=True)).max(1)
        dls, das = [mean_dl()], [da]
    else:                                              # monoclinic: s = unique axis b
        # angles between s and the other two are the 90s; the one opposite s (beta) is free
        dls = [zeros] * 3
        das = [np.delete(off90, s, axis=1).max(1) for s in range(3)]
    sc = np.stack([np.maximum(dl / SEARCH_LENGTH_RTOL, da / SETTING_ANGLE_TOL)
                   for dl, da in zip(dls, das)], 1)    # (T, n_choices)
    s_best = sc.argmin(1)
    score = sc[np.arange(nT), s_best]
    ok = (score <= 1.0) & np.isin(_TRI_IDX, _INDICES[system])
    if not ok.any():
        return []
    cand = np.flatnonzero(ok)
    # Centering compatibility is decided here, before the axis groups: a group's score cutoff is set
    # by its best member, and an incompatible cell (an A-centred candidate for a monoclinic request)
    # scoring best would otherwise set a cutoff that excludes every compatible cell of its axis and
    # take the whole axis group with it (Copilot review of #247). The sublattice, hence the
    # centering, depends only on which three lattice vectors are used, not on their order or signs.
    def _setting(t):
        """Integer P of triple ``t`` in the setting ``_project`` expects (unique axis last, or b for
        monoclinic; signs fixed), or None if it is not a lattice basis of its index or its centering
        is not one the system has. The centering translations in ``_CENTERINGS`` are written for that
        setting (R obverse, C for monoclinic), so the order matters here."""
        s = int(s_best[t])
        tri = [int(x) for x in _TRI[t]]
        if system in ("tetragonal", "hexagonal", "monoclinic"):
            p, q = sorted((x for x in range(3) if x != s), key=lambda x: l3[t, x])
            order = [tri[p], tri[q], tri[s]] if system != "monoclinic" else [tri[p], tri[s], tri[q]]
        else:
            order = [tri[x] for x in np.argsort(l3[t], kind="stable")]
        Mc = _fix_signs(V[:, order].copy(), system, near90_guard=False)
        P = np.linalg.solve(M, Mc)
        Pi = np.rint(P)
        index = int(round(abs(np.linalg.det(Pi))))
        if (np.abs(P - Pi).max() < 1e-6 and index == int(_TRI_IDX[t])
                and _centering_compatible(Pi, system, index)):
            return Pi
        return None
    settings = {int(t): _setting(t) for t in cand}
    cand = np.array([t for t in cand if settings[int(t)] is not None], int)
    if cand.size == 0:
        return []
    sig = _axis_signature(V, l3, c3, cand, s_best[cand], system)      # (n, m, 3) unit axes
    out, left = [], np.arange(len(cand))
    while left.size and len(out) < max_groups:
        b = left[np.argmin(score[cand[left]])]                       # best remaining: picks the axis
        dots = np.abs(np.einsum("nmd,kd->nmk", sig[left], sig[b]))
        same = (dots.max(2) >= np.cos(np.radians(_AXIS_TOL))).all(1)
        grp = cand[left[same & (score[cand[left]] <= _SCORE_RATIO * score[cand[b]] + _SCORE_SLACK)]]
        left = left[~same]
        # every member of grp has a valid setting (filtered above): the group's shortest cell, then
        # smallest index, then best score
        t = grp[np.lexsort((score[grp], _TRI_IDX[grp], l3[grp].sum(1)))[0]]
        out.append(settings[int(t)])
    return out


def conventional_setting(M, system):
    """The best of ``conventional_settings``: integer P with ``M @ P`` the conventional cell of
    ``system`` in the lattice of ``M``, or None if there is none within the tolerances."""
    if system == "triclinic":
        return np.eye(3)
    out = conventional_settings(M, system, max_groups=1)
    return out[0] if out else None


def _axis_signature(V, l3, c3, cand, s, system):
    """Unit symmetry-axis directions (sign-free) of candidate triples ``_TRI[cand]`` -- shape
    (n, 1, 3), or (n, 3, 3) for the orthorhombic and cubic frames.  See ``_AXIS_TOL``."""
    tri = _TRI[cand]
    cols = V[:, tri].transpose(1, 2, 0) / l3[cand][:, :, None]       # (n, 3, 3): unit axes i, j, k
    if system in ("orthorhombic", "cubic"):
        return cols
    if system == "trigonal":                           # a+b+c after the rhombohedral sign rule
        c = c3[cand]
        want = np.where(np.prod(c, axis=1) >= 0, 1.0, -1.0)[:, None]
        odd = np.sign(c) != want
        flip = (odd.sum(1) == 2)[:, None] & ~odd       # the column opposite the agreeing angle
        d = (np.where(flip, -1.0, 1.0)[:, :, None] * cols).sum(1)
    else:                                              # tetragonal / hexagonal c, monoclinic b
        d = cols[np.arange(len(cand)), s]
    return (d / np.linalg.norm(d, axis=1, keepdims=True))[:, None, :]


# ----------------------------------------------------------------------------
# the constrained refine
# ----------------------------------------------------------------------------
def _residual(R0, w, p, hkl, g, system):
    R = _expmap(w) @ R0
    C = R @ _B_cell(p, system)
    return (hkl @ C.T) - g            # (m,3) predicted - observed


def _lm_inner(R0, p, hkl, g, system, refine_cell, max_steps=8):
    """Levenberg-Marquardt on x=[w(3); p] with fixed hkl assignment. Rotation retracted
    on SO(3). Returns (R0, p, final_cost)."""
    n_p = _NP[system] if refine_cell else 0
    n = 3 + n_p
    w = np.zeros(3)
    r = _residual(R0, w, p, hkl, g, system).reshape(-1)
    cost = float(r @ r)
    mu = 1e-3
    for _ in range(max_steps):
        # finite-difference Jacobian (n params, m*3 residuals)
        m3 = r.size
        J = np.zeros((m3, n))
        for a in range(3):                        # rotation tangent
            dw = np.zeros(3); dw[a] = 1e-6
            rp = _residual(R0, w + dw, p, hkl, g, system).reshape(-1)
            rm = _residual(R0, w - dw, p, hkl, g, system).reshape(-1)
            J[:, a] = (rp - rm) / (2e-6)
        for b in range(n_p):                      # cell params
            eps = 1e-5 * max(abs(p[b]), 1.0)
            pp = p.copy(); pp[b] += eps
            pm = p.copy(); pm[b] -= eps
            rp = _residual(R0, w, pp, hkl, g, system).reshape(-1)
            rm = _residual(R0, w, pm, hkl, g, system).reshape(-1)
            J[:, 3 + b] = (rp - rm) / (2 * eps)
        JtJ = J.T @ J
        Jtr = J.T @ r
        diag = np.diag(np.diag(JtJ))
        accepted = False
        for _try in range(8):
            try:
                dx = np.linalg.solve(JtJ + mu * diag + 1e-15 * np.eye(n), -Jtr)
            except np.linalg.LinAlgError:
                mu *= 10.0; continue
            w_try = dx[:3]
            p_try = p.copy()
            if n_p:
                p_try = p + dx[3:]
                # length/volume guard: reject non-physical cells
                cell = _expand(p_try, system)
                if any(cell[i] <= 0 for i in range(3)):
                    mu *= 10.0; continue
                Ar = cell_to_Ar(*cell)
                if np.linalg.det(Ar) <= 0:
                    mu *= 10.0; continue
            r_try = _residual(R0, w_try, p_try, hkl, g, system).reshape(-1)
            cost_try = float(r_try @ r_try)
            if cost_try < cost:
                # accept: retract rotation, update p, reset w
                R0 = _expmap(w_try) @ R0
                p = p_try
                w = np.zeros(3)
                r = _residual(R0, w, p, hkl, g, system).reshape(-1)
                cost = cost_try
                mu = max(mu / 10.0, 1e-12)
                accepted = True
                break
            else:
                mu *= 10.0
        if not accepted or np.linalg.norm(dx) < 1e-9:
            break
    return R0, p, cost


def refine_bravais(g, M, system, tol_abs, n_iter=5, refine_cell=True,
                   min_inliers=6, standardize=True):
    """Bravais-constrained sibling of ``index.refine``.

    g        : (n,3) observed reciprocal peaks (1/A)
    M        : (3,3) initial real basis (columns = rotated real axes); any basis of the lattice
    system   : Bravais system name (cubic/tetragonal/orthorhombic/hexagonal/
               trigonal/monoclinic/triclinic).  ``trigonal`` is the RHOMBOHEDRAL cell of an hR
               lattice (a=b=c, al=be=ga); a primitive trigonal crystal (hP lattice, e.g. P3121)
               is ``hexagonal``, and an hR crystal may also be given as ``hexagonal``
    tol_abs  : absolute inlier tolerance in 1/A (as in index.refine)

    Returns (M, C, hkl, inliers) with the cell LOCKED to the Bravais manifold to
    machine precision.  Falls back to rotation-only refine when too few inliers
    constrain the cell.  Never leaves the manifold and guards against divergence.

    SETTING.  The parameterisation fixes the conventional cell (a=b, the 90 and 120 degree
    angles, the unique axis), and the start is projected onto it, so the start must already be in
    that setting.  ``standardize_setting`` relabels the axes; if the result is in the setting
    (``in_setting``) it is refined exactly as before this check existed.  Otherwise -- the
    Buerger-reduced primitive cells the blind indexer emits for centred lattices, a reduced
    rhombohedral cell that is not the rhombohedral one, any non-reduced basis -- projecting it
    returned a lattice that fit a few percent of the spots (glint review s4-04).  Instead
    ``conventional_settings`` lists integer P's with ``M @ P`` a conventional cell (one per candidate
    symmetry axis), each is refined, and the one whose refined lattice fits the spots best (median
    residual) is kept.  The returned basis is then

      * the refined conventional cell, when that is a basis of M's lattice (|det P| = 1: every
        primitive lattice);
      * mapped back to M's own setting, ``M_ref = Mc_ref @ inv(P)``, when the conventional cell is
        centred (|det P| = 2, 3, 4): it spans M's lattice and its conventional cell ``M_ref @ P``
        is on the manifold.  hkl and inliers are then in M's setting too.

    If the lattice has no ``system`` cell within the tolerances, M is returned unrefined (with its own
    C, hkl, inliers) and a RuntimeWarning is issued.  ``standardize=False`` skips the relabelling and
    the check: M must then already be in the conventional setting.

    LIMITS.  A start that relabels to within ``SETTING_ANGLE_TOL`` of the manifold is taken as the
    conventional cell, as it always was, even when it is a different cell that only looks symmetric:
    a pseudo-symmetric lattice (seen on random skewed monoclinic cells, and on rhombohedral cells with
    alpha just above 109.5, whose reduced cell (a+b+c, b, c) is then within ~10% / 3.5 degrees of
    rhombohedral; none of the real C2, C222_1, I222, I4, I422, I23, F432 protein cells tried, nor
    insulin H3) is still mis-refined from such a start.  And a
    lattice asked for a symmetry it only approximates is refined to it if a cell within the
    tolerances exists (tP 60/60/90 asked ``cubic`` finds its C cell, 4% from cubic).
    """
    g = np.asarray(g, float)
    if system == "triclinic":
        refine_cell = True  # 6 free params == unconstrained
    if not standardize:
        return _refine_from(g, M, system, tol_abs, n_iter, refine_cell, min_inliers)
    M_in = np.asarray(M, float)
    Ms = standardize_setting(M_in, system)
    if in_setting(Ms, system):
        return _refine_from(g, Ms, system, tol_abs, n_iter, refine_cell, min_inliers)
    best, best_r = None, np.inf
    for P in conventional_settings(M_in, system):
        try:
            out = _refine_from(g, M_in @ P, system, tol_abs, n_iter, refine_cell, min_inliers)
            if int(round(abs(np.linalg.det(P)))) != 1:
                out = _with_basis(g, out[0] @ np.linalg.inv(P), tol_abs)   # centred: back to M's lattice
            r = _median_resid(g, out[0])
        except np.linalg.LinAlgError:                   # a candidate the refine cannot start from
            continue
        if r < best_r:
            best, best_r = out, r
    if best is not None:
        return best
    dl, da = setting_deviation(Ms, system)
    hint = (" ('trigonal' means the rhombohedral cell of an hR lattice; an hP lattice is 'hexagonal')"
            if system == "trigonal" else "")
    warnings.warn(f"refine_bravais: M is not refined: its lattice has no {system} cell within "
                  f"{SETTING_ANGLE_TOL:g} deg / {100 * SEARCH_LENGTH_RTOL:g}% (M standardized is {da:.1f} deg / "
                  f"{100 * dl:.1f}% off the manifold){hint}; returning M unchanged", RuntimeWarning, stacklevel=2)
    return _with_basis(g, M_in.copy(), tol_abs)


def _assign(g, M):
    """Miller indices of the spots ``g`` in the basis ``M``, each spot given the lattice node nearest
    to it. Rounding ``g @ M`` component by component finds that node only in a reduced basis: in a
    sheared one (``30*[[-1,1,1],[1,-1,1],[1,1,-1]]`` with its first column added 50 times to the
    second) the spot at the node (1, 50, 0) rounds to (1, 49, 0), a node 0.024/A away instead of
    0.0004/A, and is rejected at tol_abs 0.02 (Copilot review of #247). So the assignment is made in
    the reduced basis ``Mr`` (``M = Mr @ U``, U integer unimodular) and carried into M's setting as
    ``h_M = h_r @ U``. The residual |g - C h| is the same in either basis."""
    Mr = _reduced(M)
    U = np.linalg.solve(Mr, M)
    Ui = np.rint(U)
    if np.abs(U - Ui).max() > 1e-6 or abs(round(np.linalg.det(Ui))) != 1:
        return np.rint(g @ M)                           # not a basis of the same lattice: as before
    return np.rint(g @ Mr) @ Ui


def _with_basis(g, M, tol_abs):
    """(M, C, hkl, inliers) of the basis M as it stands; hkl from ``_assign``."""
    C = np.linalg.inv(M).T
    hkl = _assign(g, M)
    return M, C, hkl, np.linalg.norm(g - hkl @ C.T, axis=1) < tol_abs


def _median_resid(g, M):
    """Median over the spots of |g - C h|, h from ``_assign``: how well the lattice of M fits them."""
    C = np.linalg.inv(M).T
    return float(np.median(np.linalg.norm(g - _assign(g, M) @ C.T, axis=1)))


def _refine_from(g, M, system, tol_abs, n_iter, refine_cell, min_inliers):
    """The constrained refine from a start M in the conventional setting (unchanged since #53)."""
    # initialise (R0, p0) from M: cell params are rotation-invariant, project onto manifold
    cell0 = cell_params(M)
    p = _project(cell0, system)
    Ar0 = cell_to_Ar(*_expand(p, system))
    R0 = _polar(M @ np.linalg.inv(Ar0))

    C = R0 @ _B_cell(p, system)
    hkl = np.rint(g @ np.linalg.inv(C).T)
    inliers = np.linalg.norm(g - hkl @ C.T, axis=1) < tol_abs

    for _ in range(n_iter):
        C = R0 @ _B_cell(p, system)
        Minv_t = np.linalg.inv(C).T                 # == M
        hkl = np.rint(g @ Minv_t)
        resid = np.linalg.norm(g - hkl @ C.T, axis=1)
        inliers = resid < tol_abs
        m = int(inliers.sum())
        if m < 4:
            break
        do_cell = refine_cell and (m >= min_inliers)
        R0, p, _cost = _lm_inner(R0, p, hkl[inliers], g[inliers], system, do_cell)

    C = R0 @ _B_cell(p, system)
    M_out = np.linalg.inv(C).T
    hkl = np.rint(g @ np.linalg.inv(C).T)
    inliers = np.linalg.norm(g - hkl @ C.T, axis=1) < tol_abs
    return M_out, C, hkl, inliers
