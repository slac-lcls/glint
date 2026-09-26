"""Multi-lattice / double-hit primitives.

A "double hit" is two (or more) crystals in one shot -- common when the sample is concentrated
(several crystals per drop). GLINT resolves them by DEFLATE-AND-REINDEX: index the first lattice,
remove the peaks it explains, and re-index the residual to find the second. `deflate_peaks` is the
pure (removal) half; the second-lattice search is the GPU indexer, driven from the streaming driver's
double-hit path (StreamDriver(double_hit=True)).

A rising double-hit RATE is a useful live beamline signal (sample too concentrated / jet issues), so
the driver exposes it in stats() as `double_hit_rate`.
"""
import numpy as np


def claimed_mask(q, M, tol=0.15):
    """Boolean mask of the peaks of `q` that cell `M` explains -- the assignment deflate_peaks removes.

    M has reciprocal-basis rows, so hkl = q @ M; a peak is 'assigned' to M when every hkl component is
    within `tol` of an integer. Pure numpy.
    """
    q = np.asarray(q, float)
    if q.ndim != 2 or len(q) == 0:
        return np.zeros(0, bool)
    hf = q @ np.asarray(M, float)
    return np.abs(hf - np.round(hf)).max(1) < tol


def deflate_peaks(q, M, tol=0.15):
    """Reciprocal vectors of `q` NOT explained by cell `M` -- the residual for a second-lattice search.

    The peaks claimed_mask does not assign to M (a copy). Pure numpy.
    """
    q = np.asarray(q, float)
    if q.ndim != 2 or len(q) == 0:
        return q.reshape(0, 3)
    return q[~claimed_mask(q, M, tol)]

def scramble_azimuth(q, rng):
    """Each peak independently rotated by a random azimuth about the beam axis (z).

    Preserves |q| and q_z -- hence each peak's Ewald excitation error EXACTLY -- and the radial
    intensity structure; destroys only lattice coherence. This is the null for any 'is there a
    lattice in these peaks' rule: a rule that accepts scrambled residuals at the same rate as real
    ones is measuring peak count, not crystals. Measured on mfxl1038923 r0278 (job 34409542) the
    bare deflate-and-reindex rule did exactly that: 95% real vs 93% scrambled."""
    q = np.asarray(q, float)
    phi = rng.uniform(0.0, 2.0 * np.pi, len(q))
    c, s = np.cos(phi), np.sin(phi)
    out = q.copy()
    out[:, 0] = c * q[:, 0] - s * q[:, 1]
    out[:, 1] = s * q[:, 0] + c * q[:, 1]
    return out


_UNIMOD = None
_I3 = np.eye(3, dtype=int)
_R4_C = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], int)
_R2_A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], int)
_R2_B = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], int)
_R2_C = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], int)
_R3_C_HEX = np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]], int)
_R6_C_HEX = np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 1]], int)
_R2_A_HEX = np.array([[1, 0, 0], [-1, -1, 0], [0, 0, -1]], int)
_R2_AB_HEX = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], int)
_R3_111 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], int)
_LAUE_ALIASES = {"2/m": "2/m_uab", "-3m": "-3m1"}


def _close_group(gens):
    G = [_I3.copy()]
    ch = True
    while ch:
        ch = False
        for g in list(G):
            for s in gens:
                h = (s @ g).astype(int)
                if not any(np.array_equal(h, x) for x in G):
                    G.append(h); ch = True
    return G


_PROPER_LAUE_OPS = {
    "-1": _close_group(()),
    "2/m_uaa": _close_group((_R2_A,)),
    "2/m_uab": _close_group((_R2_B,)),
    "2/m_uac": _close_group((_R2_C,)),
    "mmm": _close_group((_R2_A, _R2_B)),
    "4/m": _close_group((_R4_C,)),
    "4/mmm": _close_group((_R4_C, _R2_A)),
    "-3": _close_group((_R3_C_HEX,)),
    "-3m1": _close_group((_R3_C_HEX, _R2_A_HEX)),
    "-31m": _close_group((_R3_C_HEX, _R2_AB_HEX)),
    "-3_R": _close_group((_R3_111,)),
    "-3m_R": _close_group((_R3_111, _R2_AB_HEX)),
    "6/m": _close_group((_R6_C_HEX,)),
    "6/mmm": _close_group((_R6_C_HEX, _R2_A_HEX)),
    "m-3": _close_group((_R3_111, _R2_C)),
    "m-3m": _close_group((_R3_111, _R4_C)),
}


def _unimodular():
    """The 3480 integer 3x3 matrices with entries in {-1, 0, 1} and determinant +1 (built once)."""
    global _UNIMOD
    if _UNIMOD is None:
        from itertools import product
        U = np.array(list(product((-1, 0, 1), repeat=9)), float).reshape(-1, 3, 3)
        _UNIMOD = U[np.abs(np.linalg.det(U) - 1.0) < 0.5]
    return _UNIMOD


def _basis_change(B1, B2):
    return np.rint(np.linalg.solve(B1, B2)).astype(int)


def _conjugate_op(T, U):
    return np.rint(np.linalg.solve(T, np.asarray(U, int) @ T)).astype(int)


def _canonical_basis(M):
    """A Buerger-reduced, RIGHT-HANDED basis of the lattice M spans.

    -I maps every lattice onto itself, so negating a left-handed basis changes the basis and not the
    lattice. Both bases handed to _metric_ops must share handedness: otherwise no proper rotation
    carries one onto the other, and the polar projection returns a large spurious angle.

    buerger_reduce searches integer combinations up to +-3, which one pass does not always reach
    from a strongly skewed basis (half of random entries-up-to-2 bases of the lysozyme cell), so it
    is repeated until the lengths stop falling. Bases the indexers emit are already reduced; one pass."""
    from glint.lattice import buerger_reduce
    B = np.asarray(M, float)
    T = _I3.copy()
    for _ in range(10):
        Bn = buerger_reduce(B)
        T = T @ _basis_change(B, Bn)
        done = np.linalg.norm(Bn, axis=0).sum() >= np.linalg.norm(B, axis=0).sum() * (1 - 1e-9)
        B = Bn
        if done:
            break
    if np.linalg.det(B) < 0:
        B = -B
        T = -T
    return B, T


def _metric_ops(B1, B2, rtol=0.05, ctol=0.06):
    """Integer basis changes U (entries in {-1, 0, 1}, det +1) under which B1 @ U has B2's metric.

    B1 and B2 are right-handed reduced bases. The set holds the lattice's own proper symmetry (the
    identity alone for a generic triclinic cell; 4 ops orthorhombic, 8 tetragonal, 12 hexagonal, 24
    cubic, pinned by experiments/test_double_hit_rule.py), composed with whatever basis change relates
    the two reductions -- e.g. an axis swap when two lengths are near-equal. In a reduced basis these
    ops have entries in {-1, 0, 1} for every Bravais class the test checks.

    rtol (relative, on lengths) and ctol (absolute, on SIGNED cosines) are same_lattice's values.
    The comparison stays signed all the way through: treating every pair of near-90-degree angles as
    interchangeable invents false 2-folds for low-symmetry cells. Callers that KNOW the lattice class
    should supply that class to misorientation_deg instead of asking this metric-only fallback to infer
    higher symmetry from a noisy near-right angle. If nothing matches, the tolerances are doubled up to
    twice (looser can only admit MORE ops, so a SMALLER angle); if nothing matches even at 4x
    (different cells), the single closest U is used so the angle is still defined, but it is not
    meaningful."""
    U = _unimodular()
    C = np.einsum("ij,njk->nik", B1, U)                          # candidate bases, columns
    G = np.einsum("nji,njk->nik", C, C)                          # their Gram matrices
    G2 = B2.T @ B2
    L, L2 = np.sqrt(np.einsum("nii->ni", G)), np.sqrt(np.diag(G2))
    dl = np.abs(L - L2) / (0.5 * (L + L2))
    iu = ([0, 0, 1], [1, 2, 2])
    cos = G[:, iu[0], iu[1]] / (L[:, iu[0]] * L[:, iu[1]])
    cos2 = G2[iu] / (L2[iu[0]] * L2[iu[1]])
    dc = np.abs(cos - cos2)
    for f in (1.0, 2.0, 4.0):
        ok = (dl <= f * rtol).all(1) & (dc <= f * ctol).all(1)
        if ok.any():
            return U[ok]
    return U[[np.argmin(dl.max(1) / rtol + dc.max(1) / ctol)]]


def _symmetry_candidates(B1, B2, T1, T2, laue):
    if laue is None:
        return B1, B2, _metric_ops(B1, B2)
    from glint.lattice import standardize_axes
    key = _LAUE_ALIASES.get(str(laue).strip(), str(laue).strip())
    if key in ("mmm", "4/m", "4/mmm", "-3", "-3m1", "-31m", "6/m", "6/mmm"):
        B1n = standardize_axes(B1, laue=key)
        B2n = standardize_axes(B2, laue=key)
        T1 = T1 @ _basis_change(B1, B1n)
        T2 = T2 @ _basis_change(B2, B2n)
        B1, B2 = B1n, B2n
    rel = _metric_ops(B1, B2)
    proper = _PROPER_LAUE_OPS.get(key)
    if proper is not None:
        ops = []
        for S in (U for U in proper if int(round(np.linalg.det(U))) == 1):
            S = _conjugate_op(T1, S)
            for V in rel:
                U = np.rint(S @ V).astype(int)
                if not any(np.array_equal(U, W) for W in ops):
                    ops.append(U)
        return B1, B2, ops
    return B1, B2, rel


def misorientation_deg(M1, M2, laue=None):
    """Smallest rotation angle carrying lattice 1 onto lattice 2, modulo lattice symmetry (degrees).

    THIS IS THE GATE THAT SEPARATES A SECOND CRYSTAL FROM A MOSAIC TAIL, and it does so on the
    physics rather than on a proxy. A mosaic tail is lattice 1 rotated by a FEW DEGREES -- that is
    precisely what puts its peaks just outside the deflation tolerance and into the residual. A
    genuine second crystal lands at a generic angle, and small angles are intrinsically rare there:
    for Haar-random orientations modulo a 222 lattice symmetry, P(angle < 5 deg) = 6.5e-5.

    Measured on mfxl1038923 the distribution is sharply bimodal in both runs -- r0278: 145 second
    lattices below 5 deg, 11 in 5-15 deg, 56 above; r0058: 152 / 20 / 107 (job 39151575, which
    re-ran jobs 34464312/34468402 exactly and kept the matrices). Against 6.5e-5 the sub-5-degree
    population is four orders of magnitude over the random expectation, so it is mosaic/split
    domains, not independent crystals; and the near-empty 5-15 deg band is why the exact cut hardly
    matters.

    ⚠ Before this function compared lattices it compared BASES: it tried only signed axis
    permutations of the incoming matrices, so the same lattice in a different basis -- or in the same
    basis with the opposite handedness, which buerger_reduce returns about half the time -- could read
    far from itself (on the lysozyme cell an a/b swap read 90 degrees, an a+b basis 27;
    experiments/test_double_hit_rule.py). On mfxl1038923 it read EVERY opposite-handed pair at >= 15
    degrees (108 and 143 of them); the numbers it gave were 72 / 5 / 135 and 74 / 9 / 196. After the
    fix the sub-5-degree share is the same for same- and opposite-handed pairs (69 % vs 68 % on r0278,
    54 % vs 55 % on r0058), as it must be for a label that carries no physics.

    Both matrices are first put in a right-handed Buerger-reduced basis (_canonical_basis), so the
    result does not depend on the basis either caller used. When the caller knows the Laue class,
    `laue=` supplies fixed proper operators for that class instead of asking a noisy metric to guess
    them; otherwise B2 = R B1 U for some integer basis change U (_metric_ops: the lattice's own
    symmetry, composed with any reshuffle between the two reductions). R is projected onto the nearest
    rotation because the two cells are refined independently and differ slightly (stable to a 1% cell
    mismatch)."""
    B1, T1 = _canonical_basis(M1)
    B2, T2 = _canonical_basis(M2)
    B1, B2, ops = _symmetry_candidates(B1, B2, T1, T2, laue)
    best = 180.0
    for U in ops:
        W, _, Vt = np.linalg.svd(B2 @ np.linalg.inv(B1 @ U))
        if np.linalg.det(W @ Vt) < 0:                            # keep it a proper rotation
            W = W.copy(); W[:, -1] *= -1
        R = W @ Vt
        best = min(best, float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))))
    return best


def orientation_clone_fraction(resid, M2, M1, tol=0.15, loose=0.35):
    """Of the residual peaks M2 indexes, the fraction that are ALSO near-integer under M1 at a
    LOOSER tolerance -- the mosaic-tail signature.

    DIAGNOSTIC ONLY -- this was the clone gate until real data retired it; use misorientation_deg.
    The idea was sound (a clone re-indexes lattice-1 points sitting just outside the deflation
    tolerance, while a second crystal's peaks are generic under M1) and it separates cleanly on
    SYNTHETIC clones, which is exactly why it needed a real test. On mfxl1038923 (job 34468402),
    scored against the misorientation angle, it barely separates the two populations: the median
    fraction is 0.58 for sub-5-degree clones vs 0.39 for genuine doubles on r0278 (0.56 vs 0.40 on
    r0058), correlation only -0.28, and the distributions overlap. At the shipped cut it let 55 of
    72 clones through while killing 19 of 135 real doubles -- a marginal enrichment bought with real
    signal. Kept because the fraction is still informative in aggregate, but it no longer gates.

    Returns 0.0 when M2 indexes nothing. The raw fraction has a RANDOM BASELINE of (2*loose)^3
    (a generic point lands within loose of integer per axis with probability 2*loose) -- 0.34 at the
    default -- so it was always judged as an excess over that baseline, never absolutely."""
    resid = np.asarray(resid, float)
    if resid.ndim != 2 or len(resid) == 0:
        return 0.0
    h2 = resid @ np.asarray(M2, float)
    inl2 = np.abs(h2 - np.round(h2)).max(1) < tol
    if not inl2.any():
        return 0.0
    h1 = resid[inl2] @ np.asarray(M1, float)
    near1 = np.abs(h1 - np.round(h1)).max(1) < loose
    return float(near1.mean())


def second_lattice_verdict(resid, M1, index_fn, min_peaks=6, tol=0.15, loose=0.35, min_misorient=15.0,
                           laue=None):
    """The double-hit acceptance rule, gated so it counts crystals rather than peaks.

    resid: peaks left after deflating lattice 1 (M1, real-space columns, hkl = q @ M1).
    index_fn(resid, 1) -> [(M2, score), ...] -- the blind indexer.

    Returns dict(raw, accepted, M2, cell_match, misorientation, clone_fraction):
      raw       the ORIGINAL rule -- valid cell (|det| >= 1) with >= min_peaks residual inliers.
                Measured to be a peak-count artifact on peak-rich frames: at a median residual of
                62 peaks it accepted 95% real vs 93% azimuth-scrambled (job 34409542), and 96% vs
                96% on a second run (34468402). On its own it carries no information.
      accepted  raw AND SAME CELL as lattice 1 (same_lattice; an SFX double hit is the same protein
                at a new orientation, so a different-cell "second lattice" fitted to a residual is
                opportunistic) AND a genuinely DIFFERENT ORIENTATION, misorientation_deg >=
                min_misorient, which is what separates a second crystal from lattice 1's own mosaic
                tail. Both gates are measured, not assumed: on two runs the cell gate takes the null
                to exactly 0 (0/1449 and 0/2123 scrambled residuals) while 212 and 279 real ones
                pass it, and the misorientation distribution is bimodal with a near-empty 5-15
                degree band, so the cut is insensitive -- at the shipped 15 deg the gated rates are
                3.6% and 4.8% of scoreable frames on r0278/r0058, and moving the cut all the way
                down to 5 deg only takes them to 4.3% and 5.7%. Two independent runs agreeing.
                (Measured with the basis-independent misorientation_deg, job 39151575. The basis-
                dependent version it replaced gave 8.7% and 8.9%: it counted opposite-handed
                mosaic pairs as second crystals.)

    The clone_fraction is still reported (see orientation_clone_fraction) but no longer gates: it
    was the gate until real data showed it keeps three quarters of the clones and kills a tenth of
    the real doubles."""
    from glint.multishot import same_lattice
    out = dict(raw=False, accepted=False, M2=None, cell_match=False, misorientation=0.0,
               clone_fraction=0.0)
    resid = np.asarray(resid, float)
    if len(resid) < min_peaks:
        return out
    nb = index_fn(resid, 1)
    if not nb:
        return out
    M2 = np.asarray(nb[0][0], float)
    out["M2"] = M2
    if abs(np.linalg.det(M2)) < 1.0:
        return out
    hf = resid @ M2
    if int((np.abs(hf - np.round(hf)).max(1) < tol).sum()) < min_peaks:
        return out
    out["raw"] = True
    out["cell_match"] = bool(same_lattice(M2, np.asarray(M1, float)))
    out["clone_fraction"] = orientation_clone_fraction(resid, M2, M1, tol=tol, loose=loose)
    if out["cell_match"]:                           # an angle between different cells is meaningless
        out["misorientation"] = misorientation_deg(M1, M2, laue=laue)
    out["accepted"] = out["raw"] and out["cell_match"] and out["misorientation"] >= min_misorient
    return out
