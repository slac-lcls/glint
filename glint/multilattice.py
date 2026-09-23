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


def _lattice_ops(M, tol=0.10):
    """The lattice's own proper symmetry, as signed axis permutations that preserve its axis lengths.

    Needed so a misorientation is measured modulo operations that map the lattice onto itself: for a
    cell with distinct axes this is just the four 2-folds, but with two near-equal axes it grows, and
    ignoring it would OVERSTATE the angle."""
    from itertools import permutations, product
    L = np.linalg.norm(np.asarray(M, float), axis=0)
    ops = []
    for perm in permutations(range(3)):
        if np.any(np.abs(L[list(perm)] - L) / L > tol):
            continue
        for sg in product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for i, p in enumerate(perm):
                P[p, i] = sg[i]
            if np.linalg.det(P) > 0:
                ops.append(P)
    return ops or [np.eye(3)]


def misorientation_deg(M1, M2):
    """Smallest rotation angle carrying lattice 1 onto lattice 2, modulo lattice symmetry (degrees).

    THIS IS THE GATE THAT SEPARATES A SECOND CRYSTAL FROM A MOSAIC TAIL, and it does so on the
    physics rather than on a proxy. A mosaic tail is lattice 1 rotated by a FEW DEGREES -- that is
    precisely what puts its peaks just outside the deflation tolerance and into the residual. A
    genuine second crystal lands at a generic angle, and small angles are intrinsically rare there:
    for Haar-random orientations modulo a 222 lattice symmetry, P(angle < 5 deg) = 6.5e-5.

    Measured on mfxl1038923 (jobs 34464312, 34468402) the distribution is sharply bimodal in both
    runs -- r0278: 72 second lattices below 5 deg, 5 in 5-15 deg, 135 above; r0058: 74 / 9 / 196.
    Against 6.5e-5 the sub-5-degree population is ~5000x over the random expectation, so it is
    mosaic/split domains, not independent crystals; and the near-empty 5-15 deg band is why the exact
    cut hardly matters.

    M2 P = R M1 for some symmetry op P; R is projected onto the nearest rotation because the two
    cells are refined independently and differ slightly (stable to a 1% cell mismatch)."""
    M1 = np.asarray(M1, float); M2 = np.asarray(M2, float)
    A = np.linalg.inv(M1)
    best = 180.0
    for P in _lattice_ops(M1):
        U, _, Vt = np.linalg.svd(M2 @ P @ A)
        if np.linalg.det(U @ Vt) < 0:                            # keep it a proper rotation
            U = U.copy(); U[:, -1] *= -1
        R = U @ Vt
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


def second_lattice_verdict(resid, M1, index_fn, min_peaks=6, tol=0.15, loose=0.35, min_misorient=15.0):
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
                8.7% and 8.9% of scoreable frames on r0278/r0058, and moving the cut all the way
                down to 5 deg only takes them to 9.0% and 9.3%. Two independent runs agreeing.

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
        out["misorientation"] = misorientation_deg(M1, M2)
    out["accepted"] = out["raw"] and out["cell_match"] and out["misorientation"] >= min_misorient
    return out
