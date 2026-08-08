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


def deflate_peaks(q, M, tol=0.15):
    """Reciprocal vectors of `q` NOT explained by cell `M` -- the residual for a second-lattice search.

    M has reciprocal-basis rows, so hkl = q @ M; a peak is 'assigned' to M when every hkl component is
    within `tol` of an integer. Returns the unassigned peaks (a copy). Pure numpy.
    """
    q = np.asarray(q, float)
    if q.ndim != 2 or len(q) == 0:
        return q.reshape(0, 3)
    hf = q @ np.asarray(M, float)
    assigned = np.abs(hf - np.round(hf)).max(1) < tol
    return q[~assigned]

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


def orientation_clone_fraction(resid, M2, M1, tol=0.15, loose=0.35):
    """Of the residual peaks M2 indexes, the fraction that are ALSO near-integer under M1 at a
    LOOSER tolerance -- the mosaic-tail signature.

    same_lattice cannot police this: a genuine same-protein second crystal shares M1's CELL, so a
    cell test passes both the real double and the clone. What separates them is orientation --
    a clone's "second lattice" re-indexes peaks that are lattice-1 points sitting just outside the
    deflation tolerance (deviation in (tol, loose)), while a re-oriented second crystal's peaks are
    generic under M1. Returns 0.0 when M2 indexes nothing.

    NB the raw fraction has a RANDOM BASELINE of (2*loose)^3 (a generic point lands within loose of
    integer per axis with probability 2*loose) -- 0.34 at the default -- so it must be judged as an
    excess over that baseline, which second_lattice_verdict does. Measured: a true 25-degree double
    reads median 0.36 raw (its baseline), a mosaic-tail clone reads ~1."""
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


def second_lattice_verdict(resid, M1, index_fn, min_peaks=6, tol=0.15, loose=0.35, clone_frac=0.5):
    """The double-hit acceptance rule, gated so it counts crystals rather than peaks.

    resid: peaks left after deflating lattice 1 (M1, real-space columns, hkl = q @ M1).
    index_fn(resid, 1) -> [(M2, score), ...] -- the blind indexer.

    Returns dict(raw, accepted, M2, cell_match, clone_fraction):
      raw       the ORIGINAL rule -- valid cell (|det| >= 1) with >= min_peaks residual inliers.
                Measured to be a peak-count artifact on peak-rich frames: at a median residual of
                62 peaks it accepted 95% real vs 93% azimuth-scrambled (job 34409542).
      accepted  raw AND the second lattice is the SAME CELL as lattice 1 (same_lattice; SFX double
                hits are the same protein at a new orientation, so a different-cell "second
                lattice" on a residual is opportunistic) AND it is not an orientation clone of
                lattice 1: the clone test is on the EXCESS of orientation_clone_fraction over its
                random baseline (2*loose)^3, since a genuine double sits AT that baseline (~0.34)
                while a mosaic tail reads ~1. clone_frac is the normalized-excess threshold."""
    from glint.multishot import same_lattice
    out = dict(raw=False, accepted=False, M2=None, cell_match=False, clone_fraction=0.0)
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
    base = (2.0 * loose) ** 3                       # random baseline of the raw fraction
    is_clone = out["clone_fraction"] >= base + clone_frac * (1.0 - base)
    out["accepted"] = out["raw"] and out["cell_match"] and not is_clone
    return out
