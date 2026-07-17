# ─────────────────────────────────────────────────────────────────────────────────────────────
# HAND-OFF NOTE (Jul 2026) — parked here for the GLINT indexing collaboration; candidate to hand to
# Yuan. Deliberately kept OUT of slac-lcls/drp-benchmarks (shared more widely) — this is
# indexing-research code, not general DRP tooling.
#   Companion PRIMITIVE (general, lives in the shared repo): per-peak shape moments — height +
#   intensity-weighted covariance (sigma_major/minor, theta, eccentricity) — landed in
#   drp-benchmarks radial_integration/peakfinder8.py behind `moments=True` (commit 365a3be).
#   THIS file is the CURVED-streak EXTENSION of that idea (Kossel lines / CBXD conic arcs), where a
#   single global covariance ellipse fails. Neighbours here: cbxd_angles.py, cbxd_multishot.py.
# ─────────────────────────────────────────────────────────────────────────────────────────────
"""
ridge_moments.py -- SKETCH / prototype: a LOCAL (piecewise) 2nd-moment descriptor for CURVED
streaks (Kossel lines / CBXD conic arcs), where a single global covariance ellipse fails: on a
curved arc the curvature inflates the apparent width (sigma_minor picks up the arc's sagitta) and
the centroid falls off the curve.

Idea -- reuse the SAME intensity-weighted 2nd-moment primitive peakfinder8 uses per peak, but
apply it LOCALLY along the arc. Order the pixels of one elongated connected component by arclength,
cut into short segments (each locally straight), and take the covariance IN EACH SEGMENT. That
yields a PROFILE along the arc:
    tangent(s)   -- local major-axis direction (the streak's direction, not the chord's)
    width(s)     -- local minor sigma = true perpendicular half-width (NOT curvature-inflated)
    intensity(s) -- local integrated intensity
    curvature(s) -- d(tangent)/ds  (~ 1/R for a circular arc)
plus scalar summaries: length, median width, total turning angle (line vs arc), and the GLOBAL
sigma_minor for comparison (the value peakfinder8's single ellipse would report -- inflated on arcs).

STATUS (sketch, validated headline only): on a synthetic R=60 arc the LOCAL width recovers the true
cross-section (1.32 vs sigma0=1.5, tail-clipped) while the single-ellipse global sigma_minor reads
3.41 (2.3x inflated) -- the local-moment profile defeats the curvature inflation. BUT curvature /
total-turning are only order-of-magnitude (chord-projection ordering degrades as curvature grows).

ORDERING here = projection onto the global principal axis (PCA). Valid for OPEN, gently curved arcs
(sagitta < ~half-length). For strongly curved or CLOSED conics (a full Kossel ring), the projection
FOLDS -> replace with a real ridge tracer (skimage.morphology.skeletonize + path walk) or fit a
PARAMETRIC CONIC directly (Fitzgibbon-Pilu-Fisher 1999) to the ridge points. The local-moment
profile (local shape) and a conic fit (global parametrization) are complementary. == the next step ==

Front-end caveat: to reach these components from peakfinder8, relax its max_pix cap (long lines
exceed it) and loosen the isotropic-ring background (Kossel/CBXD features cross resolution rings).

Portable NumPy (swap np->cupy for the device path).
"""
import numpy as np


def _cov_axes(y, x, w):
    """Intensity-weighted centroid + principal axes for a pixel set (the peakfinder8 primitive)."""
    W = float(w.sum())
    if W <= 0:
        return None
    cy = float((w * y).sum() / W); cx = float((w * x).sum() / W)
    dy = y - cy; dx = x - cx
    Cyy = float((w * dy * dy).sum() / W); Cxx = float((w * dx * dx).sum() / W)
    Cxy = float((w * dy * dx).sum() / W)
    hm = 0.5 * (Cyy + Cxx); hd = np.sqrt(max((0.5 * (Cyy - Cxx)) ** 2 + Cxy * Cxy, 0.0))
    lam1 = max(hm + hd, 0.0); lam2 = max(hm - hd, 0.0)
    theta = 0.5 * np.arctan2(2 * Cxy, Cyy - Cxx)     # major-axis angle from +row(y) toward +col(x)
    return dict(cy=cy, cx=cx, sigma_major=np.sqrt(lam1), sigma_minor=np.sqrt(lam2),
                theta=theta, intensity=W)


def ridge_descriptor(ys, xs, iv, nseg=12):
    """
    ys, xs, iv : row/col coords + (background-subtracted) intensities of ONE elongated component
                 (e.g. peakfinder8 labelled-pixel gather for a streak/Kossel-line blob).
    nseg       : number of arclength segments (each should keep >=~4 pixels).
    Returns a dict of per-segment profile arrays + scalar summaries, or None if degenerate.
    """
    ys = np.asarray(ys, float); xs = np.asarray(xs, float); iv = np.asarray(iv, float)
    g = _cov_axes(ys, xs, iv)
    if g is None or ys.size < 2 * nseg:
        return None
    # order pixels along the global major axis (arclength proxy for a gently-curved open arc)
    ey, ex = np.cos(g["theta"]), np.sin(g["theta"])          # major-axis unit vector in (row, col)
    t = (ys - g["cy"]) * ey + (xs - g["cx"]) * ex
    order = np.argsort(t)
    segs = [s for s in np.array_split(order, nseg) if s.size >= 4]
    C = [c for c in (_cov_axes(ys[s], xs[s], iv[s]) for s in segs) if c is not None]
    if len(C) < 2:
        return None
    cy = np.array([c["cy"] for c in C]); cx = np.array([c["cx"] for c in C])
    width = np.array([c["sigma_minor"] for c in C])          # local perpendicular half-width
    seg_len = np.array([c["sigma_major"] for c in C])
    inten = np.array([c["intensity"] for c in C])
    theta = np.array([c["theta"] for c in C])
    # arclength between consecutive segment centroids
    ds = np.hypot(np.diff(cy), np.diff(cx))
    s_arc = np.concatenate([[0.0], np.cumsum(ds)])
    # tangent has a pi ambiguity (it's a line): align signs along the walk, then unwrap for curvature
    tv = np.stack([np.cos(theta), np.sin(theta)], 1)
    for i in range(1, len(tv)):
        if tv[i] @ tv[i - 1] < 0:
            tv[i] = -tv[i]
    tang = np.unwrap(np.arctan2(tv[:, 1], tv[:, 0]))
    kappa = np.gradient(tang, s_arc) if len(s_arc) > 2 else np.zeros_like(s_arc)
    return dict(s=s_arc, y=cy, x=cx, tangent=tang, width=width, seg_len=seg_len,
                intensity=inten, curvature=kappa,
                length=float(s_arc[-1]),
                median_width=float(np.median(width)),
                median_curvature=float(np.median(kappa)),
                total_turn_deg=float(np.degrees(tang[-1] - tang[0])),
                global_sigma_minor=float(g["sigma_minor"]))   # what a single ellipse would report


if __name__ == "__main__":
    # demo: a circular arc (radius R) with a Gaussian cross-section (sigma0) + noise.
    R, sigma0, H, W = 60.0, 1.5, 160, 160
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cy0, cx0 = 80.0, 40.0
    rr = np.hypot(yy - cy0, xx - cx0); ang = np.arctan2(yy - cy0, xx - cx0)
    span = 1.2                                                # rad of arc subtended
    img = 1000.0 * np.exp(-0.5 * ((rr - R) / sigma0) ** 2) * ((ang > -span / 2) & (ang < span / 2))
    img = img + np.random.default_rng(0).normal(0, 5, (H, W))
    m = img > 50.0
    d = ridge_descriptor(*[a[m] for a in (yy, xx)], np.clip(img[m], 0, None), nseg=12)
    print("ridge_moments demo -- circular arc R=%.0f, sigma0=%.1f, span=%.0f deg" %
          (R, sigma0, np.degrees(span)))
    print(f"  median LOCAL width   = {d['median_width']:.2f}  (true sigma0 = {sigma0})   <- not inflated")
    print(f"  GLOBAL sigma_minor   = {d['global_sigma_minor']:.2f}  (single-ellipse value)  <- inflated by curvature")
    print(f"  median curvature     = {d['median_curvature']:.4f}  (true 1/R = {1/R:.4f})")
    print(f"  total turning angle  = {d['total_turn_deg']:.1f} deg  (arc span = {np.degrees(span):.0f} deg)")
    print(f"  arc length           = {d['length']:.1f} px  (true ~ R*span = {R*span:.1f})")
