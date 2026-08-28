"""Step 0 of the ridge_moments follow-up (github.com/slac-lcls/glint issue #9): before wiring a
per-streak tangent into the accumulator, check whether a tangent is even recoverable from
cbxd_joint.simulate()'s streaks in the first place.

ridge_moments.py's ridge_descriptor() doesn't apply directly here -- it's built for PIXEL
coordinates + intensities from a rendered detector image (ys, xs, iv), and simulate() never
renders pixels: it works purely in 3-D reciprocal space (points on a Kossel circle). So this
re-derives the same idea (a robust LOCAL tangent from a small window of consecutive arc points,
not a single global fit) for our 3-D point-cloud representation, where we also get a free exact
ground truth: at any point p on the Kossel circle of lattice vector G (center G/2, in the plane
perp to G), the tangent direction is exactly Ghat x (p - G/2), normalized (perpendicular to both
the radius vector and G, i.e. within the circle's plane) -- no fitting needed for the truth.

simulate() discards per-streak grouping (returns one flattened, shuffled kobs) and doesn't expose
the clean (pre-noise) points, so _simulate_grouped below duplicates its arc-generation loop to
recover: for each streak, its ordered clean points, its lattice vector G, and noisy points.

  python tangent_validate.py
"""
import sys

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import B, COSA, HS, K, SINA, kossel_basis, rand_rot


def _simulate_grouped(R, rng, noise, thin=6):
    """Like cbxd_joint.simulate(), but keeps each streak's points GROUPED (ordered by chi) and
    keeps the CLEAN (pre-noise) points alongside the noisy ones, plus each streak's true G.
    Only the 'real' (non-spurious) streaks -- there's no tangent to validate on spurious points.

    thin=6 matches simulate()'s own decimation (kout[m][::6]); thin=1 uses every admitted sample
    -- the most points a streak could possibly have in this synthetic model, i.e. the best case
    for a tangent fit, not a realistic detector pixel density.
    """
    chi = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    cc, ss = np.cos(chi), np.sin(chi)
    streaks = []                                    # list of dict(clean=(n,3), noisy=(n,3), G=(3,))
    for h in HS:
        G = R @ (B @ h)
        a, b = kossel_basis(G)
        if a is None:
            continue
        kout = G / 2 + cc[:, None] * a + ss[:, None] * b
        kin = kout - G
        m = (kin[:, 2] > K * COSA) & (np.hypot(kin[:, 0], kin[:, 1]) < K * SINA) & (kout[:, 2] > 0)
        if m.sum() >= 2:
            clean = kout[m][::thin]
            noisy = clean + rng.normal(0, noise, clean.shape)
            streaks.append(dict(clean=clean, noisy=noisy, G=G))
    return streaks


def analytic_tangent(p, G):
    """Exact tangent at point p on the Kossel circle of lattice vector G: perpendicular to both
    the radius vector (p - G/2) and G, i.e. Ghat x (p - G/2), normalized. No fitting -- ground
    truth, valid for a point exactly ON the circle (we evaluate it at the CLEAN point)."""
    Ghat = G / np.linalg.norm(G)
    r = p - G / 2.0
    t = np.cross(Ghat, r)
    n = np.linalg.norm(t)
    return t / n if n > 0 else None


def local_tangent_pca(points, i, halfwin=3):
    """Robust local tangent estimate at points[i]: top eigenvector of the position covariance
    over a small window of consecutive (noisy) points -- the 3-D analog of ridge_moments'
    per-segment covariance ellipse (there restricted to a 2-D pixel image's row/col plane;
    here the natural 3-D generalization, since there's no pixel plane to project onto)."""
    lo, hi = max(0, i - halfwin), min(len(points), i + halfwin + 1)
    seg = points[lo:hi]
    if len(seg) < 3:
        return None
    d = seg - seg.mean(0)
    cov = d.T @ d
    w, v = np.linalg.eigh(cov)
    return v[:, -1]                                  # eigenvector of largest eigenvalue


def chord_tangent(points, i):
    """Naive baseline: first-to-last point direction of the whole streak (what you'd get from a
    single global fit, ridge_moments' 'inflated by curvature' failure mode)."""
    d = points[-1] - points[0]
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def angle_deg(u, v):
    if u is None or v is None:
        return np.nan
    c = np.clip(abs(u @ v), 0, 1)                     # abs(): tangent has a pi (sign) ambiguity
    return float(np.degrees(np.arccos(c)))


def run(ncry=15, halfwin=1, seed=7, thin=1, min_pts=3):
    """thin=1 (default here): use every admitted sample per streak, not simulate()'s ::6
    decimation -- the best-case point density this synthetic model can offer, since the ::6
    version left a median of 1 point/streak (nothing to fit a tangent to at all)."""
    noises = (0.0, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3)
    print(f"tangent recoverability on simulate() streaks  ncry={ncry}  halfwin={halfwin}  "
          f"thin={thin} (1=no decimation, raw admitted samples)")
    print(f"{'noise(1/A)':>11} {'n_streaks':>10} {'n_pts_med':>10} "
          f"{'local-PCA err(deg) p50/p90':>28} {'chord err(deg) p50/p90':>24}")
    for noise in noises:
        rng = np.random.default_rng(seed)
        local_err, chord_err, npts = [], [], []
        for _ in range(ncry):
            R = rand_rot(rng)
            streaks = _simulate_grouped(R, rng, noise, thin=thin)
            for s in streaks:
                clean, noisy, G = s["clean"], s["noisy"], s["G"]
                n = len(clean)
                if n < min_pts:
                    continue
                npts.append(n)
                i = n // 2                            # evaluate at the streak's midpoint
                t_true = analytic_tangent(clean[i], G)
                t_local = local_tangent_pca(noisy, i, halfwin=halfwin)
                t_chord = chord_tangent(noisy, i)
                local_err.append(angle_deg(t_local, t_true))
                chord_err.append(angle_deg(t_chord, t_true))
        local_err, chord_err = np.array(local_err), np.array(chord_err)
        if len(local_err) == 0:
            print(f"{noise:11.1e}          0          -   (no streaks with >= {min_pts} pts)")
            continue
        p50l, p90l = np.nanpercentile(local_err, [50, 90])
        p50c, p90c = np.nanpercentile(chord_err, [50, 90])
        print(f"{noise:11.1e} {len(local_err):10d} {int(np.median(npts)):10d} "
              f"{p50l:14.1f} / {p90l:9.1f} {p50c:12.1f} / {p90c:9.1f}")


if __name__ == "__main__":
    run()
