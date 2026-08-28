"""Spurious-peak meters -- how many of a frame's peaks are not on the lattice, and (the one that
matters) how close that pushes the frame to the blind-indexing ceiling.

"How many spurious peaks?" is really four questions, each with its own meter. None needs ground truth;
they measure different things and are meant to be read together (see `frame_report`).

  1. finder_spurious(p_peak)      -- E[spurious] = sum(1 - p_peak). The learned photon finder already
                                     rates each peak's realness; this is the literal count it implies.
                                     Knows what the FINDER thinks, not what the LATTICE says.
  2. unindexed_fraction(q, M)     -- peaks not on the best cell M. An UPPER bound: also catches weak-real
                                     peaks the cell missed and geometry error. Cheap triage.
  3. null_margin(q, M[, K])       -- THE ceiling meter. Scores the true orientation's overlap against the
                                     null of random (wrong) orientations of the same lattice. z = how many
                                     sigma the true fit sits above the by-chance floor; for a K-orientation
                                     search the extreme-value floor is mean + sigma*sqrt(2 ln K), so
                                     `wall` = the true fit does NOT clear it => spurious-limited frame.
                                     Needs no labels; per-frame; separates wall-limited from blank frames.
  4. run_orphan_rate(frames, Mc)  -- model-free, cross-frame: fraction of all peaks across a run that miss
                                     the CONSENSUS cell (corroborated across frames, so a spurious single-
                                     frame lattice can't earn credit). The reliability-selection meter.

Cell convention matches stream_driver._inliers / alias_gate: M columns are the real cell vectors, so
hkl = q @ M and a peak is "on the lattice" when q @ M is near-integer. A random crystal orientation Rrot
rotates the cell to Rrot @ M, which is how the null in meter 3 is generated.
"""
import numpy as np

from .lattice import random_rotation

HKL_TOL = 0.15                       # == stream_driver._inliers near-integer window
WALL_Z = 3.0                         # z below this: the true fit is not cleanly separated from the null


def _inlier_mask(q, M, tol=HKL_TOL):
    """Boolean mask of peaks whose q @ M is within `tol` of an integer hkl."""
    hf = np.asarray(q, float) @ np.asarray(M, float)
    return np.abs(hf - np.round(hf)).max(1) < tol


def _inliers(q, M, tol=HKL_TOL):
    return int(_inlier_mask(q, M, tol).sum())


# ------------------------------------------------------------------------------- 1. finder confidence
def finder_spurious(p_peak):
    """E[spurious] and spurious fraction implied by the finder's per-peak probabilities."""
    p = np.clip(np.asarray(p_peak, float), 0.0, 1.0)
    n = int(p.size)
    if n == 0:
        return dict(n=0, expected_spurious=0.0, spurious_frac=0.0)
    exp = float((1.0 - p).sum())
    return dict(n=n, expected_spurious=exp, spurious_frac=exp / n)


# --------------------------------------------------------------------------------- 2. unindexed count
def unindexed_fraction(q, M, tol=HKL_TOL):
    """Peaks not near any node of cell M -- an upper bound on the spurious count."""
    q = np.asarray(q, float)
    n = int(len(q))
    if n == 0:
        return dict(n=0, n_indexed=0, unindexed_frac=0.0)
    ni = _inliers(q, M, tol)
    return dict(n=n, n_indexed=ni, unindexed_frac=1.0 - ni / n)


# --------------------------------------------------------------------------- 3. the ceiling: null test
def null_margin(q, M, tol=HKL_TOL, n_null=256, K=None, rng=None):
    """Separation of the true orientation's overlap from the random-orientation null.

    Returns s_true (peaks on M), the null mean/std over `n_null` random orientations of the same lattice,
    z=(s_true-mean)/std, and -- if a search size K is given -- the extreme-value floor mean+std*sqrt(2 ln K)
    and `wall` (True when s_true does not clear it). `wall_z` flags z<WALL_Z when K is not supplied."""
    q = np.asarray(q, float)
    rng = rng or np.random.default_rng()
    M = np.asarray(M, float)
    if len(q) == 0 or not np.all(np.isfinite(M)) or abs(np.linalg.det(M)) < 1e-9:
        return dict(n=int(len(q)), s_true=0, null_mean=0.0, null_std=0.0, z=0.0,
                    null_max=0.0, evt_floor=None, wall=None, wall_z=True)
    s_true = _inliers(q, M, tol)
    null = np.empty(int(n_null))
    for i in range(int(n_null)):
        null[i] = _inliers(q, random_rotation(rng) @ M, tol)     # wrong orientation of the SAME cell
    mu, sd = float(null.mean()), float(null.std())
    z = (s_true - mu) / sd if sd > 1e-9 else float("inf")
    out = dict(n=int(len(q)), s_true=s_true, null_mean=mu, null_std=sd, z=z,
               null_max=float(null.max()), wall_z=bool(z < WALL_Z))
    if K is not None and K > 1:
        floor = mu + sd * np.sqrt(2.0 * np.log(float(K)))        # E[max] of K near-Gaussian nulls
        out["evt_floor"] = float(floor)
        out["wall"] = bool(s_true <= floor)
    else:
        out["evt_floor"] = None
        out["wall"] = None
    return out


# ----------------------------------------------------------------------- 4. cross-frame orphan rate
def run_orphan_rate(frames_q, M_consensus, tol=HKL_TOL):
    """Fraction of all peaks across a run that miss the consensus cell -- the model-free spurious rate
    relative to the corroborated lattice (a single-frame spurious cell earns no credit here)."""
    tot = orph = 0
    per_frame = []
    for q in frames_q:
        q = np.asarray(q, float)
        if len(q) == 0:
            per_frame.append(0.0); continue
        ni = _inliers(q, M_consensus, tol)
        tot += len(q); orph += len(q) - ni
        per_frame.append(1.0 - ni / len(q))
    return dict(n_peaks=int(tot), orphan_frac=(orph / tot if tot else 0.0), per_frame=per_frame)


# --------------------------------------------------------------------------------- combined per-frame
def frame_report(q, M=None, p_peak=None, tol=HKL_TOL, n_null=256, K=None, rng=None):
    """All applicable per-frame meters. Skips lattice meters when M is None and the finder meter when
    p_peak is None. The headline is `null.z` / `null.wall`: is this frame spurious-limited or just blank?"""
    rep = dict(n_peaks=int(len(np.asarray(q, float))))
    if p_peak is not None:
        rep["finder"] = finder_spurious(p_peak)
    if M is not None:
        rep["unindexed"] = unindexed_fraction(q, M, tol)
        rep["null"] = null_margin(q, M, tol=tol, n_null=n_null, K=K, rng=rng)
    return rep


def format_report(rep):
    """One-line-per-meter human summary of a frame_report dict."""
    lines = [f"peaks: {rep.get('n_peaks', 0)}"]
    if "finder" in rep:
        f = rep["finder"]
        lines.append(f"  finder    : E[spurious]={f['expected_spurious']:.1f} ({f['spurious_frac']*100:.0f}%)")
    if "unindexed" in rep:
        u = rep["unindexed"]
        lines.append(f"  unindexed : {u['unindexed_frac']*100:.0f}% ({u['n']-u['n_indexed']}/{u['n']} off-lattice)")
    if "null" in rep:
        nu = rep["null"]
        tag = "WALL" if nu.get("wall") else ("wall?" if nu["wall_z"] else "clear")
        floor = f", evt_floor={nu['evt_floor']:.1f}" if nu.get("evt_floor") is not None else ""
        lines.append(f"  null      : s_true={nu['s_true']} vs null {nu['null_mean']:.1f}+/-{nu['null_std']:.1f}"
                     f"  z={nu['z']:.1f}{floor}  -> {tag}")
    return "\n".join(lines)
