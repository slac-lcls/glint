"""Cascade: PTS by default, escalate to TAN only if PTS's own result looks unconfident.

Why: TAN beats PTS badly on the fixed dataset (18/20 vs 14/20, results_three_arms.npz) but costs
~2.5-3x the wall-clock (~45-60s/crystal vs ~20s). A #streaks threshold can't reliably predict
which crystals need TAN (two crystals with identical streak counts needed opposite arms in that
data). The fix: don't predict in advance -- run the cheap arm first, and use ITS OWN result
confidence as the escalation signal. No ground truth needed (real blind indexing doesn't have it
either), and it's free: you were always going to run PTS first.

Confidence = PTS's matched fraction of ALL observed points, score(R, kobs, tol)/len(kobs). Because
~30% of kobs is spurious (cbxd_joint.SPUR), even a PERFECT solve can't reach confidence 1.0 -- the
ceiling is roughly 1/(1+SPUR) ~= 0.77. conf_thresh is calibrated against the fixed dataset in
calibrate() below, not guessed.
"""
import sys

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from cbxd_hough_gpu import hough_seed_index
from cbxd_hough_tangent import hough_seed_index_tangent


def hough_seed_index_cascade(kobs, reprs, tangents, rng, n_coarse=5_000_000, conf_tol=0.0025,
                             conf_thresh=0.5, **kw):
    """Returns (R, arm_used, confidence). PTS is tried first; TAN only runs if PTS's own
    confidence is below conf_thresh."""
    R_pts = hough_seed_index(kobs, rng, n_coarse=n_coarse, **kw)
    s_pts = score(R_pts, kobs, conf_tol) if R_pts is not None else 0
    conf = s_pts / max(len(kobs), 1)
    if conf >= conf_thresh:
        return R_pts, "PTS", conf
    R_tan = hough_seed_index_tangent(kobs, reprs, tangents, rng, n_coarse=n_coarse, **kw)
    return R_tan, "TAN", conf


if __name__ == "__main__":
    pass
