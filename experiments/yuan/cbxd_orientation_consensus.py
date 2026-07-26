"""CBXD deliverable 4 (issue #12), take 2: RESULT-level cross-frame consensus, mirroring
glint/multishot.py::consensus_cell -- Stefano's validated SFX mechanism -- instead of the
DATA-level pooling in run_multishot.py (which just concatenates M shots' points into one bigger
accumulator call; that's what's currently running the big background sweep).

consensus_cell's actual shape: solve each frame INDEPENDENTLY (own peaks, own weak blind search,
no data shared between frames), collect the M hypotheses (many individually wrong), group them by
whether they describe the SAME lattice, and return the dominant cluster if it clears min_support.
The true cell recurs because every frame's search is aimed at it even when a given frame misses;
wrong guesses are frame-specific noise and don't cluster.

This file is the orientation analog: instead of same_lattice (rotation/basis-invariant cell
comparison), group by rotational proximity (geodesic angle < ang_tol_deg) -- same greedy
single-link-to-representative clustering as glint.multishot._grp_reduced, just swapping the
"same lattice" predicate for "same rotation".

Works with EITHER arm:
  - com_seed_index (arm 1, centroid-only -- deliberately the WEAK per-shot signal, analogous to
    a single sparse SFX frame, since it doesn't exploit streak curvature/cross-streak pooling at
    all) -- isolates how much cross-frame consensus buys on its OWN, uncontaminated by
    cross-streak pooling's own contribution.
  - hough_seed_index (arm 2/PTS, the full cross-streak accumulator) -- tests whether cross-frame
    consensus STACKS on top of cross-streak pooling, or is redundant with it.

  python cbxd_orientation_consensus.py    # runs a tiny golden check
"""
import sys

import numpy as np

sys.path.insert(0, "..")

from cbxd_arm1_com import com_seed_index
from cbxd_hough_gpu import hough_seed_index


def rot_angle_deg(Ra, Rb):
    """Geodesic angle between two rotation matrices, in degrees."""
    Rrel = Ra.T @ Rb
    c = np.clip((np.trace(Rrel) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def _grp_rotations(Rs, ang_tol_deg):
    """Greedy grouping by rotational proximity -- same shape as
    glint.multishot._grp_reduced: each hypothesis joins the FIRST existing group whose
    representative it's within tolerance of, else starts a new group. Returns the group with
    the most members as [rep_index, count]."""
    groups = []                                   # [rep_index, count]
    for idx in range(len(Rs)):
        for grp in groups:
            if rot_angle_deg(Rs[grp[0]], Rs[idx]) <= ang_tol_deg:
                grp[1] += 1
                break
        else:
            groups.append([idx, 1])
    return max(groups, key=lambda g: g[1])


def majority_support(m):
    """Strict-majority min_support for M independent shots (m//2 + 1) -- unlike a fixed
    constant, this gets HARDER to clear as M grows, so accidental agreement among a growing
    pool of independent hypotheses doesn't masquerade as stronger consensus (the bug flagged
    on issue #12: a fixed min_support=2 let coverage climb with M while conditional accuracy
    fell, 67%->59%->41%)."""
    return m // 2 + 1


def consensus_orientation(Rs, min_support=2, ang_tol_deg=10.0):
    """Group independent per-shot orientation hypotheses (some may be None) by rotational
    proximity; return (representative R, support count), or (None, 0) if the dominant cluster
    doesn't clear min_support -- mirrors glint.multishot.consensus_cell's reject-if-too-weak
    behavior (a single lucky hit isn't consensus)."""
    valid = [R for R in Rs if R is not None]
    if not valid:
        return None, 0
    ridx, cnt = _grp_rotations(valid, ang_tol_deg)
    return (valid[ridx] if cnt >= min_support else None), cnt


def multishot_consensus_pts(shots, seed_base, n_coarse=1_000_000, min_support=2,
                            ang_tol_deg=10.0, **kw):
    """Run hough_seed_index (PTS, cross-streak pooling) INDEPENDENTLY on each shot's own kobs
    (no data pooling), then consensus-cluster the M resulting orientation hypotheses."""
    Rs = [hough_seed_index(kobs, np.random.default_rng(seed_base + m), n_coarse=n_coarse, **kw)
          for m, (kobs, lab, cents) in enumerate(shots)]
    R_cons, support = consensus_orientation(Rs, min_support=min_support, ang_tol_deg=ang_tol_deg)
    return R_cons, support, Rs


def multishot_consensus_com(shots, seed_base, n_coarse=1_000_000, min_support=2,
                            ang_tol_deg=10.0, **kw):
    """Run com_seed_index (arm 1, centroid-only -- deliberately weak, no cross-streak pooling)
    INDEPENDENTLY on each shot's own cents, then consensus-cluster."""
    Rs = [com_seed_index(cents, np.random.default_rng(seed_base + m), n_coarse=n_coarse, **kw)
          for m, (kobs, lab, cents) in enumerate(shots)]
    R_cons, support = consensus_orientation(Rs, min_support=min_support, ang_tol_deg=ang_tol_deg)
    return R_cons, support, Rs


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    from cbxd_joint import rand_rot
    Rt = rand_rot(rng)
    # golden check: identical hypotheses cluster together; a scattered outlier doesn't join
    Rs = [Rt, Rt @ np.eye(3), rand_rot(rng), rand_rot(rng)]
    R_cons, support = consensus_orientation(Rs, min_support=2, ang_tol_deg=10.0)
    assert support == 2 and np.allclose(R_cons, Rt), (support, R_cons)
    R_cons2, support2 = consensus_orientation([rand_rot(rng) for _ in range(3)], min_support=2)
    assert R_cons2 is None and support2 <= 1, (support2, R_cons2)   # 3 independent randoms shouldn't cluster
    print("golden check OK")
