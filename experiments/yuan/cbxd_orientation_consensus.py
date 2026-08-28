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


# Proper (rotation-only) point-group operators for the orthorhombic P cell in cbxd_joint.CELL
# (16, 21, 25, 90, 90, 90) -- point group 222. cbxd_joint.HS is a symmetric +-h,+-k,+-l grid,
# so each of these permutes the lattice nodes exactly: score(R @ g, kobs) == score(R, kobs)
# for every g here. R and R @ g (g != I) are therefore the SAME physical orientation, not two
# different hypotheses that happen to agree -- treating them as different is issue #58.
ORTHORHOMBIC_222 = [
    np.eye(3),
    np.diag([1.0, -1.0, -1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([-1.0, -1.0, 1.0]),
]


def rot_angle_deg_sym(Ra, Rb, point_group=ORTHORHOMBIC_222):
    """Geodesic angle between two rotations, minimized over the cell's proper point-group
    symmetry -- the distance used for consensus grouping (issue #58). Falls back to the plain
    geodesic angle when point_group=[I] (e.g. for a lower-symmetry cell)."""
    return min(rot_angle_deg(Ra, Rb @ g) for g in point_group)


def _grp_rotations(Rs, ang_tol_deg, point_group=ORTHORHOMBIC_222):
    """Greedy grouping by rotational proximity -- same shape as
    glint.multishot._grp_reduced: each hypothesis joins the FIRST existing group whose
    representative it's within tolerance of (modulo the cell's point-group symmetry -- issue
    #58), else starts a new group. Returns the group with the most members as
    [rep_index, count]."""
    groups = []                                   # [rep_index, count]
    for idx in range(len(Rs)):
        for grp in groups:
            if rot_angle_deg_sym(Rs[grp[0]], Rs[idx], point_group) <= ang_tol_deg:
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


def consensus_orientation(Rs, min_support=2, ang_tol_deg=10.0, point_group=ORTHORHOMBIC_222):
    """Group independent per-shot orientation hypotheses (some may be None) by rotational
    proximity, modulo the cell's proper point-group symmetry (issue #58 -- R and R@g score
    identically for g in point_group, so they're the same hypothesis, not competing ones);
    return (representative R, support count), or (None, 0) if the dominant cluster doesn't
    clear min_support -- mirrors glint.multishot.consensus_cell's reject-if-too-weak behavior
    (a single lucky hit isn't consensus)."""
    valid = [R for R in Rs if R is not None]
    if not valid:
        return None, 0
    ridx, cnt = _grp_rotations(valid, ang_tol_deg, point_group)
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

    # issue #58 regression: the cell's four proper point-group copies of Rt are the SAME
    # orientation (score(R@g, kobs) == score(R, kobs)), so they must merge into one cluster
    # of support 4, not fragment into up to 4 singletons (which is what the pre-fix raw
    # geodesic-angle grouping did -- the three non-identity copies sit at exactly 180deg,
    # far past ang_tol_deg=10, so _grp_rotations without the point-group fix yields support 1).
    Rs_sym = [Rt @ g for g in ORTHORHOMBIC_222]
    R_cons3, support3 = consensus_orientation(Rs_sym, min_support=2, ang_tol_deg=10.0)
    assert support3 == 4, (support3, "symmetry-equivalent copies must merge into one cluster")
    print("golden check OK")
