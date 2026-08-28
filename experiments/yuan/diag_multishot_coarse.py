"""Diagnostic for the flat/worse multishot pilot result (run_multishot.py, results_multishot.jsonl):
does the COARSE ranking stage (score_batch_gpu at tol_c=0.03, before refine()) actually put the
true orientation's neighborhood in the top-K candidates, and does that get WORSE as M (pooled
shots) grows? Fixed n_coarse and tol_c were tuned for single-shot point counts (~20-50 pts);
this checks whether they still work once Np grows to ~150-400 (M=8).

  python diag_multishot_coarse.py
"""
import sys

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import rotvec

from generate_dataset_multishot import load_dataset, pooled
from cbxd_hough_gpu import rand_rot_batch, score_batch_gpu

N_COARSE = 1_000_000
KEEP = 10
TOL_C = 0.03


def rot_angle_deg(Ra, Rb):
    """Geodesic angle between two rotation matrices, in degrees."""
    Rrel = Ra.T @ Rb
    c = np.clip((np.trace(Rrel) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def diagnose(noise, crystal_idx, m_values=(1, 2, 4, 8), seed=1):
    ds = load_dataset(noise)
    c = ds[crystal_idx]
    Rt = c["Rt"]
    print(f"noise={noise:.0e} crystal={crystal_idx} n_streaks(M=1)={len(c['shots'][0][2])}")
    print(f"{'M':>3} {'n_points':>8} {'best_coarse_ang(deg)':>21} {'topK_min_ang(deg)':>18} "
          f"{'topK_count':>10} {'best_count':>10} {'2nd_best_count':>15}")
    for m in m_values:
        kobs, lab, cents = pooled(c, m)
        rng = np.random.default_rng(seed)
        Rs = rand_rot_batch(rng, N_COARSE)
        counts = score_batch_gpu(Rs, kobs, TOL_C)
        order = np.argsort(counts)[::-1]
        top = order[:KEEP]
        angs_top = np.array([rot_angle_deg(Rt, Rs[i]) for i in top])
        # closest coarse candidate to truth, ANYWHERE in the n_coarse pool (not just top-K)
        # -- sampled cheaply via the same batch (angle to a subsample, since computing angle
        # to all 1e6 is fine, it's just numpy)
        angs_all = np.array([rot_angle_deg(Rt, R) for R in Rs[order[:2000]]])  # top-2000 by vote is enough context
        best_ang_anywhere_in_topN = angs_all.min()
        print(f"{m:3d} {len(kobs):8d} {best_ang_anywhere_in_topN:21.2f} {angs_top.min():18.2f} "
              f"{int((angs_top < 10).sum()):10d} {int(counts[top[0]]):10d} {int(counts[top[1]]):15d}")


if __name__ == "__main__":
    # crystal 2 at noise=2e-4 regressed hardest in the pilot (1.00 -> 0.43 -> 0.41 -> 0.36)
    diagnose(2e-4, 2)
    print()
    diagnose(2e-4, 0)   # for contrast: this one stayed roughly flat/good
