"""Run all three arms (COM/PTS/TAN) on the fixed on-disk dataset (generate_dataset.py) -- a
genuine matched, per-crystal comparison, unlike the earlier in-session runs where crystal identity
silently drifted between arms because a search's rng draws perturbed the next crystal's generation.

Saves per-crystal fracs to results_three_arms.npz (for the streak-count threshold analysis) and
prints a summary table.

  python run_three_arms.py [n_coarse]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from generate_dataset import load_dataset
from cbxd_arm1_com import com_seed_index
from cbxd_hough_gpu import hough_seed_index
from cbxd_hough_tangent import hough_seed_index_tangent, streak_reprs_and_tangents
from tangent_validate import _simulate_grouped


def get_reprs_tangents(Rt, noise, seed):
    return streak_reprs_and_tangents(Rt, noise, np.random.default_rng(seed))


def frac_indexed(R, kobs, lab):
    if R is None:
        return 0.0
    return float(score(R, kobs, 0.0025, ret_mask=True)[lab].mean())


def run(n_coarse=5_000_000):
    ds = load_dataset()
    n = len(ds)
    fr = dict(com=np.zeros(n), pts=np.zeros(n), tan=np.zeros(n))
    n_streaks = np.array([len(c["cents"]) for c in ds])
    print(f"three-arm matched comparison, fixed dataset (n={n})  n_coarse={n_coarse}")
    print(f"{'i':>3} {'#streaks':>9} {'COM':>7} {'PTS':>7} {'TAN':>7}   time")
    for i, c in enumerate(ds):
        Rt, kobs, lab, cents, noise = c["Rt"], c["kobs"], c["lab"], c["cents"], c["noise"]
        t0 = time.perf_counter()

        R_com = com_seed_index(cents, np.random.default_rng(1000 + i), n_coarse=n_coarse)
        fr["com"][i] = frac_indexed(R_com, kobs, lab)

        R_pts = hough_seed_index(kobs, np.random.default_rng(2000 + i), n_coarse=n_coarse)
        fr["pts"][i] = frac_indexed(R_pts, kobs, lab)

        reprs, tangents = get_reprs_tangents(Rt, noise, 3000 + i)
        R_tan = hough_seed_index_tangent(kobs, reprs, tangents, np.random.default_rng(4000 + i),
                                         n_coarse=n_coarse)
        fr["tan"][i] = frac_indexed(R_tan, kobs, lab)

        dt = time.perf_counter() - t0
        print(f"{i:3d} {n_streaks[i]:9d} {fr['com'][i]:7.2f} {fr['pts'][i]:7.2f} "
              f"{fr['tan'][i]:7.2f}   {dt:5.1f}s", flush=True)

    def succ(v):
        return int((v > 0.7).sum())

    print(f"\nsuccess: COM {succ(fr['com'])}/{n}  PTS {succ(fr['pts'])}/{n}  TAN {succ(fr['tan'])}/{n}")
    np.savez("results_three_arms.npz", n_streaks=n_streaks, **fr)
    print("saved results_three_arms.npz")


if __name__ == "__main__":
    n_coarse = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
    run(n_coarse)
