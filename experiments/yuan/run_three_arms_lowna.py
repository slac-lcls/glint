"""Same matched PTS-vs-TAN comparison as run_three_arms.py, but on the low-NA (genuinely
reflection-starved) dataset -- generate_dataset_lowna.py, NA=0.016, median 11 streaks/crystal
(vs 22.5 at the NA=0.028 baseline dataset, where #streaks did NOT predict PTS-vs-TAN at all).

set_NA(NA) must be called before running ANY search on this dataset -- score_batch_gpu's cone
gate and _simulate_grouped's admission mask both read the module-level COSA/SINA, which are
NA-dependent and were patched during generation. Skipping this would score the low-NA dataset's
kobs against the WRONG (baseline 0.028) convergence cone.

  python run_three_arms_lowna.py [n_coarse]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from cbxd_sweep import set_NA
from generate_dataset_lowna import NA, load_dataset
from cbxd_hough_gpu import hough_seed_index
from cbxd_hough_tangent import hough_seed_index_tangent, streak_reprs_and_tangents


def frac_indexed(R, kobs, lab):
    if R is None:
        return 0.0
    return float(score(R, kobs, 0.0025, ret_mask=True)[lab].mean())


def run(n_coarse=5_000_000):
    set_NA(NA)
    ds = load_dataset()
    n = len(ds)
    fr = dict(pts=np.zeros(n), tan=np.zeros(n))
    n_streaks = np.array([len(c["cents"]) for c in ds])
    print(f"low-NA (NA={NA}) matched PTS-vs-TAN, n={n}  n_coarse={n_coarse}")
    print(f"{'i':>3} {'#streaks':>9} {'PTS':>7} {'TAN':>7}   time")
    for i, c in enumerate(ds):
        Rt, kobs, lab, noise = c["Rt"], c["kobs"], c["lab"], c["noise"]
        t0 = time.perf_counter()

        R_pts = hough_seed_index(kobs, np.random.default_rng(2000 + i), n_coarse=n_coarse)
        fr["pts"][i] = frac_indexed(R_pts, kobs, lab)

        reprs, tangents = streak_reprs_and_tangents(Rt, noise, np.random.default_rng(3000 + i))
        R_tan = hough_seed_index_tangent(kobs, reprs, tangents, np.random.default_rng(4000 + i),
                                         n_coarse=n_coarse)
        fr["tan"][i] = frac_indexed(R_tan, kobs, lab)

        dt = time.perf_counter() - t0
        print(f"{i:3d} {n_streaks[i]:9d} {fr['pts'][i]:7.2f} {fr['tan'][i]:7.2f}   {dt:5.1f}s",
              flush=True)

    def succ(v):
        return int((v > 0.7).sum())

    print(f"\nsuccess: PTS {succ(fr['pts'])}/{n}  TAN {succ(fr['tan'])}/{n}")
    np.savez("results_three_arms_lowna.npz", n_streaks=n_streaks, **fr)
    print("saved results_three_arms_lowna.npz")


if __name__ == "__main__":
    n_coarse = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
    run(n_coarse)
