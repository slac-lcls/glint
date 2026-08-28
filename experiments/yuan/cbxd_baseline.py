"""Frozen baseline for the CBXD blind-orientation search: what `seed_index` (coarse
centroid random-search + Nelder-Mead refine, cbxd_joint.py) gets today, on the same
synthetic crystals every time. Anything built to replace it (batched score over candidate
orientations, arc-Hough accumulator, ...) should be benchmarked against these numbers on
this same script -- same seed, same metric, same success threshold (frac_indexed >= 0.7).

  python cbxd_baseline.py [ncry]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import rand_rot, score, seed_index, simulate

NOISE_LEVELS = (1e-4, 2e-4)
SEED = 2                      # matches cbxd_joint.py's own __main__ "blind" mode


def run(ncry):
    """NOTE: crystal generation and each crystal's search use INDEPENDENT rng streams (see
    experiments/yuan/generate_dataset.py's docstring) -- seed_index's own internal 20k-candidate
    rand_rot draws must not perturb the next crystal's orientation."""
    print(f"seed_index baseline (blind: 20k-candidate cent_score search -> refine top-10)")
    print(f"ncry={ncry}  seed={SEED}")
    print(f"{'noise(1/A)':>11} {'wall/crystal(s)':>16} {'success':>9} {'median real idx':>16}")
    for noise in NOISE_LEVELS:
        rng = np.random.default_rng(SEED)
        ok = 0
        fr = []
        times = []
        for c in range(ncry):
            Rt = rand_rot(rng)
            kobs, lab, cents = simulate(Rt, rng, noise)
            t0 = time.perf_counter()
            Rh = seed_index(kobs, cents, np.random.default_rng(SEED + 1000 + c))
            times.append(time.perf_counter() - t0)
            idx = score(Rh, kobs, 0.0025, ret_mask=True)
            fr.append(idx[lab].mean())
            ok += idx[lab].mean() > 0.7
        print(f"{noise:11.1e} {np.mean(times):16.2f} {f'{ok}/{ncry}':>9} "
              f"{100*np.median(fr):15.0f}%")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run(n)
