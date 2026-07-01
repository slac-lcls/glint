"""Does the learned peakfinder beat the classical baseline?

Runs the three cuts (spot count, jitter, spurious) with the classical peakfinder
vs. the trained LearnedPeakFinder plugged into index_shot, reporting solve%.
The key questions: does re-ranking move the jitter cliff and the sub-30-spot floor
that selection tricks could not?

Eval seeds (300+) are disjoint from the training stream (rng(0)).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time

import joblib
import numpy as np

sys.path.insert(0, "..")
from glint import LearnedPeakFinder, index_shot, score, simulate_shot
from glint.peakfind import find_peaks_classical

N_TRIALS = 20
LEARNED = LearnedPeakFinder(joblib.load("detector_rf.joblib"))


def solve_rate(peakfinder, n_target, pos_sigma=0.0, frac_spurious=0.0):
    sv = 0
    for s in range(N_TRIALS):
        rng = np.random.default_rng(300 + s)
        shot = simulate_shot(rng=rng, n_target=n_target, pos_sigma=pos_sigma,
                             frac_spurious=frac_spurious)
        res = index_shot(shot.g, shot.meta["qmax"], peakfinder=peakfinder)
        sv += score(shot, res)["solved"]
    return 100 * sv / N_TRIALS


def row(label, **kw):
    c = solve_rate(find_peaks_classical, **kw)
    l = solve_rate(LEARNED, **kw)
    print(f"     {label:<16}: classical {c:>3.0f}%   learned {l:>3.0f}%   (Δ {l - c:+.0f})")


if __name__ == "__main__":
    t0 = time.time()
    print("(1) vs spot count (clean)")
    for nt in [20, 25, 30, 40, 60]:
        row(f"n={nt}", n_target=nt)

    print("\n(2) vs jitter sigma [1/A] (n=60)")
    for ps in [0.001, 0.0015, 0.002, 0.003]:
        row(f"sigma={ps}", n_target=60, pos_sigma=ps)

    print("\n(3) vs spurious fraction (n=60)")
    for fs in [0.1, 0.2, 0.4]:
        row(f"spurious={fs}", n_target=60, frac_spurious=fs)

    print(f"\nwall time: {time.time() - t0:.1f}s")
