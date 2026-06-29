"""Cross-symmetry comparison at a HARD operating point.

The mild setting in symmetry_eval.py (n=45, sigma=0.001, 10% spurious) leaves little
headroom -- classical already does ~60-80%, so the learned re-ranker is a wash. The
single-cell wins were at a hard point (sparse / past the jitter cliff). This re-runs
classical vs learn-mixed across symmetries at a hard setting to see if the learned
advantage generalizes where it should matter.
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
from fftindex import LearnedPeakFinder, index_shot, score, simulate_shot
from fftindex.lattice import SYMMETRIES, random_cell
from fftindex.peakfind import find_peaks_classical

N_TRIALS = 15
SETTING = dict(n_target=30, pos_sigma=0.0015, frac_spurious=0.2)
MIXED = LearnedPeakFinder(joblib.load("detector_mixed.joblib"))


def solve(peakfinder, sym, seed0=9000):
    sv = 0
    for s in range(N_TRIALS):
        rng = np.random.default_rng(seed0 + s)
        shot = simulate_shot(rng=rng, cell=random_cell(rng, sym), **SETTING)
        sv += score(shot, index_shot(shot.g, shot.meta["qmax"],
                                     peakfinder=peakfinder))["solved"]
    return 100 * sv / N_TRIALS


if __name__ == "__main__":
    t0 = time.time()
    print(f"HARD setting {SETTING}  (random cell per trial)")
    print(f"{'symmetry':<13}{'classical':>10}{'learn-mixed':>12}{'delta':>7}")
    for sym in SYMMETRIES:
        c = solve(find_peaks_classical, sym)
        m = solve(MIXED, sym)
        tag = " (sheared)" if sym in ("monoclinic", "triclinic") else ""
        print(f"{sym:<13}{c:>9.0f}%{m:>11.0f}%{m - c:>+6.0f}{tag}")
    print(f"\nwall time: {time.time() - t0:.0f}s")
