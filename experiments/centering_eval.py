"""Indexing across Bravais centering types (FCC, BCC, base-centered) vs primitive.

Centered lattices have systematic absences -> the observed spots form the PRIMITIVE
lattice (a rhombohedral cell for FCC/BCC), which is what the indexer recovers and
what score()'s lattice-match checks. Compares classical vs the feature re-ranker.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys

import joblib
import numpy as np

sys.path.insert(0, "..")
from glint import LearnedPeakFinder, index_shot, score, simulate_shot
from glint.peakfind import find_peaks_classical

N_TRIALS = 20
SETTING = dict(n_target=45, pos_sigma=0.001, frac_spurious=0.1)
mixed = LearnedPeakFinder(joblib.load("detector_mixed.joblib"))

# (label, centering, cell-shape generator) -- I/F on cubic, C on orthorhombic
CASES = [
    ("P  (simple cubic)", "P", lambda r: (lambda a: (a, a, a, 90, 90, 90))(r.uniform(50, 90))),
    ("I  (BCC)", "I", lambda r: (lambda a: (a, a, a, 90, 90, 90))(r.uniform(50, 90))),
    ("F  (FCC)", "F", lambda r: (lambda a: (a, a, a, 90, 90, 90))(r.uniform(50, 90))),
    ("C  (orthorhombic)", "C", lambda r: tuple(r.uniform(50, 90, 3)) + (90, 90, 90)),
]


def solve(pf, centering, cellgen, seed0=4000):
    sv = 0
    for s in range(N_TRIALS):
        rng = np.random.default_rng(seed0 + s)
        shot = simulate_shot(rng=rng, cell=cellgen(rng), centering=centering, **SETTING)
        sv += score(shot, index_shot(shot.g, shot.meta["qmax"], peakfinder=pf))["solved"]
    return 100 * sv / N_TRIALS


if __name__ == "__main__":
    print(f"solve% at {SETTING}")
    print(f"{'centering':<20}{'classical':>10}{'feature':>9}")
    for label, cen, cellgen in CASES:
        cl = solve(find_peaks_classical, cen, cellgen)
        fe = solve(mixed, cen, cellgen)
        print(f"{label:<20}{cl:>9.0f}%{fe:>8.0f}%")
