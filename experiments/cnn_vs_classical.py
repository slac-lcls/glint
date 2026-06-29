"""Test the trained 3D-CNN peakfinder inside the indexer vs the classical baseline.

Loads cnn_peakfinder.pt, wraps it as CNNPeakFinder, and compares solve% on random
cells within the CNN's training size range (30-64 A, fixed n=96 grid).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys

import numpy as np
import torch

sys.path.insert(0, "..")
from fftindex import index_shot, score, simulate_shot
from fftindex.cnn import CNNPeakFinder, UNet3D
from fftindex.lattice import random_cell
from fftindex.peakfind import find_peaks_classical

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("cnn_peakfinder.pt", map_location=device)
model = UNet3D()
model.load_state_dict(ck["model"])
cnn = CNNPeakFinder(model, n_model=ck["n"], device=device)
print(f"loaded cnn_peakfinder.pt (n={ck['n']}) on {device}")


def solve(pf, n_target, ps, fs, ntr=20, seed0=7000):
    sv = 0
    for s in range(ntr):
        rng = np.random.default_rng(seed0 + s)
        cell = random_cell(rng, "random", lo=30, hi=64)
        shot = simulate_shot(rng=rng, cell=cell, n_target=n_target,
                             pos_sigma=ps, frac_spurious=fs)
        res = index_shot(shot.g, shot.meta["qmax"], n=96, peakfinder=pf)
        sv += score(shot, res)["solved"]
    return 100 * sv / ntr


print(f"{'setting':<24}{'classical':>10}{'cnn':>6}")
for label, nt, ps, fs in [("clean n=40", 40, 0.0, 0.0),
                          ("n=30 jit.0015 spur.2", 30, 0.0015, 0.2),
                          ("n=25 spur.1", 25, 0.0, 0.1)]:
    c = solve(find_peaks_classical, nt, ps, fs)
    l = solve(cnn, nt, ps, fs)
    print(f"{label:<24}{c:>9.0f}%{l:>5.0f}%")
