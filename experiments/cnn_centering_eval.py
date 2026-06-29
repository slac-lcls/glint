"""Does the (primitive-trained) 3D-CNN transfer to centered lattices? FCC/BCC test.

The CNN trained on primitive cells with rhombohedral angle in [70,110]. BCC's
primitive cell is alpha~=109.5 (in-distribution); FCC's is alpha=60 (out). So FCC may
expose an out-of-distribution gap. Hard setting (n=30, sigma=0.0015, 20% spurious).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys

import numpy as np
import torch

sys.path.insert(0, "..")
from fftindex import index_shot, score, simulate_shot
from fftindex.cnn import CNNPeakFinder, UNet3D
from fftindex.peakfind import find_peaks_classical

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("cnn_peakfinder.pt", map_location=dev)
model = UNet3D(); model.load_state_dict(ck["model"])
cnn = CNNPeakFinder(model, n_model=ck["n"], device=dev)
print(f"loaded cnn (n={ck['n']}) on {dev}")

SETTING = dict(n_target=30, pos_sigma=0.0015, frac_spurious=0.2)


def solve(pf, cen, alo, ahi, ntr=20, seed0=4000):
    sv = 0
    for s in range(ntr):
        rng = np.random.default_rng(seed0 + s)
        a = rng.uniform(alo, ahi)
        shot = simulate_shot(rng=rng, cell=(a, a, a, 90, 90, 90),
                             centering=cen, **SETTING)
        sv += score(shot, index_shot(shot.g, shot.meta["qmax"], n=96, peakfinder=pf))["solved"]
    return 100 * sv / ntr


# conventional a chosen so the primitive edge (a/sqrt2 for F, 0.866a for I) fits n=96
print(f"hard setting {SETTING}, n=96")
print(f"{'centering':<16}{'classical':>10}{'cnn':>6}")
for label, cen, alo, ahi in [("P (cubic)", "P", 45, 68),
                             ("I (BCC)", "I", 50, 73),
                             ("F (FCC)", "F", 45, 90)]:
    c = solve(find_peaks_classical, cen, alo, ahi)
    l = solve(cnn, cen, alo, ahi)
    print(f"{label:<16}{c:>9.0f}%{l:>5.0f}%")
