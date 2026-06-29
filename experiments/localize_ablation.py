"""Ablation: how much of the v0 baseline error was just grid (dx) quantization?

Runs the spot-count sweep twice -- with and without the direct-FT |F(x)|^2
localizer -- reporting solve rate and mean cell-length error for each.
"""

import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")   # kill BLAS thread thrash on 3x3 solves
sys.path.insert(0, "..")
from fftindex import index_shot, score, simulate_shot


def sweep(localize, spot_counts, n_trials=20):
    rows = []
    for nt in spot_counts:
        sv, ae = [], []
        for s in range(n_trials):
            rng = np.random.default_rng(100 + s)
            shot = simulate_shot(rng=rng, n_target=nt, pos_sigma=0.002,
                                 frac_spurious=0.1)
            res = index_shot(shot.g, shot.meta["qmax"], n=128, localize=localize)
            sc = score(shot, res)
            sv.append(sc["solved"])              # honest: lattice match required
            if sc["lattice_match"]:
                ae.append(sc["axis_err"])        # grid-free per-axis error
        rows.append((nt, 100 * np.mean(sv), np.median(ae) if ae else float("nan")))
    return rows


if __name__ == "__main__":
    counts = [15, 20, 25, 30, 40, 60, 90]
    print("solve% = correct lattice (integer unimodular T); axisErr median on matches")
    print(f"{'spots':>6} | {'OFF solve%':>10} {'axisErr':>8} "
          f"| {'ON solve%':>10} {'axisErr':>8}")
    print("-" * 52)
    off = {r[0]: r for r in sweep(False, counts)}
    on = {r[0]: r for r in sweep(True, counts)}
    for nt in counts:
        o, n = off[nt], on[nt]
        print(f"{nt:>6} | {o[1]:>9.0f}% {o[2]:>8.3f} "
              f"| {n[1]:>9.0f}% {n[2]:>8.3f}")
