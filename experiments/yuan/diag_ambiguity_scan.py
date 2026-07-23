"""Diagnostic: does each pilot crystal have a SYSTEMATIC near-degenerate rival orientation --
one that the coarse vote-count search converges to reliably across INDEPENDENT search seeds
(not random per-draw noise)? Found by diag_multishot_coarse.py for crystal 2 (noise=2e-4):
top-1 landed ~92-96 or ~155 deg from truth across 6 independent seeds, never near truth.

This matters for issue #12: cross-frame consensus (pooling shots, or independent per-shot
result-consensus) can only avearge out RANDOM per-shot noise. A systematic rival -- one that
wins the coarse vote reproducibly regardless of which random candidates get drawn -- is not
noise, and pooling more shots of the SAME crystal won't break it (the rival is exactly as
consistent with every repeat shot as the truth is).

Single-shot (M=1), noise=2e-4 (clean geometry, degeneracy is a property of the crystal's
streak set relative to the lattice, not of the per-point noise), N_SEEDS independent coarse
searches per crystal.

  python diag_ambiguity_scan.py
"""
import sys

import numpy as np

sys.path.insert(0, "..")

from generate_dataset_multishot import load_dataset, pooled
from cbxd_hough_gpu import rand_rot_batch, score_batch_gpu

N_COARSE = 1_000_000
N_SEEDS = 4
TOL_C = 0.03


def rot_angle_deg(Ra, Rb):
    Rrel = Ra.T @ Rb
    c = np.clip((np.trace(Rrel) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def scan(noise=2e-4):
    ds = load_dataset(noise)
    print(f"noise={noise:.0e}  N_SEEDS={N_SEEDS}  n_coarse={N_COARSE}")
    print(f"{'crystal':>7} {'n_streaks':>9}  top1_angle_to_truth_per_seed")
    for i, c in enumerate(ds):
        kobs, lab, cents = pooled(c, 1)
        Rt = c["Rt"]
        angs = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            Rs = rand_rot_batch(rng, N_COARSE)
            counts = score_batch_gpu(Rs, kobs, TOL_C)
            top1 = np.argsort(counts)[::-1][0]
            angs.append(rot_angle_deg(Rt, Rs[top1]))
        angs = np.array(angs)
        near_truth = angs < 10
        # "systematic" if it NEVER lands near truth AND repeatedly lands in a tight cluster
        # (std small relative to the mean) rather than scattering
        tag = "ALWAYS_NEAR_TRUTH" if near_truth.all() else (
            "SYSTEMATIC_RIVAL" if (not near_truth.any() and angs.std() < 15) else "MIXED/RANDOM")
        print(f"{i:7d} {len(cents):9d}  {np.round(angs, 1)}  -> {tag}")


if __name__ == "__main__":
    scan()
