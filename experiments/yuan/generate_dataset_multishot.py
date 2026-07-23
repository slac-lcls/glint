"""CBXD deliverable 4 (issue #12): dataset for cross-frame consensus -- does pooling across
SHOTS of the same crystal stack with pooling across STREAKS within one shot?

Physical picture: the same crystal held in the beam for several independent CBXD exposures
(radiation-tolerant dose regime, or several sub-frames of one longer exposure) -- same true
orientation Rt every time, but each shot draws its own independent noise realization on the
real streak points AND its own independent spurious-point draw (cbxd_joint.simulate's SPUR
sphere-points, which model detector/background clutter, not the crystal). Pooling M such shots'
kobs into ONE accumulator run is mathematically exact for the vote-count objective:
score(R, kobs, tol) just counts matched points, so
    score(R, concat(kobs_1..kobs_M), tol) == sum_m score(R, kobs_m, tol)
(see test_cbxd_hough.py::test_multishot_pooling_exact). So "cross-shot pooling" and "cross-streak
pooling" are literally the same accumulator, run on a bigger point cloud -- the open empirical
question is how much noise-averaging that buys: does the noise=2e-3 WALLED boundary (0/20 success
across every NA in results_grid.jsonl) retreat as M grows?

Same frozen-dataset discipline as generate_dataset.py: crystal orientations (Rt) are fixed ONCE
per (seed, i), independent of which M or noise is later requested from the saved shots, so
different M's are exactly nested subsets of the same underlying shot sequence -- M=2 is shots
[0,1] of the SAME crystal used at M=1 (shot [0]), not a fresh draw.

SCALED UP to match deliverable 3's own rigor -- but rather than drawing a NEW independent
20-crystal pool, this REUSES the exact orientations already on disk from deliverable 3
(data/grid/orientations_na0.028.npz, generate_dataset_grid.py::get_orientation_pool), per the
lesson this repo already learned the hard way in generate_dataset.py's docstring: don't silently
re-derive "the same" crystals from a different seed and risk comparing physical crystals that
aren't actually the same. For issue #12, orientation is a FIXED input we already have on disk --
the only new axis is (noise level) x (shot count M), so nothing about crystal generation needs
touching. Shot 0 of every crystal also uses generate_dataset_grid.py::generate_cell's EXACT rng
seed (200_000 + na*1000 + i), so shot 0's kobs is BYTE-IDENTICAL to the cached grid cell at that
(na=0.028, noise, i) -- M=1 here reproduces results_grid.jsonl's PTS numbers exactly, not just
"comparably" (see test_cbxd_multishot.py::test_shot_zero_matches_grid_cell). Shots 1..M_MAX-1
continue that same rng stream (fresh, independent draws).

Noise ladder trimmed to the informative subset of results_grid.jsonl's 7-level ladder
({2e-4,5e-4,1e-3,2e-3}) -- 5e-3/1e-2/2e-2 were already 0/20 at EVERY NA in the deliverable-3 grid,
so there's no dynamic range left for an M-axis to move there.

Each crystal saved as data/simulated_data_multishot/noise_{tag}/crystal_{NNN}.npz:
  Rt (3,3), noise (scalar), and kobs_{m}/lab_{m}/cents_{m} for m in range(M_MAX) -- one triplet
  per shot, ragged lengths (fine in npz, each key is its own array).

  python generate_dataset_multishot.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "..")

from cbxd_joint import simulate

from cbxd_sweep import set_NA
from generate_dataset_grid import get_orientation_pool

NA = 0.028                                  # the standard cell used throughout deliverables 1-3
N_CRY = 20
M_MAX = 8
NOISE_LEVELS = (2e-4, 5e-4, 1e-3, 2e-3)     # informative subset of results_grid.jsonl's ladder
OUTDIR = os.path.join(os.path.dirname(__file__), "data", "simulated_data_multishot")


def generate_crystals(n_cry=N_CRY, na=NA):
    """The SAME 20 orientations deliverable 3's phase diagram used at this NA -- cached on disk,
    not re-derived, so there's no risk of silently comparing different physical crystals."""
    set_NA(na)
    return get_orientation_pool(na, n_keep=n_cry)


def generate_shots(Rt, noise, m_max, rng):
    """m_max independent simulate() calls at the SAME Rt -- each consumes rng sequentially, so
    the shot sequence is deterministic given (Rt, noise, rng-seed) and shot m doesn't depend on
    how many shots were drawn after it. Shot 0 uses generate_dataset_grid.py's own per-crystal
    rng seed (passed in by save_dataset), so it's byte-identical to the cached grid cell."""
    shots = []
    for _ in range(m_max):
        kobs, lab, cents = simulate(Rt, rng, noise)
        shots.append((kobs, lab, cents))
    return shots


def save_dataset(crystals, outdir=OUTDIR, na=NA):
    for noise in NOISE_LEVELS:
        tag = f"{noise:.0e}"
        d = os.path.join(outdir, f"noise_{tag}")
        os.makedirs(d, exist_ok=True)
        for i, Rt in enumerate(crystals):
            # matches generate_dataset_grid.py::generate_cell's seed exactly, so shot 0 here
            # reproduces that cell's kobs bit-for-bit
            rng = np.random.default_rng(200_000 + int(round(na * 1000)) + i)
            shots = generate_shots(Rt, noise, M_MAX, rng)
            kw = dict(Rt=Rt, noise=noise)
            for m, (kobs, lab, cents) in enumerate(shots):
                kw[f"kobs_{m}"] = kobs
                kw[f"lab_{m}"] = lab
                kw[f"cents_{m}"] = cents
            np.savez(os.path.join(d, f"crystal_{i:03d}.npz"), **kw)


def load_dataset(noise, outdir=OUTDIR, n=N_CRY, m_max=M_MAX):
    tag = f"{noise:.0e}"
    d = os.path.join(outdir, f"noise_{tag}")
    crystals = []
    for i in range(n):
        z = np.load(os.path.join(d, f"crystal_{i:03d}.npz"))
        shots = [(z[f"kobs_{m}"], z[f"lab_{m}"], z[f"cents_{m}"]) for m in range(m_max)]
        crystals.append(dict(Rt=z["Rt"], noise=float(z["noise"]), shots=shots))
    return crystals


def pooled(crystal, m):
    """Concatenate the first m shots' kobs/lab/cents -- shots are nested, so pooled(c, m) is a
    strict superset of pooled(c, m-1)."""
    shots = crystal["shots"][:m]
    kobs = np.vstack([s[0] for s in shots])
    lab = np.concatenate([s[1] for s in shots])
    cents = np.vstack([s[2] for s in shots])
    return kobs, lab, cents


if __name__ == "__main__":
    crystals = generate_crystals()
    save_dataset(crystals)
    for noise in NOISE_LEVELS:
        ds = load_dataset(noise)
        ns1 = [len(c["shots"][0][2]) for c in ds]
        nsM = [sum(len(s[2]) for s in c["shots"]) for c in ds]
        print(f"noise={noise:.0e}: saved {len(ds)} crystals x {M_MAX} shots -- "
              f"#streaks/shot(M=1) range {min(ns1)}-{max(ns1)}, "
              f"total #streaks(M={M_MAX}) range {min(nsM)}-{max(nsM)}")
