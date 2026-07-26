"""Frozen (NA, noise) grid datasets for deliverable 3 (issue #11) -- Stefano's scoped plan:
grid = (NA x noise), #streaks/length reported as an observed statistic (not swept, since we
showed it isn't predictive), noise laddered up until PTS/TAN actually wall.

For a given NA, ONE stratified pool of 20 orientations is drawn (same method as
generate_dataset_lowna.py: draw a large pool, keep 20 evenly spread across the observed #streaks
range) and REUSED across every noise level in that NA's ladder -- noise doesn't affect which
reflections are admitted (the cone-gate mask is noise-independent, only position jitter/spurious
counts depend on it), so this gives a clean "same 20 crystals, more noise" comparison as the
ladder climbs, the same pattern generate_dataset_noise1e4.py used for the single NA=0.028 case.

Directory layout: data/grid/orientations_na{NA}.npz (the 20 Rt's, cached, generated once per NA)
and data/grid/na{NA}_noise{NOISE}/crystal_NNN.npz (kobs/lab/cents at that cell).

  python generate_dataset_grid.py <na> <noise>          # generate one cell
  python generate_dataset_grid.py --all                 # generate the whole suggested grid
"""
import os
import sys

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import rand_rot, simulate

from cbxd_sweep import set_NA
from generate_dataset import select_stratified

N_KEEP = 20
MIN_STREAKS = 2
POOL_SIZE = 400
ORIENTATION_SEED_BASE = 100_000          # + a per-NA offset below, so each NA's pool is distinct
GRID_DIR = os.path.join(os.path.dirname(__file__), "data", "grid")

# Stefano's suggested grid (issue #11): NA brackets floor (0.010) to PTS-ceiling (0.022) and
# beyond (0.028); noise ladders geometrically from where we've already validated (2e-4) up past
# where anything should still solve.
#
# Extended 2026-07-20 per Stefano's review of the first pass: the grid stopped at NA=0.028, just
# below where COM turns over (his sandbox note: COM ~94% at 25 mrad but ~22% at 50 mrad) -- without
# the wide-NA region a reader would extrapolate "wider cone is always better", which isn't the
# actual shape. Added 0.040/0.050/0.060 (his suggested values, in our units: NA is already in
# radians, so his "40/50/60 mrad" = 0.040/0.050/0.060 here) to actually see the turnover, and
# specifically to test whether TAN (which doesn't use the centroid approximation) survives past
# where COM breaks -- a specific mechanistic prediction, not just "TAN is better on average".
NA_GRID = (0.010, 0.016, 0.022, 0.028, 0.040, 0.050, 0.060)
NOISE_LADDER = (2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2)


def _na_key(na):
    return f"{na:.4f}".rstrip("0").rstrip(".")


def _noise_key(noise):
    return f"{noise:.1e}"


def orientation_pool_path(na):
    return os.path.join(GRID_DIR, f"orientations_na{_na_key(na)}.npz")


def get_orientation_pool(na, pool_size=POOL_SIZE, n_keep=N_KEEP):
    """20 orientations stratified by #streaks at this NA, cached to disk so every noise level
    in this NA's ladder reuses the exact same 20 crystals."""
    path = orientation_pool_path(na)
    if os.path.exists(path):
        d = np.load(path)
        return [d[f"Rt_{i:03d}"] for i in range(n_keep)]

    set_NA(na)
    seed = ORIENTATION_SEED_BASE + int(round(na * 1000))
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(pool_size):
        Rt = rand_rot(rng)
        _, _, cents = simulate(Rt, rng, 2e-4)          # noise value here is irrelevant to #streaks
        if len(cents) >= MIN_STREAKS:
            pool.append(dict(Rt=Rt, n_streaks=len(cents)))
    kept = select_stratified(pool, n_keep=n_keep)
    Rts = [c["Rt"] for c in kept]

    os.makedirs(GRID_DIR, exist_ok=True)
    np.savez(path, **{f"Rt_{i:03d}": Rt for i, Rt in enumerate(Rts)},
             na=na, n_streaks=np.array([c["n_streaks"] for c in kept]))
    return Rts


def cell_dir(na, noise):
    return os.path.join(GRID_DIR, f"na{_na_key(na)}_noise{_noise_key(noise)}")


def generate_cell(na, noise, n_keep=N_KEEP):
    outdir = cell_dir(na, noise)
    if os.path.exists(os.path.join(outdir, f"crystal_{n_keep-1:03d}.npz")):
        return outdir                                    # already generated

    set_NA(na)
    Rts = get_orientation_pool(na, n_keep=n_keep)
    os.makedirs(outdir, exist_ok=True)
    for i, Rt in enumerate(Rts):
        rng = np.random.default_rng(200_000 + int(round(na * 1000)) + i)
        kobs, lab, cents = simulate(Rt, rng, noise)
        np.savez(os.path.join(outdir, f"crystal_{i:03d}.npz"),
                 Rt=Rt, kobs=kobs, lab=lab, cents=cents, noise=noise, na=na)
    return outdir


def load_cell(na, noise, n=N_KEEP):
    outdir = cell_dir(na, noise)
    # The orientations (orientations_na*.npz) are the committed frozen selection; the per-cell npz are
    # regenerable from them (generate_cell's per-crystal seed is deterministic), so materialize on demand.
    if not os.path.exists(os.path.join(outdir, f"crystal_{0:03d}.npz")):
        generate_cell(na, noise, n_keep=n)
    crystals = []
    for i in range(n):
        d = np.load(os.path.join(outdir, f"crystal_{i:03d}.npz"))
        crystals.append(dict(Rt=d["Rt"], kobs=d["kobs"], lab=d["lab"], cents=d["cents"],
                             noise=float(d["noise"]), na=float(d["na"])))
    return crystals


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        for na in NA_GRID:
            for noise in NOISE_LADDER:
                outdir = generate_cell(na, noise)
                ns = np.load(orientation_pool_path(na))["n_streaks"]
                print(f"NA={na}  noise={noise:.1e}  #streaks median={int(np.median(ns))}  "
                      f"-> {outdir}", flush=True)
    else:
        na = float(sys.argv[1])
        noise = float(sys.argv[2])
        outdir = generate_cell(na, noise)
        print(f"NA={na}  noise={noise:.1e}  -> {outdir}")
