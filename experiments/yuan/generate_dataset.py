"""Generate a fixed, ON-DISK CBXD crystal dataset for arm comparisons (COM/PTS/TAN/hybrid).

Why on-disk: earlier in this session, per-crystal comparisons across arms were silently invalid.
hough_seed_index(kobs, rng, n_coarse=...) draws millions of random candidates from the SAME rng
object used to generate the next crystal's orientation -- so crystal N's orientation depended on
how much random state crystal N-1's SEARCH consumed, not just on N. Two runs with the "same seed"
but different searches (or different n_coarse) silently produced DIFFERENT physical crystals under
the same index label. Persisting the crystals once removes the whole bug class: every arm reads
the exact same orientations/streak-clouds from disk, independent of whatever search runs on them.

Selection: draw a large POOL of candidate crystals at the standard test cell / NA=0.028 /
noise=2e-4 (cbxd_joint's defaults), then keep N_KEEP stratified evenly across the OBSERVED
#streaks range -- not just the first N_KEEP sequentially -- so the saved set actually covers the
sparse ("reflection-starved") to dense range needed to find where the tangent vote helps, rather
than whatever mix a plain sequential draw happens to produce.

Each crystal is saved as data/simulated_data/crystal_NNN.npz: Rt (3,3), kobs (Np,3), lab (Np,)
bool, cents (Ns,3), noise (scalar). load_dataset() reads them back in index order.

  python generate_dataset.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import rand_rot, simulate

NOISE = 2e-4
SEED = 42
POOL_SIZE = 200
N_KEEP = 20
OUTDIR = os.path.join(os.path.dirname(__file__), "data", "simulated_data")


def generate_pool(pool_size=POOL_SIZE, noise=NOISE, seed=SEED):
    """Plain sequential draw, no search in between -- deterministic and reproducible on its own,
    with nothing else touching the rng."""
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(pool_size):
        Rt = rand_rot(rng)
        kobs, lab, cents = simulate(Rt, rng, noise)
        pool.append(dict(Rt=Rt, kobs=kobs, lab=lab, cents=cents, n_streaks=len(cents)))
    return pool


def select_stratified(pool, n_keep=N_KEEP):
    """Keep n_keep crystals spread evenly across the observed #streaks range (by rank, i.e.
    evenly spaced quantiles of the sorted #streaks distribution) -- covers sparse-to-dense rather
    than whatever a plain sequential run happens to contain."""
    order = np.argsort([c["n_streaks"] for c in pool])
    idx = np.linspace(0, len(order) - 1, n_keep).round().astype(int)
    idx = sorted(set(idx.tolist()))
    # pad if dedup dropped below n_keep (possible with ties at the edges)
    remaining = [i for i in range(len(order)) if order[i] not in [order[j] for j in idx]]
    while len(idx) < n_keep and remaining:
        idx.append(remaining.pop(0))
    idx = sorted(idx[:n_keep])
    return [pool[order[i]] for i in idx]


def save_dataset(crystals, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    for i, c in enumerate(crystals):
        np.savez(os.path.join(outdir, f"crystal_{i:03d}.npz"),
                 Rt=c["Rt"], kobs=c["kobs"], lab=c["lab"], cents=c["cents"],
                 noise=NOISE)


def load_dataset(outdir=OUTDIR, n=N_KEEP):
    crystals = []
    for i in range(n):
        d = np.load(os.path.join(outdir, f"crystal_{i:03d}.npz"))
        crystals.append(dict(Rt=d["Rt"], kobs=d["kobs"], lab=d["lab"], cents=d["cents"],
                             noise=float(d["noise"])))
    return crystals


if __name__ == "__main__":
    pool = generate_pool()
    kept = select_stratified(pool)
    save_dataset(kept)
    ns = sorted(c["n_streaks"] for c in kept)
    print(f"pool={POOL_SIZE}  kept={len(kept)}  noise={NOISE:.0e}  seed={SEED}")
    print(f"#streaks range in saved set: min={ns[0]} p25={np.percentile(ns,25):.0f} "
          f"median={np.median(ns):.0f} p75={np.percentile(ns,75):.0f} max={ns[-1]}")
    print(f"#streaks per crystal: {ns}")
    print(f"saved to {OUTDIR}/")
