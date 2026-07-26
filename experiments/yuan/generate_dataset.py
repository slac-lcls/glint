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
# Frozen selection: the ONLY committed part of the dataset (20 x 3x3 orientations, ~1.5 KB). The bulky
# per-crystal npz (kobs/lab/cents) are regenerated on demand from these + the seeded pool, so data/ can be
# gitignored. Reproduction is deterministic to float tolerance (~1e-10 << every point-match tol), i.e.
# RESULT-identical, not byte-identical -- the rng draws are bit-stable, only cross-platform matrix
# arithmetic differs sub-tolerance. This replaces the non-reproducible np.argsort tie-order selection.
SELECTION = os.path.join(os.path.dirname(__file__), "data", "selection_simulated_data.npz")


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
    than whatever a plain sequential run happens to contain.

    Deterministic: #streaks is a small integer with heavy ties, so the sort MUST be stable
    (kind='stable') and break ties by pool index -- a plain np.argsort defaults to quicksort, whose
    tie order is numpy-version/platform dependent, which is what made the frozen datasets
    non-regenerable (a fresh run overlapped the committed set at only ~4/20). Used only to MINT a new
    frozen selection; existing datasets are reproduced via select_frozen()."""
    counts = np.asarray([c["n_streaks"] for c in pool])
    order = np.lexsort((np.arange(len(pool)), counts))       # stable: primary=counts, secondary=index
    idx = np.linspace(0, len(order) - 1, n_keep).round().astype(int)
    idx = sorted(set(idx.tolist()))
    # pad if dedup dropped below n_keep (possible with ties at the edges)
    chosen = {order[j] for j in idx}
    remaining = [i for i in range(len(order)) if order[i] not in chosen]
    while len(idx) < n_keep and remaining:
        idx.append(remaining.pop(0))
    idx = sorted(idx[:n_keep])
    return [pool[order[i]] for i in idx]


def select_frozen(pool, frozen_Rt, tol=1e-6):
    """Reproduce a pinned selection by matching each frozen orientation to its pool member.
    The pool is deterministic (seed=SEED), so this recovers the exact crystals the frozen set named --
    RESULT-identical (matched to ~1e-10, far below every point-match tol), not necessarily byte-identical
    across platforms. Order follows frozen_Rt (== the saved crystal_NNN order)."""
    pool_Rt = np.stack([c["Rt"] for c in pool])
    out = []
    for Rt in frozen_Rt:
        j = int(np.abs(pool_Rt - Rt).reshape(len(pool), -1).max(1).argmin())
        assert np.abs(pool_Rt[j] - Rt).max() < tol, "frozen Rt not in pool -- SEED/pool_size/cbxd_joint changed"
        out.append(pool[j])
    return out


def freeze_selection(crystals, path=SELECTION):
    """Persist ONLY the chosen orientations (tiny) so the bulky per-crystal npz can be gitignored."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, Rt=np.stack([c["Rt"] for c in crystals]))


def load_frozen(path=SELECTION):
    return np.load(path)["Rt"] if os.path.exists(path) else None


def build_dataset():
    """The canonical crystals. Reproduce the pinned selection if frozen (default), else mint a fresh
    deterministic one and freeze it."""
    pool = generate_pool()
    frozen = load_frozen()
    if frozen is not None:
        return select_frozen(pool, frozen)
    kept = select_stratified(pool)
    freeze_selection(kept)
    return kept


def save_dataset(crystals, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    for i, c in enumerate(crystals):
        np.savez(os.path.join(outdir, f"crystal_{i:03d}.npz"),
                 Rt=c["Rt"], kobs=c["kobs"], lab=c["lab"], cents=c["cents"],
                 noise=NOISE)


def load_dataset(outdir=OUTDIR, n=N_KEEP):
    # regenerate transparently from the frozen selection if the (gitignored) npz aren't materialized yet
    if not os.path.exists(os.path.join(outdir, f"crystal_{0:03d}.npz")):
        save_dataset(build_dataset(), outdir)
    crystals = []
    for i in range(n):
        d = np.load(os.path.join(outdir, f"crystal_{i:03d}.npz"))
        crystals.append(dict(Rt=d["Rt"], kobs=d["kobs"], lab=d["lab"], cents=d["cents"],
                             noise=float(d["noise"])))
    return crystals


if __name__ == "__main__":
    kept = build_dataset()          # reproduces the pinned selection (or mints+freezes one the first time)
    save_dataset(kept)
    ns = sorted(c["n_streaks"] for c in kept)
    print(f"pool={POOL_SIZE}  kept={len(kept)}  noise={NOISE:.0e}  seed={SEED}")
    print(f"#streaks range in saved set: min={ns[0]} p25={np.percentile(ns,25):.0f} "
          f"median={np.median(ns):.0f} p75={np.percentile(ns,75):.0f} max={ns[-1]}")
    print(f"#streaks per crystal: {ns}")
    print(f"saved to {OUTDIR}/")
