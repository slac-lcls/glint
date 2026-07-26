"""Low-NA variant of generate_dataset.py: a fixed, on-disk dataset genuinely in the
"reflection-starved" regime (few streaks), to test the plan's literal criterion --
"In the reflection-starved corner (few streaks / small cells), add each streak's tangent..."
-- rather than the NA=0.028 baseline dataset, where #streaks turned out NOT to predict
PTS-vs-TAN at all (Spearman -0.33, not significant; PTS's sparsest crystals solved perfectly,
its worst failure had the MOST streaks in the set).

NA=0.016 chosen from the earlier cbxd_sweep.py run: median #streaks=11.5 there (vs. 22.5 at the
baseline NA=0.028), and it was the one setting where PTS/TAN clearly separated from COM and from
each other (COM 11%, PTS 84%, TAN 87% median -- small sample, ncry=6). This dataset gives a
proper N=20 stratified sample in that same regime to test whether #streaks predicts anything here.

NA is a module-level constant patched at runtime (cbxd_sweep.set_NA) -- must be applied before
BOTH generation (this script) AND before running any arm's search on the resulting dataset
(score_batch_gpu / _simulate_grouped also read COSA/SINA and must see the same NA).

  python generate_dataset_lowna.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import rand_rot, simulate

from cbxd_sweep import set_NA
from generate_dataset import select_stratified, select_frozen, freeze_selection, load_frozen

NA = 0.016
NOISE = 2e-4
SEED = 43                                   # different from generate_dataset.py's 42 -- new pool
POOL_SIZE = 300                             # bigger pool: low NA means many crystals have too few
                                             # streaks to be usable at all (filtered below)
N_KEEP = 20
MIN_STREAKS = 3                             # need at least a few streaks for a search to mean anything
OUTDIR = os.path.join(os.path.dirname(__file__), "data", "simulated_data_lowna")
# Frozen selection (see generate_dataset.py): commit only these 20 orientations; regenerate the npz
# on demand. RESULT-identical, not byte-identical across platforms.
SELECTION = os.path.join(os.path.dirname(__file__), "data", "selection_simulated_data_lowna.npz")


def generate_pool(pool_size=POOL_SIZE, noise=NOISE, seed=SEED):
    set_NA(NA)
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(pool_size):
        Rt = rand_rot(rng)
        kobs, lab, cents = simulate(Rt, rng, noise)
        if len(cents) >= MIN_STREAKS:
            pool.append(dict(Rt=Rt, kobs=kobs, lab=lab, cents=cents, n_streaks=len(cents)))
    return pool


def save_dataset(crystals, outdir=OUTDIR):
    """Like generate_dataset.save_dataset, but also stores NA -- a downstream reader must
    set_NA() to this value before running any search on the dataset (the cone-gate constants
    COSA/SINA are NA-dependent, and score()/simulate() consulted them when this was made)."""
    os.makedirs(outdir, exist_ok=True)
    for i, c in enumerate(crystals):
        np.savez(os.path.join(outdir, f"crystal_{i:03d}.npz"),
                 Rt=c["Rt"], kobs=c["kobs"], lab=c["lab"], cents=c["cents"],
                 noise=NOISE, na=NA)


def build_dataset():
    """Reproduce the pinned selection if frozen (default), else mint+freeze a fresh deterministic one."""
    pool = generate_pool()
    frozen = load_frozen(SELECTION)
    if frozen is not None:
        return select_frozen(pool, frozen)
    kept = select_stratified(pool, n_keep=N_KEEP)
    freeze_selection(kept, SELECTION)
    return kept


def load_dataset(outdir=OUTDIR, n=N_KEEP):
    if not os.path.exists(os.path.join(outdir, f"crystal_{0:03d}.npz")):
        save_dataset(build_dataset(), outdir)            # regenerate from the frozen selection on demand
    crystals = []
    for i in range(n):
        d = np.load(os.path.join(outdir, f"crystal_{i:03d}.npz"))
        crystals.append(dict(Rt=d["Rt"], kobs=d["kobs"], lab=d["lab"], cents=d["cents"],
                             noise=float(d["noise"]), na=float(d["na"])))
    return crystals


if __name__ == "__main__":
    kept = build_dataset()
    print(f"kept {len(kept)} crystals (frozen selection reproduced or minted)")
    save_dataset(kept, outdir=OUTDIR)
    ns = sorted(c["n_streaks"] for c in kept)
    print(f"NA={NA}  kept={len(kept)}  noise={NOISE:.0e}  seed={SEED}")
    print(f"#streaks range in saved set: min={ns[0]} p25={np.percentile(ns,25):.0f} "
          f"median={np.median(ns):.0f} p75={np.percentile(ns,75):.0f} max={ns[-1]}")
    print(f"#streaks per crystal: {ns}")
    print(f"saved to {OUTDIR}/")
