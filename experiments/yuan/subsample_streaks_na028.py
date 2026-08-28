"""Deconfound NA from #streaks (Stefano's review of the phase diagram): in the grid, NA and
#streaks are perfectly confounded (median #streaks 4->49 as NA goes 0.010->0.060, since streak
count is an emergent property of NA in the current simulate() model). The #9 finding says
#streaks doesn't predict which arm wins -- but that was tested by comparing DIFFERENT NA values,
which also changes the cone geometry. This settles it properly: hold NA FIXED at 0.028 (same
cone/geometry) and subsample down to the SAME streak counts NA=0.010 (~4) and NA=0.016 (~11)
naturally produce, then compare against the ACTUAL NA=0.010/0.016 results already in
results_grid.jsonl. If subsampled-NA=0.028 behaves like the natural low-NA cells, streak count is
what matters. If it stays close to NA=0.028's own dense behavior, geometry/cone-width is what
matters, not raw count.

Subsampling: pick a random subset of N_TARGET streaks (by identity, same subset used for both the
thin=6 kobs/cents construction and the thin=1 tangent extraction, since both come from the same
underlying admitted-reflection list for a given (Rt, NA) -- the admission mask is deterministic,
independent of rng, so two independent _simulate_grouped calls at different thinning produce the
same streak list in the same order). Spurious point count is rescaled to
cbxd_joint.SPUR * (subsampled real point count), matching simulate()'s own convention, so the
subsampled dataset has the same spurious-to-real ratio a natively-sparse crystal would.

Fixed at noise=2e-4 (the value most of today's findings are anchored to). Checkpointed
(results_subsample.jsonl), safe to resume.

  python subsample_streaks_na028.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import SPUR, K, score

from cbxd_sweep import set_NA
from generate_dataset_grid import get_orientation_pool
from tangent_validate import _simulate_grouped, local_tangent_pca
from cbxd_arm1_com import com_seed_index
from cbxd_hough_gpu import hough_seed_index
from cbxd_hough_tangent import hough_seed_index_tangent

NA = 0.028
NOISE = 2e-4
N_COARSE = 5_000_000
N_TARGETS = [4, 11]                          # match NA=0.010's and NA=0.016's median #streaks
LOG = os.path.join(os.path.dirname(__file__), "results_subsample.jsonl")


def build_subsampled(Rt, n_target, seed):
    """One subsampled dataset: n_target streaks kept (by identity) out of NA=0.028's full set,
    same subset used for both the kobs/cents construction and the tangent reprs."""
    rng_pick = np.random.default_rng(seed)
    streaks6 = _simulate_grouped(Rt, np.random.default_rng(seed + 1), NOISE, thin=6)
    streaks1 = _simulate_grouped(Rt, np.random.default_rng(seed + 2), NOISE, thin=1)
    n_total = len(streaks6)
    if n_total <= n_target:
        idx = np.arange(n_total)
    else:
        idx = rng_pick.choice(n_total, size=n_target, replace=False)

    real = np.vstack([streaks6[i]["noisy"] for i in idx])
    cents = np.array([streaks6[i]["noisy"].mean(0) for i in idx])

    rng_spur = np.random.default_rng(seed + 3)
    n_spur = int(round(SPUR * len(real)))
    sp = rng_spur.normal(size=(n_spur, 3))
    sp /= np.linalg.norm(sp, axis=1, keepdims=True)
    sp *= K
    sp = sp[sp[:, 2] > 0]
    kobs = np.vstack([real, sp])
    lab = np.concatenate([np.ones(len(real), bool), np.zeros(len(sp), bool)])
    perm = rng_spur.permutation(len(kobs))
    kobs, lab = kobs[perm], lab[perm]

    reprs, tangents = [], []
    for i in idx:
        clean, noisy = streaks1[i]["clean"], streaks1[i]["noisy"]
        if len(clean) < 3:
            continue
        j = len(clean) // 2
        t = local_tangent_pca(noisy, j, halfwin=1)
        if t is not None:
            reprs.append(noisy[j])
            tangents.append(t)
    reprs = np.array(reprs) if reprs else np.zeros((0, 3))
    tangents = np.array(tangents) if tangents else np.zeros((0, 3))

    return kobs, lab, cents, reprs, tangents, len(idx)


def frac_indexed(R, kobs, lab):
    if R is None:
        return 0.0
    return float(score(R, kobs, 0.0025, ret_mask=True)[lab].mean())


def already_done():
    done = set()
    if os.path.exists(LOG):
        with open(LOG) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["n_target"], r["i"], r["arm"]))
    return done


def append_result(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run():
    set_NA(NA)
    Rts = get_orientation_pool(NA)
    done = already_done()
    print(f"resuming: {len(done)} (n_target,i,arm) trials already logged", flush=True)

    for n_target in N_TARGETS:
        for i, Rt in enumerate(Rts):
            kobs, lab, cents, reprs, tangents, n_kept = build_subsampled(Rt, n_target, seed=10_000 * n_target + i)

            if (n_target, i, "COM") not in done:
                t0 = time.perf_counter()
                R = com_seed_index(cents, np.random.default_rng(1000 + i), n_coarse=N_COARSE)
                frac = frac_indexed(R, kobs, lab)
                append_result(dict(n_target=n_target, i=i, arm="COM", frac=frac, n_kept=n_kept,
                                   dt=time.perf_counter() - t0))
                print(f"n_target={n_target} i={i:2d} COM  frac={frac:.2f}", flush=True)

            if (n_target, i, "PTS") not in done:
                t0 = time.perf_counter()
                R = hough_seed_index(kobs, np.random.default_rng(2000 + i), n_coarse=N_COARSE)
                frac = frac_indexed(R, kobs, lab)
                append_result(dict(n_target=n_target, i=i, arm="PTS", frac=frac, n_kept=n_kept,
                                   dt=time.perf_counter() - t0))
                print(f"n_target={n_target} i={i:2d} PTS  frac={frac:.2f}", flush=True)

            if (n_target, i, "TAN") not in done:
                t0 = time.perf_counter()
                R = hough_seed_index_tangent(kobs, reprs, tangents, np.random.default_rng(3000 + i),
                                             n_coarse=N_COARSE)
                frac = frac_indexed(R, kobs, lab)
                append_result(dict(n_target=n_target, i=i, arm="TAN", frac=frac, n_kept=n_kept,
                                   dt=time.perf_counter() - t0))
                print(f"n_target={n_target} i={i:2d} TAN  frac={frac:.2f}", flush=True)

        fracs = {arm: [] for arm in ("COM", "PTS", "TAN")}
        for line in open(LOG):
            r = json.loads(line)
            if r["n_target"] == n_target:
                fracs[r["arm"]].append(r["frac"])
        summary = "  ".join(f"{arm} {sum(f>0.7 for f in v)}/{len(v)}" for arm, v in fracs.items())
        print(f"n_target={n_target}  {summary}", flush=True)

    print("\nALL DONE.", flush=True)


if __name__ == "__main__":
    run()
