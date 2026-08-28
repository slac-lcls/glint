"""CBXD deliverable 4 (issue #12): does cross-frame consensus (pooling M independent shots of
the SAME crystal) stack with cross-streak pooling (the existing single-shot accumulator)?

Mechanically this is trivial -- test_cbxd_multishot.py::test_multishot_pooling_exact proves
pooling M shots into one accumulator run is EXACT for the vote-count objective, so this file
just runs the existing PTS arm (hough_seed_index, unchanged) on pooled(crystal, m) for
m in {1,2,4,8}. The empirical question is how much noise-averaging that buys: does the
noise=2e-3 WALLED point (0/20 success at every NA in results_grid.jsonl) retreat as M grows,
and does noise=1e-3's TRANSITION point (partial success at M=1) climb toward 100%?

REVISION (after v1 in results_multishot_v1_seedbug.jsonl): two things wrong with the first
pass, found by diag_multishot_coarse.py / diag_ambiguity_scan.py:
  1. v1 drew a DIFFERENT coarse-candidate rng seed per (crystal, m) cell -- so the apparent
     "M=4 worse than M=1" swings were partly search-seed lottery, not a clean M effect.
  2. At n_coarse=1M (chosen for pilot speed, 5x below the 5M used for this repo's headline
     numbers), a truth-in-top-10 check across independent seeds showed most crystals hit only
     ~50-75% of the time by DRAW LUCK ALONE (crystal 2 was a genuine outlier at 0/4, converging
     on a specific systematic rival ~93-155deg from truth across every seed -- a real geometric
     near-degeneracy, not noise, which pooling can't fix by construction). That draw-lottery
     variance is bigger than any M-driven signal a single-seed-per-cell pilot could detect.
Fix: REPEAT each (crystal, m) cell over N_REPEATS independent coarse-candidate seeds (fixed
across m within a repeat -- still a paired comparison), and report the hit RATE over
N_CRY x N_REPEATS trials per (noise, m) cell instead of a single pass/fail per crystal.

Checkpointed like run_four_arms_full.py: one JSON line per (noise_tag, i, m, rep) trial, safe
to resume.

  python run_multishot.py [n_coarse] [n_repeats]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from generate_dataset_multishot import NOISE_LEVELS, M_MAX, N_CRY, load_dataset, pooled
from cbxd_hough_gpu import hough_seed_index

LOG = os.path.join(os.path.dirname(__file__), "results_multishot.jsonl")
M_VALUES = (1, 2, 4, 8)
N_REPEATS = 3


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
                done.add((r["noise_tag"], r["i"], r["m"], r["rep"]))
    return done


def append_result(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(n_coarse=1_000_000, n_repeats=N_REPEATS, noise_levels=NOISE_LEVELS):
    done = already_done()
    print(f"resuming: {len(done)} (noise,i,m,rep) trials already logged in {LOG}", flush=True)
    print(f"n_coarse={n_coarse}  N_CRY={N_CRY}  M_MAX={M_MAX}  M_VALUES={M_VALUES}  "
          f"n_repeats={n_repeats}  noise_levels={noise_levels}", flush=True)

    for noise in noise_levels:
        noise_tag = f"{noise:.0e}"
        ds = load_dataset(noise)
        for i, c in enumerate(ds):
            for rep in range(n_repeats):
                for m in M_VALUES:
                    if (noise_tag, i, m, rep) in done:
                        continue
                    kobs, lab, cents = pooled(c, m)
                    t0 = time.perf_counter()
                    # candidate-draw seed is FIXED per (crystal, rep), independent of m: a
                    # paired comparison (same random SO(3) samples, growing point cloud) within
                    # a repeat, so the M-trend within a repeat is attributable to M itself; the
                    # rep index gives independent draws to average out search-seed lottery.
                    R = hough_seed_index(kobs, np.random.default_rng(9000 + 100 * i + 17 * rep),
                                         n_coarse=n_coarse)
                    frac = frac_indexed(R, kobs, lab)
                    dt = time.perf_counter() - t0
                    append_result(dict(noise_tag=noise_tag, i=i, m=m, rep=rep,
                                       n_points=len(kobs), n_streaks=len(cents), frac=frac, dt=dt))
                    print(f"[{noise_tag}] i={i} rep={rep} m={m} n_points={len(kobs):4d} "
                          f"frac={frac:.2f} dt={dt:5.1f}s", flush=True)

    print("\nALL DONE. Summarizing...", flush=True)
    summarize()


def summarize(log=LOG):
    import collections
    rows = [json.loads(l) for l in open(log)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["noise_tag"], r["m"])].append(r["frac"])
    print(f"\n{'noise':>7} {'M':>3} {'n':>4} {'success':>9} {'median%':>9}")
    for (noise_tag, m), fracs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        fracs = np.array(fracs)
        succ = int((fracs > 0.7).sum())
        print(f"{noise_tag:>7} {m:3d} {len(fracs):4d} {succ:>6d}/{len(fracs)} "
              f"{100*np.median(fracs):8.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
        reps = int(sys.argv[2]) if len(sys.argv) > 2 else N_REPEATS
        run(n, reps)
