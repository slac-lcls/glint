"""Issue #59 matched control: is "TAN 37/40 vs PTS 30/40" (run_four_arms_full.py) really
curvature, or partly just TAN's tangent evidence coming from a fresh, denser (thin=1 vs
simulate()'s thin=6), spurious-free, independently-noised re-simulation of the crystal that PTS
never sees?

Adds a third arm, PTS_MATCHED, using hough_seed_index_points_matched: identical to TAN except
the bonus from `reprs` drops the tangent-alignment gate (require_tangent=False), so it counts
extra points-only matches from the SAME reprs TAN's tangent bonus is built from. PTS_MATCHED
uses the exact same reprs seed (3000+i) as TAN below, so both arms see byte-identical fresh
data -- only the tangent gate differs between them.

Reading the three-way comparison:
  PTS vs PTS_MATCHED  -- isolates the "extra/cleaner/independent data" effect alone
  PTS_MATCHED vs TAN  -- isolates curvature specifically, holding the extra data constant
  PTS vs TAN          -- the original (confounded) headline comparison

Same frozen datasets/seeds as run_four_arms_full.py (20 crystals x {1e-4, 2e-4}, 5M candidates)
so results are directly comparable to the existing four-arm numbers. Checkpointed JSONL, safe to
resume.

  python run_tan_matched_control.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from generate_dataset import load_dataset as load_2e4
from generate_dataset_noise1e4 import load_dataset as load_1e4
from cbxd_hough_gpu import hough_seed_index
from cbxd_hough_tangent import (hough_seed_index_points_matched, hough_seed_index_tangent,
                                streak_reprs_and_tangents)

N_COARSE = 5_000_000
LOG = os.path.join(os.path.dirname(__file__), "results_tan_matched_control.jsonl")


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
                done.add((r["noise_tag"], r["i"], r["arm"]))
    return done


def append_result(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run():
    datasets = {"1e-4": load_1e4(), "2e-4": load_2e4()}
    done = already_done()
    print(f"resuming: {len(done)} (noise,i,arm) trials already logged in {LOG}", flush=True)

    for noise_tag, ds in datasets.items():
        for i, c in enumerate(ds):
            Rt, kobs, lab, noise = c["Rt"], c["kobs"], c["lab"], c["noise"]

            if (noise_tag, i, "PTS") not in done:
                t0 = time.perf_counter()
                R = hough_seed_index(kobs, np.random.default_rng(2000 + i), n_coarse=N_COARSE)
                frac = frac_indexed(R, kobs, lab)
                dt = time.perf_counter() - t0
                append_result(dict(noise_tag=noise_tag, i=i, arm="PTS", frac=frac, dt=dt))
                print(f"[{noise_tag}] i={i:2d} PTS          frac={frac:.2f} dt={dt:5.1f}s", flush=True)

            if (noise_tag, i, "TAN") not in done:
                reprs, tangents = streak_reprs_and_tangents(Rt, noise, np.random.default_rng(3000 + i))
                t0 = time.perf_counter()
                R = hough_seed_index_tangent(kobs, reprs, tangents, np.random.default_rng(4000 + i),
                                             n_coarse=N_COARSE)
                frac = frac_indexed(R, kobs, lab)
                dt = time.perf_counter() - t0
                append_result(dict(noise_tag=noise_tag, i=i, arm="TAN", frac=frac, dt=dt,
                                   n_reprs=len(reprs)))
                print(f"[{noise_tag}] i={i:2d} TAN          frac={frac:.2f} dt={dt:5.1f}s", flush=True)

            if (noise_tag, i, "PTS_MATCHED") not in done:
                # SAME reprs seed (3000+i) as TAN above -- byte-identical fresh data, only the
                # tangent gate differs.
                reprs, _ = streak_reprs_and_tangents(Rt, noise, np.random.default_rng(3000 + i))
                t0 = time.perf_counter()
                R = hough_seed_index_points_matched(kobs, reprs, np.random.default_rng(4000 + i),
                                                    n_coarse=N_COARSE)
                frac = frac_indexed(R, kobs, lab)
                dt = time.perf_counter() - t0
                append_result(dict(noise_tag=noise_tag, i=i, arm="PTS_MATCHED", frac=frac, dt=dt,
                                   n_reprs=len(reprs)))
                print(f"[{noise_tag}] i={i:2d} PTS_MATCHED  frac={frac:.2f} dt={dt:5.1f}s", flush=True)

    print("\nALL DONE. Summarizing...", flush=True)
    summarize()


def summarize(log=LOG):
    import collections
    rows = [json.loads(l) for l in open(log)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["noise_tag"], r["arm"])].append(r["frac"])
    print(f"\n{'noise':>6} {'arm':>12} {'n':>4} {'success':>9} {'median%':>9} {'mean%':>7}")
    for (noise_tag, arm), fracs in sorted(by.items()):
        fracs = np.array(fracs)
        succ = int((fracs > 0.7).sum())
        print(f"{noise_tag:>6} {arm:>12} {len(fracs):4d} {succ:>6d}/{len(fracs)} "
              f"{100*np.median(fracs):8.0f}% {100*np.mean(fracs):6.0f}%")

    print("\ncombined across both noise levels:")
    by_arm = collections.defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r["frac"])
    for arm, fracs in sorted(by_arm.items()):
        fracs = np.array(fracs)
        succ = int((fracs > 0.7).sum())
        print(f"{arm:>12} {len(fracs):4d} {succ:>6d}/{len(fracs)} ({100*succ/len(fracs):.1f}%) "
              f"mean%={100*np.mean(fracs):.0f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        run()
