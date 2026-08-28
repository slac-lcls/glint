"""Does TAN (arm 3, cross-streak accumulator + per-streak tangent bonus) benefit from M-shot
data pooling where PTS (arm 2, points-only) didn't? results_multishot.jsonl showed PTS pooling
flat at every noise level tested -- this checks whether TAN's extra tangent signal changes that.

Same data-pooling mechanism as run_multishot.py (concatenate M shots into one bigger accumulator
call, exact for the vote-count objective per test_cbxd_multishot.py). kobs_pooled reuses the SAME
frozen shots from generate_dataset_multishot.py (byte-identical to PTS's data). For the tangent
side, reprs/tangents are extracted per-shot via streak_reprs_and_tangents(Rt, noise, rng) --
matching the established single-shot convention (run_four_arms_full.py's TAN arm: independent rng
draw from kobs, NOT tied to the same noise realization) -- called M times with independent draws
and concatenated, so pooling gives M independent tangent estimates per lattice plane, same
noise-averaging logic as the point pooling.

  python run_tan_pooling_test.py [n_coarse] [n_crystals]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from generate_dataset_multishot import load_dataset, pooled
from cbxd_hough_tangent import hough_seed_index_tangent, streak_reprs_and_tangents

NOISE = 1e-3
M_VALUES = (1, 2, 4, 8)
LOG = os.path.join(os.path.dirname(__file__), "results_tan_pooling.jsonl")


def frac_indexed(R, kobs, lab):
    if R is None:
        return 0.0
    return float(score(R, kobs, 0.0025, ret_mask=True)[lab].mean())


def pooled_tangents(Rt, noise, m, seed_base):
    """M independent streak_reprs_and_tangents draws, concatenated -- the tangent-side analog
    of pooled() for kobs."""
    reprs_list, tang_list = [], []
    for j in range(m):
        r, t = streak_reprs_and_tangents(Rt, noise, np.random.default_rng(seed_base + j))
        if len(r):
            reprs_list.append(r)
            tang_list.append(t)
    if not reprs_list:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack(reprs_list), np.vstack(tang_list)


def already_done():
    done = set()
    if os.path.exists(LOG):
        with open(LOG) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["i"], r["m"]))
    return done


def append_result(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(n_coarse=5_000_000, n_crystals=4):
    done = already_done()
    print(f"resuming: {len(done)} trials logged in {LOG}", flush=True)
    print(f"noise={NOISE:.0e}  n_coarse={n_coarse}  n_crystals={n_crystals}  M_VALUES={M_VALUES}",
          flush=True)

    ds = load_dataset(NOISE)[:n_crystals]
    for i, c in enumerate(ds):
        Rt = c["Rt"]
        for m in M_VALUES:
            if (i, m) in done:
                continue
            kobs, lab, cents = pooled(c, m)
            reprs, tangents = pooled_tangents(Rt, NOISE, m, seed_base=8000 + 100 * i)
            t0 = time.perf_counter()
            R = hough_seed_index_tangent(kobs, reprs, tangents,
                                         np.random.default_rng(9000 + 100 * i + m),
                                         n_coarse=n_coarse)
            frac = frac_indexed(R, kobs, lab)
            dt = time.perf_counter() - t0
            append_result(dict(i=i, m=m, n_points=len(kobs), n_tangents=len(reprs),
                               frac=frac, dt=dt))
            print(f"i={i} m={m} n_points={len(kobs):4d} n_tangents={len(reprs):3d} "
                  f"frac={frac:.2f} dt={dt:5.1f}s", flush=True)

    print("\nALL DONE. Summarizing...", flush=True)
    summarize()


def summarize(log=LOG):
    import collections
    rows = [json.loads(l) for l in open(log)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["m"]].append(r["frac"])
    print(f"\n{'M':>3} {'n':>4} {'success':>9} {'median%':>9} {'mean%':>7}")
    for m, fracs in sorted(by.items()):
        fracs = np.array(fracs)
        succ = int((fracs > 0.7).sum())
        print(f"{m:3d} {len(fracs):4d} {succ:>6d}/{len(fracs)} "
              f"{100*np.median(fracs):8.0f}% {100*np.mean(fracs):6.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
        ncry = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        run(n, ncry)
