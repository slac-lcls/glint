"""Full reproduction of yesterday's PR'ed experiment (20 crystals x {1e-4, 2e-4} noise, 5M
candidates, github.com/slac-lcls/glint branch yuan/cbxd-hough-accumulator-wip) -- but on the
FIXED on-disk dataset (no rng-crystal-identity bug this time) and with all four things built
today: COM (arm 1), PTS (arm 2), TAN (arm 3), and the confidence-based Cascade.

Checkpointed: appends one JSON line to results_four_arms_full.jsonl after EVERY (crystal, noise,
arm) trial, not just at the end -- a lost background process (this happened once already today)
loses at most one in-flight trial, not the whole ~90-minute run. Safe to re-run: skips any
(crystal, noise, arm) triple already present in the log.

  python run_four_arms_full.py
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
from cbxd_arm1_com import com_seed_index
from cbxd_hough_gpu import hough_seed_index
from cbxd_hough_tangent import hough_seed_index_tangent, streak_reprs_and_tangents
from cbxd_hough_cascade import hough_seed_index_cascade

N_COARSE = 5_000_000
CONF_THRESH = 0.6
LOG = os.path.join(os.path.dirname(__file__), "results_four_arms_full.jsonl")


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
            Rt, kobs, lab, cents, noise = c["Rt"], c["kobs"], c["lab"], c["cents"], c["noise"]
            n_streaks = len(cents)

            if (noise_tag, i, "COM") not in done:
                t0 = time.perf_counter()
                R = com_seed_index(cents, np.random.default_rng(1000 + i), n_coarse=N_COARSE)
                frac = frac_indexed(R, kobs, lab)
                dt = time.perf_counter() - t0
                append_result(dict(noise_tag=noise_tag, i=i, arm="COM", n_streaks=n_streaks,
                                   frac=frac, dt=dt))
                print(f"[{noise_tag}] i={i:2d} COM  frac={frac:.2f} dt={dt:5.1f}s", flush=True)

            if (noise_tag, i, "PTS") not in done:
                t0 = time.perf_counter()
                R = hough_seed_index(kobs, np.random.default_rng(2000 + i), n_coarse=N_COARSE)
                frac = frac_indexed(R, kobs, lab)
                dt = time.perf_counter() - t0
                append_result(dict(noise_tag=noise_tag, i=i, arm="PTS", n_streaks=n_streaks,
                                   frac=frac, dt=dt))
                print(f"[{noise_tag}] i={i:2d} PTS  frac={frac:.2f} dt={dt:5.1f}s", flush=True)

            if (noise_tag, i, "TAN") not in done:
                reprs, tangents = streak_reprs_and_tangents(Rt, noise, np.random.default_rng(3000 + i))
                t0 = time.perf_counter()
                R = hough_seed_index_tangent(kobs, reprs, tangents, np.random.default_rng(4000 + i),
                                             n_coarse=N_COARSE)
                frac = frac_indexed(R, kobs, lab)
                dt = time.perf_counter() - t0
                append_result(dict(noise_tag=noise_tag, i=i, arm="TAN", n_streaks=n_streaks,
                                   frac=frac, dt=dt))
                print(f"[{noise_tag}] i={i:2d} TAN  frac={frac:.2f} dt={dt:5.1f}s", flush=True)

            if (noise_tag, i, "CASCADE") not in done:
                reprs, tangents = streak_reprs_and_tangents(Rt, noise, np.random.default_rng(5000 + i))
                t0 = time.perf_counter()
                R, arm_used, conf = hough_seed_index_cascade(
                    kobs, reprs, tangents, np.random.default_rng(2000 + i),
                    n_coarse=N_COARSE, conf_thresh=CONF_THRESH)
                frac = frac_indexed(R, kobs, lab)
                dt = time.perf_counter() - t0
                append_result(dict(noise_tag=noise_tag, i=i, arm="CASCADE", n_streaks=n_streaks,
                                   frac=frac, dt=dt, arm_used=arm_used, conf=conf))
                print(f"[{noise_tag}] i={i:2d} CASC arm_used={arm_used} conf={conf:.2f} "
                      f"frac={frac:.2f} dt={dt:5.1f}s", flush=True)

    print("\nALL DONE. Summarizing...", flush=True)
    summarize()


def summarize(log=LOG):
    import collections
    rows = [json.loads(l) for l in open(log)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["noise_tag"], r["arm"])].append(r["frac"])
    print(f"\n{'noise':>6} {'arm':>8} {'n':>4} {'success':>9} {'median%':>9}")
    for (noise_tag, arm), fracs in sorted(by.items()):
        fracs = np.array(fracs)
        succ = int((fracs > 0.7).sum())
        print(f"{noise_tag:>6} {arm:>8} {len(fracs):4d} {succ:>6d}/{len(fracs)} "
              f"{100*np.median(fracs):8.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        run()
