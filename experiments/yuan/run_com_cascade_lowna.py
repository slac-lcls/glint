"""Fills the gap in the NA=0.016 (low-NA / reflection-starved) study: run COM and the
confidence-based Cascade (PTS by default, escalate to TAN only if unconfident -- NOT always-on
TAN) on the SAME fixed dataset run_three_arms_lowna.py already used (data/simulated_data_lowna/,
20 crystals, #streaks 4-18, noise=2e-4). PTS/TAN references already live in
results_three_arms_lowna.npz. n_coarse=5M, matching every other "final" run today.

Checkpointed the same way as run_four_arms_full.py -- append one JSON line per (i, arm) trial so
a lost session costs at most one in-flight trial.

  python run_com_cascade_lowna.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from cbxd_sweep import set_NA
from generate_dataset_lowna import NA, load_dataset
from cbxd_arm1_com import com_seed_index
from cbxd_hough_tangent import streak_reprs_and_tangents
from cbxd_hough_cascade import hough_seed_index_cascade

N_COARSE = 5_000_000
CONF_THRESH = 0.6
LOG = os.path.join(os.path.dirname(__file__), "results_com_cascade_lowna.jsonl")


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
                done.add((r["i"], r["arm"]))
    return done


def append_result(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run():
    set_NA(NA)
    ds = load_dataset()
    done = already_done()
    print(f"resuming: {len(done)} (i,arm) trials already logged", flush=True)

    for i, c in enumerate(ds):
        Rt, kobs, lab, cents, noise = c["Rt"], c["kobs"], c["lab"], c["cents"], c["noise"]
        n_streaks = len(cents)

        if (i, "COM") not in done:
            t0 = time.perf_counter()
            R = com_seed_index(cents, np.random.default_rng(1000 + i), n_coarse=N_COARSE)
            frac = frac_indexed(R, kobs, lab)
            dt = time.perf_counter() - t0
            append_result(dict(i=i, arm="COM", n_streaks=n_streaks, frac=frac, dt=dt))
            print(f"i={i:2d} COM  frac={frac:.2f} dt={dt:5.1f}s", flush=True)

        if (i, "CASCADE") not in done:
            reprs, tangents = streak_reprs_and_tangents(Rt, noise, np.random.default_rng(5000 + i))
            t0 = time.perf_counter()
            R, arm_used, conf = hough_seed_index_cascade(
                kobs, reprs, tangents, np.random.default_rng(2000 + i),
                n_coarse=N_COARSE, conf_thresh=CONF_THRESH)
            frac = frac_indexed(R, kobs, lab)
            dt = time.perf_counter() - t0
            append_result(dict(i=i, arm="CASCADE", n_streaks=n_streaks, frac=frac, dt=dt,
                               arm_used=arm_used, conf=conf))
            print(f"i={i:2d} CASC arm_used={arm_used} conf={conf:.2f} frac={frac:.2f} "
                  f"dt={dt:5.1f}s", flush=True)

    print("\nALL DONE.", flush=True)


if __name__ == "__main__":
    run()
