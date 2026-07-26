"""Deliverable 3 (issue #11) phase-diagram sweep, per Stefano's scoped plan: grid = (NA x noise),
arms = COM (floor reference) + PTS + TAN (cascade deliberately excluded -- "a deployment result,
not a phase-boundary variable"), #streaks/length reported as an observed statistic per cell (not
swept). NA in {0.010, 0.016, 0.022, 0.028}, noise laddered 2e-4 -> 2e-2 until PTS/TAN wall.

Adaptive per NA: stop climbing that NA's noise ladder once BOTH PTS and TAN hit 0/20 for two
consecutive levels -- clearly walled, no point burning GPU time deeper past the boundary. COM
still gets its floor-reference measurement at every visited cell.

Checkpointed: one JSON line per (na, noise, arm) trial to results_grid.jsonl, safe to resume --
re-running skips anything already logged.

  python run_grid_arms.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from generate_dataset_grid import NA_GRID, NOISE_LADDER, load_cell, orientation_pool_path
from cbxd_arm1_com import com_seed_index
from cbxd_hough_gpu import hough_seed_index
from cbxd_hough_tangent import hough_seed_index_tangent, streak_reprs_and_tangents

N_COARSE = 5_000_000
LOG = os.path.join(os.path.dirname(__file__), "results_grid.jsonl")


def frac_indexed(R, kobs, lab):
    if R is None:
        return 0.0
    return float(score(R, kobs, 0.0025, ret_mask=True)[lab].mean())


def already_done():
    done = {}
    if os.path.exists(LOG):
        with open(LOG) as f:
            for line in f:
                r = json.loads(line)
                done.setdefault((r["na"], r["noise"]), {})[(r["i"], r["arm"])] = r["frac"]
    return done


def append_result(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_cell(na, noise, done, n_coarse=N_COARSE, n=20):
    from cbxd_sweep import set_NA
    set_NA(na)
    crystals = load_cell(na, noise, n=n)
    cell_done = done.get((na, noise), {})
    pts_fracs, tan_fracs = [], []
    for i, c in enumerate(crystals):
        Rt, kobs, lab, cents, cnoise = c["Rt"], c["kobs"], c["lab"], c["cents"], c["noise"]

        if (i, "COM") in cell_done:
            pass
        else:
            t0 = time.perf_counter()
            R = com_seed_index(cents, np.random.default_rng(1000 + i), n_coarse=n_coarse)
            frac = frac_indexed(R, kobs, lab)
            append_result(dict(na=na, noise=noise, i=i, arm="COM", frac=frac,
                               dt=time.perf_counter() - t0, n_streaks=len(cents)))

        if (i, "PTS") in cell_done:
            pts_fracs.append(cell_done[(i, "PTS")])
        else:
            t0 = time.perf_counter()
            R = hough_seed_index(kobs, np.random.default_rng(2000 + i), n_coarse=n_coarse)
            frac = frac_indexed(R, kobs, lab)
            pts_fracs.append(frac)
            append_result(dict(na=na, noise=noise, i=i, arm="PTS", frac=frac,
                               dt=time.perf_counter() - t0, n_streaks=len(cents)))

        if (i, "TAN") in cell_done:
            tan_fracs.append(cell_done[(i, "TAN")])
        else:
            reprs, tangents = streak_reprs_and_tangents(Rt, cnoise, np.random.default_rng(3000 + i))
            t0 = time.perf_counter()
            R = hough_seed_index_tangent(kobs, reprs, tangents, np.random.default_rng(4000 + i),
                                         n_coarse=n_coarse)
            frac = frac_indexed(R, kobs, lab)
            tan_fracs.append(frac)
            append_result(dict(na=na, noise=noise, i=i, arm="TAN", frac=frac,
                               dt=time.perf_counter() - t0, n_streaks=len(cents)))

        print(f"  na={na} noise={noise:.1e} i={i:2d} done", flush=True)

    pts_succ = sum(f > 0.7 for f in pts_fracs)
    tan_succ = sum(f > 0.7 for f in tan_fracs)
    return pts_succ, tan_succ


def run():
    done = already_done()
    for na in NA_GRID:
        consecutive_walled = 0
        for noise in NOISE_LADDER:
            print(f"=== na={na} noise={noise:.1e} ===", flush=True)
            pts_succ, tan_succ = run_cell(na, noise, done)
            print(f"na={na} noise={noise:.1e}  PTS {pts_succ}/20  TAN {tan_succ}/20", flush=True)
            if pts_succ == 0 and tan_succ == 0:
                consecutive_walled += 1
                if consecutive_walled >= 2:
                    print(f"na={na}: walled at noise={noise:.1e} (2 consecutive 0/20) -- "
                          f"skipping remaining higher noise levels for this NA", flush=True)
                    break
            else:
                consecutive_walled = 0
    print("\nALL DONE.", flush=True)


if __name__ == "__main__":
    run()
