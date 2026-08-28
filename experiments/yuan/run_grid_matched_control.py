"""Issue #59 matched control, extended to deliverable 3's actual phase diagram (issue #11,
run_grid_arms.py / results_grid.jsonl) -- not cbxd_sweep.py, which is a superseded, deliberately
reduced-scope pilot.

run_grid_arms.py's TAN arm has the same confound run_four_arms_full.py did: its tangent evidence
comes from a fresh, denser (thin=1 vs simulate()'s thin=6), spurious-free, independently-noised
re-simulation (streak_reprs_and_tangents), while PTS/COM vote on the on-disk kobs. Streak length
scales with NA, so the "extra data" advantage baked into that fresh re-sim likely GROWS with NA
too -- meaning the phase diagram's headline "TAN pulls away from PTS as NA grows" could be
partly a growing data-quality artifact, not a growing curvature effect.

This does NOT re-run PTS/COM/TAN (already in results_grid.jsonl) or extend into new NA/noise
territory -- it adds ONE new arm, PTS_MATCHED (cbxd_hough_tangent.hough_seed_index_points_matched),
at every (na, noise) cell ALREADY explored, using each cell's TAN reprs seed exactly (byte-identical
fresh data, matching run_grid_arms.py's own reprs seed convention: np.random.default_rng(3000+i)).

Separate output file (results_grid_matched.jsonl) so it can't disturb anything already reading
results_grid.jsonl (e.g. plot_phase_diagram.py).

Checkpointed JSONL, safe to resume.

  python run_grid_matched_control.py
"""
import json
import os
import sys
import time
import collections

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from generate_dataset_grid import load_cell
from cbxd_sweep import set_NA
from cbxd_hough_tangent import hough_seed_index_points_matched, streak_reprs_and_tangents

N_COARSE = 5_000_000
GRID_LOG = os.path.join(os.path.dirname(__file__), "results_grid.jsonl")
LOG = os.path.join(os.path.dirname(__file__), "results_grid_matched.jsonl")


def frac_indexed(R, kobs, lab):
    if R is None:
        return 0.0
    return float(score(R, kobs, 0.0025, ret_mask=True)[lab].mean())


def explored_cells(log=GRID_LOG):
    """(na, noise) cells run_grid_arms.py already visited -- the exact grid to extend, not a
    fresh sweep (no new NA/noise territory)."""
    cells = set()
    with open(log) as f:
        for line in f:
            r = json.loads(line)
            cells.add((r["na"], r["noise"]))
    return cells


def already_done(log=LOG):
    done = set()
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["na"], r["noise"], r["i"]))
    return done


def append_result(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(n_coarse=N_COARSE, n=20):
    cells = sorted(explored_cells())
    done = already_done()
    print(f"resuming: {len(done)} (na,noise,i) trials logged in {LOG}", flush=True)
    print(f"extending {len(cells)} already-explored grid cells with PTS_MATCHED", flush=True)

    for na, noise in cells:
        set_NA(na)
        crystals = load_cell(na, noise, n=n)
        for i, c in enumerate(crystals):
            if (na, noise, i) in done:
                continue
            Rt, kobs, lab, cnoise = c["Rt"], c["kobs"], c["lab"], c["noise"]
            # SAME reprs seed as run_grid_arms.py's TAN arm at this cell -- byte-identical data.
            reprs, _tangents = streak_reprs_and_tangents(Rt, cnoise, np.random.default_rng(3000 + i))
            t0 = time.perf_counter()
            R = hough_seed_index_points_matched(kobs, reprs, np.random.default_rng(4000 + i),
                                                n_coarse=n_coarse)
            frac = frac_indexed(R, kobs, lab)
            dt = time.perf_counter() - t0
            append_result(dict(na=na, noise=noise, i=i, arm="PTS_MATCHED", frac=frac, dt=dt,
                               n_reprs=len(reprs)))
            print(f"na={na} noise={noise:.1e} i={i:2d} PTS_MATCHED frac={frac:.2f} dt={dt:5.1f}s",
                  flush=True)

    print("\nALL DONE. Summarizing...", flush=True)
    summarize()


def summarize(grid_log=GRID_LOG, matched_log=LOG):
    grid_rows = [json.loads(l) for l in open(grid_log)]
    matched_rows = [json.loads(l) for l in open(matched_log)] if os.path.exists(matched_log) else []

    by_cell_arm = collections.defaultdict(list)
    for r in grid_rows:
        by_cell_arm[(r["na"], r["noise"], r["arm"])].append(r["frac"])
    for r in matched_rows:
        by_cell_arm[(r["na"], r["noise"], r["arm"])].append(r["frac"])

    cells = sorted(set((na, noise) for na, noise, arm in by_cell_arm))
    print(f"\n{'NA':>6} {'noise':>8} {'PTS':>10} {'PTS_MATCHED':>14} {'TAN':>10} "
          f"{'data-effect':>12} {'curv-effect':>12}")
    for na, noise in cells:
        pts = by_cell_arm.get((na, noise, "PTS"), [])
        matched = by_cell_arm.get((na, noise, "PTS_MATCHED"), [])
        tan = by_cell_arm.get((na, noise, "TAN"), [])
        if not (pts and matched and tan):
            continue
        pts_s = sum(f > 0.7 for f in pts)
        matched_s = sum(f > 0.7 for f in matched)
        tan_s = sum(f > 0.7 for f in tan)
        n = len(pts)
        data_effect = matched_s - pts_s
        curv_effect = tan_s - matched_s
        print(f"{na:6.3f} {noise:8.1e} {pts_s:>4d}/{n:<4d} {matched_s:>9d}/{n:<4d} "
              f"{tan_s:>4d}/{n:<4d} {data_effect:>+11d} {curv_effect:>+11d}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        run()
