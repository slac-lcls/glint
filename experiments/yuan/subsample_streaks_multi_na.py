"""Multi-geometry streak-count deconfound (extends subsample_streaks_na028.py). One NA and one
streak count isn't enough to conclude "geometry matters at low counts, less so at high counts" --
this tests the SAME streak-count targets subsampled from SEVERAL distinct NA geometries, so the
NA-dependence at fixed count can be seen directly rather than inferred from a single comparison.

NA classes: 0.016, 0.028, 0.040, 0.060 (well-spaced across the tested grid range -- distinct cone
geometries, not near-duplicates). Streak-count targets: 4, 11 (matching NA=0.010's and NA=0.016's
natural medians, as before). Each of the 20 crystals per (NA, target) cell already gets its own
INDEPENDENTLY random-drawn subset of streaks (see build_subsampled in subsample_streaks_na028.py
-- a fresh rng per crystal, not a shared/repeated draw), so the 20 trials at one (NA, target) cell
are already 20 distinct geometric realizations at that count, not repeats.

NA=0.028's two targets were already run in results_subsample.jsonl (single-NA pass) -- migrated
into this run's log on first launch so they aren't recomputed.

Fixed at noise=2e-4. Checkpointed (results_subsample_multi.jsonl), safe to resume.

  python subsample_streaks_multi_na.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")

from cbxd_sweep import set_NA
from generate_dataset_grid import get_orientation_pool
from subsample_streaks_na028 import NOISE, N_COARSE, build_subsampled, frac_indexed
from cbxd_arm1_com import com_seed_index
from cbxd_hough_gpu import hough_seed_index
from cbxd_hough_tangent import hough_seed_index_tangent

NA_LIST = [0.016, 0.028, 0.040, 0.060]
N_TARGETS = [4, 11]
LOG = os.path.join(os.path.dirname(__file__), "results_subsample_multi.jsonl")
OLD_LOG = os.path.join(os.path.dirname(__file__), "results_subsample.jsonl")


def migrate_old_log():
    """Port NA=0.028's already-run trials (single-NA pass) into this log, tagged with na=0.028,
    so they aren't recomputed."""
    if not os.path.exists(OLD_LOG) or os.path.exists(LOG):
        return
    with open(LOG, "a") as out:
        for line in open(OLD_LOG):
            r = json.loads(line)
            r["na"] = 0.028
            out.write(json.dumps(r) + "\n")
    print(f"migrated {sum(1 for _ in open(OLD_LOG))} trials from {OLD_LOG} (na=0.028)", flush=True)


def already_done():
    done = set()
    if os.path.exists(LOG):
        with open(LOG) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["na"], r["n_target"], r["i"], r["arm"]))
    return done


def append_result(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run():
    migrate_old_log()
    done = already_done()
    print(f"resuming: {len(done)} (na,n_target,i,arm) trials already logged", flush=True)

    for na in NA_LIST:
        set_NA(na)
        Rts = get_orientation_pool(na)
        for n_target in N_TARGETS:
            for i, Rt in enumerate(Rts):
                if (na, n_target, i, "COM") in done and (na, n_target, i, "PTS") in done \
                        and (na, n_target, i, "TAN") in done:
                    continue
                seed = int(round(na * 100_000)) + 10_000 * n_target + i
                kobs, lab, cents, reprs, tangents, n_kept = build_subsampled(Rt, n_target, seed=seed)

                if (na, n_target, i, "COM") not in done:
                    t0 = time.perf_counter()
                    R = com_seed_index(cents, np.random.default_rng(1000 + i), n_coarse=N_COARSE)
                    frac = frac_indexed(R, kobs, lab)
                    append_result(dict(na=na, n_target=n_target, i=i, arm="COM", frac=frac,
                                       n_kept=n_kept, dt=time.perf_counter() - t0))
                    print(f"na={na} n_target={n_target} i={i:2d} COM  frac={frac:.2f}", flush=True)

                if (na, n_target, i, "PTS") not in done:
                    t0 = time.perf_counter()
                    R = hough_seed_index(kobs, np.random.default_rng(2000 + i), n_coarse=N_COARSE)
                    frac = frac_indexed(R, kobs, lab)
                    append_result(dict(na=na, n_target=n_target, i=i, arm="PTS", frac=frac,
                                       n_kept=n_kept, dt=time.perf_counter() - t0))
                    print(f"na={na} n_target={n_target} i={i:2d} PTS  frac={frac:.2f}", flush=True)

                if (na, n_target, i, "TAN") not in done:
                    t0 = time.perf_counter()
                    R = hough_seed_index_tangent(kobs, reprs, tangents,
                                                 np.random.default_rng(3000 + i), n_coarse=N_COARSE)
                    frac = frac_indexed(R, kobs, lab)
                    append_result(dict(na=na, n_target=n_target, i=i, arm="TAN", frac=frac,
                                       n_kept=n_kept, dt=time.perf_counter() - t0))
                    print(f"na={na} n_target={n_target} i={i:2d} TAN  frac={frac:.2f}", flush=True)

            fracs = {arm: [] for arm in ("COM", "PTS", "TAN")}
            for line in open(LOG):
                r = json.loads(line)
                if r["na"] == na and r["n_target"] == n_target:
                    fracs[r["arm"]].append(r["frac"])
            summary = "  ".join(f"{arm} {sum(f>0.7 for f in v)}/{len(v)}" for arm, v in fracs.items())
            print(f"na={na} n_target={n_target}  {summary}", flush=True)

    print("\nALL DONE.", flush=True)


if __name__ == "__main__":
    run()
