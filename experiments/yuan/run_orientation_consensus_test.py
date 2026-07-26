"""Small test (per Stefano's SFX consensus mechanism, glint/multishot.py::consensus_cell):
does RESULT-level cross-frame consensus (independent per-shot searches, then vote on the
orientation) buy anything on its own, isolated from cross-streak pooling?

Compares four things per crystal, at noise=1e-3 (transition zone) x M in {2,4,8}, on the SAME
NA=0.028 crystals deliverable 3 used (data/grid/orientations_na0.028.npz via
generate_dataset_multishot.load_dataset):

  COM_alone       -- arm 1 (centroid-only, no cross-streak pooling), single shot (M=1)
  COM_consensus   -- COM run INDEPENDENTLY on each of M shots, then orientation-consensus
                     (cbxd_orientation_consensus.multishot_consensus_com) -- isolates cross-frame
                     consensus's OWN contribution, since COM alone has none of PTS's cross-streak
                     pooling to confound it
  PTS_alone       -- arm 2 (cross-streak accumulator), single shot (M=1)
  PTS_consensus   -- PTS run independently per shot, then consensus -- does cross-frame consensus
                     STACK on top of cross-streak pooling, or is it redundant with it?

frac is always measured against pooled(crystal, m)'s kobs/lab (or shot 0 alone for the M=1
baselines) -- same convention as run_multishot.py's frac_indexed.

Checkpointed JSONL, safe to resume.

  python run_orientation_consensus_test.py [n_coarse] [n_crystals]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import score

from generate_dataset_multishot import load_dataset, pooled
from cbxd_arm1_com import com_seed_index
from cbxd_hough_gpu import hough_seed_index
from cbxd_orientation_consensus import majority_support, multishot_consensus_com, multishot_consensus_pts

NOISE = 1e-3
M_VALUES = (2, 4, 8)
# v2: min_support scales with M (majority_support) instead of the fixed constant 2 used in
# the N=8 pilot (results_orientation_consensus.jsonl) -- separate log file, not resumed from
# the pilot's stale fixed-threshold results (see issue #12 comment).
LOG = os.path.join(os.path.dirname(__file__), "results_orientation_consensus_v2_majority.jsonl")


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
                done.add((r["i"], r["method"], r["m"]))
    return done


def append_result(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(n_coarse=1_000_000, n_crystals=8):
    done = already_done()
    print(f"resuming: {len(done)} trials logged in {LOG}", flush=True)
    print(f"noise={NOISE:.0e}  n_coarse={n_coarse}  n_crystals={n_crystals}  M_VALUES={M_VALUES}",
          flush=True)

    ds = load_dataset(NOISE)[:n_crystals]
    for i, c in enumerate(ds):
        kobs0, lab0, cents0 = c["shots"][0]

        if (i, "COM_alone", 1) not in done:
            t0 = time.perf_counter()
            R = com_seed_index(cents0, np.random.default_rng(1000 + i), n_coarse=n_coarse)
            frac = frac_indexed(R, kobs0, lab0)
            append_result(dict(i=i, method="COM_alone", m=1, frac=frac,
                               dt=time.perf_counter() - t0))
            print(f"i={i} COM_alone       frac={frac:.2f}", flush=True)

        if (i, "PTS_alone", 1) not in done:
            t0 = time.perf_counter()
            R = hough_seed_index(kobs0, np.random.default_rng(2000 + i), n_coarse=n_coarse)
            frac = frac_indexed(R, kobs0, lab0)
            append_result(dict(i=i, method="PTS_alone", m=1, frac=frac,
                               dt=time.perf_counter() - t0))
            print(f"i={i} PTS_alone       frac={frac:.2f}", flush=True)

        for m in M_VALUES:
            shots = c["shots"][:m]
            kobs_p, lab_p, _ = pooled(c, m)

            maj = majority_support(m)

            if (i, "COM_consensus", m) not in done:
                t0 = time.perf_counter()
                R, support, _ = multishot_consensus_com(shots, 3000 + 100 * i, n_coarse=n_coarse,
                                                         min_support=maj)
                frac = frac_indexed(R, kobs_p, lab_p)
                append_result(dict(i=i, method="COM_consensus", m=m, frac=frac, support=support,
                                   min_support=maj, dt=time.perf_counter() - t0))
                print(f"i={i} COM_consensus m={m} support={support}/{m} (need>={maj}) frac={frac:.2f}", flush=True)

            if (i, "PTS_consensus", m) not in done:
                t0 = time.perf_counter()
                R, support, _ = multishot_consensus_pts(shots, 4000 + 100 * i, n_coarse=n_coarse,
                                                         min_support=maj)
                frac = frac_indexed(R, kobs_p, lab_p)
                append_result(dict(i=i, method="PTS_consensus", m=m, frac=frac, support=support,
                                   min_support=maj, dt=time.perf_counter() - t0))
                print(f"i={i} PTS_consensus m={m} support={support}/{m} (need>={maj}) frac={frac:.2f}", flush=True)

    print("\nALL DONE. Summarizing...", flush=True)
    summarize()


def summarize(log=LOG):
    import collections
    rows = [json.loads(l) for l in open(log)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["method"], r["m"])].append(r["frac"])
    print(f"\n{'method':>15} {'M':>3} {'n':>4} {'success':>9} {'median%':>9} {'mean%':>7}")
    for (method, m), fracs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        fracs = np.array(fracs)
        succ = int((fracs > 0.7).sum())
        print(f"{method:>15} {m:3d} {len(fracs):4d} {succ:>6d}/{len(fracs)} "
              f"{100*np.median(fracs):8.0f}% {100*np.mean(fracs):6.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
        ncry = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        run(n, ncry)
