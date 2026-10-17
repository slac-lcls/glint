#!/usr/bin/env python3
"""Harness check (CPU, seconds): j5_common.replay against the repo's own pinned replay.

experiments/test_seqstop_replay.py replays RunningConsensus(min_support=3, gap=2) over the cached
cxidb-120 N-best (experiments/nbest_120.npz) in 400 orders drawn from default_rng(1) and pins
FACTS poolgate_clean_lock_med 6, poolgate_clean_lock_max 18, poolgate_clean_maxpool 54. Its docstring
and the FACTS comment say the shipped pool-share and ratio tests change none of those orders
(poolgate_clean_diff_72 = 0). The replay here must reproduce all four with the same cells and orders, or
its numbers on the real runs mean nothing.

    python j5_check120.py            # exit 0 = harness agrees with the repo's test
    python j5_check120.py --time 3   # also: CPU ms/frame of index_blind_nbest on 3 refined-r0016 frames
                                     # (and the same frames scrambled), for the CPU fallback estimate
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import j5_common as C                                                       # noqa: E402

FACTS = dict(poolgate_clean_lock_med=6, poolgate_clean_lock_max=18, poolgate_clean_maxpool=54,
             poolgate_clean_diff_72=0, poolgate_clean_trials=400)        # check_numbers.py at 95d51ce


def check():
    """The n = 120 harness check; returns a dict with 'pass'."""
    z = np.load(os.path.join(C.GLINT, "experiments", "nbest_120.npz"))
    cells, fid, lyso, n = z["cells"], z["fid"], z["lyso"], int(z["nframes"])
    per_frame = [list(cells[fid == i]) for i in range(n)]
    voted = np.ones(n, bool)
    rng = np.random.default_rng(1)
    orders = [rng.permutation(n) for _ in range(FACTS["poolgate_clean_trials"])]
    res = {v: [C.replay(per_frame, voted, o, v) for o in orders] for v in ("shipped", "gaponly")}
    out = {}
    for v, rows in res.items():
        fr = np.array([r["voting"] for r in rows if r["decided"]])
        pools = np.array([r["npool"] for r in rows if r["decided"]])
        false = sum(1 for r in rows if r["decided"] and not C.same_lattice(r["M"], lyso))
        out[v] = dict(decided=int(sum(r["decided"] for r in rows)), median=float(np.median(fr)),
                      p90=float(np.percentile(fr, 90)), max=int(fr.max()), maxpool=int(pools.max()), false=false)
    diff = sum(1 for a, b in zip(res["shipped"], res["gaponly"])
               if (a["decided"], a["pushed"], a["support"], a["runner"]) != (b["decided"], b["pushed"], b["support"], b["runner"])
               or (a["M"] is not None and b["M"] is not None and not np.allclose(a["M"], b["M"])))
    print("rule defaults at", C.commit(), C.shipped_defaults())
    for v, s in out.items():
        print(f"  {v:8s}: {s}")
    print(f"  orders differing shipped vs gaponly: {diff}")
    ok = (out["gaponly"]["decided"] == 400 and out["gaponly"]["median"] == FACTS["poolgate_clean_lock_med"]
          and out["gaponly"]["max"] == FACTS["poolgate_clean_lock_max"]
          and out["gaponly"]["maxpool"] == FACTS["poolgate_clean_maxpool"]
          and out["gaponly"]["false"] == 0 and diff == FACTS["poolgate_clean_diff_72"])
    print("HARNESS CHECK", "PASS" if ok else "FAIL", "(vs FACTS", FACTS, ")")
    return dict(passed=bool(ok), facts=FACTS, shipped=out["shipped"], gaponly=out["gaponly"],
                orders_differing=int(diff))


def main():
    ok = check()["passed"]
    if "--time" in sys.argv:
        k = int(sys.argv[sys.argv.index("--time") + 1])
        from glint.glint_fast import index_blind_nbest, DEV
        from glint.multilattice import scramble_azimuth
        q6 = np.load(os.path.join(C.GT, "q6000_geom.npz"))
        fr = [np.asarray(q6[f"q{i}"], float) for i in range(k)]
        index_blind_nbest(fr[0], 3)                                          # warm-up
        for tag, frames in (("real", fr), ("scrambled", [scramble_azimuth(q, np.random.default_rng([1, i]))
                                                         for i, q in enumerate(fr)])):
            t = time.time()
            nb = [index_blind_nbest(q, 3) for q in frames]
            print(f"  {tag}: {k} frames on {DEV}, {1e3 * (time.time() - t) / k:.0f} ms/frame, "
                  f"peaks {[len(q) for q in frames]}, cells per frame {[len(x) for x in nb]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
