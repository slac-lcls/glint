#!/usr/bin/env python3
"""Does hybrid_index(escalate=True) reproduce the measurement it ships?

The escalation's rule and defaults come from exp/joint-ceiling (RESULTS_escalation_k32.md, S3DF job 39181473):
on the cxidb-17 sets the shipped path indexes 366/480 [92/120] at the strict gate, and a T2 deep search on its
misses, accepted only when the fit beats all 32 of its own azimuth-scrambled copies, adds 21 [1] for 387/480
[93/120]. This runs the PRODUCT path twice on the same frames -- escalate=None, then escalate=True -- scores both
with the strict gate, and, given the experiment's tier file, checks that the escalated frames are exactly the
frames the experiment accepted (same seeding [20260926, frame, copy], so they must be).

  PYTHONPATH=. python experiments/validate_escalation.py --frames ~/q480_fix.txt \\
      [--expect-total 387] [--expect-tier .../k32/tier_480_T2_k32.json] [--out validate_480.json] \\
      [--escalate '{"batch": false}']      # any hybrid_index(escalate=...) dict, as JSON; default True

--escalate picks the mode: the default is the batched escalation (escalate_batch over index_known_deep_batch);
{"batch": false} is the per-frame arm. The batched search runs at the engine's working precision (KC_FP), the
per-frame one always in fp64, so KC_FP=64 is the configuration in which both must reproduce the experiment.

GPU (torch). Exit 1 on any mismatch with an --expect-* value.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np                                                               # noqa: E402

from glint.glint_fast import GATE_FRAC, GATE_MIN, LYSO, load, matched_strict   # noqa: E402
from glint.hybrid_stream import hybrid_index                                     # noqa: E402
from glint.multishot import same_lattice                                         # noqa: E402


def strict(M, q):
    if M is None:
        return False
    m = matched_strict(M, q)
    return bool(m >= GATE_MIN and m >= GATE_FRAC * len(q) and same_lattice(M, np.asarray(LYSO, float)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--expect-total", type=int)
    ap.add_argument("--expect-tier", help="the experiment's tier_<tag>_T2_k32.json: escalated frames must match")
    ap.add_argument("--out")
    ap.add_argument("--escalate", default="true", help="JSON for hybrid_index(escalate=...): true or a dict")
    a = ap.parse_args()
    esc_arg = json.loads(a.escalate)
    frames = [np.asarray(q, float) for q in load(a.frames) if len(q) >= 6]     # escalate.py's frame list
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except Exception:                                                          # noqa: BLE001
        dev, sync = "cpu", (lambda: None)

    t = time.time(); r0, s0 = hybrid_index(frames); sync(); t_base = time.time() - t
    t = time.time(); r1, s1 = hybrid_index(frames, escalate=esc_arg); sync(); t_esc = time.time() - t
    ok0 = [strict(r["M"], q) for r, q in zip(r0, frames)]
    ok1 = [strict(r["M"], q) for r, q in zip(r1, frames)]
    esc = sorted(i for i, r in enumerate(r1) if r.get("escalated"))
    lost = [i for i in range(len(frames)) if ok0[i] and not ok1[i]]
    try:
        import glint.replica_gpu_batch as _rgb
        prec = str(_rgb.FP)
    except Exception:                                                          # noqa: BLE001
        prec = None
    out = dict(frames=os.path.abspath(a.frames), n=len(frames), device=dev, escalate=s1.get("escalation"),
               batch_precision=prec,
               git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                  cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip(),
               strict_base=int(sum(ok0)), strict_escalated=int(sum(ok1)), escalated=esc,
               escalated_strict=int(sum(ok1[i] for i in esc)), lost=lost,
               candidates=s1.get("n_escalation_candidates"), searches=s1.get("escalation_searches"),
               t_base_s=round(t_base, 1), t_escalate_total_s=round(t_esc, 1),
               t_escalation_stage_s=round(t_esc - t_base, 1))
    print(f"{os.path.basename(a.frames)} on {dev} (escalate {out['escalate']}, engine {prec}): strict {out['strict_base']} -> {out['strict_escalated']} / {len(frames)}"
          f" | escalated {len(esc)} ({out['escalated_strict']} strict) of {out['candidates']} candidates, "
          f"{out['searches']} searches | lost {lost} | escalation stage ~{out['t_escalation_stage_s']} s")
    bad = []
    if lost:
        bad.append(f"escalation lost strict frames {lost}")
    if a.expect_total is not None and out["strict_escalated"] != a.expect_total:
        bad.append(f"total {out['strict_escalated']} != expected {a.expect_total}")
    if a.expect_tier:
        R = json.load(open(a.expect_tier))["records"]
        want = sorted(r["i"] for r in R if r["strict"] and r["obs_ok"] and r["m"] > max(r["null_m"]))
        out["experiment_accepted"] = want
        if want != esc:
            bad.append(f"escalated {esc} != experiment {want}")
        print(f"  experiment accepted {len(want)}: {'IDENTICAL' if want == esc else 'DIFFERENT'}")
    out["mismatches"] = bad
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    for b in bad:
        print("  MISMATCH:", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
