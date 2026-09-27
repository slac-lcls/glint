"""The known-cell engine's precision defaults: fp32 working precision, fp64 3x3 solve on the torch path.

WHY THIS EXISTS. replica_gpu_batch reads KC_FP / KC_SOLVE_FP at import. The default moved from fp64 to fp32 on
26 Sep 2026: fp32 was measured rate/lattice-identical (all lattice systems, 18 Jul; the cxidb-17 120 and 480 on
exclusive A100s, and the published streaming replay decision-identical, 26 Sep) and is 12-19 % faster at B=120 on a
warm A100 and far faster on fp32-strong GPUs. This pins what the
defaults ARE, that KC_FP=64 still gives the fp64 path the published timings were measured on, that the torch
path's solve runs in fp64 under the fp32 default (the hedge) and returns the working dtype, and -- on real frames
-- that the two precisions make the same strict decisions.

WHAT IT CHECKS (each configuration in a fresh interpreter, since the module reads the environment at import):
  * unset -> FP float32, _SOLVE_FP float64;  KC_FP=64 -> float64 / float64;  KC_FP=32 KC_SOLVE_FP=32 -> both float32;
  * solve3x3 under the default: float32 in, float32 out, and -- on an ill-conditioned regression family --
    measurably different from a separate KC_SOLVE_FP=32 probe;
  * index_known_gpu_cell_batch (the torch path; CPU here) on the first 30 cxidb-17 benchmark frames with the textbook
    cell: the same strict pass/fail per frame under the default and under KC_FP=64.

SKIPS, exit 0, without torch or below pyproject's torch floor (>= 1.12), like test_nbest_prefix.py.

  PYTHONPATH=. python experiments/test_kc_precision_default.py
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

try:
    import torch
except ImportError:
    print(f"SKIP {os.path.basename(__file__)} -- no torch: glint.replica_gpu_batch cannot import here")
    sys.exit(0)
_m = re.match(r"(\d+)\.(\d+)", torch.__version__)
if _m is None:
    print(f"SKIP {os.path.basename(__file__)} -- unparseable torch version: {torch.__version__}")
    sys.exit(0)
_ver = tuple(int(x) for x in _m.groups())
if _ver < (1, 12):
    print(f"SKIP {os.path.basename(__file__)} -- torch {torch.__version__} is below pyproject's floor (>= 1.12)")
    sys.exit(0)

PROBE = r"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.environ["ROOT"])
import glint.replica_gpu_batch as rgb
from glint.glint_fast import LYSO, gpass, load
out = dict(fp=str(rgb.FP), solve=str(rgb._SOLVE_FP))
rhs = torch.tensor([[1.0, 0.0, 2.0], [0.5, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=rgb.FP)[None]
cases = [[[1.0, 1.0, 1.0], [1.0, 1.0000001, 1.0], [1.0, 1.0, 1.0000002]],
         [[1.0, 1.0, 1.0], [1.0, 1.0000002, 1.0], [1.0, 1.0, 1.0000004]],
         [[1.0, 1.0, 1.0], [1.0, 1.0000005, 1.0], [1.0, 1.0, 1.0000010]]]
X = [rgb.solve3x3(torch.tensor(A, dtype=rgb.FP)[None], rhs) for A in cases]
out["solve_dtype"] = str(X[0].dtype); out["solve_case"] = [x.cpu().tolist()[0] for x in X]
if os.environ.get("FRAMES"):
    fr = [q for q in load(os.environ["FRAMES"]) if len(q) >= 6][:30]
    Ms = rgb.index_known_gpu_cell_batch(fr, LYSO)
    out["strict"] = [int(a and b) for a, b in (gpass(M, q) for M, q in zip(Ms, fr))]
print("RESULT " + json.dumps(out))
"""


def probe(env_extra, frames=False):
    env = {k: v for k, v in os.environ.items() if k not in ("KC_FP", "KC_SOLVE_FP")}
    env.update(env_extra, ROOT=ROOT, OMP_NUM_THREADS="1")
    if frames:
        env["FRAMES"] = os.path.join(HERE, "frames_cxidb_clean.txt")
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"probe failed ({env_extra}) rc={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(f"probe failed ({env_extra}) missing RESULT line\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    return json.loads(line[len("RESULT "):])


fails = []


def check(name, cond, msg=""):
    print(f"  {name:70s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def main():
    d = probe({}, frames=True)
    check("default: working precision float32", d["fp"] == "torch.float32", d["fp"])
    check("default: torch-path 3x3 solve float64 (the hedge)", d["solve"] == "torch.float64", d["solve"])
    check("default: solve3x3 returns the working dtype", d["solve_dtype"] == "torch.float32", d["solve_dtype"])
    e = probe({"KC_FP": "64"}, frames=True)
    check("KC_FP=64: working precision float64", e["fp"] == "torch.float64", e["fp"])
    check("KC_FP=64: solve float64", e["solve"] == "torch.float64", e["solve"])
    f = probe({"KC_FP": "32", "KC_SOLVE_FP": "32"})
    check("KC_FP=32 KC_SOLVE_FP=32: both float32", f["fp"] == f["solve"] == "torch.float32", (f["fp"], f["solve"]))
    solve_probe_gap = max(abs(a - b)
                          for case_a, case_b in zip(d["solve_case"], f["solve_case"])
                          for row_a, row_b in zip(case_a, case_b)
                          for a, b in zip(row_a, row_b))
    solve_probe_cases = sum(any(abs(a - b) > 0.0
                                for row_a, row_b in zip(case_a, case_b)
                                for a, b in zip(row_a, row_b))
                            for case_a, case_b in zip(d["solve_case"], f["solve_case"]))
    check("default: solve3x3 differs from a KC_SOLVE_FP=32 probe on the ill-conditioned regression family",
          solve_probe_cases > 0, f"{solve_probe_cases}/{len(d['solve_case'])} cases differ; max gap {solve_probe_gap:.2e}")
    check("real frames: the same strict decision per frame in fp32 (default) and fp64",
          d["strict"] == e["strict"] and len(d["strict"]) == 30, (d["strict"], e["strict"]))
    check("real frames: the check is not vacuous (some pass, some fail)",
          0 < sum(e["strict"]) < 30, e["strict"])
    print(f"{'FAILURES: ' + ', '.join(fails) if fails else 'ALL PASS'}  (torch {torch.__version__})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
