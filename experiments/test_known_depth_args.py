"""The known-cell search depth (replica_gpu: index_known_gpu_cell's topa and nc) is checked, not coerced.

WHY THIS EXISTS. nc and topa became public parameters when the escalation arm (glint#208) started asking
for a deeper search (topa 128, nc 32). Before the check, nc=0 or a negative nc still kept ONE anchor
direction (the limit is tested after the first append), and a float was truncated by int(), so a caller
got a depth other than the one asked for without being told; a float topa reached torch.topk and raised,
which the escalation arm reads as an ordinary missed frame (Copilot review of #208). Both now raise
ValueError before any search, and the directions kept really are the nc asked for.

WHAT IT CHECKS. Bad values (0, negative, float, bool) raise for nc and topa; valid ints, numpy ints and
the defaults do not; on a real cxidb-17 frame axis_candidates_t keeps exactly 1 direction at nc=1 and at
most 3 at nc=3.

The batched engine takes the same depth (replica_gpu_batch: index_known_gpu_cell_batch's nc / topa /
full_grid and index_known_deep_batch, the batched escalation's search): the same bad values raise, as
does a bad budget; a chunk whose search raises comes back as misses flagged in `errors`; and -- in a fresh
interpreter at KC_FP=64, the precision the per-frame search always runs at -- index_known_deep_batch on
the full grid gives the same matched count as the per-frame index_known_gpu_cell on 12 real frames at
nc=1 AND at nc=16, while nc=1 and nc=16 differ on some of them (so a depth that failed to reach the
batched search would show). nc is capped at the 120-direction anchor pool (ANCHOR_POOL): above it the
batched dedup loop only refilled copies of the first anchor while its tensors kept growing with nc
(Copilot review of #211). The cap is checked directly, and the equivalence also runs at nc=500, past both
the pool and the distinct directions that survive the dedup, so the copies must not change the answer.

SKIPS, exit 0, without torch (replica_gpu imports it unconditionally; the CPU CI job installs none) or
below pyproject's torch floor (>= 1.12), like test_nbest_prefix.py.

  PYTHONPATH=. python experiments/test_known_depth_args.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    import torch
except ImportError:
    print(f"SKIP {os.path.basename(__file__)} -- no torch: glint.replica_gpu cannot import here")
    sys.exit(0)
_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _ver < (1, 12):
    print(f"SKIP {os.path.basename(__file__)} -- torch {torch.__version__} is below pyproject's "
          f"floor (>= 1.12); the indexer is not supported there")
    sys.exit(0)

import json                                                                # noqa: E402
import subprocess                                                          # noqa: E402
import warnings                                                            # noqa: E402

import numpy as np                                                         # noqa: E402
import glint.replica_gpu_batch as rgb                                      # noqa: E402
from glint.glint_fast import load                                          # noqa: E402
from glint.lattice import cell_to_Ar                                       # noqa: E402
from glint.replica_gpu import DEV, axis_candidates_t, index_known_gpu_cell  # noqa: E402

CELL = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
fails = []


def check(name, cond, msg=""):
    print(f"  {name:64s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def raises(fn):
    """True iff fn raises ValueError (the checked rejection); any other outcome, including a different
    exception from deeper in the search, is a failed check rather than a crash."""
    try:
        fn()
    except ValueError:
        return True
    except Exception:                                                   # noqa: BLE001
        return False
    return False


EQUIV = r"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["ROOT"])
import glint.replica_gpu as rg, glint.replica_gpu_batch as rgb
from glint.glint_fast import LYSO, load, matched_strict
fr = [q for q in load(os.environ["FRAMES"]) if len(q) >= 6][:12]
cnt = lambda M, q: -1 if M is None else int(matched_strict(np.asarray(M, float), q))
out = dict(fp=str(rgb.FP))
for nc in (1, 16, 500):
    out[f"pf{nc}"] = [cnt(rg.index_known_gpu_cell(q, LYSO, topa=2, nc=nc), q) for q in fr]
    out[f"b{nc}"] = [cnt(M, q) for M, q in zip(rgb.index_known_deep_batch(fr, LYSO, topa=2, nc=nc), fr)]
print("RESULT " + json.dumps(out))
"""


def main():
    few = np.zeros((3, 3))                          # < 6 peaks: a valid call returns None before searching
    for name in ("nc", "topa"):
        for bad in (0, -1, 1.5, 32.0, True):
            check(f"index_known_gpu_cell({name}={bad!r}) raises ValueError",
                  raises(lambda: index_known_gpu_cell(few, CELL, **{name: bad})))
    check("axis_candidates_t(nc=0) raises ValueError", raises(lambda: axis_candidates_t(None, 37.98, nc=0)))
    for kw in (dict(), dict(topa=128, nc=32), dict(topa=np.int64(16), nc=np.int64(4))):
        try:
            out = index_known_gpu_cell(few, CELL, **kw)
            check(f"index_known_gpu_cell({kw}) accepted", out is None, out)
        except ValueError as exc:
            check(f"index_known_gpu_cell({kw}) accepted", False, exc)

    q = next(np.asarray(f, float) for f in load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(f) >= 30)
    Q = torch.as_tensor(q, dtype=torch.float64, device=DEV)
    n1 = axis_candidates_t(Q, 37.98, nc=1)
    n3 = axis_candidates_t(Q, 37.98, nc=3)
    check("real frame: nc=1 keeps exactly 1 anchor direction", n1 is not None and len(n1) == 1,
          None if n1 is None else len(n1))
    check("real frame: nc=3 keeps 1 to 3", n3 is not None and 1 <= len(n3) <= 3, None if n3 is None else len(n3))

    # the batched engine: the same depth checks, a budget check, errors reported per frame
    for name in ("nc", "topa"):
        for bad in (0, -1, 1.5, True):
            check(f"index_known_gpu_cell_batch({name}={bad!r}) raises ValueError",
                  raises(lambda: rgb.index_known_gpu_cell_batch([q], CELL, **{name: bad})))
    for bad in (0, -5, 2.5, True):
        check(f"index_known_deep_batch(budget={bad!r}) raises ValueError",
              raises(lambda: rgb.index_known_deep_batch([q], CELL, 2, 1, budget=bad)))
    for asked, kept in ((7, 7), (rgb.ANCHOR_POOL, rgb.ANCHOR_POOL), (500, rgb.ANCHOR_POOL), (None, min(rgb.NC, rgb.ANCHOR_POOL))):
        got = rgb._cell_params(CELL, 2, asked)[-1]
        check(f"batched nc={asked!r} keeps {kept} anchor slots (capped at the pool)", got == kept, got)
    res, err = rgb.index_known_deep_batch([few, few], CELL, 2, 1, return_errors=True)
    check("index_known_deep_batch: frames with < 6 peaks are misses, not errors", res == [None, None] and err == [False, False])
    real = rgb.index_known_gpu_cell_batch
    try:
        def boom(*a, **k):
            raise RuntimeError("scripted search failure")
        rgb.index_known_gpu_cell_batch = boom
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            res, err = rgb.index_known_deep_batch([q, q], CELL, 2, 1, return_errors=True)
        check("a chunk whose search raises: misses, flagged as errors, with a warning",
              res == [None, None] and err == [True, True] and any("failed" in str(w.message) for w in wl), (res, err))
    finally:
        rgb.index_known_gpu_cell_batch = real
    env = {k: v for k, v in os.environ.items() if k not in ("KC_FP", "KC_SOLVE_FP")}
    env.update(KC_FP="64", ROOT=os.path.dirname(HERE), FRAMES=os.path.join(HERE, "frames_cxidb_clean.txt"),
               OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", EQUIV], capture_output=True, text=True, env=env)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        check("batched vs per-frame equivalence probe ran", False, r.stderr.strip().splitlines()[-3:])
    else:
        d = json.loads(line[len("RESULT "):])
        check("probe runs at fp64", d["fp"] == "torch.float64", d["fp"])
        for nc in (1, 16, 500):
            check(f"index_known_deep_batch == per-frame index_known_gpu_cell, 12 real frames, nc={nc}",
                  d[f"b{nc}"] == d[f"pf{nc}"], (d[f"b{nc}"], d[f"pf{nc}"]))
        diff = sum(a != b for a, b in zip(d["pf1"], d["pf16"]))
        check("nc changes the result on some frames (the equivalence can fail)", diff > 0, f"{diff}/12 differ")
    print(f"{'FAILURES: ' + ', '.join(fails) if fails else 'ALL PASS'}  (device {DEV}, torch {torch.__version__})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
