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

import numpy as np                                                         # noqa: E402
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
    print(f"{'FAILURES: ' + ', '.join(fails) if fails else 'ALL PASS'}  (device {DEV}, torch {torch.__version__})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
