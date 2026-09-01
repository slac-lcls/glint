"""Prefix contract of glint_fast.index_blind_nbest: the top-3 of an N=10 call IS the N=3 call.

WHY THIS EXISTS. hybrid_stream, StreamDriver's warm-up vote and the retry cascade (glint#75) all
lean on `index_blind_nbest(q, N)` returning the SAME ranked hypotheses for any N and merely
truncating -- #145 reused one N-best call for the cascade on exactly that promise ("slice == fresh
solve, bit-for-bit; N only truncates"). Nothing pinned it. The function's body is N-independent up to
its final dedup loop by construction, but a future early-exit keyed on N (say, stopping the anneal
once N distinct cells are in hand) would break the promise silently: every caller would still get
plausible cells, just not the same ones the vote was taken over.

WHAT IT CHECKS, on 3 real frames from experiments/frames_cxidb_clean.txt (the first three with >= 6
peaks): `index_blind_nbest(q, 10)[:3]` equals `index_blind_nbest(q, 3)` element-wise -- same order,
cells np.allclose, scores allclose -- and the check is not vacuous: each frame must yield at least
one hypothesis, and at least one frame must yield MORE than 3 at N=10, so the truncation is actually
exercised. The contract is device-independent, so this runs on whatever device glint_fast picked
(CPU torch indexes one of these frames in ~0.5 s; a GPU run is the owner's to do).

SKIPS, exit 0, when torch is not importable (glint_fast imports it unconditionally; the CPU CI job
installs none) or when torch is older than pyproject's floor (>= 1.12: the 1.11 CI leg exists for
glint_index's scatter_reduce shim, not for the indexer).

  PYTHONPATH=. python experiments/test_nbest_prefix.py
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")        # hybrid_stream's / gen_nbest's producing configuration

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:
    print(f"SKIP {os.path.basename(__file__)} -- no torch: glint.glint_fast cannot import here")
    sys.exit(0)
_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if _ver < (1, 12):
    print(f"SKIP {os.path.basename(__file__)} -- torch {torch.__version__} is below pyproject's "
          f"floor (>= 1.12); the indexer is not supported there")
    sys.exit(0)

import numpy as np                                                         # noqa: E402
from glint.glint_fast import DEV, index_blind_nbest, load                  # noqa: E402

FRAMES = os.path.join(HERE, "frames_cxidb_clean.txt")
N_FRAMES, N_SMALL, N_BIG = 3, 3, 10
fails = []


def check(name, cond, msg=""):
    print(f"  {name:64s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def main():
    frames = [np.asarray(q, float) for q in load(FRAMES) if len(q) >= 6][:N_FRAMES]
    print(f"device {DEV}, torch {torch.__version__}; {len(frames)} frames "
          f"({', '.join(str(len(q)) for q in frames)} peaks)")
    check(f"{N_FRAMES} real frames loaded", len(frames) == N_FRAMES, len(frames))
    index_blind_nbest(frames[0], N_SMALL)                    # warm-up (allocations), not timed
    any_truncated = False
    for k, q in enumerate(frames):
        t0 = time.time()
        big = index_blind_nbest(q, N_BIG)
        small = index_blind_nbest(q, N_SMALL)
        dt = time.time() - t0
        print(f"  frame {k}: N={N_BIG} -> {len(big)} hypotheses, N={N_SMALL} -> {len(small)}  "
              f"({dt:.1f} s both calls)")
        check(f"frame {k}: indexes at all (>= 1 hypothesis, else the contract is vacuous)",
              len(small) >= 1 and len(big) >= 1)
        check(f"frame {k}: len(N={N_SMALL}) == min({N_SMALL}, len(N={N_BIG}))",
              len(small) == min(N_SMALL, len(big)), f"{len(small)} vs {len(big)}")
        any_truncated |= len(big) > N_SMALL
        for i, ((cs, ss), (cb, sb)) in enumerate(zip(small, big)):
            check(f"frame {k} rank {i}: cell of N={N_SMALL} == cell of N={N_BIG} (same order)",
                  np.allclose(cs, cb), f"max |diff| {np.abs(np.asarray(cs) - np.asarray(cb)).max():.3g}")
            check(f"frame {k} rank {i}: score equal", np.allclose(ss, sb), f"{ss:.6f} vs {sb:.6f}")
    check(f"at least one frame yields > {N_SMALL} hypotheses at N={N_BIG} (truncation exercised)",
          any_truncated)
    return 0 if not fails else 1


if __name__ == "__main__":
    rc = main()
    print()
    print("ALL PASS" if not fails else f"FAIL: {len(fails)} check(s): {fails}")
    sys.exit(rc)
