"""A5: does nonlinear-CG M3 refine reach the same maxima in fewer steps at equal rate? Sweep REFINER x
STEPS on the real 120 cxidb frames (blind rate + ms/frame). GD-8 is the baseline; CG-4 vs GD-4 is the
tell (if CG-4 holds 84/120 while GD-4 drops -> CG wins, ~2x on M3). M3 is compute-saturated -> ms scales
with steps."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import fftindex.glint_fast as gf
from fftindex.glint_fast import index_blind_fast
from fftindex.multishot import same_lattice
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)


def run(refiner, steps):
    gf.REFINER = refiner; gf.STEPS = steps
    index_blind_fast(frames[0]); torch.cuda.synchronize()          # warmup (per config)
    t0 = time.perf_counter(); sl = 0
    for q in frames:
        M = index_blind_fast(q); sl += (M is not None and same_lattice(M, gf.LYSO))
    torch.cuda.synchronize()
    return sl, 1e3 * (time.perf_counter() - t0) / n


print("refiner steps  blind_rate  ms/frame")
for refiner, steps in [("grad", 8), ("cg", 8), ("cg", 6), ("cg", 4), ("cg", 3), ("grad", 4), ("grad", 3)]:
    sl, ms = run(refiner, steps)
    print(f"  {refiner:4s}  {steps:2d}    {sl}/{n} = {100*sl//n}%   {ms:.1f}", flush=True)
