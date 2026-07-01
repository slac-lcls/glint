"""A5b: BB (Barzilai-Borwein) + LM (adaptive Levenberg-Marquardt) M3 refiners vs momentum-GD. Do either
reach the same maxima in fewer steps at equal blind rate? (M3 compute-saturated -> ms ~ steps.) grad-8 =
baseline (84/120). BB is gradient-direction (cheap/gentle); LM is steepest->Newton with a safety valve."""
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
    index_blind_fast(frames[0]); torch.cuda.synchronize()
    t0 = time.perf_counter(); sl = 0
    for q in frames:
        M = index_blind_fast(q); sl += (M is not None and same_lattice(M, gf.LYSO))
    torch.cuda.synchronize()
    return sl, 1e3 * (time.perf_counter() - t0) / n


print("refiner steps  blind_rate  ms/frame")
for refiner, steps in [("grad", 8), ("bb", 8), ("bb", 6), ("bb", 4), ("bb", 3),
                       ("lm", 8), ("lm", 6), ("lm", 4), ("lm", 3), ("grad", 4)]:
    sl, ms = run(refiner, steps)
    print(f"  {refiner:4s}  {steps:2d}    {sl}/{n} = {100*sl//n}%   {ms:.1f}", flush=True)
