"""Does the EXACT fused line-search M3 (refine_vec_ls) reach the M2 maxima in FEWER outer steps than
momentum-GD? Each outer step is ONE matmul for both (GD and LS), but LS takes the curvature-exact stride
along the gradient (f/f'/f'' fused from the same phases, matmul-free), safeguard-capped against basin
jumps. The tell: ls-4 vs gd-8 -- if ls-4 holds ~84/120 it wins ~2x on matmuls. cg-4 (the fixed-schedule
loser) is shown for contrast. Same 120 cxidb frames + gate as cg_test.py."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_fast import index_blind_fast
from glint.multishot import same_lattice
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
for refiner, steps in [("grad", 8), ("ls", 8), ("ls", 6), ("ls", 4), ("ls", 3), ("ls", 2),
                       ("grad", 4), ("cg", 4)]:
    sl, ms = run(refiner, steps)
    print(f"  {refiner:4s}  {steps:2d}    {sl}/{n} = {100*sl//n}%   {ms:.1f}", flush=True)
