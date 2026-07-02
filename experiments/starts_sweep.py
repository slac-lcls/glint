"""Does pruning the M1 start grid hold the blind rate? M3 is compute-saturated (cost proportional to
start count), so fewer starts = proportional M3-time cut IF the rate holds. STARTS = sample(n_dir) x 32
length shells (default n_dir=2200 -> 70400 starts). Sweep n_dir DOWN on the 120 cxidb frames; the tell
is the smallest n_dir that still solves 84/120. Mirror of cg_test.py (same frames, same gate).
Run (S3DF ampere): srun -p ampere -A lcls:default@ampere -q preemptable --gres=gpu:a100:1 -t 12 \
    python starts_sweep.py > ~/starts.txt; cat ~/starts.txt"""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_fast import index_blind_fast
from glint.glint_index import sample
from glint.multishot import same_lattice

frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)
gf.REFINER = "grad"; gf.STEPS = 8                       # the validated default point


def run(n_dir):
    gf.STARTS = sample(n_dir=n_dir).to(gf.DEV)          # rebind the module-global start grid
    nstart = gf.STARTS.shape[0]
    index_blind_fast(frames[0]); torch.cuda.synchronize()          # warmup (per config)
    t0 = time.perf_counter(); sl = 0
    for q in frames:
        M = index_blind_fast(q); sl += (M is not None and same_lattice(M, gf.LYSO))
    torch.cuda.synchronize()
    return nstart, sl, 1e3 * (time.perf_counter() - t0) / n


print("n_dir  starts   blind_rate   ms/frame")
for n_dir in [2200, 1600, 1100, 700, 400, 200]:
    ns, sl, ms = run(n_dir)
    print(f"{n_dir:5d}  {ns:6d}   {sl}/{n} = {100*sl//n:2d}%    {ms:.1f}", flush=True)
