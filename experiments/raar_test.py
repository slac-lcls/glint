"""RAAR indexer (indexing cast as phase retrieval) vs momentum-GD on the 120 blind cxidb frames.
P_data = round inlier projections to integer Miller indices; P_support = least-squares refit onto range(Q).
RAAR adds HIO-style reflection FEEDBACK -- the question is whether it escapes the spurious basins that wall
GD at 84/120. Sweeps beta (feedback) and step count (RAAR usually needs more iterations than GD's 8). The
tell: any (beta, steps) that beats 84. Same frames + gate as cg_test/ls_test."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf
import glint.glint_index as gi
from glint.glint_fast import index_blind_fast
from glint.multishot import same_lattice
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)


def run(refiner, steps, beta=0.7):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = beta
    index_blind_fast(frames[0]); torch.cuda.synchronize()          # warmup (per config)
    t0 = time.perf_counter(); sl = 0
    for q in frames:
        M = index_blind_fast(q); sl += (M is not None and same_lattice(M, gf.LYSO))
    torch.cuda.synchronize()
    return sl, 1e3 * (time.perf_counter() - t0) / n


print("refiner steps  beta   blind_rate   ms/frame")
cfgs = [("grad", 8, 0.0),
        ("raar", 8, 0.5), ("raar", 8, 0.7), ("raar", 8, 0.9),
        ("raar", 16, 0.7), ("raar", 30, 0.7), ("raar", 30, 0.9),
        ("raar", 16, 0.5)]
for r, s, b in cfgs:
    sl, ms = run(r, s, beta=b)
    print(f"  {r:4s}  {s:2d}   {b:.1f}    {sl}/{n} = {100*sl//n:2d}%    {ms:.1f}", flush=True)
