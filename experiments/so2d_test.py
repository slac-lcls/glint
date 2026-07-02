"""SO2D saddle-point indexer vs the RAAR winner vs momentum-GD on the 120 blind cxidb frames. SO2D chooses
the RAAR feedback per seed per step by an analytic solve of the L=e_d^2 - w_s e_s^2 saddle. Question: does
the adaptive saddle step beat fixed-beta RAAR (88/120) and GD (84/120)? Sweeps steps and the support weight
SO_WS. Includes a RAAR-16 repeat to gauge the shared-node noise on the +4. Same frames + gate as raar_test."""
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


def run(refiner, steps, beta=0.7, ws=1.0):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = beta; gi.SO_WS = ws
    index_blind_fast(frames[0]); torch.cuda.synchronize()          # warmup (per config)
    t0 = time.perf_counter(); sl = 0
    for q in frames:
        M = index_blind_fast(q); sl += (M is not None and same_lattice(M, gf.LYSO))
    torch.cuda.synchronize()
    return sl, 1e3 * (time.perf_counter() - t0) / n


print("refiner steps  ws    blind_rate   ms/frame")
cfgs = [("grad", 8, {}),
        ("raar", 16, {"beta": 0.7}),           # the winner (reference)
        ("raar", 16, {"beta": 0.7}),           # repeat -> noise on the +4
        ("so2d", 8, {"ws": 1.0}),
        ("so2d", 16, {"ws": 1.0}),
        ("so2d", 16, {"ws": 0.5}),
        ("so2d", 16, {"ws": 2.0}),
        ("so2d", 30, {"ws": 1.0})]
for r, s, kw in cfgs:
    sl, ms = run(r, s, beta=kw.get("beta", 0.7), ws=kw.get("ws", 1.0))
    tag = kw.get("ws", "-") if r == "so2d" else ("b%.1f" % kw["beta"] if r == "raar" else "-")
    print(f"  {r:4s}  {s:2d}  {str(tag):>4}   {sl}/{n} = {100*sl//n:2d}%    {ms:.1f}", flush=True)
