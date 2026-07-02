"""Does RAAR's single-frame edge (+4 blind, +3 support) become a HYBRID win where the consensus net is
THIN? On the full 120 the net absorbs it (117 vs 116). Here we shrink N and, over K random subsets (PAIRED:
same subsets for both refiners), measure how often the CORRECT consensus cell is derived (Mc ~ LYSO) and
the indexed rate -- GD-8 vs RAAR-16. Tell: does RAAR's advantage grow as N falls (thin net = few-frame x
hard, where consensus is documented to pay most)?"""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf
import glint.glint_index as gi
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)
rng = np.random.default_rng(0)
K = 12                                                              # random subsets per N (paired GD/RAAR)


def eval_cfg(subs, refiner, steps):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = 0.7
    hybrid_index(subs[0], nbest=3, warmup=True)                     # warm this config
    corr = []; idx = []
    for s in subs:
        _, st = hybrid_index(s, nbest=3, warmup=False)
        Mc = st["Mc"]
        corr.append(1.0 if (Mc is not None and same_lattice(Mc, gf.LYSO)) else 0.0)
        idx.append(st["n_idx"] / len(s))
    return 100 * np.mean(corr), 100 * np.mean(idx)


print(" N    correct-cell %%        indexed/N %%       (mean of K=%d paired subsets)" % K)
print("        GD    RAAR   d        GD    RAAR   d")
for N in [8, 12, 16, 24, 40]:
    subs = [[frames[i] for i in rng.choice(n, N, replace=False)] for _ in range(K)]
    gc, gi_ = eval_cfg(subs, "grad", 8)
    rc, ri = eval_cfg(subs, "raar", 16)
    print(f"{N:3d}   {gc:5.0f} {rc:5.0f} {rc-gc:+5.0f}    {gi_:5.1f} {ri:5.1f} {ri-gi_:+5.1f}", flush=True)
