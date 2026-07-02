"""Firmer small-N: N in {5,6,8,10,14}, K=40 paired random subsets -- nail whether RAAR derives the correct
consensus cell more reliably than GD in the extreme thin-net regime (the +8 seen at N=8, K=12)."""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf, glint.glint_index as gi
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames); rng = np.random.default_rng(0); K = 40


def evalc(subs, refiner, steps):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = 0.7
    hybrid_index(subs[0], nbest=3, warmup=True)
    c = []
    for s in subs:
        _, st = hybrid_index(s, nbest=3, warmup=False)
        Mc = st["Mc"]; c.append(1.0 if (Mc is not None and same_lattice(Mc, gf.LYSO)) else 0.0)
    return 100 * np.mean(c)


print(" N   correct-cell %% (K=%d)   GD   RAAR    d" % K)
for N in [5, 6, 8, 10, 14]:
    subs = [[frames[i] for i in rng.choice(n, N, replace=False)] for _ in range(K)]
    g = evalc(subs, "grad", 8); r = evalc(subs, "raar", 16)
    print(f"{N:3d}                        {g:5.0f} {r:5.0f}  {r-g:+5.0f}", flush=True)
