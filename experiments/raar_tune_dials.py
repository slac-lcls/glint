"""Is the RAAR rich-data regression tunable away, or is exploration<->easy-data a genuine tension? Sweep
gentler RAAR (fewer iters / lower beta) on dials60. The cxidb win needed beta0.7/16; does any setting tie
GD's 56/60 on rich data while still helping sparse?"""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf, glint.glint_index as gi
from glint.glint_fast import index_blind_fast
from glint.multishot import same_lattice
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_dials60.txt") if len(q) >= 6]
n = len(frames); LYSO = gf.LYSO


def blind(refiner, steps, beta=0.7):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = beta
    index_blind_fast(frames[0]); torch.cuda.synchronize()
    return sum(1 for q in frames if (lambda M: M is not None and same_lattice(M, LYSO))(index_blind_fast(q)))


print("dials60 (rich): config  blind_rate")
for r, s, b in [("grad", 8, 0), ("raar", 8, 0.7), ("raar", 16, 0.7), ("raar", 8, 0.5), ("raar", 8, 0.3), ("raar", 4, 0.5)]:
    x = blind(r, s, b if b else 0.7)
    print(f"  {r}-{s} b{b}   {x}/{n} = {100*x//n}%", flush=True)
