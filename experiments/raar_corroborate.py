"""Corroborate the RAAR blind win beyond cxidb-120: same LYSO gate on a 2nd sparse dataset. GD-8 vs
RAAR-16, blind + hybrid. Usage: python raar_corroborate.py [frames_file]"""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf, glint.glint_index as gi
from glint.glint_fast import index_blind_fast
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice
path = sys.argv[1] if len(sys.argv) > 1 else "frames_dials60.txt"
frames = [q for q in gf.load(path) if len(q) >= 6]; n = len(frames); LYSO = gf.LYSO


def blind(refiner, steps):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = 0.7
    index_blind_fast(frames[0]); torch.cuda.synchronize()
    return sum(1 for q in frames if (lambda M: M is not None and same_lattice(M, LYSO))(index_blind_fast(q)))


def hyb(refiner, steps):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = 0.7
    _, st = hybrid_index(frames, nbest=3, warmup=False)
    return st["n_idx"], st["support"]


print(f"{path}: {n} frames  (LYSO gate)")
for r, s in [("grad", 8), ("raar", 16)]:
    b = blind(r, s); h, sup = hyb(r, s)
    print(f"  {r:4s}-{s:2d}   blind {b}/{n} = {100*b//n:2d}%    hybrid {h}/{n}   support {sup}", flush=True)
