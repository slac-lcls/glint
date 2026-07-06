"""Test the annealed round-to-hkl in refine_vec_raar (soft->hard schedule) against the ~71% wall.
Blind + hybrid on cxidb-120, LYSO gate. Conditions: grad-8 (default ref), raar-16 (baseline), and
raar-16 with RAAR_ANNEAL over a small (a0,tolk) sweep. If the wall is spurious-sublattice-lock, a
soft-then-hard assignment should let RAAR escape before committing -> a higher blind rate.
  module load pytorch/2.6.0 ; srun ... python anneal_test.py [frames_file]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")   # dir containing glint/
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf, glint.glint_index as gi
from glint.glint_fast import index_blind_fast
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice

path = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)) + "/frames_cxidb_clean.txt"
frames = [q for q in gf.load(path) if len(q) >= 6]; n = len(frames); LYSO = gf.LYSO
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{os.path.basename(path)}: {n} frames  device={dev}  (LYSO gate)", flush=True)


def _set(refiner, steps, anneal, a0, tolk):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = 0.7
    gi.RAAR_ANNEAL = anneal; gi.RAAR_A0 = a0; gi.RAAR_TOLK = tolk


def blind(refiner, steps, anneal=0, a0=0.3, tolk=2.0):
    _set(refiner, steps, anneal, a0, tolk)
    index_blind_fast(frames[0])                                   # warmup/compile
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    b = sum(1 for q in frames if (lambda M: M is not None and same_lattice(M, LYSO))(index_blind_fast(q)))
    if dev == "cuda": torch.cuda.synchronize()
    return b, time.time() - t0


def hyb(refiner, steps, anneal=0, a0=0.3, tolk=2.0):
    _set(refiner, steps, anneal, a0, tolk)
    _, st = hybrid_index(frames, nbest=3, warmup=False)
    return st["n_idx"], st["support"]


conds = [
    ("grad", 8, 0, 0.3, 2.0, "default ref"),
    ("raar", 16, 0, 0.3, 2.0, "RAAR baseline (hard round)"),
    ("raar", 16, 1, 0.3, 2.0, "anneal a0=.3 tolk=2"),
    ("raar", 16, 1, 0.0, 3.0, "anneal a0=.0 tolk=3"),
    ("raar", 16, 1, 0.5, 2.0, "anneal a0=.5 tolk=2"),
    ("raar", 16, 1, 0.2, 1.0, "anneal a0=.2 tolk=1 (round-only)"),
]
print(f"{'cond':34s} {'blind':>10s} {'hybrid':>9s} {'support':>8s}  {'s':>5s}", flush=True)
for r, s, a, a0, tk, lab in conds:
    b, dt = blind(r, s, a, a0, tk)
    h, sup = hyb(r, s, a, a0, tk)
    print(f"{lab:34s} {b:3d}/{n} {100*b//n:2d}% {h:4d}/{n}   {sup:4d}    {dt:5.1f}", flush=True)
