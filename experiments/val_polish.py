"""Validate the gate-matched polish in index_known_gpu_cell: standalone rescue >=25% should rise
(was 55%/58%), lattice + >=10refl held, hybrid unchanged."""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf
from glint.replica_gpu import index_known_gpu_cell
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames); LYSO = gf.LYSO


def stats(M, q):
    if M is None or not same_lattice(M, LYSO):
        return (0, 0, 0)
    r = np.asarray(q) @ M; r = r - np.rint(r); m = int((np.abs(r).max(1) < 0.15).sum())
    return (1, int(m / len(q) >= 0.25), int(m >= 10))


index_known_gpu_cell(frames[0], LYSO)
S = np.array([stats(index_known_gpu_cell(q, LYSO), q) for q in frames])
print(f"POLISHED standalone rescue: lattice {S[:,0].sum()}/{n}  >=25% {S[:,1].sum()}/{n} = {100*S[:,1].sum()//n}%  "
      f">=10refl {S[:,2].sum()}/{n} = {100*S[:,2].sum()//n}%   (was lattice 95% / >=25% 55% / >=10refl 93%)")
results, st = hybrid_index(frames, nbest=3)
print(f"HYBRID n_idx={st['n_idx']}/{st['n']}  support={st['support']}  edges={st['edges']}  "
      f"n_resc={st['n_resc']}  (expected 116/120, unchanged)")
