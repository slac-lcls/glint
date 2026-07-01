"""GLINT on the DIALS-60 rich stills, same gate: blind-fast, GLINT-(1) known-cell, GLINT-(1) blind hybrid."""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint"); sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np
import glint.glint_fast as gf
from glint.glint_fast import index_blind_fast
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice
LYSO = gf.LYSO
frames = list(gf.load(sys.argv[1] if len(sys.argv) > 1 else "frames_dials60.txt")); n = len(frames)
def gate(M, q):
    if M is None or not same_lattice(M, LYSO): return (0, 0)
    r = np.asarray(q) @ M; m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return (int(m / len(q) >= 0.25), int(m >= 10))
index_blind_fast(frames[0])
gb = [gate(index_blind_fast(q), q) for q in frames]
res, st = hybrid_index(frames, Mc_known=LYSO, warmup=True)
hh = [gate(r["M"], q) for r, q in zip(res, frames)]
res2, st2 = hybrid_index(frames, warmup=False)
hb = [gate(r["M"], q) for r, q in zip(res2, frames)]
S = lambda xs, i: sum(x[i] for x in xs)
print("DIALS-60 GLINT blind-fast      >=25%% %d/%d=%d%%  >=10refl %d/%d=%d%%" % (S(gb,0),n,100*S(gb,0)//n,S(gb,1),n,100*S(gb,1)//n))
print("DIALS-60 GLINT-1 known-cell    >=25%% %d/%d=%d%%  >=10refl %d/%d=%d%%" % (S(hh,0),n,100*S(hh,0)//n,S(hh,1),n,100*S(hh,1)//n))
print("DIALS-60 GLINT-1 blind-hybrid  >=25%% %d/%d=%d%%  >=10refl %d/%d=%d%%  support %s" % (S(hb,0),n,100*S(hb,0)//n,S(hb,1),n,100*S(hb,1)//n,st2['support']))
