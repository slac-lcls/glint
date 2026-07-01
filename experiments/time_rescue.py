"""Task (a): time the POLISHED known-cell rescue on the 120 cxidb frames -> gated rate (confirm 60%) + ms/frame."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import fftindex.glint_fast as gf
from fftindex.replica_gpu import index_known_gpu_cell
from fftindex.multishot import same_lattice
LYSO = gf.LYSO
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)
def gate(M, q):
    if M is None or not same_lattice(M, LYSO): return (0, 0)
    r = np.asarray(q) @ M; m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return (int(m/len(q) >= 0.25), int(m >= 10))
index_known_gpu_cell(frames[0], LYSO)              # warmup
t = []; g25 = g10 = 0
for q in frames:
    t0 = time.perf_counter(); M = index_known_gpu_cell(q, LYSO); t.append(time.perf_counter()-t0)
    a, b = gate(M, q); g25 += a; g10 += b
print(f"POLISHED known-cell rescue N={n}: >=25%(Table1) {g25}/{n}={100*g25//n}%  >=10refl {g10}/{n}={100*g10//n}%  "
      f"mean {1e3*np.mean(t):.1f} ms/frame  ({1000/(1e3*np.mean(t)):.0f} f/s)  median {1e3*np.median(t):.1f} ms")
