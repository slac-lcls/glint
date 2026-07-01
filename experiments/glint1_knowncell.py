"""GLINT-(1) GIVEN the cell (hybrid_index(Mc_known=LYSO)) at the SAME strict Table-1 gate as every
other row (correct lattice + >=25% spots + >=10 refl) -- the fair 'GLINT known-cell' number, vs the
bare rescue engine (index_known_gpu_cell) 60%. No cross-frame consensus is used when the cell is given,
so this is still per-frame comparable to ffbidx/xgandalf."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import fftindex.glint_fast as gf
from fftindex.replica_gpu import index_known_gpu_cell
from fftindex.hybrid_stream import hybrid_index
from fftindex.multishot import same_lattice
LYSO = gf.LYSO
frames = list(gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt"))
n = len(frames)


def gate(M, q):
    if M is None or not same_lattice(M, LYSO):
        return (0, 0)
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return (int(m / len(q) >= 0.25), int(m >= 10))


# bare rescue engine (the current row)
index_known_gpu_cell(frames[0], LYSO)
rr = [gate(index_known_gpu_cell(q, LYSO), q) for q in frames]
r25 = sum(a for a, _ in rr); r10 = sum(b for _, b in rr)

# GLINT-(1) given the cell: full hybrid with Mc_known=LYSO, timed
t0 = time.perf_counter()
results, stats = hybrid_index(frames, Mc_known=LYSO, warmup=True)
ms = 1e3 * (time.perf_counter() - t0) / n
hh = [gate(res["M"], q) for res, q in zip(results, frames)]
h25 = sum(a for a, _ in hh); h10 = sum(b for _, b in hh)

print(f"bare rescue engine  (index_known_gpu_cell): >=25% {r25}/{n}={100*r25//n}%  >=10refl {r10}/{n}={100*r10//n}%")
print(f"GLINT-(1) KNOWN-CELL (hybrid Mc_known=LYSO): >=25% {h25}/{n}={100*h25//n}%  >=10refl {h10}/{n}={100*h10//n}%  {ms:.1f} ms/frame")
print(f"  (hybrid n_idx cell-consistent = {stats['n_idx']}/{n}; n_resc {stats['n_resc']}, n_nbest {stats.get('n_nbest')})")
