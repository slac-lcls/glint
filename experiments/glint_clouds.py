"""GLINT side of the head-to-head: index the shared clouds with the cluster-FFT front-end (GPU).
Reports rate + median wall-time per regime; identical clouds are indexed by DIALS/labelit separately.

  module load pytorch/2.6.0 ; srun ... --gpus 1 python glint_clouds.py
"""
import os, sys, time
sys.path.insert(0, "/pscratch/sd/s/smarches/glint_real"); os.environ.setdefault("STEPS", "8")
import numpy as np
from fftindex.glint_fast import index_blind_cluster_seeded

D = np.load("/pscratch/sd/s/smarches/glint_real/clouds.npz")
CELL = np.sort(D["cell"]); REPS = int(D["reps"]); regimes = [str(x) for x in D["regimes"]]


def ok(M):
    if M is None:
        return False
    L = np.sort(np.linalg.norm(np.asarray(M, float), axis=0))
    return bool(np.all(np.abs(L - CELL) <= 0.05 * CELL))


index_blind_cluster_seeded(D[f"{regimes[0]}_0"])            # warmup
print(f"GLINT cluster-FFT (GPU)   cell {CELL.tolist()}")
print(f"{'regime':10}{'n_rlps':>8}{'rate':>7}{'ms':>9}")
for g in regimes:
    ts, oks, ns = [], 0, []
    for r in range(REPS):
        q = D[f"{g}_{r}"]; ns.append(len(q))
        t0 = time.perf_counter(); M = index_blind_cluster_seeded(q); ts.append(time.perf_counter() - t0)
        oks += ok(M)
    print(f"  {g:8}{int(np.median(ns)):>8}{100*oks//REPS:>6}%{1e3*np.median(ts):>9.1f}", flush=True)
