"""GLINT cluster-FFT across a range of cells (corroboration). Same clouds indexed by dials_cells.py."""
import os, sys, time
sys.path.insert(0, "/pscratch/sd/s/smarches/glint_real"); os.environ.setdefault("STEPS", "8")
import numpy as np
from glint.glint_fast import index_blind_cluster_seeded

D = np.load("/pscratch/sd/s/smarches/glint_real/cells.npz")
names = [str(x) for x in D["names"]]; REPS = int(D["reps"])


def ok(M, axes):
    if M is None:
        return False
    L = np.sort(np.linalg.norm(np.asarray(M, float), axis=0))
    return bool(np.all(np.abs(L - axes) <= 0.05 * axes))


index_blind_cluster_seeded(D["%s_0" % names[0]])           # warmup
print("GLINT cluster-FFT (GPU)")
print("%-11s%8s%7s%9s" % ("cell", "n_rlps", "rate", "ms"))
for n in names:
    axes = D["%s_axes" % n]; ts, oks, ns = [], 0, []
    for r in range(REPS):
        q = D["%s_%d" % (n, r)]; ns.append(len(q))
        t0 = time.perf_counter(); M = index_blind_cluster_seeded(q); ts.append(time.perf_counter() - t0)
        oks += ok(M, axes)
    print("  %-9s%8d%6d%%%9.1f" % (n, int(np.median(ns)), 100 * oks // REPS, 1e3 * np.median(ts)), flush=True)
