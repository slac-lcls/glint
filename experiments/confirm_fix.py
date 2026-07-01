"""Confirm the new cluster-FFT defaults (fov=200, n_grid=96) through the PRODUCTION entry
index_blind_cluster_seeded: all 6 corroboration cells + the lyso density sweep, no regression."""
import os, sys, time
sys.path.insert(0, "/pscratch/sd/s/smarches/glint_real"); os.environ.setdefault("STEPS", "8")
import numpy as np
from glint.glint_fast import index_blind_cluster_seeded, _cluster_fft_seeds

R = "/pscratch/sd/s/smarches/glint_real"


def ok(M, axes):
    if M is None:
        return False
    L = np.sort(np.linalg.norm(np.asarray(M, float), axis=0))
    return bool(np.all(np.abs(L - axes) <= 0.05 * axes))


print("cluster-FFT defaults (n_clusters,cpts,n_grid,fov,...):", _cluster_fft_seeds.__defaults__)
D = np.load(f"{R}/cells.npz"); names = [str(x) for x in D["names"]]; REPS = int(D["reps"])
index_blind_cluster_seeded(D["lyso_tet_0"])  # warmup

print("\n=== 6 cells (production entry, new defaults) ===")
for n in names:
    axes = np.sort(D[f"{n}_axes"]); oks = 0; ts = []
    for r in range(REPS):
        q = D[f"{n}_{r}"]; t0 = time.perf_counter(); M = index_blind_cluster_seeded(q)
        ts.append(time.perf_counter() - t0); oks += ok(M, axes)
    print(f"  {n:9s} {100*oks//REPS:3d}%  {1e3*np.median(ts):5.0f}ms", flush=True)

C = np.load(f"{R}/clouds.npz"); regimes = [str(x) for x in C["regimes"]]; CR = int(C["reps"])
axes = np.sort(C["cell"])
print(f"\n=== lyso density sweep (production entry) truth {axes} ===")
for g in regimes:
    oks = 0; ts = []; ns = []
    for r in range(CR):
        q = C[f"{g}_{r}"]; ns.append(len(q)); t0 = time.perf_counter(); M = index_blind_cluster_seeded(q)
        ts.append(time.perf_counter() - t0); oks += ok(M, axes)
    print(f"  {g:9s} n~{int(np.median(ns)):6d}  {100*oks//CR:3d}%  {1e3*np.median(ts):5.0f}ms", flush=True)
