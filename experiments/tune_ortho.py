"""Tune the cluster-FFT front end on the large-anisotropic ortho cell (25% vs DPS 50%).
Diagnose generation (do seeds cover the 3 truth axes?) vs end-to-end, sweep grid/FOV/clusters,
then regression-check the best config across all 6 cells. GPU (pytorch/2.6.0)."""
import os, sys, time
sys.path.insert(0, "/pscratch/sd/s/smarches/glint_real")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf

D = np.load("/pscratch/sd/s/smarches/glint_real/cells.npz")
names = [str(x) for x in D["names"]]; REPS = int(D["reps"])


def cell_ok(M, axes):
    if M is None:
        return False
    L = np.sort(np.linalg.norm(np.asarray(M, float), axis=0))
    return bool(np.all(np.abs(L - axes) <= 0.05 * axes))


def seeds_cover(s, axes, tol=0.05):
    if int(s.shape[0]) == 0:
        return False
    L = s.norm(dim=1).cpu().numpy()
    return all(np.any(np.abs(L - a) <= tol * a) for a in axes)


def run_cfg(name, kw, cells):
    for cn in cells:
        axes = np.sort(D[f"{cn}_axes"]); gen = e2e = 0; ts = []; nsd = []
        for r in range(REPS):
            q = D[f"{cn}_{r}"]
            t0 = time.perf_counter()
            s = gf._cluster_fft_seeds(q, **kw)
            qq = q
            if len(qq) > gf.FFTSEED_CAP:
                qq = qq[np.argsort(np.linalg.norm(qq, axis=1))[:gf.FFTSEED_CAP]]
            M = gf.index_blind_fast(qq, starts=s)
            ts.append(time.perf_counter() - t0)
            nsd.append(int(s.shape[0])); gen += seeds_cover(s, axes); e2e += cell_ok(M, axes)
        print(f"{name:26s} {cn:9s} gen {gen}/{REPS} e2e {e2e}/{REPS} "
              f"seeds~{int(np.median(nsd))} {1e3*np.median(ts):.0f}ms", flush=True)


gf._cluster_fft_seeds(D["lyso_tet_0"]); gf.index_blind_fast(D["lyso_tet_0"][:700])  # warmup
print("truth ortho axes:", np.sort(D["ortho_lg_axes"]))
print("=== ORTHO sweep (generation vs end-to-end) ===")
cfgs = {
    "default(f160,g64,28,300)": dict(),
    "g96": dict(n_grid=96),
    "g128": dict(n_grid=128),
    "f200_g96": dict(fov=200.0, n_grid=96),
    "f220_g128": dict(fov=220.0, n_grid=128),
    "f200_g128_48c_500p": dict(fov=200.0, n_grid=128, n_clusters=48, cpts=500),
    "f220_g128_48c_500p": dict(fov=220.0, n_grid=128, n_clusters=48, cpts=500),
}
for n, kw in cfgs.items():
    run_cfg(n, kw, ("ortho_lg",))

print("\n=== full 6-cell regression @ f200_g128_48c_500p ===")
best = dict(fov=200.0, n_grid=128, n_clusters=48, cpts=500)
run_cfg("best", best, tuple(names))
