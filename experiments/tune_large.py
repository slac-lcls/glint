"""Characterize the large_tet (140/140/150, 456k rlps) 50% failure: is it the fixed cluster FOV
(recoverable with a bigger grid) or deeper? Sweep (fov, n_grid) via the seed generator + full index."""
import os, sys, time
sys.path.insert(0, "/pscratch/sd/s/smarches/glint_real"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf

D = np.load("/pscratch/sd/s/smarches/glint_real/cells.npz")
name = "large_tet"; axes = np.sort(D[f"{name}_axes"]); REPS = int(D["reps"])
print("truth axes", axes)


def cell_ok(M, axes):
    if M is None:
        return False
    L = np.sort(np.linalg.norm(np.asarray(M, float), axis=0))
    return bool(np.all(np.abs(L - axes) <= 0.05 * axes))


def seeds_cover(s, axes, tol=0.06):
    if int(s.shape[0]) == 0:
        return False
    L = s.norm(dim=1).cpu().numpy()
    return all(np.any(np.abs(L - a) <= tol * a) for a in axes)


gf._cluster_fft_seeds(D["lyso_tet_0"]); gf.index_blind_fast(D["lyso_tet_0"][:700])  # warmup
for kw in [dict(), dict(fov=240.0, n_grid=128), dict(fov=280.0, n_grid=128), dict(fov=320.0, n_grid=160)]:
    gen = e2e = 0; ts = []
    for r in range(REPS):
        q = D[f"{name}_{r}"]
        t0 = time.perf_counter()
        s = gf._cluster_fft_seeds(q, **kw)
        qq = q[np.argsort(np.linalg.norm(q, axis=1))[:gf.FFTSEED_CAP]] if len(q) > gf.FFTSEED_CAP else q
        M = gf.index_blind_fast(qq, starts=s)
        ts.append(time.perf_counter() - t0)
        gen += seeds_cover(s, axes); e2e += cell_ok(M, axes)
    tag = "default(f200,g96)" if not kw else f"f{int(kw['fov'])},g{kw['n_grid']}"
    print(f"  {tag:16s} gen {gen}/{REPS}  e2e {e2e}/{REPS}  {1e3*np.median(ts):.0f}ms", flush=True)
