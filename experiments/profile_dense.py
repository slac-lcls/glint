"""Profile the DENSE (local-cluster 3D-FFT) GLINT path per-stage (M1-M6), the dense analogue of
profile_glint.py. M1 here is the cluster-FFT seed generation (not the Fibonacci grid); M3-M6 are the
same shared core. Self-contained: generates the 10-cell dense rotation clouds inline (gen_cells.py
logic). Run on a GPU: srun -p ampere ... python experiments/profile_dense.py"""
import os, sys, time, itertools
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_index import (objective, refine_vec, distinct_maxima, invq_weight,
                               buerger_reduce, primitivize)
from glint.glint_fast import anneal_batch_t, score_batch_t
DEV = gf.DEV


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


# ---- inline dense-cloud generator (from gen_cells.py) ----
DMIN = 3.0; QMAX = 1.0 / DMIN; REPS = 2
CELLS = [("lyso_tet",(79.,79.,38.,90,90,90)), ("prok_tet",(68.5,68.5,109.,90,90,90)),
         ("thaum_tet",(58.,58.,130.,90,90,90)), ("hex_P6",(105.,105.,75.,90,90,120)),
         ("cubic_ins",(78.,78.,78.,90,90,90)), ("ortho_lg",(60.,110.,135.,90,90,90)),
         ("tricl",(45.,55.,65.,80,85,95)), ("mono",(60.,70.,90.,90,105,90)),
         ("large_tet",(140.,140.,150.,90,90,90)), ("trig_ob",(70.,70.,70.,100,100,100))]


def cell_to_B(a,b,c,al,be,ga):
    al,be,ga = np.radians([al,be,ga])
    va=[a,0,0]; vb=[b*np.cos(ga),b*np.sin(ga),0]
    cx=c*np.cos(be); cy=c*(np.cos(al)-np.cos(be)*np.cos(ga))/np.sin(ga)
    vc=[cx,cy,np.sqrt(max(c*c-cx*cx-cy*cy,1e-9))]
    return np.linalg.inv(np.array([va,vb,vc]).T).T


def rand_rot(r):
    u1,u2,u3=r.random(3)
    q=np.array([np.sqrt(1-u1)*np.sin(2*np.pi*u2),np.sqrt(1-u1)*np.cos(2*np.pi*u2),
                np.sqrt(u1)*np.sin(2*np.pi*u3),np.sqrt(u1)*np.cos(2*np.pi*u3)])
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def gen_clouds():
    rng = np.random.default_rng(0); clouds = []
    for name, cp in CELLS:
        B = cell_to_B(*cp)
        H = (np.ceil(QMAX/np.linalg.norm(B,axis=0))+1).astype(int)
        g = np.mgrid[-H[0]:H[0]+1,-H[1]:H[1]+1,-H[2]:H[2]+1].reshape(3,-1).T
        g = g[np.any(g!=0,axis=1)]
        for r in range(REPS):
            A = rand_rot(rng)@B; q = g@A.T
            clouds.append((name, q[np.linalg.norm(q,axis=1) <= QMAX].astype(np.float64)))
    return clouds


acc = dict(m1_fft=0.0, m3_refine=0.0, m2_obj=0.0, m4_build=0.0, m5_anneal=0.0, m6_score=0.0)


def index_dense_profiled(q, acc):
    """Mirror of profile_glint.index_profiled, but M1 = cluster-FFT seeds (dense front end)."""
    q = np.asarray(q, float)
    if len(q) < 6:
        return None
    sync(); t = time.perf_counter()                                        # M1: cluster-FFT seeds
    if len(q) > gf.BIGCELL_RLPS:
        S0 = gf._cluster_fft_seeds(q, fov=280.0, n_grid=128)               # adaptive FOV (large cells)
    else:
        S0 = gf._cluster_fft_seeds(q)
    sync(); acc['m1_fft'] += time.perf_counter() - t
    if int(S0.shape[0]) < 3:
        return None
    if len(q) > gf.FFTSEED_CAP:                                            # cap rlps for the O(N) core
        q = q[np.argsort(np.linalg.norm(q, axis=1))[:gf.FFTSEED_CAP]]
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    t = time.perf_counter()                                                # M3: gradient ascent
    T = refine_vec(S0, Q, w, qmax, steps=gf.STEPS, tol=gf.TOL)
    sync(); acc['m3_refine'] += time.perf_counter() - t; t = time.perf_counter()
    f, _ = objective(T, Q, w, sharp=True, tol=gf.TOL)                      # M2: objective + distinct
    cands = distinct_maxima(T.cpu().numpy(), f.cpu().numpy(), keep=gf.KEEP)[:gf.NTOP]
    sync(); acc['m2_obj'] += time.perf_counter() - t
    if len(cands) < 3:
        return None
    t = time.perf_counter()                                                # M4: triplet build
    nrm = np.linalg.norm(cands, axis=1)
    tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
    M0 = np.transpose(cands[tris], (0, 2, 1)); sc = nrm[tris].prod(1); det = np.abs(np.linalg.det(M0))
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    sync(); acc['m4_build'] += time.perf_counter() - t
    if len(M0) == 0:
        return None
    Qd = Q.double(); M0t = torch.as_tensor(M0, dtype=torch.float64, device=DEV)
    t = time.perf_counter(); Mt = anneal_batch_t(M0t, Qd, max_iter=gf.ANNEAL_ITERS); sync()          # M5: batched anneal
    acc['m5_anneal'] += time.perf_counter() - t; t = time.perf_counter()
    key, ni = score_batch_t(Mt, Qd); b = int(torch.argmax(key).item())     # M6: score + reduce
    cell = None if float(key[b]) <= -1e8 else primitivize(buerger_reduce(Mt[b].cpu().numpy()), q)
    sync(); acc['m6_score'] += time.perf_counter() - t
    return cell


clouds = gen_clouds()
index_dense_profiled(clouds[0][1], acc)                                    # warmup
for k in acc:
    acc[k] = 0.0
sync(); t0 = time.perf_counter(); n = 0
for name, q in clouds:
    sync(); ts = time.perf_counter()
    index_dense_profiled(q, acc); n += 1
    sync()
tot = time.perf_counter() - t0
tot_stage = sum(acc.values())
print(f"=== GLINT DENSE profile, {n} clouds ({len(CELLS)} cells x{REPS}), device={DEV}, STEPS={gf.STEPS} ===")
print(f"DENSE total {1e3*tot/n:.1f} ms/cloud ({n/tot:.1f} clouds/s)")
print("--- per-stage (M1 cluster-FFT + M3-M6 core) ---")
for k in ['m1_fft', 'm3_refine', 'm2_obj', 'm4_build', 'm5_anneal', 'm6_score']:
    print(f"  {k:10s} {1e3*acc[k]/n:7.2f} ms  {100*acc[k]/tot_stage:5.1f}%")
