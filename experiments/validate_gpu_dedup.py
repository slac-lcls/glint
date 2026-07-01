"""Validate the on-device M2 dedup + M4 build: (1) distinct_maxima_gpu == distinct_maxima (same
inputs), (2) full blind rate unchanged (84/120), (3) re-profile the new on-device stages."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_index import (objective, refine_vec, distinct_maxima, distinct_maxima_gpu,
                                  invq_weight, buerger_reduce, primitivize, STARTS)
from glint.glint_fast import anneal_batch_t, score_batch_t, index_blind_fast
from glint.multishot import same_lattice
DEV = gf.DEV


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
index_blind_fast(frames[0])                                     # warmup
n = len(frames)

# 1) correctness: distinct_maxima_gpu vs distinct_maxima on identical T,f
mism = 0
for q in frames:
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    T = refine_vec(STARTS.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL)
    f, _ = objective(T, Q, w, sharp=True, tol=gf.TOL)
    old = distinct_maxima(T.cpu().numpy(), f.cpu().numpy(), keep=gf.KEEP)[:gf.NTOP]
    new = distinct_maxima_gpu(T, f, keep=gf.KEEP)[:gf.NTOP].cpu().numpy()
    so = set(map(tuple, np.round(old, 2))); sn = set(map(tuple, np.round(new, 2)))
    mism += (len(so ^ sn) > 0)
print(f"(1) cands set-mismatch frames: {mism}/{n}  (0 = exact match)")

# 2) full blind rate unchanged
sl = 0; sync(); t0 = time.perf_counter()
for q in frames:
    M = index_blind_fast(q); sl += (M is not None and same_lattice(M, gf.LYSO))
sync(); tot = time.perf_counter() - t0
print(f"(2) NEW blind-solved (LYSO) {sl}/{n} = {100*sl//n}%   {1e3*tot/n:.1f} ms/frame  (was 84/120=70%, 29.6ms)")

# 3) re-profile new on-device stages
acc = dict(m1_seed=0.0, m3_refine=0.0, m2_obj=0.0, m4_build=0.0, m5_anneal=0.0, m6_score=0.0)


def prof(q):
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    sync(); t = time.perf_counter(); S0 = STARTS.clone(); sync(); acc['m1_seed'] += time.perf_counter() - t
    t = time.perf_counter(); T = refine_vec(S0, Q, w, qmax, steps=gf.STEPS, tol=gf.TOL); sync(); acc['m3_refine'] += time.perf_counter() - t
    t = time.perf_counter(); f, _ = objective(T, Q, w, sharp=True, tol=gf.TOL); cands = distinct_maxima_gpu(T, f, keep=gf.KEEP)[:gf.NTOP]; sync(); acc['m2_obj'] += time.perf_counter() - t
    if int(cands.shape[0]) < 3:
        return
    t = time.perf_counter(); Qd = Q.double(); cd = cands.double()
    tri = torch.combinations(torch.arange(cd.shape[0], device=DEV), 3); M0 = cd[tri].permute(0, 2, 1)
    nrm = cd.norm(dim=1); sc = nrm[tri].prod(1); det = torch.linalg.det(M0).abs(); M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    sync(); acc['m4_build'] += time.perf_counter() - t
    if int(M0.shape[0]) == 0:
        return
    t = time.perf_counter(); Mt = anneal_batch_t(M0, Qd); sync(); acc['m5_anneal'] += time.perf_counter() - t
    t = time.perf_counter(); key, ni = score_batch_t(Mt, Qd); b = int(torch.argmax(key).item())
    _ = None if float(key[b]) <= -1e8 else primitivize(buerger_reduce(Mt[b].cpu().numpy()), np.asarray(q, float))
    sync(); acc['m6_score'] += time.perf_counter() - t


prof(frames[0])
for k in acc:
    acc[k] = 0.0
for q in frames:
    prof(q)
ts = sum(acc.values())
print("(3) NEW per-stage (on-device M2 dedup + M4 build):")
for k in ['m1_seed', 'm3_refine', 'm2_obj', 'm4_build', 'm5_anneal', 'm6_score']:
    print(f"    {k:10s} {1e3*acc[k]/n:7.2f} ms/f  {100*acc[k]/ts:5.1f}%")
print(f"    TOTAL stage {1e3*ts/n:.1f} ms/f  (was 28.7)")
