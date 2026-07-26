"""Profile GLINT: per-stage M1-M6 breakdown, blind-vs-rescued split, latency distributions,
consensus cost, and CUDA kernel-launch count (for CUDA-graph / fusion assessment).
Real 120 cxidb lysozyme frames, S3DF A100."""
import os, sys, time, itertools, json
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")                                 # PRODUCTION M3 steps (80 over-iterates)
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_index import (objective, refine_vec, distinct_maxima, invq_weight,
                                   buerger_reduce, primitivize, STARTS)
from glint.glint_fast import anneal_batch_t, score_batch_t
from glint.multishot import same_lattice, consensus_cell
from glint.replica_gpu import index_known_gpu_cell
DEV = gf.DEV


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


FR = "/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt"
frames = [q for q in gf.load(FR) if len(q) >= 6]

acc = dict(m1_seed=0.0, m3_refine=0.0, m2_obj=0.0, m4_build=0.0, m5_anneal=0.0, m6_score=0.0)


def index_profiled(q):
    q = np.asarray(q, float)
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    sync(); t = time.perf_counter()
    S0 = STARTS.clone()                                              # M1 seeds
    sync(); acc['m1_seed'] += time.perf_counter() - t; t = time.perf_counter()
    T = refine_vec(S0, Q, w, qmax, steps=gf.STEPS, tol=gf.TOL)       # M3 ascent (dominant?)
    sync(); acc['m3_refine'] += time.perf_counter() - t; t = time.perf_counter()
    f, _ = objective(T, Q, w, sharp=True, tol=gf.TOL)               # M2 objective + pick
    cands = distinct_maxima(T.cpu().numpy(), f.cpu().numpy(), keep=gf.KEEP)[:gf.NTOP]
    sync(); acc['m2_obj'] += time.perf_counter() - t
    if len(cands) < 3:
        return None
    t = time.perf_counter()                                         # M4 triplet build (CPU)
    nrm = np.linalg.norm(cands, axis=1)
    tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
    M0 = np.transpose(cands[tris], (0, 2, 1)); sc = nrm[tris].prod(1); det = np.abs(np.linalg.det(M0))
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    sync(); acc['m4_build'] += time.perf_counter() - t
    if len(M0) == 0:
        return None
    Qd = Q.double(); M0t = torch.as_tensor(M0, dtype=torch.float64, device=DEV)
    t = time.perf_counter(); Mt = anneal_batch_t(M0t, Qd, max_iter=gf.ANNEAL_ITERS); sync()   # M5 batched anneal (GPU)
    acc['m5_anneal'] += time.perf_counter() - t; t = time.perf_counter()
    key, ni = score_batch_t(Mt, Qd); b = int(torch.argmax(key).item())  # M6 score + reduce
    cell = None if float(key[b]) <= -1e8 else primitivize(buerger_reduce(Mt[b].cpu().numpy()), q)
    sync(); acc['m6_score'] += time.perf_counter() - t
    return cell


index_profiled(frames[0])                                           # warmup
for k in acc:
    acc[k] = 0.0
blind_times = []; blind_cells = []; blind_ok = []
sync(); t0 = time.perf_counter()
for q in frames:
    sync(); ts = time.perf_counter()
    M = index_profiled(q)
    sync(); blind_times.append(1e3 * (time.perf_counter() - ts))
    blind_cells.append(M); blind_ok.append(M is not None and same_lattice(M, gf.LYSO))
blind_tot = time.perf_counter() - t0
n = len(frames)

print(f"=== GLINT profile, N={n} real cxidb frames, device={DEV} ===")
print(f"BLIND total {1e3*blind_tot/n:.1f} ms/frame ({n/blind_tot:.1f} f/s)")
tot_stage = sum(acc.values())
print("--- per-stage (M1-M6) ---")
for k in ['m1_seed', 'm3_refine', 'm2_obj', 'm4_build', 'm5_anneal', 'm6_score']:
    print(f"  {k:10s} {1e3*acc[k]/n:7.2f} ms/f  {100*acc[k]/tot_stage:5.1f}%")

# consensus (one-time) + rescue of blind failures
solved = [M for M, ok in zip(blind_cells, blind_ok) if ok]
sync(); tc = time.perf_counter()
res = consensus_cell(solved)
sync(); t_cons = 1e3 * (time.perf_counter() - tc)
Mc = res[0] if isinstance(res, tuple) else res                      # consensus_cell -> (cell, support)
if Mc is None:
    Mc = gf.LYSO
Mc = np.asarray(Mc, float)
fails = [q for q, ok in zip(frames, blind_ok) if not ok]
resc_times = []; resc_ok = 0
index_known_gpu_cell(frames[0], Mc)                                 # warmup rescue
for q in fails:
    sync(); ts = time.perf_counter()
    Mr = index_known_gpu_cell(q, Mc)
    sync(); resc_times.append(1e3 * (time.perf_counter() - ts))
    resc_ok += (Mr is not None and same_lattice(Mr, gf.LYSO))

n_ok = sum(blind_ok)
bt = np.array(blind_times); rt = np.array(resc_times) if resc_times else np.array([0.0])
print("--- blind-vs-rescued split ---")
print(f"  blind-solved (fast) {n_ok}/{n} = {100*n_ok//n}%   -> need rescue {n-n_ok}")
print(f"  rescued now {resc_ok}/{len(fails)}  final {n_ok+resc_ok}/{n} = {100*(n_ok+resc_ok)//n}%")
print(f"  consensus (one-time) {t_cons:.1f} ms")
print("--- blind latency ms: min/median/mean/p90/max ---")
print(f"  {bt.min():.1f} / {np.median(bt):.1f} / {bt.mean():.1f} / {np.percentile(bt,90):.1f} / {bt.max():.1f}")
print("--- rescue latency ms: min/median/mean/p90/max ---")
print(f"  {rt.min():.1f} / {np.median(rt):.1f} / {rt.mean():.1f} / {np.percentile(rt,90):.1f} / {rt.max():.1f}")
# histogram of per-frame TOTAL latency (fast=blind only ; slow=blind+rescue)
fast_lat = [t for t, ok in zip(blind_times, blind_ok) if ok]
slow_lat = [t + np.median(rt) for t, ok in zip(blind_times, blind_ok) if not ok]
edges = list(range(0, 121, 10))
hist_fast, _ = np.histogram(fast_lat, bins=edges)
hist_slow, _ = np.histogram(slow_lat, bins=edges)

# CUDA kernel-launch count on one frame (CUDA-graph / fusion signal)
prof_txt = ""
try:
    from torch.profiler import profile, ProfilerActivity
    q0 = frames[len(frames) // 2]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        index_profiled(q0)
    evts = prof.key_averages()
    n_launch = sum(e.count for e in evts if e.device_type.name == 'CUDA') if evts else 0
    top = sorted([e for e in evts], key=lambda e: getattr(e, 'cuda_time_total', 0), reverse=True)[:6]
    prof_txt = f"CUDA kernel launches / frame ~ {n_launch}\n" + "\n".join(
        f"    {e.key[:34]:34s} calls={e.count:5d} cuda={getattr(e,'cuda_time_total',0)/1e3:7.2f}ms" for e in top)
except Exception as e:
    prof_txt = f"(profiler unavailable: {e})"
print("--- kernel-launch profile (one frame) ---")
print(prof_txt)

# dump machine-readable for the charts
out = dict(n=n, blind_ms_per_frame=1e3 * blind_tot / n, stages={k: 1e3 * acc[k] / n for k in acc},
           stage_pct={k: 100 * acc[k] / tot_stage for k in acc}, blind_ok=n_ok,
           n_fail=n - n_ok, resc_ok=resc_ok, t_cons_ms=t_cons,
           blind_lat=dict(min=float(bt.min()), median=float(np.median(bt)), mean=float(bt.mean()),
                          p90=float(np.percentile(bt, 90)), max=float(bt.max())),
           resc_lat=dict(median=float(np.median(rt)), mean=float(rt.mean()), max=float(rt.max())),
           hist_edges=edges, hist_fast=hist_fast.tolist(), hist_slow=hist_slow.tolist())
open("/sdf/home/s/smarches/profile_glint.json", "w").write(json.dumps(out, indent=1))
print("\nJSON -> ~/profile_glint.json")
