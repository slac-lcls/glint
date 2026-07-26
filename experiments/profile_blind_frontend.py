"""#28 step 0: is the BLIND front end dispatch-bound or compute-bound?

The rescue's 3.1-7.4x came from collapsing host kernel dispatch (batch -> CUDA graph -> fused
kernels). That only pays if the stage is LAUNCH-bound. Before writing any batching code for M1/M3,
measure which it is, on the production path (`index_blind_fast`), on the real 120 cxidb frames.

Reports, per stage:
  wall ms/frame          -- where the time actually goes
  CUDA kernel launches   -- the dispatch-bound tell (rescue was ~5215 kernels/frame)
  kernel GPU time        -- summed device time from the profiler
  occupancy proxy        -- kernel_gpu_time / wall_time. Far below 1 => the GPU is idling between
                            launches, i.e. dispatch-bound and batching will pay. Near 1 => compute-
                            bound, and cross-frame batching buys little.

Also sweeps peak count: compute-bound work scales with P, dispatch-bound work does not.

  python profile_blind_frontend.py [frames.txt] [N]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glint.glint_fast as gf
from glint.glint_index import objective, refine_vec, distinct_maxima_gpu, invq_weight, STARTS
from glint.glint_fast import anneal_batch_t, score_batch_t, index_blind_fast

DEV = gf.DEV
HERE = os.path.dirname(os.path.abspath(__file__))
FR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "frames_cxidb_clean.txt")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
frames = [q for q in gf.load(FR) if len(q) >= 6]
if N:
    frames = frames[:N]


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def stages(q, acc):
    """Production index_blind_fast, instrumented per stage (same ops, same order)."""
    q = np.asarray(q, float)
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max())
    w = invq_weight(Q)
    sync(); t = time.perf_counter()
    S0 = STARTS.clone()                                                    # M1
    sync(); acc["M1 seed"] += time.perf_counter() - t; t = time.perf_counter()
    T = refine_vec(S0, Q, w, qmax, steps=gf.STEPS, tol=gf.TOL)             # M3
    sync(); acc["M3 refine"] += time.perf_counter() - t; t = time.perf_counter()
    f, _ = objective(T, Q, w, sharp=True, tol=gf.TOL)                      # M2 objective
    cands = distinct_maxima_gpu(T, f, keep=gf.KEEP)[:gf.NTOP]              # M2 dedup (on device)
    sync(); acc["M2 obj+dedup"] += time.perf_counter() - t; t = time.perf_counter()
    if int(cands.shape[0]) < 3:
        return
    Qd = Q.to(gf.ADT); cd = cands.to(gf.ADT)                               # M4 build
    tri = torch.combinations(torch.arange(cd.shape[0], device=DEV), 3)
    M0 = cd[tri].permute(0, 2, 1)
    nrm = cd.norm(dim=1); sc = nrm[tri].prod(1)
    det = torch.linalg.det(M0).abs()
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    sync(); acc["M4 build"] += time.perf_counter() - t; t = time.perf_counter()
    if int(M0.shape[0]) == 0:
        return
    Mt = anneal_batch_t(M0, Qd, max_iter=gf.ANNEAL_ITERS)                  # M5
    sync(); acc["M5 anneal"] += time.perf_counter() - t; t = time.perf_counter()
    key, ni = score_batch_t(Mt, Qd)                                        # M6
    b = int(torch.argmax(key).item()); _ = Mt[b].cpu().numpy()
    sync(); acc["M6 score"] += time.perf_counter() - t
    acc["ntri"] += int(M0.shape[0])


ORDER = ["M1 seed", "M3 refine", "M2 obj+dedup", "M4 build", "M5 anneal", "M6 score"]
acc = {k: 0.0 for k in ORDER}; acc["ntri"] = 0
index_blind_fast(frames[0])                                                # warmup
sync(); t0 = time.perf_counter()
for q in frames:
    stages(q, acc)
sync(); total = time.perf_counter() - t0
n = len(frames)

print(f"blind front end, {n} real cxidb frames, {DEV}, STEPS={gf.STEPS} NTOP={gf.NTOP} "
      f"KEEP={gf.KEEP} starts={tuple(STARTS.shape)}")
print(f"median peaks/frame {int(np.median([len(f) for f in frames]))}\n")
print(f"{'stage':<15}{'ms/frame':>10}{'% total':>9}")
tot = sum(acc[k] for k in ORDER)
for k in ORDER:
    print(f"{k:<15}{1e3*acc[k]/n:10.2f}{100*acc[k]/tot:8.1f}%")
print(f"{'TOTAL':<15}{1e3*tot/n:10.2f}{100:8.1f}%   (wall {1e3*total/n:.2f} ms/frame, "
      f"{acc['ntri']//n} triplets/frame)")

# ---- dispatch-bound or compute-bound? ------------------------------------------------------
if DEV == "cuda":
    from torch.profiler import profile, ProfilerActivity
    def prof_stage(fn, label):
        fn()  # warm
        sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            t = time.perf_counter(); fn(); sync(); wall = time.perf_counter() - t
        # torch renamed cuda_time_total -> device_time_total after 2.1; accept either.
        def _gpu_us(e):
            return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0) or 0
        evs = [e for e in p.key_averages() if _gpu_us(e) > 0]
        klaunch = sum(e.count for e in evs)
        kgpu = sum(_gpu_us(e) for e in evs) * 1e-6               # us -> s
        print(f"{label:<15}{klaunch:>10}{1e3*kgpu:11.2f}{1e3*wall:11.2f}{kgpu/max(wall,1e-9):11.2f}")
        return klaunch, kgpu, wall

    q = frames[len(frames) // 2]
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    print(f"\n--- per-frame dispatch profile (1 frame, {len(q)} peaks) ---")
    print(f"{'stage':<15}{'kernels':>10}{'gpu ms':>11}{'wall ms':>11}{'occupancy':>11}")
    prof_stage(lambda: refine_vec(STARTS.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL), "M3 refine")
    T = refine_vec(STARTS.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL)
    prof_stage(lambda: distinct_maxima_gpu(T, objective(T, Q, w, sharp=True, tol=gf.TOL)[0],
                                           keep=gf.KEEP), "M2 obj+dedup")
    prof_stage(lambda: index_blind_fast(q), "WHOLE frame")
    print("  occupancy = kernel GPU time / wall. <<1 => GPU idles between launches (dispatch-bound,")
    print("  batching pays, as it did for the rescue). ~1 => compute-bound, batching buys little.")

# ---- does M3 cost scale with PEAK COUNT? (compute-bound would) ------------------------------
if DEV == "cuda":
    print("\n--- M3 cost vs peak count (compute-bound scales with P; dispatch-bound is flat) ---")
    print(f"{'peaks':>8}{'M3 ms':>9}")
    big = max(frames, key=len)
    for P in (16, 32, 64, 128, 256, min(512, len(big))):
        if P > len(big):
            continue
        Qp = torch.as_tensor(np.asarray(big[:P], float), dtype=torch.float32, device=DEV)
        wp = invq_weight(Qp); qm = float(Qp.norm(dim=1).max())
        refine_vec(STARTS.clone(), Qp, wp, qm, steps=gf.STEPS, tol=gf.TOL); sync()
        t = time.perf_counter()
        for _ in range(5):
            refine_vec(STARTS.clone(), Qp, wp, qm, steps=gf.STEPS, tol=gf.TOL)
        sync()
        print(f"{P:>8}{1e3*(time.perf_counter()-t)/5:9.2f}")
