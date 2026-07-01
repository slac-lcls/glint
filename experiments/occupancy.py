"""Is the GPU well occupied by one frame, and would batching/streams help? Measures:
(1) M3 refine throughput vs #starts, (2) M5 anneal throughput vs #triplets -> saturation knee,
(3) sustained GPU utilization during the real per-frame loop."""
import os, sys, time, subprocess, threading
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_index import objective, refine_vec, distinct_maxima_gpu, invq_weight, STARTS
from glint.glint_fast import anneal_batch_t, index_blind_fast
DEV = gf.DEV


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
q0 = frames[len(frames) // 2]
Q = torch.as_tensor(np.asarray(q0, float), dtype=torch.float32, device=DEV)
qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)


def timed(fn, iters=30):
    fn(); sync(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(); return 1e3 * (time.perf_counter() - t) / iters


print(f"GPU: {torch.cuda.get_device_name(0)}  (one frame's M3 starts={STARTS.shape[0]}, peaks={Q.shape[0]})")
print("=== (1) M3 refine: throughput vs #starts (flat throughput at small N = underoccupied) ===")
base = STARTS
for mult in [0.125, 0.25, 0.5, 1, 2, 4, 8]:
    n = max(1, int(base.shape[0] * mult))
    S = base.repeat(int(np.ceil(mult)) + 1, 1)[:n]
    ms = timed(lambda: refine_vec(S.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL))
    print(f"  starts {n:8d}  {ms:7.2f} ms  {n/ms/1e3:7.1f} k-starts/ms")

# real M0 triplet batch from q0
T = refine_vec(STARTS.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL)
f, _ = objective(T, Q, w, sharp=True, tol=gf.TOL)
cands = distinct_maxima_gpu(T, f, keep=gf.KEEP)[:gf.NTOP].double()
tri = torch.combinations(torch.arange(cands.shape[0], device=DEV), 3)
M0b = cands[tri].permute(0, 2, 1)
nrm = cands.norm(dim=1); sc = nrm[tri].prod(1); det = torch.linalg.det(M0b).abs()
M0b = M0b[(sc > 0) & (det >= 0.1 * sc)]; Qd = Q.double()
print(f"=== (2) M5 anneal: throughput vs #triplets (one frame ~ {M0b.shape[0]}) ===")
for B in [150, 300, 600, 1200, 2400, 4800, 9600, 19200]:
    reps = int(np.ceil(B / M0b.shape[0])); M0 = M0b.repeat(reps, 1, 1)[:B]
    ms = timed(lambda: anneal_batch_t(M0, Qd))
    print(f"  triplets {B:6d}  {ms:7.2f} ms  {B/ms:8.0f} tri/ms")

print("=== (3) sustained GPU utilization during the real per-frame loop ===")
samples = []; stop = [False]


def sampler():
    while not stop[0]:
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=1)
            samples.append(int(o.stdout.strip().split("\n")[0]))
        except Exception:
            pass


index_blind_fast(frames[0])
th = threading.Thread(target=sampler); th.start()
t0 = time.perf_counter(); nf = 0
while time.perf_counter() - t0 < 15:
    for q in frames:
        index_blind_fast(q); nf += 1
stop[0] = True; th.join()
dt = time.perf_counter() - t0
s = np.array(samples) if samples else np.array([0])
print(f"  ran {nf} frames in {dt:.1f}s = {1e3*dt/nf:.1f} ms/frame")
print(f"  GPU busy(util.gpu) avg {s.mean():.0f}%  median {np.median(s):.0f}%  p10 {np.percentile(s,10):.0f}%  (n={len(s)} samples)")
