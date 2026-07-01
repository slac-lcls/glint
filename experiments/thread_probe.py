"""A1 probe v2: multi-THREADED issue. K worker threads each run a frame's GPU stage on its own stream;
torch releases the GIL during CUDA syncs, so one thread's mid-pipeline sync lets another issue -> real
overlap without removing the data-dependent syncs. Measure throughput + rate vs serial."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
import glint.glint_fast as gf
from glint.glint_index import objective, refine_vec, distinct_maxima_gpu, invq_weight, buerger_reduce, primitivize, STARTS
from glint.glint_fast import anneal_batch_t, score_batch_t, index_blind_fast
from glint.multishot import same_lattice
DEV = gf.DEV
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)


def index_on_stream(q, stream):
    with torch.cuda.stream(stream):
        Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32, device=DEV)
        qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
        T = refine_vec(STARTS.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL)
        f, _ = objective(T, Q, w, sharp=True, tol=gf.TOL)
        cands = distinct_maxima_gpu(T, f, keep=gf.KEEP)[:gf.NTOP]
        if int(cands.shape[0]) < 3:
            return None
        cd = cands.double(); Qd = Q.double()
        tri = torch.combinations(torch.arange(cd.shape[0], device=DEV), 3)
        M0 = cd[tri].permute(0, 2, 1); nrm = cd.norm(dim=1); sc = nrm[tri].prod(1); det = torch.linalg.det(M0).abs()
        M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
        if int(M0.shape[0]) == 0:
            return None
        Mt = anneal_batch_t(M0, Qd); key, _ = score_batch_t(Mt, Qd)
        b = int(torch.argmax(key).item())
        if float(key[b]) <= -1e8:
            return None
        return primitivize(buerger_reduce(Mt[b].cpu().numpy()), np.asarray(q, float))


index_blind_fast(frames[0]); torch.cuda.synchronize()                          # warmup
t0 = time.perf_counter()
serial = [index_blind_fast(q) for q in frames]
torch.cuda.synchronize(); serial_ms = 1e3 * (time.perf_counter() - t0) / n
sl_ser = sum(1 for r in serial if r is not None and same_lattice(r, gf.LYSO))

for K in (2, 4, 8):
    streams = [torch.cuda.Stream() for _ in range(K)]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=K) as ex:
        out = list(ex.map(lambda iq: index_on_stream(iq[1], streams[iq[0] % K]), list(enumerate(frames))))
    torch.cuda.synchronize(); ms = 1e3 * (time.perf_counter() - t0) / n
    sl = sum(1 for r in out if r is not None and same_lattice(r, gf.LYSO))
    print(f"K={K}  threaded {ms:5.1f} ms/frame  rate {sl}/{n}   (serial {serial_ms:.1f} ms/frame, {sl_ser}/{n})", flush=True)
