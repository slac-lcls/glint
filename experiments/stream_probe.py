"""A1 probe: does a chunked CUDA-stream runner overlap frames enough to beat the serial loop, given
GLINT's data-dependent syncs? Split index_blind_fast into a GPU stage (no host sync until the end) +
a CPU reduce; run K frames per chunk each on its own stream, sync once, then reduce. Compare to serial
+ verify same blind rate (84/120)."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import fftindex.glint_fast as gf
from fftindex.glint_index import objective, refine_vec, distinct_maxima_gpu, invq_weight, buerger_reduce, primitivize, STARTS
from fftindex.glint_fast import anneal_batch_t, score_batch_t, index_blind_fast
from fftindex.multishot import same_lattice
DEV = gf.DEV
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)


def gpu_stage(q):
    """M1-M5 only; return (Mt,key) on DEVICE. Only sync is the data-dependent shape reads in the middle
    (distinct_maxima_gpu mask-index, det-filter) -- no argmax.item()/.cpu() (deferred to reduce)."""
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
    return (Mt, key, np.asarray(q, float))


def reduce_cell(res):
    if res is None:
        return None
    Mt, key, q = res
    b = int(torch.argmax(key).item())
    if float(key[b]) <= -1e8:
        return None
    return primitivize(buerger_reduce(Mt[b].cpu().numpy()), q)


index_blind_fast(frames[0]); torch.cuda.synchronize()                         # warmup
# serial baseline
t0 = time.perf_counter()
serial = [index_blind_fast(q) for q in frames]
torch.cuda.synchronize(); serial_ms = 1e3 * (time.perf_counter() - t0) / n
sl_ser = sum(1 for r in serial if r is not None and same_lattice(r, gf.LYSO))

for K in (4, 8, 16):
    streams = [torch.cuda.Stream() for _ in range(K)]
    torch.cuda.synchronize(); t0 = time.perf_counter(); out = []
    for i in range(0, n, K):
        chunk = frames[i:i + K]
        res = []
        for j, q in enumerate(chunk):
            with torch.cuda.stream(streams[j]):
                res.append(gpu_stage(q))
        torch.cuda.synchronize()
        for r in res:
            out.append(reduce_cell(r))
    torch.cuda.synchronize(); ms = 1e3 * (time.perf_counter() - t0) / n
    sl = sum(1 for r in out if r is not None and same_lattice(r, gf.LYSO))
    print(f"K={K:2d}  streamed {ms:5.1f} ms/frame  rate {sl}/{n}   (serial {serial_ms:.1f} ms/frame, {sl_ser}/{n})", flush=True)
