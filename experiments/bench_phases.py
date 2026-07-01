"""③ phase profiler: split GLINT index_blind into (GPU front-end refine_vec) vs
(numpy back-end M4 triplet loop) to see where per-frame latency goes on the A100.
This tells us whether 'fast GPU blind' holds as-is, or needs the M4 loop batched.

  python bench_phases.py [frames.txt] [N]
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
import glint_index as G
from glint_index import (objective, refine_vec, distinct_maxima, anneal, score_defect,
                         invq_weight, buerger_reduce, primitivize, STARTS, DEV)
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)


def index_blind_timed(q, acc):
    q = np.asarray(q, float)
    if len(q) < 6:
        return None
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    if DEV == "cuda": torch.cuda.synchronize()
    t = time.time()
    T = refine_vec(STARTS.clone(), Q, w, qmax)
    f, _ = objective(T, Q, w, sharp=True)
    cands = distinct_maxima(T.cpu().numpy(), f.cpu().numpy())[:30]
    if DEV == "cuda": torch.cuda.synchronize()
    acc["gpu"] += time.time() - t
    if len(cands) < 3:
        return None
    t = time.time()
    best = None; bk = None; ntri = 0
    for tri in itertools.combinations(range(len(cands)), 3):
        M0 = cands[list(tri)].T
        sc = np.prod([np.linalg.norm(cands[t2]) for t2 in tri])
        if sc <= 0 or abs(np.linalg.det(M0)) < 0.1 * sc:
            continue
        ntri += 1
        M = anneal(M0, q); key = score_defect(M, q)
        if key is None:
            continue
        if bk is None or key > bk:
            bk = key; best = buerger_reduce(M)
    out = primitivize(best, q) if best is not None else None
    acc["m4"] += time.time() - t; acc["ntri"] += ntri
    return out


def load(p):
    fr = []; L = open(p).read().split("\n"); i = 0
    while i < len(L):
        if not L[i].startswith("FRAME"): i += 1; continue
        _, fid, n = L[i].split(); n = int(n)
        fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
    return fr


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]
    acc = {"gpu": 0.0, "m4": 0.0, "ntri": 0}
    index_blind_timed(frames[0], {"gpu": 0.0, "m4": 0.0, "ntri": 0})   # warmup
    ok = 0; t0 = time.time()
    for q in frames:
        M = index_blind_timed(q, acc)
        ok += M is not None and same_lattice(M, LYSO)
    tot = time.time() - t0; n = len(frames)
    print(f"device={DEV}  starts={STARTS.shape[0]}  N={n}  rate(same_lattice)={ok}/{n} ({100*ok/n:.0f}%)")
    print(f"  TOTAL          {1e3*tot/n:7.1f} ms/frame   ({n/tot:.1f} f/s)")
    print(f"  GPU front-end  {1e3*acc['gpu']/n:7.1f} ms/frame   ({100*acc['gpu']/tot:.0f}% of time)")
    print(f"  M4 numpy loop  {1e3*acc['m4']/n:7.1f} ms/frame   ({100*acc['m4']/tot:.0f}% of time)  ~{acc['ntri']//n} triplets/frame")
