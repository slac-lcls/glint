"""Fix test: the known-cell rescue finds the right lattice (95%) but a COARSE orientation (58% >=25%
vs the blind path's 92%). Polish = snap the rescue's 3 axes to the objective maxima with the blind
refiner (refine_vec), then re-anneal to the gate tolerance. Target: lift 58% toward the blind's 92%."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import fftindex.glint_fast as gf
from fftindex.glint_fast import anneal_batch_t
from fftindex.glint_index import refine_vec, invq_weight
from fftindex.replica_gpu import index_known_gpu_cell
from fftindex.multishot import same_lattice
DEV = gf.DEV
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames); LYSO = gf.LYSO


def mfrac(M, q):
    if M is None:
        return 0.0
    r = np.asarray(q) @ M; r = r - np.rint(r)
    return float((np.abs(r).max(1) < 0.15).sum()) / len(q)


def refine_polish(M, q, steps=8, min_thr=0.15):
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32, device=DEV)
    w = invq_weight(Q); qmax = float(Q.norm(dim=1).max())
    axes = torch.as_tensor(np.asarray(M, float).T, dtype=torch.float32, device=DEV)   # (3,3) rows=axes
    T = refine_vec(axes, Q, w, qmax, steps=steps, tol=gf.TOL)                          # snap to objective maxima
    Mt = anneal_batch_t(T.t().double()[None], Q.double(), thr0=0.30, contract=0.85, max_iter=15, min_thr=min_thr)
    return Mt[0].cpu().numpy()


index_known_gpu_cell(frames[0], LYSO); refine_polish(index_known_gpu_cell(frames[0], LYSO), frames[0])  # warmup
resc = [index_known_gpu_cell(q, LYSO) for q in frames]
sl = [r is not None and same_lattice(r, LYSO) for r in resc]
base = np.array([mfrac(r, q) for r, q, s in zip(resc, frames, sl) if s])
print(f"correct-lattice {sum(sl)}/{n}; baseline >=25%: {int((base>=0.25).sum())}/{len(base)} = {100*int((base>=0.25).sum())//len(base)}%")

for steps, mt in [(8, 0.15), (4, 0.12), (8, 0.12)]:
    keep_sl = 0; ge25 = 0; tot = 0; t0 = time.perf_counter()
    for r, q, s in zip(resc, frames, sl):
        if not s:
            continue
        tot += 1
        rp = refine_polish(r, q, steps=steps, min_thr=mt)
        keep_sl += same_lattice(rp, LYSO)                          # polish must NOT break the lattice
        ge25 += (max(mfrac(r, q), mfrac(rp, q)) >= 0.25)
    print(f"refine-polish steps={steps} min_thr={mt}: >=25% {ge25}/{tot} = {100*ge25//tot}%  "
          f"(lattice kept {keep_sl}/{tot})  {1e3*(time.perf_counter()-t0)/tot:.1f} ms/frame")
print("(blind reference on its correct-lattice frames was 92%)")
