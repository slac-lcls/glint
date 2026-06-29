"""② generation sweep: does keeping MORE candidate maxima (or denser M1 sampling) lift
the reachable ceiling? The oracle showed 28% GENERATION-miss (true cell not among top-30
maxima). Here we vary NTOP (candidates kept) and the M1 start grid, and report the
reachable ceiling = fraction of frames where SOME annealed triplet is same_lattice(LYSO)
with frac>=0.25. Reachability is GPU-vectorized (batched matched-count) so it scales.

  NTOP=50 NDIR=2200 LENSTEP=3.0 python oracle_gen.py [frames.txt] [N]
"""
import os, sys, itertools, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_index import objective, refine_vec, distinct_maxima, invq_weight, buerger_reduce, fib_sphere, DEV
from glint_fast import anneal_batch_t, load, matched
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
NTOP = int(os.environ.get("NTOP", "30"))
KEEP = int(os.environ.get("KEEP", "60"))
NDIR = int(os.environ.get("NDIR", "2200"))
LENSTEP = float(os.environ.get("LENSTEP", "3.0"))
LMIN, LMAX = 30.0, 126.0
_lengths = np.arange(LMIN, LMAX, LENSTEP)
_D = torch.as_tensor(fib_sphere(NDIR), dtype=torch.float32)
STARTS = torch.cat([float(L) * _D for L in _lengths], 0).to(DEV)


def candidates(q):
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    T = refine_vec(STARTS.clone(), Q, w, qmax)
    f, _ = objective(T, Q, w, sharp=True)
    return distinct_maxima(T.cpu().numpy(), f.cpu().numpy(), keep=KEEP)[:NTOP], Q


def reachable(q):
    cands, Q = candidates(q)
    if len(cands) < 3:
        return False
    nrm = np.linalg.norm(cands, axis=1)
    tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
    M0 = np.transpose(cands[tris], (0, 2, 1))
    sc = nrm[tris].prod(1); det = np.abs(np.linalg.det(M0))
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    if len(M0) == 0:
        return False
    Qd = Q.double()
    Mt = anneal_batch_t(torch.as_tensor(M0, dtype=torch.float64, device=DEV), Qd)
    H = torch.einsum('pc,bcd->bpd', Qd, Mt)
    frac = (torch.abs(H - torch.round(H)).amax(2) < 0.15).to(Qd.dtype).mean(1)   # (B,)
    surv = (frac >= 0.25).nonzero(as_tuple=True)[0]
    Mn = Mt.cpu().numpy()
    for b in surv.tolist():
        if same_lattice(buerger_reduce(Mn[b]), LYSO):
            return True
    return False


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]; n = len(frames)
    reachable(frames[0])
    t0 = time.time(); reach = sum(reachable(q) for q in frames); dt = time.time() - t0
    print(f"NTOP={NTOP} KEEP={KEEP} NDIR={NDIR} LENSTEP={LENSTEP} starts={STARTS.shape[0]}  "
          f"N={n}  {1e3*dt/n:.0f} ms/frame")
    print(f"  reachable ceiling: {reach}/{n} ({100*reach//n}%)")
