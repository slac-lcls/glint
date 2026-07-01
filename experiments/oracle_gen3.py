"""② decisive test: does an INDEPENDENT candidate source (projection_axis_seeds, the
MOSFLM/DPS periodicity axes that 'lift the candidate ceiling on sparse frames' per its
docstring) raise the blind ceiling AND the actual solve rate above the cos-objective
maxima alone? Pool proj-seeds with the cos-maxima, anneal all triplets, report BOTH the
reachable ceiling and the cover-scorer solve (the metric that degraded under naive pool
expansion). SEEDS=none|proj  PROJK=12  NTOP=30

  SEEDS=proj PROJK=12 python oracle_gen3.py [frames.txt] [N]
"""
import os, sys, itertools, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
from glint_index import (objective, refine_vec, distinct_maxima, invq_weight, buerger_reduce,
                         primitivize, anneal, STARTS, DEV)
from glint_fast import anneal_batch_t, load, matched
from glint.seed import projection_axis_seeds
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
NTOP = int(os.environ.get("NTOP", "30"))
KEEP = int(os.environ.get("KEEP", "44"))
SEEDS = os.environ.get("SEEDS", "none")
PROJK = int(os.environ.get("PROJK", "12"))


def candidates(q):
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    T = refine_vec(STARTS.clone(), Q, w, qmax)
    f, _ = objective(T, Q, w, sharp=True)
    cands = distinct_maxima(T.cpu().numpy(), f.cpu().numpy(), keep=KEEP)[:NTOP]
    if SEEDS == "proj":
        ps = projection_axis_seeds(np.asarray(q, float), topk=PROJK)
        if len(ps):
            cands = np.vstack([cands, ps])
    return cands, Q


def evaluate(q):
    cands, Q = candidates(q)
    if len(cands) < 3:
        return False, False
    nrm = np.linalg.norm(cands, axis=1)
    tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
    M0 = np.transpose(cands[tris], (0, 2, 1))
    sc = nrm[tris].prod(1); det = np.abs(np.linalg.det(M0))
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    if len(M0) == 0:
        return False, False
    Qd = Q.double()
    Mt = anneal_batch_t(torch.as_tensor(M0, dtype=torch.float64, device=DEV), Qd)
    H = torch.einsum('pc,bcd->bpd', Qd, Mt); dd = torch.abs(H - torch.round(H))
    dmax = dd.amax(2); inl = (dmax < 0.15).to(Qd.dtype); ni = inl.sum(1)
    P = len(q); frac = ni / P
    md = (dd * inl[:, :, None]).sum((1, 2)) / (3 * ni).clamp(min=1)
    cov = (ni >= 0.30 * P).to(Qd.dtype)
    key = torch.where(ni >= 8, cov * 1e3 - md, torch.full_like(md, -1e9))
    Mn = Mt.cpu().numpy()
    # reachable ceiling
    surv = (frac >= 0.25).nonzero(as_tuple=True)[0].tolist()
    reach = any(same_lattice(buerger_reduce(Mn[b]), LYSO) for b in surv)
    # cover-scorer solve (with primitivize, like index_blind_fast)
    b = int(torch.argmax(key).item())
    solved = False
    if float(key[b]) > -1e8:
        M = primitivize(buerger_reduce(Mn[b]), q)
        solved = same_lattice(M, LYSO) and matched(M, q) / P >= 0.25
    return reach, solved


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]; n = len(frames)
    evaluate(frames[0])
    t0 = time.time(); R = S = 0
    for q in frames:
        r, s = evaluate(q); R += r; S += s
    dt = time.time() - t0
    print(f"SEEDS={SEEDS} PROJK={PROJK} NTOP={NTOP} KEEP={KEEP}  N={n}  {1e3*dt/n:.0f} ms/frame")
    print(f"  reachable ceiling: {R}/{n} ({100*R//n}%)   cover-scorer SOLVE: {S}/{n} ({100*S//n}%)")
