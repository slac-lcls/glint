"""M4 TORO-style JOINT-BASIS assembly (the never-built swap). Instead of enumerating
C(N,3) triplets and scoring each by spots-indexed, treat the converged maxima as
generating vectors of the real-space lattice, AUGMENT with their pairwise differences
(lattice closure -> the primitive axis is often a difference of two found maxima even when
it is not itself a maximum), and build cells by Minkowski reduction (3 shortest independent
vectors). This attacks the 28% GENERATION-miss: a true short axis missed by the cos-objective
can reappear as cand_i - cand_j. Hypotheses then anneal + coverage-score exactly like the
triplet path, so it is a drop-in M4 swap.

  ASSEMBLE=triplet|joint|both  WITHDIFF=1  python glint_joint.py [frames] [N]
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_index import (objective, refine_vec, distinct_maxima, invq_weight, buerger_reduce,
                         primitivize, STARTS, DEV)
from glint_fast import anneal_batch_t, load, matched
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
ASSEMBLE = os.environ.get("ASSEMBLE", "both")
WITHDIFF = os.environ.get("WITHDIFF", "1") == "1"
NTOP = int(os.environ.get("NTOP", "30"))
LMIN, LMAX = 20.0, 130.0


def _dedup(V, tol=2.0, minlen=LMIN, maxlen=LMAX):
    L = np.linalg.norm(V, axis=1)
    V = V[(L >= minlen) & (L <= maxlen)]
    s = np.where(V[:, 0] != 0, np.sign(V[:, 0]), 1.0)            # fold +/- (lattice vectors are signless here)
    Vc = V * s[:, None]
    seen = {}; out = []
    order = np.argsort(np.linalg.norm(Vc, axis=1))               # shortest first
    for j in order:
        key = tuple(np.round(Vc[j] / tol).astype(int))
        if key in seen:
            continue
        seen[key] = 1; out.append(Vc[j])
    return np.array(out) if out else np.zeros((0, 3))


def joint_hypotheses(cands, K=8):
    """build up to K reduced bases from the (diff-augmented) candidate set. For each of
    the K shortest distinct vectors as a1, greedily complete with the shortest vectors
    independent / non-coplanar with it."""
    V = cands
    if WITHDIFF and len(cands) >= 2:
        iu, ju = np.triu_indices(len(cands), 1)
        D = cands[iu] - cands[ju]
        V = np.vstack([cands, D])
    V = _dedup(V)                                                # length-filtered, deduped, shortest-first
    if len(V) < 3:
        return []
    L = np.linalg.norm(V, axis=1)
    hyps = []
    for i1 in range(min(K, len(V))):
        v1 = V[i1]; n1 = L[i1]
        v2 = None
        for k in range(len(V)):
            if k == i1:
                continue
            if np.linalg.norm(np.cross(V[k], v1)) > 0.10 * L[k] * n1:   # non-collinear (>~6 deg)
                v2 = V[k]; break
        if v2 is None:
            continue
        nrm = np.cross(v1, v2)
        v3 = None
        for k in range(len(V)):
            if abs(V[k] @ nrm) > 0.10 * L[k] * np.linalg.norm(nrm):     # non-coplanar
                v3 = V[k]; break
        if v3 is None:
            continue
        hyps.append(np.column_stack([v1, v2, v3]))
    return hyps


def hypotheses(cands):
    H = []
    if ASSEMBLE in ("triplet", "both"):
        tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
        M0 = np.transpose(cands[tris], (0, 2, 1))
        nrm = np.linalg.norm(cands, axis=1); sc = nrm[tris].prod(1)
        det = np.abs(np.linalg.det(M0))
        H.append(M0[(sc > 0) & (det >= 0.1 * sc)])
    if ASSEMBLE in ("joint", "both"):
        jh = joint_hypotheses(cands)
        if jh:
            Mj = np.stack(jh)
            det = np.abs(np.linalg.det(Mj))
            H.append(Mj[det >= 1e3])
    H = [h for h in H if len(h)]
    return np.concatenate(H, 0) if H else np.zeros((0, 3, 3))


def index_blind_joint(q):
    q = np.asarray(q, float)
    if len(q) < 6:
        return None
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    T = refine_vec(STARTS.clone(), Q, w, qmax)
    f, _ = objective(T, Q, w, sharp=True)
    cands = distinct_maxima(T.cpu().numpy(), f.cpu().numpy())[:NTOP]
    if len(cands) < 3:
        return None
    M0 = hypotheses(cands)
    if len(M0) == 0:
        return None
    Qd = Q.double()
    Mt = anneal_batch_t(torch.as_tensor(M0, dtype=torch.float64, device=DEV), Qd)
    H = torch.einsum('pc,bcd->bpd', Qd, Mt); dd = torch.abs(H - torch.round(H))
    dmax = dd.amax(2); inl = (dmax < 0.15).to(Qd.dtype); ni = inl.sum(1)
    md = (dd * inl[:, :, None]).sum((1, 2)) / (3 * ni).clamp(min=1)
    cov = (ni >= 0.30 * len(q)).to(Qd.dtype)
    key = torch.where(ni >= 8, cov * 1e3 - md, torch.full_like(md, -1e9))
    b = int(torch.argmax(key))
    if float(key[b]) <= -1e8:
        return None
    return primitivize(buerger_reduce(Mt[b].cpu().numpy()), q)


def gpass(M, q):
    if M is None or not same_lattice(M, LYSO):
        return (0, 0)
    m = matched(M, q); return (int(m / len(q) >= 0.25), int(m >= 10))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]; n = len(frames)
    index_blind_joint(frames[0])
    if DEV == "cuda": torch.cuda.synchronize()
    g = np.zeros(2, int); sl = 0; t0 = time.time()
    for q in frames:
        M = index_blind_joint(q)
        g += gpass(M, q); sl += (M is not None and same_lattice(M, LYSO))
    if DEV == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"ASSEMBLE={ASSEMBLE} WITHDIFF={WITHDIFF} NTOP={NTOP}  N={n}  {1e3*dt/n:.1f} ms/frame ({n/dt:.1f} f/s)")
    print(f"  same_lattice {sl}/{n} ({100*sl//n}%)  gated frac>=.25 {g[0]}/{n} ({100*g[0]//n}%)  >=10refl {g[1]}/{n} ({100*g[1]//n}%)")
