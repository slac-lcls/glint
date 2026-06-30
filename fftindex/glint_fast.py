"""③ GLINT-FAST: GPU-batched blind indexer. Same M1-M3 front-end (refine_vec) as
glint_index, but the M4 triplet assembly (the ~0.3 s/frame numpy bottleneck: ~1200
independent 3x3 OLS-with-masking 'anneal' solves run in a Python loop) is replaced by
ONE batched anneal on the GPU. Bit-identical to the scalar anneal (validated). Plus a
batched scorer. Goal: turn ~1.8 s/frame into tens of ms -> a GPU blind indexer that is
both FAST and accurate (xgandalf-class rate at >100x the throughput on sparse SFX).

  python glint_fast.py [frames.txt] [N] [--scalar]
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
from fftindex.glint_index import (objective, refine_vec, distinct_maxima, anneal, score_defect,
                         invq_weight, buerger_reduce, primitivize, STARTS, DEV, index_blind)
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
NTOP = int(os.environ.get("NTOP", "30"))                    # ② candidate-pool size (M4 width)
KEEP = int(os.environ.get("KEEP", "44"))                    # distinct_maxima retained
STEPS = int(os.environ.get("STEPS", "80"))                  # M3 ascent steps (K-sweep: 8 ties 80 at 3.6x)
QDIST = os.environ.get("QDIST", "0") == "1"                  # D2: reciprocal-distance inlier (sigma-matched)
QDTOL = float(os.environ.get("QDTOL", "0.004"))             # inlier radius in 1/A (q-space)
DETREJ = os.environ.get("DETREJ", "0") == "1"               # D1: reject degenerate cell (OFF: regressed deflate)
QPOW = float(os.environ.get("QPOW", "1.0"))                 # M2 weight w_i=|q_i|^-QPOW (GLINT 1; xgandalf paper 2)
TOL = float(os.environ.get("TOL", "0.18"))                  # M2/M3 hard inlier window |q.v-round|<TOL (xgandalf eps)
QHI = float(os.environ.get("QHI", "0"))                     # erf^2 high-q apodize: taper edge / qmax (0=off)
QLO = float(os.environ.get("QLO", "0"))                     # erf^2 low-q (beamstop) apodize: edge / qmax (0=off)
QAPSIG = float(os.environ.get("QAPSIG", "0.08"))            # apodization taper width / qmax


def qband_apod(qn, qmax):
    """Smooth erf^2 band-pass in |q| (apodize high-q resolution edge + low-q beamstop). The
    square removes the first-order kink: window and slope both ->0 at each edge."""
    a = torch.ones_like(qn); sig = QAPSIG * qmax
    if QHI > 0:
        qhi = QHI * qmax
        a = a * torch.where(qn < qhi, torch.erf((qhi - qn) / sig) ** 2, torch.zeros_like(qn))
    if QLO > 0:
        qlo = QLO * qmax
        a = a * torch.where(qn > qlo, torch.erf((qn - qlo) / sig) ** 2, torch.zeros_like(qn))
    return a


def anneal_batch_t(M0, Q, thr0=0.25, contract=0.85, max_iter=15, min_thr=0.02):
    """Batched residual-threshold anneal on GPU. M0:(B,3,3) real-axis cols, Q:(P,3) torch.
    Masked normal equations  (Q^T W Q) M = Q^T W hkl  solved as B 3x3 systems / iter."""
    B = M0.shape[0]; M = M0.clone(); thr = thr0; thr_q = 0.02
    eyes = 1e-9 * torch.eye(3, device=Q.device, dtype=Q.dtype)[None]
    for _ in range(max_iter):
        H = torch.einsum('pc,bcd->bpd', Q, M)               # (B,P,3)
        hkl = torch.round(H)
        if QDIST:                                           # isotropic reciprocal-distance ball (1/A)
            rq = torch.einsum('bpc,bcd->bpd', H - hkl, torch.linalg.pinv(M))
            inl = (rq.norm(dim=2) < thr_q).to(Q.dtype)      # (B,P)
        else:
            inl = (torch.abs(H - hkl).amax(2) < thr).to(Q.dtype)# (B,P)
        cnt = inl.sum(1)                                     # (B,)
        WQ = inl[:, :, None] * Q[None]                       # (B,P,3)
        A = torch.einsum('bpc,pd->bcd', WQ, Q) + eyes        # (B,3,3)
        rhs = torch.einsum('bpc,bpd->bcd', WQ, hkl)          # (B,3,3)
        Msol = torch.linalg.solve(A, rhs)
        ok = (cnt >= 6)[:, None, None]
        M = torch.where(ok, Msol, M)
        thr = max(thr * contract, min_thr)
        thr_q = max(thr_q * contract, QDTOL)
    return M


SCORER = os.environ.get("SCORER", "cover")                  # cover | count | covcount


def score_batch_t(M, Q, inl_tol=0.15, cover=0.30):
    """Batched cell scorer (② SWAP). Returns rank key (B,) and inlier counts (B,).
      cover    : cover_flag*1e3 - mean_inl_dist        (current; binary gate then tightest)
      count    : n_inliers - mean_inl_dist             (most spots indexed wins; defect tie-break)
      covcount : cover_flag*1e5 + n_inliers - md       (gate <30%, then most spots above it)
    'count'/'covcount' exploit the coverage GRADIENT the binary gate discards -- the lever
    that lets a richer candidate pool help instead of fooling the selector."""
    H = torch.einsum('pc,bcd->bpd', Q, M); r = H - torch.round(H); dd = torch.abs(r)
    P = Q.shape[0]
    if QDIST:
        rq = torch.einsum('bpc,bcd->bpd', r, torch.linalg.pinv(M)); dist = rq.norm(dim=2)
        inl = (dist < QDTOL).to(Q.dtype); ni = inl.sum(1)
        md = (dist * inl).sum(1) / ni.clamp(min=1)
    else:
        dmax = dd.amax(2)                                    # (B,P)
        inl = (dmax < inl_tol).to(Q.dtype); ni = inl.sum(1)  # (B,)
        md = (dd * inl[:, :, None]).sum((1, 2)) / (3 * ni).clamp(min=1)
    cov = (ni >= cover * P).to(Q.dtype)
    if SCORER == "count":
        key = ni - md
    elif SCORER == "covcount":
        key = cov * 1e5 + ni - md
    else:
        key = cov * 1e3 - md
    key = torch.where(ni >= 8, key, torch.full_like(key, -1e9))
    return key, ni


def index_blind_fast(q, acc=None):
    q = np.asarray(q, float)
    if len(q) < 6:
        return None
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max())
    w = invq_weight(Q) if QPOW == 1.0 else Q.norm(dim=1).clamp_min(1e-9) ** (-QPOW)
    if QHI > 0 or QLO > 0:
        w = w * qband_apod(Q.norm(dim=1), qmax)
    sync = (DEV == "cuda")
    if sync: torch.cuda.synchronize()
    t = time.time()
    T = refine_vec(STARTS.clone(), Q, w, qmax, steps=STEPS, tol=TOL)
    f, _ = objective(T, Q, w, sharp=True, tol=TOL)
    cands = distinct_maxima(T.cpu().numpy(), f.cpu().numpy(), keep=KEEP)[:NTOP]
    if sync: torch.cuda.synchronize()
    if acc is not None: acc["gpu_front"] += time.time() - t
    if len(cands) < 3:
        return None
    # M4: build all valid triplets, batched anneal + score on GPU
    t = time.time()
    nrm = np.linalg.norm(cands, axis=1)
    tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
    M0 = np.transpose(cands[tris], (0, 2, 1))               # (B,3,3) cols=axes
    sc = nrm[tris].prod(1)
    det = np.abs(np.linalg.det(M0))
    keep = (sc > 0) & (det >= 0.1 * sc)
    M0 = M0[keep]
    if len(M0) == 0:
        return None
    # M4 in float64 to match the scalar numpy anneal exactly (A100 fp64 is cheap here)
    Qd = Q.double()
    M0t = torch.as_tensor(M0, dtype=torch.float64, device=DEV)
    Mt = anneal_batch_t(M0t, Qd)
    key, ni = score_batch_t(Mt, Qd)
    b = int(torch.argmax(key).item())
    best = Mt[b].cpu().numpy()
    if sync: torch.cuda.synchronize()
    if acc is not None: acc["m4_gpu"] += time.time() - t; acc["ntri"] += len(M0)
    if float(key[b]) <= -1e8:
        return None
    cell = primitivize(buerger_reduce(best), q)
    if cell is None:
        return None
    if DETREJ and abs(np.linalg.det(np.asarray(cell, float))) < 1.0:
        return None                                         # D1: reject degenerate (det~0) cell
    return cell


def index_blind_nbest(q, N=5):
    """Return up to N DISTINCT candidate cells (primitivized, deduped by same_lattice) ranked by
    the M4 score, as [(cell, score), ...] -- the n-best HYPOTHESES per frame. Sparse single-shot
    indexing is rank-deficient (orientation ambiguous about the unobserved axis), so the true cell
    is often a reachable-but-not-top-1 hypothesis that single-pass discards as wrong_cell/sel_miss.
    Keep them (score-tagged) and let cross-frame consensus arbitrate (aliases don't recur)."""
    q = np.asarray(q, float)
    if len(q) < 6:
        return []
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max())
    w = invq_weight(Q) if QPOW == 1.0 else Q.norm(dim=1).clamp_min(1e-9) ** (-QPOW)
    if QHI > 0 or QLO > 0:
        w = w * qband_apod(Q.norm(dim=1), qmax)
    T = refine_vec(STARTS.clone(), Q, w, qmax, steps=STEPS, tol=TOL)
    f, _ = objective(T, Q, w, sharp=True, tol=TOL)
    cands = distinct_maxima(T.cpu().numpy(), f.cpu().numpy(), keep=KEEP)[:NTOP]
    if len(cands) < 3:
        return []
    nrm = np.linalg.norm(cands, axis=1)
    tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
    M0 = np.transpose(cands[tris], (0, 2, 1))
    sc = nrm[tris].prod(1); det = np.abs(np.linalg.det(M0))
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    if len(M0) == 0:
        return []
    Qd = Q.double()
    Mt = anneal_batch_t(torch.as_tensor(M0, dtype=torch.float64, device=DEV), Qd)
    key, ni = score_batch_t(Mt, Qd)
    order = torch.argsort(key, descending=True).cpu().numpy()
    Mtn = Mt.cpu().numpy(); keyn = key.cpu().numpy()
    out = []
    for idx in order:
        if keyn[idx] <= -1e8:
            break
        cell = primitivize(buerger_reduce(Mtn[idx]), q)
        if cell is None or any(same_lattice(cell, c) for c, _ in out):
            continue
        out.append((cell, float(keyn[idx])))
        if len(out) >= N:
            break
    return out


def load(p):
    fr = []; L = open(p).read().split("\n"); i = 0
    while i < len(L):
        if not L[i].startswith("FRAME"): i += 1; continue
        _, fid, n = L[i].split(); n = int(n)
        fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
    return fr


def matched(M, q, tol=0.15):
    if M is None:
        return 0
    r = q @ M - np.rint(q @ M)
    if QDIST:
        return int((np.linalg.norm(r @ np.linalg.pinv(M), axis=1) < QDTOL).sum())
    return int((np.abs(r).max(1) < tol).sum())


def gpass(M, q):
    if M is None or not same_lattice(M, LYSO): return (0, 0)
    m = matched(M, q); return (int(m / len(q) >= 0.25), int(m >= 10))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scalar = "--scalar" in sys.argv
    path = args[0] if args else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(args[1]) if len(args) > 1 else len(frames)
    frames = frames[:N]; n = len(frames)
    fn = index_blind if scalar else index_blind_fast
    name = "SCALAR (numpy M4)" if scalar else "FAST (GPU-batched M4)"
    acc = {"gpu_front": 0.0, "m4_gpu": 0.0, "ntri": 0}
    fn(frames[0], acc if not scalar else None)              # warmup
    acc = {"gpu_front": 0.0, "m4_gpu": 0.0, "ntri": 0}
    g25 = g10 = sl = 0; t0 = time.time()
    for q in frames:
        M = fn(q, acc) if not scalar else fn(q)
        a, b = gpass(M, q); g25 += a; g10 += b
        sl += (M is not None and same_lattice(M, LYSO))
    tot = time.time() - t0
    print(f"=== GLINT {name}  device={DEV}  N={n} ===")
    print(f"  THROUGHPUT  {1e3*tot/n:7.1f} ms/frame   ({n/tot:.1f} frames/s)")
    if not scalar:
        print(f"    GPU front-end (M1-M3) {1e3*acc['gpu_front']/n:7.1f} ms/frame")
        print(f"    GPU M4 batched anneal {1e3*acc['m4_gpu']/n:7.1f} ms/frame  (~{acc['ntri']//n} triplets/frame)")
    print(f"  ACCURACY    same_lattice {sl}/{n} ({100*sl//n}%)  gated frac>=.25 {g25}/{n} ({100*g25//n}%)  >=10refl {g10}/{n} ({100*g10//n}%)")
