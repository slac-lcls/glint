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
from fftindex.glint_index import (objective, refine_vec, refine_vec_newton, refine_vec_cg, distinct_maxima,
                         distinct_maxima_gpu, distinct_cells_gpu, anneal, score_defect, invq_weight,
                         buerger_reduce, primitivize, STARTS, DEV, index_blind)


def _refine(S0, Q, w, qmax):
    """M3 dispatch: grad (default GD+momentum) | cg (nonlinear conjugate-gradient) | newton (damped 3x3)."""
    if REFINER == "newton":
        return refine_vec_newton(S0, Q, w, qmax, steps=NEWTON_STEPS, tol=TOL)
    if REFINER == "cg":
        return refine_vec_cg(S0, Q, w, qmax, steps=STEPS, tol=TOL)
    return refine_vec(S0, Q, w, qmax, steps=STEPS, tol=TOL)
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
FFTSEED_CAP = int(os.environ.get("FFTSEED_CAP", "700"))     # FFT-seeded: cap rlps fed to O(N) objective/M4
REFINER = os.environ.get("REFINER", "grad")                 # M3: grad (default GD+momentum) | newton (damped 3x3)
NEWTON_STEPS = int(os.environ.get("NEWTON_STEPS", "4"))     # damped-Newton iterations
CLUSTER_MIN = int(os.environ.get("CLUSTER_MIN", "3000"))    # cluster-FFT only for genuinely DENSE (rotation) clouds; thin/moderate -> Fibonacci (fast+robust there; SFX <3000 unaffected)
ANNEAL_FP32 = os.environ.get("ANNEAL_FP32", "0") == "1"     # M5 anneal/score dtype: fp64 (default, bit-matched scalar) | fp32 (faster; validate rate)
ADT = torch.float32 if ANNEAL_FP32 else torch.float64


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


def index_blind_fast(q, acc=None, starts=None):
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
    # M1 seeds: blind Fibonacci grid (default) OR caller-supplied candidate vectors (e.g. FFT-predicted)
    S0 = STARTS.clone() if starts is None else starts
    T = _refine(S0, Q, w, qmax)                            # M3 (grad | cg | newton via REFINER)
    f, _ = objective(T, Q, w, sharp=True, tol=TOL)
    cands = distinct_maxima_gpu(T, f, keep=KEEP)[:NTOP]     # M2 dedup ON DEVICE (no host round-trip)
    if sync: torch.cuda.synchronize()
    if acc is not None: acc["gpu_front"] += time.time() - t
    if int(cands.shape[0]) < 3:
        return None
    # M4: build all valid triplets ON DEVICE, batched anneal + score on GPU (no numpy/host hop)
    t = time.time()
    Qd = Q.to(ADT); cd = cands.to(ADT)                      # M4/M5 dtype (fp64 default; ANNEAL_FP32 flag)
    tri = torch.combinations(torch.arange(cd.shape[0], device=DEV), 3)  # (B,3) lexicographic
    M0 = cd[tri].permute(0, 2, 1)                           # (B,3,3) cols=axes
    nrm = cd.norm(dim=1); sc = nrm[tri].prod(1)
    det = torch.linalg.det(M0).abs()
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    if int(M0.shape[0]) == 0:
        return None
    Mt = anneal_batch_t(M0, Qd)
    key, ni = score_batch_t(Mt, Qd)
    b = int(torch.argmax(key).item())
    best = Mt[b].cpu().numpy()
    if sync: torch.cuda.synchronize()
    if acc is not None: acc["m4_gpu"] += time.time() - t; acc["ntri"] += int(M0.shape[0])
    if float(key[b]) <= -1e8:
        return None
    cell = primitivize(buerger_reduce(best), q)
    if cell is None:
        return None
    if DETREJ and abs(np.linalg.det(np.asarray(cell, float))) < 1.0:
        return None                                         # D1: reject degenerate (det~0) cell
    return cell


def _torch_fft_seeds(q, qmax, n=None, min_len=3.0, max_peaks=300, rel=0.06):
    """GPU FFT seed generator (torch): trilinear-deposit the rlps into an n^3 reciprocal grid,
    cuFFT to real space, local-maxima peak-find -> candidate lattice vectors (Angstrom). All on
    device; only the small (m,3) peak list stays on GPU as seeds. Mirrors transform.fft_volume."""
    from fftindex.transform import estimate_grid_n
    n = int(estimate_grid_n(q, qmax) if n is None else n)
    dq = 2.0 * qmax / n
    Qt = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    gi = Qt / dq + n / 2.0
    base = torch.floor(gi).long(); frac = gi - base.float()
    rho = torch.zeros(n * n * n, device=DEV)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = base + torch.tensor([dx, dy, dz], device=DEV)
                inb = ((idx >= 0) & (idx < n)).all(1)
                wx = frac[:, 0] if dx else 1 - frac[:, 0]
                wy = frac[:, 1] if dy else 1 - frac[:, 1]
                wz = frac[:, 2] if dz else 1 - frac[:, 2]
                w = (wx * wy * wz)[inb]; ii = idx[inb]
                flat = (ii[:, 0] * n + ii[:, 1]) * n + ii[:, 2]
                rho.scatter_add_(0, flat, w)
    rho = rho.view(n, n, n)
    vol = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(rho))).abs()
    x = torch.fft.fftshift(torch.fft.fftfreq(n, d=dq)).to(DEV)
    mx = torch.nn.functional.max_pool3d(vol[None, None], 3, 1, 1)[0, 0]
    ispk = (vol == mx) & (vol > rel * vol.max())
    ijk = torch.nonzero(ispk)
    vecs = torch.stack([x[ijk[:, 0]], x[ijk[:, 1]], x[ijk[:, 2]]], 1)
    lens = vecs.norm(dim=1); keep = lens >= min_len
    vecs, amps = vecs[keep], vol[ispk][keep]
    order = torch.argsort(amps, descending=True)[:max_peaks]
    return vecs[order].float()


def _cluster_fft_seeds(q, n_clusters=28, cpts=300, n_grid=96, fov=200.0, min_len=3.0, rel=0.12, seed=0):
    """LOCAL-CLUSTER autocorrelation seeds (user idea): instead of one global O(N) FFT (whose peak
    extraction explodes on dense data), FFT several SMALL centered Bragg-peak clusters on a fine local
    grid. |F| is translation-invariant so each cluster can be centered at the origin; its
    autocorrelation peaks at the (short) lattice vectors. O(K*cpts) + K tiny FFTs -> flat in density.
    The cluster extent sets seed resolution (coarse); GLINT's refine then polishes against the data."""
    q = np.asarray(q, float)
    rng = np.random.default_rng(seed)
    dq = 1.0 / (2.0 * fov)                                   # grid spacing -> real-space FOV = fov A
    QT = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    centers = rng.choice(len(q), min(n_clusters, len(q)), replace=False)
    half = n_grid // 2
    out = []
    for ci in centers:
        d = (QT - QT[ci]).norm(dim=1)
        idx = torch.argsort(d)[:cpts]
        c = QT[idx]; c = c - c.mean(0)                      # center the cluster
        gi = torch.round(c / dq).long() + half
        inb = ((gi >= 0) & (gi < n_grid)).all(1); gi = gi[inb]
        if len(gi) < 6:
            continue
        rho = torch.zeros(n_grid ** 3, device=DEV)
        rho.scatter_add_(0, (gi[:, 0] * n_grid + gi[:, 1]) * n_grid + gi[:, 2],
                         torch.ones(len(gi), device=DEV))
        vol = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(rho.view(n_grid, n_grid, n_grid)))).abs()
        x = torch.fft.fftshift(torch.fft.fftfreq(n_grid, d=dq)).to(DEV)
        mx = torch.nn.functional.max_pool3d(vol[None, None], 3, 1, 1)[0, 0]
        ispk = (vol == mx) & (vol > rel * vol.max())
        ijk = torch.nonzero(ispk)
        vecs = torch.stack([x[ijk[:, 0]], x[ijk[:, 1]], x[ijk[:, 2]]], 1)
        L = vecs.norm(dim=1); keep = (L >= min_len) & (L <= 0.9 * fov)
        vecs, amps = vecs[keep], vol[ispk][keep]
        out.append(vecs[torch.argsort(amps, descending=True)[:24]])
    return torch.cat(out, 0) if out else torch.zeros((0, 3), device=DEV)


def index_blind_cluster_seeded(q, acc=None):
    """Index using LOCAL-CLUSTER FFT seeds -> GLINT refine/M4 (scales flat with rlp density). Below
    CLUSTER_MIN rlps the autocorrelation is starved (thin slice) -> fall back to the Fibonacci grid."""
    q = np.asarray(q, float)
    if len(q) < 6:
        return None
    if len(q) < CLUSTER_MIN:
        return index_blind_fast(q, acc)                     # sparse -> Fibonacci owns this regime
    t = time.time()
    starts = _cluster_fft_seeds(q)
    if acc is not None: acc["fft_seed"] = acc.get("fft_seed", 0.0) + time.time() - t
    if int(starts.shape[0]) < 3:
        return index_blind_fast(q, acc)                     # starved -> Fibonacci fallback
    if len(q) > FFTSEED_CAP:
        q = q[np.argsort(np.linalg.norm(q, axis=1))[:FFTSEED_CAP]]
    return index_blind_fast(q, acc, starts=starts)


def index_blind_fft_seeded(q, also_fib=False, acc=None):
    """Hybrid (user idea): use the OLD 3D-FFT method as a SEED GENERATOR for GLINT's GPU pipeline.

    fft_volume(q) -> find_peaks gives the handful of data-driven candidate lattice vectors (sharp on
    dense/rotation data); these replace the ~70k blind Fibonacci starts feeding refine_vec -> M4. On
    dense data this is far fewer, better seeds; when the FFT is starved (sparse single shot) it falls
    back to the Fibonacci grid (also_fib unions both). The seed FFT runs on CPU (one cheap FFT/frame);
    the expensive refine + GPU M4 assembly are unchanged -- so the old method's CPU basis-search blowup
    on dense data is avoided (M4 is the batched-GPU 220x replacement)."""
    q = np.asarray(q, float)
    if len(q) < 6:
        return None
    qmax = float(np.linalg.norm(q, axis=1).max())
    t = time.time()
    if DEV == "cuda":
        starts = _torch_fft_seeds(q, qmax)                  # GPU cuFFT seeds (fast, scales to dense)
        n_seed = int(starts.shape[0])
    else:
        from fftindex.transform import fft_volume, estimate_grid_n
        from fftindex.peakfind import find_peaks_classical
        vol, x = fft_volume(q, qmax, n=estimate_grid_n(q, qmax), gpu=False)
        vecs, _ = find_peaks_classical(vol, x, g=q, qmax=qmax, min_len=3.0)
        starts = torch.as_tensor(np.asarray(vecs, float), dtype=torch.float32, device=DEV)
        n_seed = len(vecs)
    if acc is not None: acc["fft_seed"] = acc.get("fft_seed", 0.0) + time.time() - t
    if n_seed < 3:
        return index_blind_fast(q, acc)                     # FFT starved -> Fibonacci fallback
    if also_fib:
        starts = torch.cat([starts, STARTS], 0)
    # the FFT used ALL rlps; the objective/refine/M4 only need enough points to verify -> cap to the
    # strongest (lowest-|q|) so the O(N) GLINT stages stay fast on dense rotation data (FFTSEED_CAP).
    if len(q) > FFTSEED_CAP:
        q = q[np.argsort(np.linalg.norm(q, axis=1))[:FFTSEED_CAP]]
    return index_blind_fast(q, acc, starts=starts)


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
    T = _refine(STARTS.clone(), Q, w, qmax)                   # M3 (grad | cg | newton via REFINER)
    f, _ = objective(T, Q, w, sharp=True, tol=TOL)
    cands = distinct_maxima_gpu(T, f, keep=KEEP)[:NTOP]        # M2 dedup ON DEVICE
    if int(cands.shape[0]) < 3:
        return []
    Qd = Q.to(ADT); cd = cands.to(ADT)                        # M4 triplet build ON DEVICE (ANNEAL_FP32 flag)
    tri = torch.combinations(torch.arange(cd.shape[0], device=DEV), 3)
    M0 = cd[tri].permute(0, 2, 1)
    nrm = cd.norm(dim=1); sc = nrm[tri].prod(1); det = torch.linalg.det(M0).abs()
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    if int(M0.shape[0]) == 0:
        return []
    Mt = anneal_batch_t(M0, Qd)
    key, ni = score_batch_t(Mt, Qd)
    reps = distinct_cells_gpu(Mt, key)                        # GPU metric-dedup: reduce only DISTINCT cells
    if int(reps.numel()) == 0:
        return []
    Mtn = Mt[reps].cpu().numpy(); keyn = key[reps].cpu().numpy()   # only the few reps hit the CPU reduce
    out = []
    for i in range(len(reps)):
        if keyn[i] <= -1e8:
            break
        cell = primitivize(buerger_reduce(Mtn[i]), q)
        if cell is None or any(same_lattice(cell, c) for c, _ in out):
            continue
        out.append((cell, float(keyn[i])))
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
