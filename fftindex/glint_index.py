"""GLINT modular indexer: the SOTA pieces (xgandalf / ffbidx / TORO), disassembled
into swappable modules we OWN. Grounded in: xgandalf paper (Gevorkov 2019), ffbidx
C++/CUDA source (PSI, read on S3DF), TORO paper (Gasparotto 2024).

Pipeline (each module independent + swappable):
  M1 sample      candidate vectors: Fibonacci half-sphere x lengths
                   blind  -> length range;  known -> cell-prior axis lengths
  M2 OBJECTIVE   proximity score of a candidate vector v over nodes q  [SWAP: w_i]
                   xgandalf : sum_i w_i cos(2pi q_i.v),   w_i = 1/|q_i|   (+cos^2 sharpen)
                   ffbidx   : trimmed-log2 dist2int + inlier count
                   tolerance: drop node i if dist2int(q_i.v) > tol
  M3 refine_vec  robust optimization of candidate vectors (gradient ascent of M2)
                   = xgandalf extended-GD / ffbidx VCR_ROPT
  M4 assemble    candidate vectors -> cells.  triplet  | (TODO) TORO joint-basis spin
  M5 anneal      cell refine = residual-threshold-annealing (TORO) / ifss (ffbidx):
                   closed-form OLS fit to inliers -> trim residuals -> anneal tau down
  M6 SCORE       pick the cell  [SWAP: weak-peak completeness]
                   blind : defect  = min mean residual of inliers (tightest fit)
                   known : |S| (#inliers) + shape penalty vs cell prior

The default config reproduces the 61% blind result on real cxidb lysozyme. The two
SWAP points (M2 weight, M6 scorer) are where weak-peak/photon signal plugs in.
"""
import os, sys, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
from fftindex.lattice import buerger_reduce

PI = np.pi
DEV = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")

# M2 proximity-function form (climb + score). "" = default cos / cos^2 with HARD tol mask.
# Smooth forms (gauss/vonmises/softcos) carry their OWN falloff, so the hard mask is dropped
# -> they replace the sharp indicator with a smooth window (sigma/kappa = soft tol).
OBJFORM = os.environ.get("OBJFORM", "")
OBJSIG = float(os.environ.get("OBJSIG", "0.12"))            # wrapped-Gaussian / soft-window width
OBJKAP = float(os.environ.get("OBJKAP", "8.0"))             # von Mises concentration


# ---- M1: sample -----------------------------------------------------------------
def fib_sphere(D):
    i = np.arange(D); phi = PI * (3 - np.sqrt(5)) * i
    z = 1.0 - (i + 0.5) / D; r = np.sqrt(np.clip(1 - z * z, 0, 1))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)

def sample(n_dir=2200, lengths=None):
    """blind start grid: Fibonacci dirs x length shells. lengths=None -> protein range."""
    if lengths is None:
        lengths = np.arange(30.0, 126.0, 3.0)
    D = torch.as_tensor(fib_sphere(n_dir), dtype=torch.float32)
    return torch.cat([float(L) * D for L in lengths], 0)


# ---- M2: objective (SWAP POINT: weight w) ---------------------------------------
def invq_weight(Q):
    """xgandalf Stage-1 weight 1/|q| (suppress high-res spurious maxima)."""
    return 1.0 / Q.norm(dim=1)

def ones_weight(Q):
    return torch.ones(Q.shape[0], device=Q.device, dtype=Q.dtype)

def objective(T, Q, w, tol=0.18, sharp=False):
    """M2: sum_i w_i c(q_i . T), c=cos(2pi x) | cos^2(pi x); tolerance-masked. Returns
    (f, grad). w is the SWAP POINT -- pass a photon/weak-peak-aware weight here."""
    proj = T @ Q.t()
    if OBJFORM == "":                                          # default: hard mask + cos / cos^2
        mask = (torch.abs(proj - torch.round(proj)) < tol).to(T.dtype)
        if sharp:
            c = torch.cos(PI * proj) ** 2; dc = -PI * torch.sin(2 * PI * proj)
        else:
            c = torch.cos(2 * PI * proj); dc = -2 * PI * torch.sin(2 * PI * proj)
        wm = w.unsqueeze(0) * mask
        return (c * wm).sum(1), (dc * wm) @ Q
    # smooth proximity forms: built-in falloff replaces the sharp indicator (no hard mask)
    d = proj - torch.round(proj)                              # signed dist to nearest int [-0.5,0.5]
    if OBJFORM == "gauss":                                    # wrapped Gaussian (smooth all-positive bump)
        c = torch.exp(-(d * d) / (2 * OBJSIG ** 2)); dc = -(d / OBJSIG ** 2) * c
    elif OBJFORM == "vonmises":                               # smooth cosine comb
        c = torch.exp(OBJKAP * (torch.cos(2 * PI * proj) - 1)); dc = -2 * PI * OBJKAP * torch.sin(2 * PI * proj) * c
    elif OBJFORM == "softcos":                                # cos x Gaussian window (smooth mask on cos)
        win = torch.exp(-(d * d) / (2 * OBJSIG ** 2)); cc = torch.cos(2 * PI * proj)
        c = cc * win; dc = (-2 * PI * torch.sin(2 * PI * proj)) * win + cc * (-(d / OBJSIG ** 2) * win)
    elif OBJFORM == "tent":                                   # xgandalf linear proximity (1 at int, -1 midway)
        c = 1 - 4 * torch.abs(d); dc = -4 * torch.sign(d)
    else:
        raise ValueError(f"unknown OBJFORM {OBJFORM}")
    w0 = w.unsqueeze(0)
    return (c * w0).sum(1), (dc * w0) @ Q


# ---- M3: robust optimization of candidate vectors -------------------------------
def refine_vec(T, Q, w, qmax, steps=80, tol=0.18, sharp_last=25, mom=0.5):
    """M3: ascend the M2 objective (normalized-grad + momentum ~ xgandalf extended-GD
    zigzag damping); sharpen with cos^2 near the end."""
    vel = torch.zeros_like(T); step0 = 0.25 / qmax
    for s in range(steps):
        lr = step0 * (1 - 0.7 * s / steps)
        _, g = objective(T, Q, w, tol, sharp=(s >= steps - sharp_last))
        vel = mom * vel + g / (g.norm(dim=1, keepdim=True) + 1e-12)
        T = T + lr * vel
    return T


def refine_vec_newton(T, Q, w, qmax, steps=4, tol=0.18, mu=None, sharp_last=1):
    """M3 variant: damped-NEWTON ascent of the default hard-masked cos objective. Analytic per-seed
    3x3 gradient + Hessian (g = -2pi (sin.wm)@Q ; H = -(2pi)^2 sum (cos.wm) q q^T); damped H-mu*I to
    stay negative-definite (ascent) and avoid basin-jumping on the multimodal comb. Batched 3x3 solve.
    Converges in ~2-4 steps where GD takes 8-80 -- tested for the dense/cluster path (small #seeds)."""
    B = T.shape[0]
    eye = torch.eye(3, dtype=T.dtype, device=T.device)
    mu = (2 * PI) ** 2 * 0.05 if mu is None else mu       # damping ~ small fraction of |H| scale
    for s in range(steps):
        proj = T @ Q.t()
        mask = (torch.abs(proj - torch.round(proj)) < tol).to(T.dtype)
        if s >= steps - sharp_last:                        # cos^2 sharpen at the end (as in refine_vec)
            sin2 = PI * torch.sin(2 * PI * proj); cos2 = (2 * PI) * PI * torch.cos(2 * PI * proj)
        else:
            sin2 = 2 * PI * torch.sin(2 * PI * proj); cos2 = (2 * PI) ** 2 * torch.cos(2 * PI * proj)
        wm = w.unsqueeze(0) * mask
        g = -((sin2 * wm) @ Q)                             # (B,3)
        H = -torch.einsum("bn,ni,nj->bij", cos2 * wm, Q, Q)  # (B,3,3)
        Hd = H - mu * eye                                  # damp toward neg-definite
        step = torch.linalg.solve(Hd, g.unsqueeze(-1)).squeeze(-1)
        T = T - step
    return T


def distinct_maxima(Tn, fn, tol=2.0, keep=44, minlen=20.0):
    """cluster converged vectors, rank by OBJECTIVE VALUE f (not basin count)."""
    s = np.where(Tn[:, 0] != 0, np.sign(Tn[:, 0]), 1.0); Tc = Tn * s[:, None]
    seen = {}; out = []
    for j in np.argsort(-fn):
        if np.linalg.norm(Tc[j]) < minlen: continue
        key = tuple(np.round(Tc[j] / tol).astype(int))
        if key in seen: continue
        seen[key] = 1; out.append(Tc[j])
        if len(out) >= keep: break
    return np.array(out)


def distinct_maxima_gpu(T, f, tol=2.0, keep=44, minlen=20.0):
    """GPU port of distinct_maxima: stays ON DEVICE (no host round-trip). Canonicalize the sign,
    drop |v|<minlen, keep the highest-f vector per tol-voxel, return the top-`keep` by f (device
    tensor, f-descending). Exact match to distinct_maxima up to argsort ties on equal f."""
    s = torch.where(T[:, 0] != 0, torch.sign(T[:, 0]), torch.ones_like(T[:, 0]))
    Tc = T * s[:, None]
    m = Tc.norm(dim=1) >= minlen
    Tc, fm = Tc[m], f[m]
    if Tc.shape[0] == 0:
        return Tc
    order = torch.argsort(fm, descending=True)                # process high-f first (greedy)
    Tc = Tc[order]
    key = torch.round(Tc / tol).long(); key = key - key.amin(0)
    b1 = int(key[:, 1].max()) + 1; b2 = int(key[:, 2].max()) + 1
    K = (key[:, 0] * b1 + key[:, 1]) * b2 + key[:, 2]         # 3D voxel -> 1D hash
    uniq, inv = torch.unique(K, return_inverse=True)
    N = K.shape[0]
    firstpos = torch.full((uniq.shape[0],), N, device=T.device, dtype=torch.long)
    firstpos.scatter_reduce_(0, inv, torch.arange(N, device=T.device), reduce="amin", include_self=True)
    keepmask = torch.zeros(N, dtype=torch.bool, device=T.device)
    keepmask[firstpos] = True                                 # first (=max-f) row per voxel
    return Tc[keepmask][:keep]                                # position order == f-descending


def distinct_cells_gpu(M, key, tol=1.0):
    """Group annealed cells M (B,3,3) by a rotation+permutation-invariant METRIC signature -- the
    sorted sqrt-eigenvalues of the Gram G=M^T M (principal-axis lengths; capture both lengths AND
    angles), rounded to `tol` A -- and return the MAX-key representative index per group, sorted by
    key descending (invalid key<=-1e8 dropped). Lets index_blind_nbest reduce only the few DISTINCT
    cells instead of all ~1200 triplets: ~1000 triplets converge to the identical setting -> one
    signature. Exactness preserved because same_lattice still arbitrates the reps in the loop, and a
    candidate dropped here shares a rep of >= its score that reduces to the same lattice."""
    G = torch.einsum('bji,bjk->bik', M, M)                    # Gram = M^T M  (B,3,3)
    ev = torch.linalg.eigvalsh(G).clamp_min(0.0).sqrt()       # (B,3) principal lengths, ascending
    s = torch.round(ev / tol).long()                          # rounded signature (already sorted asc)
    s = s - s.amin(0)
    b1 = int(s[:, 1].max()) + 1; b2 = int(s[:, 2].max()) + 1
    K = (s[:, 0] * b1 + s[:, 1]) * b2 + s[:, 2]               # 3-int signature -> 1D hash
    order = torch.argsort(key, descending=True)               # score desc
    Ks = K[order]
    uniq, inv = torch.unique(Ks, return_inverse=True)
    N = Ks.shape[0]
    firstpos = torch.full((uniq.shape[0],), N, device=M.device, dtype=torch.long)
    firstpos.scatter_reduce_(0, inv, torch.arange(N, device=M.device), reduce="amin", include_self=True)
    reps = order[firstpos]                                     # max-key rep per signature group
    reps = reps[key[reps] > -1e8]                             # drop invalid cells
    return reps[torch.argsort(key[reps], descending=True)]


# ---- M5: residual-threshold annealing (cell refine) -----------------------------
def anneal(M, Q, thr0=0.25, contract=0.85, max_iter=15, min_thr=0.02):
    """M5: closed-form OLS fit to inliers; trim residuals; anneal threshold down.
    (TORO residual-threshold-annealing == ffbidx ifss.) M columns = real axes a,b,c."""
    thr = thr0
    for _ in range(max_iter):
        H = Q @ M; hkl = np.rint(H)
        inl = np.abs(H - hkl).max(1) < thr
        if inl.sum() < 6: break
        try:
            M = np.linalg.lstsq(Q[inl], hkl[inl], rcond=None)[0]   # closed-form OLS
        except np.linalg.LinAlgError:
            break
        thr = max(thr * contract, min_thr)
    return M


# D1: (condition on inlier hkl, conventional->primitive column transform M_p = M @ P).
# A non-primitive (centered) cell's reflections ALL satisfy a parity condition; transform
# to the primitive cell (halves/quarters the volume). Order F(4x) before the 2x centerings.
_CENTERINGS = [
    (lambda h, k, l: (h % 2 == k % 2) & (k % 2 == l % 2),                       # F
     np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])),
    (lambda h, k, l: (h + k + l) % 2 == 0,                                      # I
     np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]])),
    (lambda h, k, l: (h + k) % 2 == 0, np.array([[.5, .5, 0], [.5, -.5, 0], [0, 0, 1.]])),   # C
    (lambda h, k, l: (k + l) % 2 == 0, np.array([[1., 0, 0], [0, .5, .5], [0, .5, -.5]])),   # A
    (lambda h, k, l: (h + l) % 2 == 0, np.array([[.5, 0, .5], [0, 1., 0], [.5, 0, -.5]])),   # B
]
_CENTER = os.environ.get("CENTER", "0") == "1"      # D1 centering reduce: OFF by default
                                                    # (regressed D3/D4 deflate 90->53; opt-in)


def primitivize(M, Q):
    """M5b: DATA-DRIVEN supercell reduction. If the inlier hkl along a real axis are
    ALL even, that axis is doubled (a supercell that over-fits inter-layer noise) ->
    halve it and re-anneal. Iterate. Catches the doubled-axis selection miss (e.g.
    75.5=2x37.8) that geometric buerger reduction alone cannot. cxidb 65%->68% blind,
    neutral on rich frames. D1: also detect CENTERING (h+k even etc. -- a combination,
    not a single axis -> the sqrt2 face-diagonal / centered supercell leaks) and apply
    the conventional->primitive transform."""
    M = M.copy()
    for _ in range(3):
        H = Q @ M; hkl = np.rint(H); inl = np.abs(H - hkl).max(1) < 0.15
        n0 = int(inl.sum())
        if n0 < 8:
            break
        Mt = M.copy(); reduced = False
        for j in range(3):
            nz = hkl[inl, j].astype(int); nz = nz[nz != 0]
            if len(nz) >= 5 and np.all(nz % 2 == 0):
                Mt[:, j] = Mt[:, j] / 2.0; reduced = True
        if _CENTER and not reduced:                         # D1: centering (combination parity)
            hk = hkl[inl].astype(int); hk = hk[np.any(hk != 0, 1)]
            if len(hk) >= 10:
                h, k, l = hk[:, 0], hk[:, 1], hk[:, 2]
                for cond, P in _CENTERINGS:
                    if cond(h, k, l).mean() > 0.9:          # centered ~1.0 vs primitive ~0.5
                        Mt = M @ P; reduced = True; break
        if not reduced:
            break
        Mt = anneal(Mt, Q)
        Hn = Q @ Mt; n1 = int((np.abs(Hn - np.rint(Hn)).max(1) < 0.15).sum())
        if n1 < 0.9 * n0:                                   # GUARD: spurious reduction (multi-lattice
            break                                           # false trigger) loses spots -> revert
        M = Mt
    return buerger_reduce(M)


# ---- M6: cell score (SWAP POINT: weak-peak-aware) -------------------------------
def score_defect(M, Q, inl_tol=0.15, cover=0.30):
    """M6 blind: COVERAGE-GATED defect. key = (covers>=30%? , -defect). The ORACLE
    diagnostic showed the true cell is reachable ~82% but pure-defect picks only ~58%:
    the spurious winners are TIGHT sub-lattices indexing FEW spots. Preferring cells
    that index >=`cover` of spots, THEN tightest, filters those out (cxidb 55%->65%
    blind) with graceful fallback when no cell reaches cover. SWAP POINT for weak-peak
    completeness."""
    H = Q @ M; dd = np.abs(H - np.rint(H)); inl = dd.max(1) < inl_tol
    ni = int(inl.sum())
    if ni < 8:
        return None
    frac = ni / len(Q)
    return (1 if frac >= cover else 0, -float(dd[inl].mean()))    # high coverage first, then tightest


# ---- M4 + pipeline --------------------------------------------------------------
def index_blind(q, weight_fn=invq_weight, scorer=score_defect, n_top=30, starts=None):
    """Blind index: M1->M3 (ascend) -> rank by f -> M4 triplets -> M5 anneal -> M6 score."""
    q = np.asarray(q, float)
    if len(q) < 6:
        return None
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = weight_fn(Q)
    if starts is None:
        starts = STARTS
    T = refine_vec(starts.clone(), Q, w, qmax)
    f, _ = objective(T, Q, w, sharp=True)
    cands = distinct_maxima(T.cpu().numpy(), f.cpu().numpy())[:n_top]
    if len(cands) < 3:
        return None
    best = None; bk = None
    for tri in itertools.combinations(range(len(cands)), 3):     # M4: triplet assembly
        M0 = cands[list(tri)].T
        sc = np.prod([np.linalg.norm(cands[t]) for t in tri])
        if sc <= 0 or abs(np.linalg.det(M0)) < 0.1 * sc:
            continue
        M = anneal(M0, q)                                        # M5
        key = scorer(M, q)                                      # M6 (swap point)
        if key is None:
            continue
        if bk is None or key > bk:
            bk = key; best = buerger_reduce(M)
    return primitivize(best, q) if best is not None else None       # M5b: de-double supercells


STARTS = sample().to(DEV)


if __name__ == "__main__":
    import time
    from fftindex.lattice import cell_to_Ar
    from fftindex.multishot import same_lattice
    LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
    def load(p):
        fr = []; L = open(p).read().split("\n"); i = 0
        while i < len(L):
            if not L[i].startswith("FRAME"): i += 1; continue
            _, fid, n = L[i].split(); n = int(n)
            fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
        return fr
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/frames_cxidb.txt"
    frames = load(path); lim = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:lim]
    ok = n = 0; t0 = time.time()
    for q in frames:
        if len(q) < 6: continue
        n += 1; M = index_blind(q); ok += M is not None and same_lattice(M, LYSO)
    print(f"device={DEV} starts={STARTS.shape[0]}  GLINT modular BLIND: {ok}/{n} "
          f"({100*ok/n:.0f}%)  {1e3*(time.time()-t0)/n:.0f} ms/frame")
