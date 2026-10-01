"""Batched-across-frames known-cell engine (production). Same algorithm as
replica_gpu.index_known_gpu_cell, but F frames processed in one batched pass
(pad peaks to Pmax + mask), collapsing the ~5215 tiny kernels/frame ~F-fold.
Keeps replica_gpu (per-frame) and the ffbidx handoff as the other two options.

index_all_graph() captures the whole GPU stage as a CUDA graph (one replay per batch instead of
thousands of host kernel dispatches -- the engine is host-dispatch-bound). It needs a fixed shape,
so it sorts frames by peak count and routes each batch to the smallest of a few Pmax buckets; this
is BIT-IDENTICAL to index_known_gpu_cell_batch (per-frame independence + masked padding) and ~1.3x
faster on real cxidb / ~1.5x at deployment peak-caps. Enabled by the analytic solve/det (cuSOLVER
is uncapturable and forces a stream sync)."""
import os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("CDIRS", "16384")
import numpy as np, torch
from glint.replica_gpu import (DIRS, TRIML, TRIMH, DELTA, NC, NANG, AXIS0_DEDUP_COS, _axes_from_cell,
                               _third_axis, _fib_halfsphere, _azimuth_grid, _depth,
                               _both_hands, _ref_setting)
from glint.multishot import same_lattice

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PI = np.pi
# The anchor pool: the best 120 grid directions, refined, then deduplicated down to nc (replica_gpu.
# axis_candidates_t, the per-frame search, keeps at most the distinct survivors of the same 120).
ANCHOR_POOL = 120
# Working precision (KC_FP: "32" default | "64") and the torch path's 3x3-solve precision (KC_SOLVE_FP: "64"
# default | "32"). fp32 is measured rate/lattice-IDENTICAL to fp64: across all lattice systems + sparse frames
# (2026-07-18), and on the cxidb-17 120 and 480 on exclusive A100s (26 Sep 2026, jobs 39211371, 39212408 and
# 39218788/39218965/39219158 on three nodes: 80/120 and 308/480 strict in both precisions at every batch size; the
# published streaming replay, 333/480 with 10 watchdog rescues and 1 relock, is decision-identical in both, job
# 39218734), at 12-19 % less time at B=120 on a warm A100 (3-10 % at B=32-64) and far less on fp32-strong GPUs (RTX Blackwell, L40S: fp64 at 1/32-1/64 of the fp32 rate). The default is fp32 since
# 26 Sep 2026; KC_FP=64 restores the fp64 path the published timings (0.17 ms/frame at B=120) were measured on.
# KC_SOLVE_FP=64 keeps the tiny 3x3 normal-equation solve in fp64 as a hedge for ill-conditioned (near-coplanar,
# sparse) frames on the TORCH path (index_known_gpu_cell_batch, index_all_graph, the CPU and no-cupy fallbacks).
# The fused kernels (index_fused) solve inside their anneal kernel at the working precision: KC_FP=64 is theirs.
_W = os.environ.get("KC_FP", "32"); _S = os.environ.get("KC_SOLVE_FP", "64")
FP = torch.float32 if _W == "32" else torch.float64
_SOLVE_FP = torch.float32 if _S == "32" else torch.float64
_DIRS = DIRS.to(FP)          # azimuth grid is per-cell now: _cell_params -> _azimuth_grid(c01)

# Analytic 3x3 solve + det (pure elementwise, no cuSOLVER). cuSOLVER's linalg.solve/det each force
# a STREAM SYNC, which serializes _gpu_stage and makes it impossible to overlap the host tail with
# the next batch (and impossible to CUDA-graph). These closed forms remove the sync; bit-safe vs
# cuSOLVER (~1e-13, argmax-selection stable, rate-neutral 2026-07-17). KC_ANALYTIC=0 restores cuSOLVER.
_ANALYTIC = os.environ.get("KC_ANALYTIC", "1") != "0"


def solve3x3(A, rhs):
    """Solve A X = rhs for A:(...,3,3), rhs:(...,3,k) via closed-form inverse (adjugate/det). The
    linear algebra runs at _SOLVE_FP (default fp64); result cast back to A's dtype.
    A fp64 solve under fp32 working precision (KC_SOLVE_FP=64) is the 'mixed' hedge for ill-conditioned
    (near-coplanar / sparse) frames."""
    wdt = A.dtype
    if wdt != _SOLVE_FP:
        A = A.to(_SOLVE_FP); rhs = rhs.to(_SOLVE_FP)
    a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    a10, a11, a12 = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    a20, a21, a22 = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    c00 = a11 * a22 - a12 * a21; c01 = a12 * a20 - a10 * a22; c02 = a10 * a21 - a11 * a20
    c10 = a02 * a21 - a01 * a22; c11 = a00 * a22 - a02 * a20; c12 = a01 * a20 - a00 * a21
    c20 = a01 * a12 - a02 * a11; c21 = a02 * a10 - a00 * a12; c22 = a00 * a11 - a01 * a10
    det = a00 * c00 + a01 * c01 + a02 * c02
    inv = torch.stack([torch.stack([c00, c10, c20], -1),
                       torch.stack([c01, c11, c21], -1),
                       torch.stack([c02, c12, c22], -1)], -2) / det[..., None, None]
    out = inv @ rhs
    return out.to(wdt) if out.dtype != wdt else out


def det3(M):
    """Determinant of M:(...,3,3), elementwise (no cuSOLVER)."""
    return (M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 1])
            - M[..., 0, 1] * (M[..., 1, 0] * M[..., 2, 2] - M[..., 1, 2] * M[..., 2, 0])
            + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1] - M[..., 1, 1] * M[..., 2, 0]))


# Constant tensors hoisted to module scope: creating them from host lists (torch.tensor/eye/arange)
# is a host->device copy that is NOT permitted inside a CUDA-graph capture, so build them once.
_EX = torch.tensor([1., 0., 0.], dtype=FP, device=DEV)
_EY = torch.tensor([0., 1., 0.], dtype=FP, device=DEV)
_EYE3 = torch.eye(3, dtype=FP, device=DEV)
_ARANGE = {}


def _arange(F):
    a = _ARANGE.get(F)
    if a is None:
        a = _ARANGE[F] = torch.arange(F, device=DEV)
    return a

# --- adaptive orientation-grid density (cross-cell-validated 2026-07-17) ---
# 4096 dirs is rate-neutral for orthogonal cells (all angles ~90: cubic/tet/ortho, incl. the
# anisotropic long-axis case) at ~1.3x speed, but regresses triclinic ~8pts. So: coarse grid for
# all-90 cells, full CDIRS grid for oblique (mono/tricl/rhomb). KC_ADAPTIVE_DIRS=0 forces full.
_ADAPT = os.environ.get("KC_ADAPTIVE_DIRS", "1") != "0"
_DIRS_LO = torch.as_tensor(_fib_halfsphere(min(4096, int(os.environ.get("CDIRS", "16384")))), dtype=FP, device=DEV)


def _adaptive_dirs(Mc):
    """Coarse (4096) Fibonacci half-sphere for orthogonal known cells; full grid for oblique."""
    if not _ADAPT:
        return _DIRS
    M = np.asarray(Mc, float); a, b, c = M[:, 0], M[:, 1], M[:, 2]

    def _ang(u, v):
        return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1.0, 1.0)))
    orthogonal = all(abs(x - 90.0) < 2.0 for x in (_ang(b, c), _ang(a, c), _ang(a, b)))
    return _DIRS_LO if orthogonal else _DIRS


def pad(frames, Pmax):
    F = len(frames); Q = torch.zeros(F, Pmax, 3, dtype=FP, device=DEV)
    m = torch.zeros(F, Pmax, dtype=torch.bool, device=DEV)
    for i, f in enumerate(frames):
        k = len(f); Q[i, :k] = torch.as_tensor(np.asarray(f, float), dtype=FP, device=DEV); m[i, :k] = True
    return Q, m


def obj_b(V, Q, m):                                   # V:(F,K,3) Q:(F,P,3) m:(F,P)
    proj = torch.einsum('fkc,fpc->fkp', V, Q); d = torch.abs(proj - torch.round(proj))
    inl = ((d < TRIMH) & m[:, None, :]).sum(2)
    dcl = torch.log2(torch.clamp(d, TRIML, TRIMH) + DELTA)
    sub = (dcl * m[:, None, :]).sum(2) / m.sum(1, keepdim=True).clamp(min=1)
    return inl, sub


def refine_b(V, Q, m, steps=30):
    qmax = (Q.norm(dim=2) * m).amax(1).clamp(min=1e-9); npk = m.sum(1).clamp(min=1)
    lr = (1.0 / (4 * PI ** 2 * npk * qmax ** 2))[:, None, None]
    for _ in range(steps):
        s = torch.sin(2 * PI * torch.einsum('fkc,fpc->fkp', V, Q)) * m[:, None, :]
        V = V - lr * (2 * PI * torch.einsum('fkp,fpc->fkc', s, Q))
    return V


def anneal_b(M0, Q, m, thr0=0.25, contract=0.85, max_iter=15, min_thr=0.02):
    M = M0.clone(); thr = thr0
    eyes = 1e-9 * _EYE3[None, None]; mk = m[:, None, :]
    for _ in range(max_iter):
        H = torch.einsum('fpc,fkcd->fkpd', Q, M); hkl = torch.round(H)
        inl = ((torch.abs(H - hkl).amax(3) < thr) & mk).to(FP)
        cnt = inl.sum(2); WQ = inl[..., None] * Q[:, None, :, :]
        A = torch.einsum('fkpc,fpd->fkcd', WQ, Q) + eyes
        rhs = torch.einsum('fkpc,fkpd->fkcd', WQ, hkl)
        sol = solve3x3(A, rhs) if _ANALYTIC else torch.linalg.solve(A, rhs)
        M = torch.where((cnt >= 6)[..., None, None], sol, M)
        thr = max(thr * contract, min_thr)
    return M


def _cell_params(Mc, topa=8, nc=None, full_grid=False):
    """Host-side cell geometry (numpy) + the orientation grid, hoisted out of the compute so _stage_compute
    is pure-torch (and CUDA-graph capturable for a fixed cell). The search depth rides along in P: topa
    (azimuths kept per anchor), nc (anchor directions kept; default the module's NC) and the grid -- the
    adaptive one (4096 directions for orthogonal cells) unless full_grid, which is the CDIRS grid the
    per-frame search uses (the escalation's deep search needs it: the adaptive grid changes which misses
    it recovers, exp/batched-escalation RESULTS_batched_deep.md)."""
    topa = _depth("topa", topa)
    # nc is capped at the pool: the per-frame search cannot keep more anchors than the pool has, and past it
    # the dedup loop below would only fill copies of the first anchor (argmax of an all-false row is 0)
    # while every tensor still grew with nc (Copilot review of #211).
    nc = min(NC if nc is None else _depth("nc", nc), ANCHOR_POOL)
    L, c01, c02, c12, sgn = _axes_from_cell(Mc)
    ca, sa = _azimuth_grid(c01)                     # half turn iff perpendicular (see replica_gpu)
    return (float(L[0]), float(L[1]), float(L[2]), c01, c02, c12, sgn,
            _DIRS if full_grid else _adaptive_dirs(Mc), topa, ca.to(FP), sa.to(FP), nc)


def _stage_compute(Q, m, P):
    """Pure-torch known-cell compute on padded (F,Pmax,3) Q + (F,Pmax) m; returns GPU tensors
    (best, pol, mp, mainb). No cuSOLVER (analytic solve/det) and static-shape => CUDA-graph
    capturable, which collapses the ~thousands of host kernel dispatches (the dominant cost)."""
    L0, L1, L2, c01, c02, c12, sgn, dirs, topa, _ca, _sa, nc = P
    F = Q.shape[0]
    # --- anchor search (shortest axis) + greedy dedup to nc ---
    V0 = (L0 * dirs)[None].expand(F, -1, -1)
    inl, sub = obj_b(V0, Q, m); top = (inl.double() * 100 - sub).topk(ANCHOR_POOL, 1).indices
    Vsel = torch.gather(V0, 1, top[:, :, None].expand(-1, -1, 3))
    Vref = refine_b(Vsel, Q, m, 30); Vref = Vref / Vref.norm(dim=2, keepdim=True) * L0
    inl2, sub2 = obj_b(Vref, Q, m); order = (inl2.double() * 100 - sub2).argsort(1, descending=True)
    Vs = torch.gather(Vref, 1, order[:, :, None].expand(-1, -1, 3)); dirs2 = Vs / L0
    alive = torch.ones(F, ANCHOR_POOL, dtype=torch.bool, device=DEV)
    chosen = torch.zeros(F, nc, 3, dtype=FP, device=DEV)
    ar = _arange(F)
    for k in range(nc):
        idx = alive.int().argmax(1)
        chosen[:, k, :] = Vs[ar, idx]
        dk = dirs2[ar, idx]                                # dedup radius: see AXIS0_DEDUP_COS (replica_gpu.py)
        alive = alive & (torch.abs(torch.einsum('fpc,fc->fp', dirs2, dk)) < AXIS0_DEDUP_COS)
    # --- orientation sweep per anchor ---
    C = chosen; cn = C / C.norm(dim=2, keepdim=True)
    tmp = torch.where(cn[..., :1].abs() < 0.9, _EX, _EY)
    u = torch.cross(cn, tmp, dim=2); u = u / u.norm(dim=2, keepdim=True); v = torch.cross(cn, u, dim=2)
    s01 = float(np.sqrt(max(1.0 - c01 * c01, 0.0)))
    a1 = L1 * (c01 * cn[:, :, None, :] +
               s01 * (_ca[None, None, :, None] * u[:, :, None, :] + _sa[None, None, :, None] * v[:, :, None, :]))
    a0 = C[:, :, None, :].expand(-1, -1, NANG, -1)
    inl1, _ = obj_b(a1.reshape(F, nc * NANG, 3), Q, m); inl1 = inl1.reshape(F, nc, NANG)
    ta = min(topa, NANG); topi = inl1.topk(ta, 2).indices
    a0s = torch.gather(a0, 2, topi[..., None].expand(-1, -1, -1, 3)).reshape(F, nc * ta, 3)
    a1s = torch.gather(a1, 2, topi[..., None].expand(-1, -1, -1, 3)).reshape(F, nc * ta, 3)
    a2s = _third_axis(a0s.reshape(-1, 3), a1s.reshape(-1, 3), L2, c02, c12, sgn).reshape(F, nc * ta, 3)
    both = _both_hands(L2, c01, c02, c12)              # host bool, fixed per cell: graph capture unaffected
    if both:                                           # s4-01: seed the other hand too (see replica_gpu._both_hands)
        a2m = _third_axis(a0s.reshape(-1, 3), a1s.reshape(-1, 3), L2, c02, c12, -sgn).reshape(F, nc * ta, 3)
        a0s = torch.cat([a0s, a0s], 1); a1s = torch.cat([a1s, a1s], 1); a2s = torch.cat([a2s, a2m], 1)
    M0 = torch.stack([a0s, a1s, a2s], dim=3)
    det = torch.abs(det3(M0) if _ANALYTIC else torch.linalg.det(M0)); validM = det >= 1e3
    M0 = torch.where(validM[..., None, None], M0, _EYE3[None, None])
    Mt = anneal_b(M0, Q, m, thr0=0.30, contract=0.82, max_iter=20, min_thr=0.02)
    H = torch.einsum('fpc,fkcd->fkpd', Q, Mt); dd = torch.abs(H - torch.round(H))
    main = ((dd.amax(3) < 0.15) & m[:, None, :]).sum(2)
    subm = (torch.log2(torch.clamp(dd, TRIML, TRIMH) + DELTA).mean(3) * m[:, None, :]).sum(2) / m.sum(1, keepdim=True).clamp(min=1)
    key = torch.where(validM, main.double() * 1000 - subm, torch.full_like(main.double(), -1e18))
    bidx = key.argmax(1); best = Mt[ar, bidx]; mainb = main[ar, bidx]
    if both:                                           # back to the reference hand: -M is hkl -> -hkl, same spots
        best = torch.where((det3(best) * sgn < 0)[:, None, None], -best, best)
    # --- guarded polish, batched ---
    pol = anneal_b(best[:, None], Q, m, thr0=0.30, contract=0.85, max_iter=10, min_thr=0.15)[:, 0]
    Hp = torch.einsum('fpc,fcd->fpd', Q, pol); mp = ((torch.abs(Hp - torch.round(Hp)).amax(2) < 0.15) & m).sum(1)
    return best, pol, mp, mainb


def _gpu_stage(frames, Mc, topa=8, nc=None, full_grid=False):
    """Eager wrapper: pad this batch then run the pure-torch stage compute."""
    P = _cell_params(Mc, topa, nc, full_grid)        # validates the depth before any GPU work
    Pmax = max(len(f) for f in frames)
    Q, m = pad(frames, Pmax)
    return _stage_compute(Q, m, P)


def _cpu_stage(best, pol, mp, mainb, Mc):
    """Host tail: D2H copy + the per-frame same_lattice(Buerger) guard choosing polished vs
    pre-polish cell. Pure host work (~25% of wall) that blocks nothing on the GPU."""
    bn = best.cpu().numpy(); pn = pol.cpu().numpy(); mpv = mp.cpu().numpy(); mbv = mainb.cpu().numpy()
    Mcn = np.asarray(Mc, float); out = []
    for i in range(len(bn)):
        if mpv[i] >= mbv[i] and same_lattice(pn[i], Mcn):
            out.append(pn[i])
        else:
            out.append(bn[i])
    return out


def index_known_gpu_cell_batch(frames, Mc, topa=8, nc=None, full_grid=False):
    """Returns list of M (3x3 np) or None, one per frame -- single batched pass over all F.
    topa / nc / full_grid set the search depth (see _cell_params); the defaults are the shipped search."""
    return _settle(_cpu_stage(*_gpu_stage(frames, Mc, topa, nc, full_grid), Mc), Mc)


def _settle(res, Mc):
    """replica_gpu._ref_setting on every frame, for the cells that take the two-handed seed (s4-01). Applied
    after _cpu_stage, which the fused kernels replace with their own (fused_kernels.patch(cpu=True))."""
    L, c01, c02, c12, _ = _axes_from_cell(Mc)
    return [_ref_setting(M, Mc) for M in res] if _both_hands(float(L[2]), c01, c02, c12) else res


def index_known_deep_batch(frames, Mc, topa, nc, full_grid=True, budget=12000, return_errors=False):
    """The escalation's deep known-cell search, batched: frames sorted by peak count and chunked so that
    frames x Pmax <= budget, each chunk one index_known_gpu_cell_batch call at the given depth, on the full
    CDIRS grid by default. On a GPU with cupy the fused kernels are patched in, as index_fused does; frames
    too large for the fused kernels' shared memory go through the stock path one at a time. Elsewhere it is
    the torch path. A frame whose search raises comes back as a miss (None), as in index_fused; with
    return_errors=True the call returns (results, errors), errors[j] True where frame j's search raised,
    because a caller running a NULL must not read an unevaluated control as a copy that matched nothing
    (glint.retry_cascade.escalate_batch fails those frames closed).

    Equivalence (exp/batched-escalation RESULTS_batched_deep.md, S3DF job 39203178, fp64): at topa 128,
    nc 32 on the full grid, every one of 4,653 searches (141 real frames, 4,512 scrambled copies) gave the
    same matched count as the per-frame replica_gpu.index_known_gpu_cell, fused 0.9-1.8 ms per search
    against about 20 ms per frame. The per-frame search is always fp64; this one runs at the module's
    working precision (KC_FP), so KC_FP=64 is the configuration that equivalence was measured in."""
    topa = _depth("topa", topa); nc = _depth("nc", nc)
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError(f"budget must be an integer >= 1, got {budget!r}")
    frames = [np.asarray(f, float) for f in frames]
    out = [None] * len(frames)
    err = [False] * len(frames)
    live = [j for j in range(len(frames)) if len(frames[j]) >= 6]
    fk = None
    if DEV == "cuda":
        try:
            from glint import fused_kernels as fk
        except Exception:                          # noqa: BLE001 -- no cupy: the torch path
            fk = None
    cap = fk.max_peaks() if fk is not None else None
    order = sorted(live, key=lambda j: len(frames[j]))
    fits = [j for j in order if cap is None or len(frames[j]) <= cap]
    over = [j for j in order if cap is not None and len(frames[j]) > cap]
    if fk is not None:
        fk.patch(anneal=True, obj=True, refine=True, cpu=True)
    try:
        s = 0
        while s < len(fits):
            e = s + 1
            while e < len(fits) and (e + 1 - s) * len(frames[fits[e]]) <= budget:
                e += 1
            chunk = fits[s:e]
            try:
                res = index_known_gpu_cell_batch([frames[j] for j in chunk], Mc, topa, nc, full_grid)
            except Exception as ex:                # noqa: BLE001 -- a miss, not a death
                warnings.warn(f"index_known_deep_batch: a chunk of {len(chunk)} frames failed "
                              f"({type(ex).__name__}: {ex}); returning them as misses", RuntimeWarning)
                res = [None] * len(chunk)
                for j in chunk:
                    err[j] = True
            for j, r in zip(chunk, res):
                out[j] = r
            s = e
    finally:
        if fk is not None:
            fk.unpatch()
    for j in over:                                 # the stock ops, one frame at a time (as index_fused)
        try:
            out[j] = index_known_gpu_cell_batch([frames[j]], Mc, topa, nc, full_grid)[0]
        except Exception as ex:                    # noqa: BLE001
            err[j] = True
            warnings.warn(f"index_known_deep_batch: frame with {len(frames[j])} peaks failed on the "
                          f"non-fused path ({type(ex).__name__}: {ex}); returning it as a miss", RuntimeWarning)
    return (out, err) if return_errors else out


def _pad_fixed(frames, F, Pmax):
    """Pad a (<=F)-frame batch to EXACTLY (F, Pmax): missing frames are all-zero+masked dummies
    (their garbage outputs are sliced off); peaks beyond Pmax are dropped (Pmax = global max so
    no real drop). Fixed shape is required for a single reusable CUDA graph."""
    Q = torch.zeros(F, Pmax, 3, dtype=FP, device=DEV)
    m = torch.zeros(F, Pmax, dtype=torch.bool, device=DEV)
    for i, f in enumerate(frames):
        k = min(len(f), Pmax); Q[i, :k] = torch.as_tensor(np.asarray(f, float)[:k], dtype=FP, device=DEV); m[i, :k] = True
    return Q, m


class _StageGraph:
    """Captures _stage_compute at fixed (F, Pmax) for a fixed cell: one graph replay per batch
    instead of the ~thousands of host kernel dispatches the engine is bound by."""
    def __init__(self, P, F, Pmax):
        self.Q = torch.zeros(F, Pmax, 3, dtype=FP, device=DEV)
        self.m = torch.zeros(F, Pmax, dtype=torch.bool, device=DEV)
        _arange(F)                                        # pre-cache before capture (arange is H2D)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):                       # warm the caching allocator before capture
            for _ in range(3):
                _stage_compute(self.Q, self.m, P)
        torch.cuda.current_stream().wait_stream(st)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self.out = _stage_compute(self.Q, self.m, P)

    def run(self, Q, m):
        self.Q.copy_(Q); self.m.copy_(m); self.g.replay()
        return self.out                                   # STATIC tensors: consume before next run


_GRAPHS = {}
_BUCKETS = (64, 128, 256, 576)                            # Pmax buckets for the graph pool


def _bucket(p, buckets):
    for bkt in buckets:
        if p <= bkt:
            return bkt
    return None                                           # oversized -> eager fallback (no cap)


def index_all_graph(frames, Mc, B=32, buckets=_BUCKETS):
    """Graphed known-cell indexer. Sorts frames by peak count and routes each B-batch to the
    smallest Pmax bucket that fits, replaying a per-bucket captured stage graph (host kernel
    dispatch collapsed). Because each frame indexes INDEPENDENTLY (known cell, no cross-frame
    coupling) and padding is masked out, this is BIT-IDENTICAL to index_known_gpu_cell_batch and
    full-rate -- the sort just makes each graph's fixed Pmax tight instead of over-padding the
    largest frame. Oversized frames (> max bucket) fall back to the eager path. cuSOLVER-free
    (analytic solve, ~1e-13 vs linalg). CPU / capture-unsupported -> eager."""
    if DEV != "cuda":
        return [M for i in range(0, len(frames), B) for M in index_known_gpu_cell_batch(frames[i:i + B], Mc)]
    P = _cell_params(Mc); cell = tuple(np.asarray(Mc, float).ravel().round(6))
    order = sorted(range(len(frames)), key=lambda i: len(frames[i]))   # size-homogeneous batches
    out = [None] * len(frames)
    for s in range(0, len(order), B):
        chunk = order[s:s + B]; fb = [frames[j] for j in chunk]; Fb = len(fb)
        bkt = _bucket(max(len(f) for f in fb), buckets)
        if bkt is None:                                   # frame(s) bigger than any bucket
            res = index_known_gpu_cell_batch(fb, Mc)
        else:
            G = _GRAPHS.get((cell, B, bkt))
            if G is None:
                G = _GRAPHS[(cell, B, bkt)] = _StageGraph(P, B, bkt)
            best, pol, mp, mainb = G.run(*_pad_fixed(fb, B, bkt))
            res = _settle(_cpu_stage(best[:Fb], pol[:Fb], mp[:Fb], mainb[:Fb], Mc), Mc)
        for j, r in zip(chunk, res):
            out[j] = r
    return out


index_batch = index_known_gpu_cell_batch   # alias


def _unfused(frames, Mc, B, topa, nc, full_grid):
    """index_fused without the fused kernels (CPU, or no cupy): the CUDA-graph path at the shipped depth, as
    before; at any other depth the graph is not built for it, so the torch batch path runs in B-sized chunks
    (sorted by peak count, as the fused path pads) at the depth asked for -- slower, same answer."""
    if topa == 8 and (nc is None or int(nc) == NC) and not full_grid:   # the shipped depth, written out or not
        return index_all_graph(frames, Mc, B)
    order = sorted(range(len(frames)), key=lambda i: len(frames[i]))
    out = [None] * len(frames)
    for s in range(0, len(order), B):
        chunk = order[s:s + B]
        res = index_known_gpu_cell_batch([frames[j] for j in chunk], Mc, topa, nc, full_grid)
        for j, r in zip(chunk, res):
            out[j] = r
    return out


def index_fused(frames, Mc, B=32, topa=8, nc=None, full_grid=False):
    """Fully-fused known-cell indexer: custom fused CUDA kernels (cupy RawKernel, nvrtc-JIT) for the
    anneal/obj/refine per-candidate hot loops + an on-device cpu_stage (batched buerger same_lattice),
    replacing the many small per-stage torch kernels AND the host tail. At B=32 on one A100: 0.31
    ms/frame fp32 / 0.33 fp64 -- 6.6x (fp32) / 4.3x (fp64) over index_all_graph -- with per-frame output
    numerically equivalent in fp64 (max|ΔM| 1.42e-13; rate + lattice identical in both precisions,
    80/115 on 120 cxidb) to the stock engine.
    Sorts frames by peak count so each batch pads to its own tight Pmax. Requires cupy on a GPU; falls
    back to index_all_graph (graph path) when cupy is unavailable or on CPU.

    Throughput scales with the batch B: anneal and refine give each frame one thread-block, so B
    still sets their occupancy (obj splits its candidates across blockIdx.y since #165, but that did
    NOT remove the need to batch -- measured, B=16 is 0.579 ms/fr against B=120's 0.170).
    120 cxidb frames: B=32 -> 0.31/0.33 ms/fr fp32/fp64; B=64 -> 0.19/0.21; B=120 -> 0.14/0.17.
    B=96 is no better than B=64 in either precision, but that is batch-count quantisation, not
    occupancy -- 120 frames at B=96 is a ragged 96+24 while B=120 is one batch. Output is
    batch-invariant -- the kernels loop each frame's real peak count, not Pmax, so a looser
    per-batch pad costs no ARITHMETIC (fp64 bit-identical across B). It is not unconditionally free:
    every block reserves Pmax*3*_IB dynamic shared memory, which caps resident blocks per SM.
    MEASURED (A100, fp64, K=4096, real P pinned at 200, only the pad varied): flat to Pmax 800
    (1.00x), 1.02x at 1600, 1.29x at 3200 -- the reservation only bites once residency falls to ~4
    blocks/SM. At the peak counts this code sees (cxidb-17 tops out at 554: ~13 KB/block, ~12
    blocks/SM) the pad really is free.

    Each frame's peaks are staged in dynamic shared memory, so a frame with more peaks than the
    device can hold in one block (fused_kernels.max_peaks(): ~6954 fp64 / 13909 fp32 on an A100)
    cannot use the fused kernels. Such frames are split out and run ONE AT A TIME through the stock
    torch path instead, which has no shared-memory limit -- same answer, just slower. Nothing about
    an oversized frame raises: an indexer that dies on a dense frame takes StreamDriver.flush() with
    it, so a frame that cannot be indexed is returned as a miss (None).

    topa / nc / full_grid set the search depth as in index_known_gpu_cell_batch. The defaults are the shipped
    search, and with them every call is the call it was. Deeper tiers (StreamDriver's effort policy): the
    depth sweep on the cxidb-17 480 gave 308 / 341 / 365 / 379 frames at 0.15 / 0.27 / 0.40 / 0.94 ms per
    frame for (topa, nc) = (8, 16), (32, 16), (32, 32) and (128, 32) on the full grid, B=120, one A100
    (branch exp/batched-escalation)."""
    if DEV != "cuda":
        return _unfused(frames, Mc, B, topa, nc, full_grid)
    try:
        from glint import fused_kernels as _fk
    except Exception:
        return _unfused(frames, Mc, B, topa, nc, full_grid)
    cap = _fk.max_peaks()
    order = sorted(range(len(frames)), key=lambda i: len(frames[i]))   # tight per-batch Pmax
    fits = [j for j in order if len(frames[j]) <= cap]
    over = [j for j in order if len(frames[j]) > cap]
    out = [None] * len(frames)
    _fk.patch(anneal=True, obj=True, refine=True, cpu=True)
    try:
        for s in range(0, len(fits), B):
            chunk = fits[s:s + B]
            res = index_known_gpu_cell_batch([frames[j] for j in chunk], Mc, topa, nc, full_grid)
            for j, r in zip(chunk, res):
                out[j] = r
    finally:
        _fk.unpatch()
    for j in over:                     # unpatched: the stock ops, one frame at a time. The stock
        try:                           # path materialises (F, NC*NANG, Pmax) tensors, so at these
            out[j] = index_known_gpu_cell_batch([frames[j]], Mc, topa, nc, full_grid)[0]   # peak counts F must stay 1.
        except Exception as e:                                        # OOM, etc: a miss, not a death
            warnings.warn(f"index_fused: frame with {len(frames[j])} peaks failed on the non-fused "
                          f"fallback ({type(e).__name__}: {e}); returning it as a miss", RuntimeWarning)
            out[j] = None
    return out
