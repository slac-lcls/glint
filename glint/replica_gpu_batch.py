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
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("CDIRS", "16384")
import numpy as np, torch
from glint.replica_gpu import (DIRS, TRIML, TRIMH, DELTA, NC, NANG, AXIS0_DEDUP_COS, _axes_from_cell,
                               _third_axis, _fib_halfsphere, _azimuth_grid)
from glint.multishot import same_lattice

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PI = np.pi
# Working precision (KC_FP: "64" default | "32") and 3x3-solve precision (KC_SOLVE_FP: defaults to
# KC_FP). fp32 is measured rate/lattice-IDENTICAL to fp64 across all lattice systems + sparse frames
# (2026-07-18), so KC_FP=32 unlocks fp32-strong GPUs (e.g. RTX Blackwell, ~2x fp32 / half price) at no
# accuracy cost; KC_SOLVE_FP=64 keeps just the tiny 3x3 solve in fp64 as a hedge. Default stays fp64.
_W = os.environ.get("KC_FP", "64"); _S = os.environ.get("KC_SOLVE_FP", _W)
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
    linear algebra runs at _SOLVE_FP (default = working precision); result cast back to A's dtype.
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


def _cell_params(Mc, topa=8):
    """Host-side cell geometry (numpy) + the adaptive orientation grid, hoisted out of the compute
    so _stage_compute is pure-torch (and CUDA-graph capturable for a fixed cell)."""
    L, c01, c02, c12, sgn = _axes_from_cell(Mc)
    ca, sa = _azimuth_grid(c01)                     # half turn iff perpendicular (see replica_gpu)
    return (float(L[0]), float(L[1]), float(L[2]), c01, c02, c12, sgn, _adaptive_dirs(Mc), topa,
            ca.to(FP), sa.to(FP))


def _stage_compute(Q, m, P):
    """Pure-torch known-cell compute on padded (F,Pmax,3) Q + (F,Pmax) m; returns GPU tensors
    (best, pol, mp, mainb). No cuSOLVER (analytic solve/det) and static-shape => CUDA-graph
    capturable, which collapses the ~thousands of host kernel dispatches (the dominant cost)."""
    L0, L1, L2, c01, c02, c12, sgn, dirs, topa, _ca, _sa = P
    F = Q.shape[0]
    # --- anchor search (shortest axis) + greedy dedup to NC ---
    V0 = (L0 * dirs)[None].expand(F, -1, -1)
    inl, sub = obj_b(V0, Q, m); top = (inl.double() * 100 - sub).topk(120, 1).indices
    Vsel = torch.gather(V0, 1, top[:, :, None].expand(-1, -1, 3))
    Vref = refine_b(Vsel, Q, m, 30); Vref = Vref / Vref.norm(dim=2, keepdim=True) * L0
    inl2, sub2 = obj_b(Vref, Q, m); order = (inl2.double() * 100 - sub2).argsort(1, descending=True)
    Vs = torch.gather(Vref, 1, order[:, :, None].expand(-1, -1, 3)); dirs2 = Vs / L0
    alive = torch.ones(F, 120, dtype=torch.bool, device=DEV)
    chosen = torch.zeros(F, NC, 3, dtype=FP, device=DEV)
    ar = _arange(F)
    for k in range(NC):
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
    inl1, _ = obj_b(a1.reshape(F, NC * NANG, 3), Q, m); inl1 = inl1.reshape(F, NC, NANG)
    ta = min(topa, NANG); topi = inl1.topk(ta, 2).indices
    a0s = torch.gather(a0, 2, topi[..., None].expand(-1, -1, -1, 3)).reshape(F, NC * ta, 3)
    a1s = torch.gather(a1, 2, topi[..., None].expand(-1, -1, -1, 3)).reshape(F, NC * ta, 3)
    a2s = _third_axis(a0s.reshape(-1, 3), a1s.reshape(-1, 3), L2, c02, c12, sgn).reshape(F, NC * ta, 3)
    M0 = torch.stack([a0s, a1s, a2s], dim=3)
    det = torch.abs(det3(M0) if _ANALYTIC else torch.linalg.det(M0)); validM = det >= 1e3
    M0 = torch.where(validM[..., None, None], M0, _EYE3[None, None])
    Mt = anneal_b(M0, Q, m, thr0=0.30, contract=0.82, max_iter=20, min_thr=0.02)
    H = torch.einsum('fpc,fkcd->fkpd', Q, Mt); dd = torch.abs(H - torch.round(H))
    main = ((dd.amax(3) < 0.15) & m[:, None, :]).sum(2)
    subm = (torch.log2(torch.clamp(dd, TRIML, TRIMH) + DELTA).mean(3) * m[:, None, :]).sum(2) / m.sum(1, keepdim=True).clamp(min=1)
    key = torch.where(validM, main.double() * 1000 - subm, torch.full_like(main.double(), -1e18))
    bidx = key.argmax(1); best = Mt[ar, bidx]; mainb = main[ar, bidx]
    # --- guarded polish, batched ---
    pol = anneal_b(best[:, None], Q, m, thr0=0.30, contract=0.85, max_iter=10, min_thr=0.15)[:, 0]
    Hp = torch.einsum('fpc,fcd->fpd', Q, pol); mp = ((torch.abs(Hp - torch.round(Hp)).amax(2) < 0.15) & m).sum(1)
    return best, pol, mp, mainb


def _gpu_stage(frames, Mc, topa=8):
    """Eager wrapper: pad this batch then run the pure-torch stage compute."""
    Pmax = max(len(f) for f in frames)
    Q, m = pad(frames, Pmax)
    return _stage_compute(Q, m, _cell_params(Mc, topa))


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


def index_known_gpu_cell_batch(frames, Mc, topa=8):
    """Returns list of M (3x3 np) or None, one per frame -- single batched pass over all F."""
    return _cpu_stage(*_gpu_stage(frames, Mc, topa), Mc)


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
            res = _cpu_stage(best[:Fb], pol[:Fb], mp[:Fb], mainb[:Fb], Mc)
        for j, r in zip(chunk, res):
            out[j] = r
    return out


index_batch = index_known_gpu_cell_batch   # alias


def index_fused(frames, Mc, B=32):
    """Fully-fused known-cell indexer: custom fused CUDA kernels (cupy RawKernel, nvrtc-JIT) for the
    anneal/obj/refine per-candidate hot loops + an on-device cpu_stage (batched buerger same_lattice),
    replacing the many small per-stage torch kernels AND the host tail. ~0.36 ms/frame fp32 / ~0.63
    fp64 on one A100 -- 2.2x (fp64) to 6.7x (fp32) over index_all_graph -- with per-frame output
    IDENTICAL (bit-exact fp64; rate + lattice identical fp32, 75/114 on 120 cxidb) to the stock engine.
    Sorts frames by peak count so each batch pads to its own tight Pmax. Requires cupy on a GPU; falls
    back to index_all_graph (graph path) when cupy is unavailable or on CPU.

    Throughput scales with the batch B: each frame is one thread-block, so B sets GPU occupancy.
    B>=64 saturates an A100 (120 cxidb frames: B=32 -> 0.33/0.45 ms/fr fp32/fp64; B=64 -> 0.21/0.31;
    B=120 -> 0.16/0.26). Output is batch-invariant -- the kernels loop each frame's real peak count,
    not Pmax, so a looser per-batch pad costs no work (fp64 bit-identical across B)."""
    if DEV != "cuda":
        return index_all_graph(frames, Mc, B)
    try:
        from glint import fused_kernels as _fk
    except Exception:
        return index_all_graph(frames, Mc, B)
    order = sorted(range(len(frames)), key=lambda i: len(frames[i]))   # tight per-batch Pmax
    out = [None] * len(frames)
    _fk.patch(anneal=True, obj=True, refine=True, cpu=True)
    try:
        for s in range(0, len(order), B):
            chunk = order[s:s + B]
            res = index_known_gpu_cell_batch([frames[j] for j in chunk], Mc)
            for j, r in zip(chunk, res):
                out[j] = r
    finally:
        _fk.unpatch()
    return out
