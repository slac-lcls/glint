"""Batched-across-frames known-cell engine (production). Same algorithm as
replica_gpu.index_known_gpu_cell, but F frames processed in one batched pass
(pad peaks to Pmax + mask), collapsing the ~5215 tiny kernels/frame ~F-fold.
Keeps replica_gpu (per-frame) and the ffbidx handoff as the other two options."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("CDIRS", "16384")
import numpy as np, torch
from glint.replica_gpu import DIRS, CA, SA, TRIML, TRIMH, DELTA, NC, NANG, _axes_from_cell, _third_axis, _fib_halfsphere
from glint.multishot import same_lattice

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FP = torch.float64; PI = np.pi
_CA = CA.to(FP); _SA = SA.to(FP); _DIRS = DIRS.to(FP)

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
    eyes = 1e-9 * torch.eye(3, device=DEV, dtype=FP)[None, None]; mk = m[:, None, :]
    for _ in range(max_iter):
        H = torch.einsum('fpc,fkcd->fkpd', Q, M); hkl = torch.round(H)
        inl = ((torch.abs(H - hkl).amax(3) < thr) & mk).to(FP)
        cnt = inl.sum(2); WQ = inl[..., None] * Q[:, None, :, :]
        A = torch.einsum('fkpc,fpd->fkcd', WQ, Q) + eyes
        rhs = torch.einsum('fkpc,fkpd->fkcd', WQ, hkl)
        M = torch.where((cnt >= 6)[..., None, None], torch.linalg.solve(A, rhs), M)
        thr = max(thr * contract, min_thr)
    return M


def index_known_gpu_cell_batch(frames, Mc, topa=8):
    """Returns list of M (3x3 np) or None, one per frame -- batched over all F."""
    L, c01, c02, c12, sgn = _axes_from_cell(Mc)
    F = len(frames); Pmax = max(len(f) for f in frames)
    Q, m = pad(frames, Pmax)
    # --- anchor search (shortest axis) + greedy dedup to NC ---
    V0 = (float(L[0]) * _adaptive_dirs(Mc))[None].expand(F, -1, -1)
    inl, sub = obj_b(V0, Q, m); top = (inl.double() * 100 - sub).topk(120, 1).indices
    Vsel = torch.gather(V0, 1, top[:, :, None].expand(-1, -1, 3))
    Vref = refine_b(Vsel, Q, m, 30); Vref = Vref / Vref.norm(dim=2, keepdim=True) * float(L[0])
    inl2, sub2 = obj_b(Vref, Q, m); order = (inl2.double() * 100 - sub2).argsort(1, descending=True)
    Vs = torch.gather(Vref, 1, order[:, :, None].expand(-1, -1, 3)); dirs = Vs / float(L[0])
    alive = torch.ones(F, 120, dtype=torch.bool, device=DEV)
    chosen = torch.zeros(F, NC, 3, dtype=FP, device=DEV); valid = torch.zeros(F, NC, dtype=torch.bool, device=DEV)
    ar = torch.arange(F, device=DEV)
    for k in range(NC):
        idx = alive.int().argmax(1)
        chosen[:, k, :] = Vs[ar, idx]; valid[:, k] = alive.any(1)
        dk = dirs[ar, idx]; alive = alive & (torch.abs(torch.einsum('fpc,fc->fp', dirs, dk)) < 0.985)
    # --- orientation sweep per anchor ---
    C = chosen; cn = C / C.norm(dim=2, keepdim=True)
    ex = torch.tensor([1., 0, 0], dtype=FP, device=DEV); ey = torch.tensor([0, 1., 0], dtype=FP, device=DEV)
    tmp = torch.where(cn[..., :1].abs() < 0.9, ex, ey)
    u = torch.cross(cn, tmp, dim=2); u = u / u.norm(dim=2, keepdim=True); v = torch.cross(cn, u, dim=2)
    s01 = float(np.sqrt(max(1.0 - c01 * c01, 0.0)))
    a1 = float(L[1]) * (c01 * cn[:, :, None, :] +
                        s01 * (_CA[None, None, :, None] * u[:, :, None, :] + _SA[None, None, :, None] * v[:, :, None, :]))
    a0 = C[:, :, None, :].expand(-1, -1, NANG, -1)
    inl1, _ = obj_b(a1.reshape(F, NC * NANG, 3), Q, m); inl1 = inl1.reshape(F, NC, NANG)
    ta = min(topa, NANG); topi = inl1.topk(ta, 2).indices
    a0s = torch.gather(a0, 2, topi[..., None].expand(-1, -1, -1, 3)).reshape(F, NC * ta, 3)
    a1s = torch.gather(a1, 2, topi[..., None].expand(-1, -1, -1, 3)).reshape(F, NC * ta, 3)
    a2s = _third_axis(a0s.reshape(-1, 3), a1s.reshape(-1, 3), float(L[2]), c02, c12, sgn).reshape(F, NC * ta, 3)
    M0 = torch.stack([a0s, a1s, a2s], dim=3)
    det = torch.abs(torch.linalg.det(M0)); validM = det >= 1e3
    M0 = torch.where(validM[..., None, None], M0, torch.eye(3, dtype=FP, device=DEV)[None, None])
    Mt = anneal_b(M0, Q, m, thr0=0.30, contract=0.82, max_iter=20, min_thr=0.02)
    H = torch.einsum('fpc,fkcd->fkpd', Q, Mt); dd = torch.abs(H - torch.round(H))
    main = ((dd.amax(3) < 0.15) & m[:, None, :]).sum(2)
    subm = (torch.log2(torch.clamp(dd, TRIML, TRIMH) + DELTA).mean(3) * m[:, None, :]).sum(2) / m.sum(1, keepdim=True).clamp(min=1)
    key = torch.where(validM, main.double() * 1000 - subm, torch.full_like(main.double(), -1e18))
    bidx = key.argmax(1); best = Mt[ar, bidx]; mainb = main[ar, bidx]
    # --- guarded polish, batched ---
    pol = anneal_b(best[:, None], Q, m, thr0=0.30, contract=0.85, max_iter=10, min_thr=0.15)[:, 0]
    Hp = torch.einsum('fpc,fcd->fpd', Q, pol); mp = ((torch.abs(Hp - torch.round(Hp)).amax(2) < 0.15) & m).sum(1)
    out = []; bn = best.cpu().numpy(); pn = pol.cpu().numpy(); mpv = mp.cpu().numpy(); mbv = mainb.cpu().numpy()
    Mcn = np.asarray(Mc, float)
    for i in range(F):
        if mpv[i] >= mbv[i] and same_lattice(pn[i], Mcn):
            out.append(pn[i])
        else:
            out.append(bn[i])
    return out


index_batch = index_known_gpu_cell_batch   # alias
