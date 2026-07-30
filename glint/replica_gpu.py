"""GPU known-cell indexer = faithful torch port of replica_v2.index_known (the ① rescue
engine), closing the per-frame gap to ffbidx. Same algorithm: GPU c-axis candidate search
(objective + gradient refine over CDIRS dirs) -> per c-candidate sweep NANG a-orientations
-> batched ifss (anneal_batch_t) on the top oriented cells -> pick best by ifss score.
The numpy replica is ~127 ms/frame; this targets a few ms on the A100.

  python replica_gpu.py [frames.txt] [N]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
from glint.glint_fast import anneal_batch_t
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice

DEV = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
LA, LC = 79.02, 37.98
TRIML, TRIMH, DELTA = 0.05, 0.30, 0.10
CDIRS = int(os.environ.get("CDIRS", "16384"))
NANG = int(os.environ.get("NANG", "360"))
NC = int(os.environ.get("NC", "16"))
# Greedy dedup radius (cos-angle) for the axis0 candidate pool (axis_candidates_t / c_candidates_t /
# index_fused): candidates within this angle of an already-accepted (higher-scoring) direction are
# excluded. At 0.985 (~10 deg) a spurious-boosted nearby impostor can claim the TRUE axis0's
# neighborhood before the greedy scan reaches it, permanently excluding it -- RADIUS-based, so
# widening NC never helps. 0.9995 (~1.8 deg): known-cell 494->600/600 at f=0.8 severe spurious load,
# zero regressions across f=0.3-0.9 (recovers 47-106 frames/point).
AXIS0_DEDUP_COS = float(os.environ.get("AXIS0_DEDUP_COS", "0.9995"))
PI = np.pi


def _fib_halfsphere(D):
    i = np.arange(D); phi = PI * (3 - np.sqrt(5)) * i
    z = 1.0 - (i + 0.5) / D; r = np.sqrt(np.clip(1 - z * z, 0, 1))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


DIRS = torch.as_tensor(_fib_halfsphere(CDIRS), dtype=torch.float64, device=DEV)
TH = torch.as_tensor(np.linspace(0, PI, NANG, endpoint=False), dtype=torch.float64, device=DEV)
CA, SA = torch.cos(TH), torch.sin(TH)
# Full-turn twin of TH, at the SAME sample count (1 deg steps instead of 0.5) -- see _azimuth_grid.
TH2 = torch.as_tensor(np.linspace(0, 2 * PI, NANG, endpoint=False), dtype=torch.float64, device=DEV)
CA2, SA2 = torch.cos(TH2), torch.sin(TH2)
AZ_TOL = float(os.environ.get("AZ_TOL", "1e-9"))          # |c01| below this counts as perpendicular


def _azimuth_grid(c01):
    """(cos, sin) samples for the axis1 azimuth sweep, given the anchor/axis1 unit-cosine c01.

    axis1 is swept on the cone  a1(th) = L1*(c01*cn + s01*(cos th * u + sin th * v)).  The map
    th -> th+pi sends a1 -> 2*c01*L1*cn - a1, which equals -a1 -- the SAME lattice vector, so a
    HALF turn covers the cone -- only when c01 == 0.  For an oblique pair (c01 != 0) the -a1
    partner lies on the supplementary cone  x.cn = -L1*c01,  which the sweep never visits, so a
    half turn generates only half the cone and the true axis is missed for ~50% of orientations
    (measured: 50% of orientations off by the full cone offset, 10 deg on triclinic / 20 deg on
    rhombohedral-oblique -- far outside the annealer's basin).

    So: half turn when perpendicular (bit-identical to the original, and the finer 0.5 deg step),
    full turn otherwise.  Same sample count either way, so this is COST-NEUTRAL -- oblique cells
    trade angular step (0.5 -> 1 deg) for the missing half, which the annealer absorbs."""
    return (CA, SA) if abs(c01) < AZ_TOL else (CA2, SA2)


def objective_t(V, Q):
    """replica module 2 (torch): inlier count + trimmed-log2 sub-score. V:(B,3) Q:(P,3)."""
    proj = V @ Q.T
    d = torch.abs(proj - torch.round(proj))
    inl = (d < TRIMH).sum(1)
    sub = torch.log2(torch.clamp(d, TRIML, TRIMH) + DELTA).mean(1)
    return inl, sub


def refine_vec_t(V, Q, steps=30):
    qmax = float(Q.norm(dim=1).max())
    lr = 1.0 / (4 * PI ** 2 * len(Q) * qmax ** 2)
    for _ in range(steps):
        th = 2 * PI * (V @ Q.T)
        V = V - lr * (2 * PI * (torch.sin(th) @ Q))
    return V


RESCUE_CG = os.environ.get("RESCUE_CG", "0") == "1"   # flag: rescue uses conjugate-gradient refiner (wider basin)


def refine_vec_cg_t(V, Q, steps=30):
    """Nonlinear conjugate-gradient (Polak-Ribiere+) analog of refine_vec_t -- same objective, but the
    conjugate direction has a WIDER basin than plain GD (measured on synthetic M3: a CG restart-ensemble
    reaches full recovery at ~5x fewer restarts than GD). Normalized-direction step control
    (step0=0.25/qmax, decaying) as the validated blind refine_vec_cg."""
    qmax = float(Q.norm(dim=1).max())
    step0 = 0.25 / qmax
    d = torch.zeros_like(V); a_prev = None
    for s in range(steps):
        lr = step0 * (1 - 0.7 * s / steps)
        a = -(2 * PI * (torch.sin(2 * PI * (V @ Q.T)) @ Q))       # ascend dir (= -descent grad of refine_vec_t)
        if a_prev is None:
            d = a
        else:
            num = (a * (a - a_prev)).sum(1, keepdim=True)         # Polak-Ribiere
            den = (a_prev * a_prev).sum(1, keepdim=True) + 1e-12
            d = a + (num / den).clamp_min(0.0) * d                # PR+ (restart when beta<0)
        a_prev = a
        V = V + lr * d / (d.norm(dim=1, keepdim=True) + 1e-12)
    return V


def _resc_refine(V, Q, steps=30):
    return refine_vec_cg_t(V, Q, steps) if RESCUE_CG else refine_vec_t(V, Q, steps)


def c_candidates_t(Q):
    inl, sub = objective_t(LC * DIRS, Q)
    key = inl.double() * 100.0 - sub                      # max inl, then min sub
    idx = torch.argsort(key, descending=True)[:120]
    ref = _resc_refine((LC * DIRS)[idx], Q, steps=30)
    ref = ref / ref.norm(dim=1, keepdim=True) * LC
    inl2, sub2 = objective_t(ref, Q)
    order = torch.argsort(inl2.double() * 100.0 - sub2, descending=True)
    refc = ref.cpu().numpy(); out = []
    for j in order.tolist():
        d = refc[j] / LC
        if all(abs(d @ (o / LC)) < AXIS0_DEDUP_COS for o in out):
            out.append(refc[j])
        if len(out) >= NC:
            break
    return torch.as_tensor(np.array(out), dtype=torch.float64, device=DEV) if out else None


def index_known_gpu(q, topa=8):
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float64, device=DEV)
    if len(Q) < 6:
        return None
    C = c_candidates_t(Q)                                  # (NC,3)
    if C is None:
        return None
    # per c-candidate: orthonormal basis u,v perp c; sweep a = LA*(ca*u + sa*v)
    cn = C / C.norm(dim=1, keepdim=True)
    tmp = torch.where(cn[:, :1].abs() < 0.9,
                      torch.tensor([1.0, 0, 0], dtype=torch.float64, device=DEV),
                      torch.tensor([0, 1.0, 0], dtype=torch.float64, device=DEV))
    u = torch.cross(cn, tmp, dim=1); u = u / u.norm(dim=1, keepdim=True)
    v = torch.cross(cn, u, dim=1)
    A = LA * (CA[None, :, None] * u[:, None, :] + SA[None, :, None] * v[:, None, :])  # (NC,NANG,3)
    inl, _ = objective_t(A.reshape(-1, 3), Q)
    inl = inl.reshape(len(C), NANG)
    ta = min(topa, NANG)
    topi = torch.topk(inl, ta, dim=1).indices                                         # (NC,ta)
    a_sel = torch.gather(A, 1, topi[:, :, None].expand(-1, -1, 3))                     # (NC,ta,3)
    c_exp = C[:, None, :].expand(-1, ta, -1)                                           # (NC,ta,3)
    b = torch.cross(c_exp, a_sel, dim=2); b = b / b.norm(dim=2, keepdim=True) * LA
    M0 = torch.stack([a_sel, b, c_exp], dim=3).reshape(-1, 3, 3)                       # (NC*ta,3,3) cols a,b,c
    det = torch.abs(torch.linalg.det(M0))
    M0 = M0[det >= 1e3]
    if len(M0) == 0:
        return None
    Mt = anneal_batch_t(M0, Q, thr0=0.30, contract=0.82, max_iter=20, min_thr=0.02)    # ifss
    H = torch.einsum('pc,bcd->bpd', Q, Mt); dd = torch.abs(H - torch.round(H))
    main = (dd.amax(2) < 0.15).sum(1)                                                  # ffbidx score_thr
    sub = torch.log2(torch.clamp(dd, TRIML, TRIMH) + DELTA).mean((1, 2))
    key = main.double() * 1000.0 - sub
    b = int(torch.argmax(key)); best = Mt[b]
    # GUARDED gate-matched polish (see index_known_gpu_cell): re-anneal the winner to the 0.15 gate
    # tolerance for completeness; accept only if it indexes >= spots AND stays the known (LYSO) lattice.
    pol = anneal_batch_t(best[None], Q, thr0=0.30, contract=0.85, max_iter=10, min_thr=0.15)[0]
    Hp = Q @ pol; mp = int((torch.abs(Hp - torch.round(Hp)).amax(1) < 0.15).sum())
    poln = pol.cpu().numpy()
    if mp >= int(main[b]) and same_lattice(poln, LYSO):                     # more spots AND still LYSO
        return poln
    return best.cpu().numpy()


# ----- cell-GENERAL known-cell rescue (productization: not tied to lysozyme geometry) --------
# index_known_gpu above bakes in LYSO's tetragonal sweep (anchor=short axis, perpendicular
# equal-length a/b). The functions below take an ARBITRARY consensus basis Mc and rescue against
# its true metric: anchor = shortest real axis (found from data), 2nd axis placed at the cell's
# actual inter-axis angle (azimuth swept), 3rd axis fixed by the two metric constraints
# a2.e0 = L2 cos02, a2.e1 = L2 cos12 with |a2|=L2 (handedness picked from det(Mc)). Reduces
# EXACTLY to index_known_gpu when Mc is tetragonal (cos=0, L1=L2) -> built-in correctness check.

def _axes_from_cell(Mc):
    """Ordered axes (shortest first = data-robust anchor) from real-space basis Mc (cols a,b,c):
    returns (L[3] lengths, c01, c02, c12 unit-cosines, sgn handedness)."""
    A = np.asarray(Mc, float)
    cols = [A[:, i] for i in range(3)]
    order = list(np.argsort([np.linalg.norm(c) for c in cols]))
    v0, v1, v2 = (cols[order[0]], cols[order[1]], cols[order[2]])
    L = np.array([np.linalg.norm(v0), np.linalg.norm(v1), np.linalg.norm(v2)])
    un = lambda v: v / np.linalg.norm(v)
    c01 = float(un(v0) @ un(v1)); c02 = float(un(v0) @ un(v2)); c12 = float(un(v1) @ un(v2))
    sgn = float(np.sign(np.linalg.det(np.stack([v0, v1, v2], axis=1))) or 1.0)
    return L, c01, c02, c12, sgn


def axis_candidates_t(Q, L0):
    """c_candidates_t generalized to an arbitrary anchor length L0 (was hardcoded LC)."""
    inl, sub = objective_t(L0 * DIRS, Q)
    idx = torch.argsort(inl.double() * 100.0 - sub, descending=True)[:120]
    ref = _resc_refine((L0 * DIRS)[idx], Q, steps=30)
    ref = ref / ref.norm(dim=1, keepdim=True) * L0
    inl2, sub2 = objective_t(ref, Q)
    order = torch.argsort(inl2.double() * 100.0 - sub2, descending=True)
    refc = ref.cpu().numpy(); out = []
    for j in order.tolist():
        d = refc[j] / L0
        if all(abs(d @ (o / L0)) < AXIS0_DEDUP_COS for o in out):
            out.append(refc[j])
        if len(out) >= NC:
            break
    return torch.as_tensor(np.array(out), dtype=torch.float64, device=DEV) if out else None


def _third_axis(a0, a1, L2, c02, c12, sgn):
    """Place a2 (|a2|=L2) at fixed metric angles to a0,a1: a2.e0=L2 c02, a2.e1=L2 c12, in the
    {e0,e1,e0xe1} frame; gamma sign = handedness. Infeasible azimuths -> 0 (dropped by det filter)."""
    e0 = a0 / a0.norm(dim=1, keepdim=True)
    e1 = a1 / a1.norm(dim=1, keepdim=True)
    g = (e0 * e1).sum(1)                                     # = c01 per row
    denom = 1.0 - g * g
    alpha = L2 * (c02 - g * c12) / denom
    beta = L2 * (c12 - g * c02) / denom
    w = torch.cross(e0, e1, dim=1); wn = w.norm(dim=1, keepdim=True)
    w = w / wn.clamp_min(1e-12)
    g2 = L2 * L2 - (alpha ** 2 + beta ** 2 + 2 * alpha * beta * g)
    feas = (denom.abs() > 1e-6) & (g2 > 0) & (wn.squeeze(1) > 1e-6)
    gamma = sgn * torch.sqrt(g2.clamp_min(0.0))
    a2 = alpha[:, None] * e0 + beta[:, None] * e1 + gamma[:, None] * w
    return torch.where(feas[:, None], a2, torch.zeros_like(a2))


def index_known_gpu_cell(q, Mc, topa=8):
    """GPU known-cell rescue against an ARBITRARY consensus cell Mc (3x3 real-space cols)."""
    L, c01, c02, c12, sgn = _axes_from_cell(Mc)
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float64, device=DEV)
    if len(Q) < 6:
        return None
    C = axis_candidates_t(Q, float(L[0]))                   # anchor = shortest axis
    if C is None:
        return None
    cn = C / C.norm(dim=1, keepdim=True)
    tmp = torch.where(cn[:, :1].abs() < 0.9,
                      torch.tensor([1.0, 0, 0], dtype=torch.float64, device=DEV),
                      torch.tensor([0, 1.0, 0], dtype=torch.float64, device=DEV))
    u = torch.cross(cn, tmp, dim=1); u = u / u.norm(dim=1, keepdim=True)
    v = torch.cross(cn, u, dim=1)
    s01 = float(np.sqrt(max(1.0 - c01 * c01, 0.0)))
    ca, sa = _azimuth_grid(c01)                             # half turn iff perpendicular
    a1 = float(L[1]) * (c01 * cn[:, None, :] +
                        s01 * (ca[None, :, None] * u[:, None, :] + sa[None, :, None] * v[:, None, :]))  # (NC,NANG,3)
    a0 = C[:, None, :].expand(-1, NANG, -1)
    inl1, _ = objective_t(a1.reshape(-1, 3), Q); inl1 = inl1.reshape(len(C), NANG)
    ta = min(topa, NANG)
    topi = torch.topk(inl1, ta, dim=1).indices
    a0s = torch.gather(a0, 1, topi[:, :, None].expand(-1, -1, 3)).reshape(-1, 3)
    a1s = torch.gather(a1, 1, topi[:, :, None].expand(-1, -1, 3)).reshape(-1, 3)
    a2s = _third_axis(a0s, a1s, float(L[2]), c02, c12, sgn)
    M0 = torch.stack([a0s, a1s, a2s], dim=2)                # cols = anchor, axis1, axis2
    det = torch.abs(torch.linalg.det(M0))
    M0 = M0[det >= 1e3]
    if len(M0) == 0:
        return None
    Mt = anneal_batch_t(M0, Q, thr0=0.30, contract=0.82, max_iter=20, min_thr=0.02)
    H = torch.einsum('pc,bcd->bpd', Q, Mt); dd = torch.abs(H - torch.round(H))
    main = (dd.amax(2) < 0.15).sum(1)
    sub = torch.log2(torch.clamp(dd, TRIML, TRIMH) + DELTA).mean((1, 2))
    key = main.double() * 1000.0 - sub
    b = int(torch.argmax(key)); best = Mt[b]
    # GUARDED gate-matched polish: the winner was annealed tight (min_thr 0.02) and over-fits a few spots;
    # re-anneal it to the 0.15 gate tolerance so it indexes MORE spots (completeness). Accept the looser
    # refit ONLY if it indexes at least as many 0.15-inliers (reverts on drift -> never loses the lattice
    # or a >=10-refl frame; lifts the strict >=25%-of-spots gate).
    pol = anneal_batch_t(best[None], Q, thr0=0.30, contract=0.85, max_iter=10, min_thr=0.15)[0]
    Hp = Q @ pol; mp = int((torch.abs(Hp - torch.round(Hp)).amax(1) < 0.15).sum())
    poln = pol.cpu().numpy()
    if mp >= int(main[b]) and same_lattice(poln, np.asarray(Mc, float)):    # more spots AND still the known cell
        return poln
    return best.cpu().numpy()


def load(p):
    fr = []; L = open(p).read().split("\n"); i = 0
    while i < len(L):
        if not L[i].startswith("FRAME"): i += 1; continue
        _, fid, n = L[i].split(); n = int(n)
        fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
    return fr


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]; n = len(frames)
    index_known_gpu(frames[0])                              # warmup
    if DEV == "cuda": torch.cuda.synchronize()
    f25 = fN = 0; t0 = time.time()
    for q in frames:
        M = index_known_gpu(q)
        if M is None or not same_lattice(M, LYSO):
            continue
        m = int((np.abs(q @ M - np.rint(q @ M)).max(1) < 0.15).sum())
        f25 += m / len(q) >= 0.25; fN += m >= 10
    if DEV == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"replica_GPU (known-cell)  device={DEV}  N={n}  {1e3*dt/n:.1f} ms/frame  ({n/dt:.0f} f/s)")
    print(f"  frac>=.25: {f25}/{n} ({100*f25//n}%)   >=10refl: {fN}/{n} ({100*fN//n}%)")
    print(f"  numpy replica baseline: ~127 ms/frame ; ffbidx ~4.4 ms/frame")
