"""GPU known-cell indexer = faithful torch port of replica_v2.index_known (the ① rescue
engine), closing the per-frame gap to ffbidx. Same algorithm: GPU c-axis candidate search
(objective + gradient refine over CDIRS dirs) -> per c-candidate sweep NANG a-orientations
-> batched ifss (anneal_batch_t) on the top oriented cells -> pick best by ifss score.
The numpy replica is ~127 ms/frame; this targets a few ms on the A100.

  python replica_gpu.py [frames.txt] [N]
"""
import itertools, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
from glint.glint_fast import anneal_batch_t, matched_strict, GATE_FRAC, GATE_MIN
from glint.lattice import LOW_LAUE, cell_to_Ar
from glint.geom import clean_q
from glint.multishot import same_lattice

# cuda -> cpu, with NO mps rung. Apple's MPS backend does not implement float64, and this module
# computes in float64 throughout (DIRS below is the first of eleven sites), so selecting mps raised
# `Cannot convert a MPS Tensor to float64` at IMPORT time on every Apple-silicon machine -- and
# `--device cpu` did not rescue it, because that flag only clears CUDA_VISIBLE_DEVICES. mps was
# therefore never a working path, only a way for the cpu fallback to be skipped. Same in
# glint/glint_index.py. (The old ladder was getattr-guarded for torch<1.12, which has no
# torch.backends.mps at all; removing the rung removes that concern with it.)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
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
    q = clean_q(np.asarray(q, float))          # drop NaN/inf and zero-length rows: a NaN row loses the frame
    Q = torch.as_tensor(q, dtype=torch.float64, device=DEV)
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


def _depth(name, v):
    """A search-depth argument (nc, topa) as an int >= 1. Checked up front: nc=0 would otherwise still keep
    one direction (the limit is tested after the first append) and a float would be truncated silently, so
    the caller would get a depth other than the one asked for."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {v!r}")
    return int(v)


def axis_candidates_t(Q, L0, nc=None):
    """c_candidates_t generalized to an arbitrary anchor length L0 (was hardcoded LC).

    nc: how many distinct anchor directions to keep (default: the module's NC, env-set at import). The
    escalation arm (glint.retry_cascade.arm_known_deep) asks for more without re-importing the module."""
    nc = NC if nc is None else _depth("nc", nc)
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
        if len(out) >= nc:
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


MIRROR_TOL_DEG = float(os.environ.get("KC_MIRROR_TOL_DEG", "1.0"))   # see _both_hands


def _both_hands(L2, c01, c02, c12, laue=None):
    """True when a2 must be seeded at BOTH gamma signs (review r2 s4-01).

    The anchor pool is a half-sphere (DIRS), so on a frame whose true shortest axis points to z<0 the anchor
    is -v0; the axis-1 sweep then finds -v1, and the lattice's own third axis is -v2, whose handedness is the
    OPPOSITE of the reference's. _third_axis at the reference's sign then builds the mirror image of -v2
    through the (v0, v1) plane, which is +v2 displaced by twice a2's in-plane component. When a2 is
    perpendicular to both that displacement is 0 and the seed IS a lattice vector. Otherwise the anneal can
    return a basis with the reference metric that indexes ~40 % of the spots (triclinic, and monoclinic with
    the unique axis shortest or in the middle), and same_lattice and the live gate both pass it. Seeding the
    other sign too gives (-v0, -v1, -v2), which is negated back to the reference hand after the pick.

    Which cells take this path depends on what is known about the crystal's symmetry:
    - `laue` is a low-symmetry class (glint.lattice.LOW_LAUE: -1, a 2/m setting, or a rhombohedral-axes class):
      always. No rotation of the class makes the mirror seed a lattice vector, however close to 90 deg the
      angles are; and only this
      path runs _ref_setting, which is what puts a near-orthogonal cell (triclinic 50/60/70/89.5/90/90.2) in
      the reference setting instead of a 2-fold sign-flipped one (review of #225).
    - Otherwise (a higher class, or no class): when a2 tilts more than KC_MIRROR_TOL_DEG (1 deg) from the
      (v0, v1) plane's normal. A consensus or lock cell of an orthogonal lattice is skewed -- StreamDriver's
      GPU lock on cxidb-17 lysozyme is 78.71/78.79/37.81 at 89.81/90.08/90.19 deg, a 0.27 deg tilt -- and its
      2-fold makes either seed valid, so the second seed only reshuffles marginal frames (333 -> 334 strict on
      the 480-frame replay when this gate was 0.2 A). Such cells keep exactly the code they ran before. Every
      clearly oblique cell (triclinic, monoclinic at beta >~ 91 deg) takes the path with or without a class.
    A near-orthogonal low-symmetry cell with NO class given stays on the one-handed path: the metric alone
    cannot tell it from the skewed lock cell of an orthogonal lattice. Its frames are then a lattice basis in
    a possibly sign-flipped setting (not a non-lattice basis). Callers that know the class pass it:
    StreamDriver (its laue=, or the class derived from stream_symmetry) and hybrid_index(laue=)."""
    if laue is not None and str(laue).strip() in LOW_LAUE:
        return True
    g = c01
    alpha = L2 * (c02 - g * c12) / (1.0 - g * g)
    beta = L2 * (c12 - g * c02) / (1.0 - g * g)
    inplane = np.sqrt(max(alpha * alpha + beta * beta + 2.0 * alpha * beta * g, 0.0))
    return float(np.degrees(np.arcsin(min(inplane / L2, 1.0)))) > MIRROR_TOL_DEG


def _canonical_laue(laue):
    if laue is None:
        return None
    from glint.stream_driver import laue_name
    return laue_name(laue)


# Proper unimodular changes of basis with entries in {-1, 0, 1}: enough to reach every setting of a lattice
# basis whose axes are sums or differences of the reference's (a+c for c, a swapped pair, ...).
_UNIMOD = np.array(list(itertools.product((-1, 0, 1), repeat=9)), dtype=float).reshape(-1, 3, 3)
_UNIMOD = _UNIMOD[np.rint(np.linalg.det(_UNIMOD)) == 1]                     # 3480 of them
SETTING_TOL = float(os.environ.get("KC_SETTING_TOL", "0.01"))   # relative metric deviation counted as "off"


# The four proper 2-fold sign flips of a basis, identity first: (a,b,c) -> (+-a, +-b, +-c) with det +1.
_FLIPS = np.array([np.diag(d) for d in ((1., 1., 1.), (-1., -1., 1.), (-1., 1., -1.), (1., -1., -1.))])
_FLIP_SIGN = np.array([[d[0] * d[1], d[0] * d[2], d[1] * d[2]] for d in np.diagonal(_FLIPS, axis1=1, axis2=2)])
# Only the reference's cosines whose sign a flip changes by at least 2 sin(KC_FLIP_MIN_DEG) -- angles at least
# this far from 90 deg -- take part in the sign-flip step. Closer to 90 the choice would follow rounding (an exact
# 90 deg has cos ~6e-17) or per-frame noise: on real mfx101343025 r199 (beta 90.07 deg, per-frame DIALS beta
# 88.5-91.7) it only reshuffled frames at chance. The price: a genuine 0-0.1 deg skew is not corrected.
FLIP_MIN = 2.0 * np.sin(np.radians(float(os.environ.get("KC_FLIP_MIN_DEG", "0.1"))))


def _cosines(M):
    """cos(a,b), cos(a,c), cos(b,c) of the basis columns: the scale-free part of the metric."""
    G = M.T @ M
    d = np.sqrt(np.diag(G))
    return np.array([G[0, 1] / (d[0] * d[1]), G[0, 2] / (d[0] * d[2]), G[1, 2] / (d[1] * d[2])])


def _ref_setting(M, Mc):
    """The engines' output basis, moved to the setting of the same lattice whose metric is closest to the
    reference's (review r2 s4-01, the INEQUIV half).

    The engines return the frame's axes shortest-first, and StreamDriver's _relabel_like undoes only that sort.
    On an oblique cell the anneal sometimes converges to another basis of the right lattice: (-a, -c, -b) on a
    triclinic HEWL cell (metric 13 % off), or (a, -b, -a-c) on a monoclinic one (3 % off). Every spot is
    indexed, same_lattice passes, and hkl come out in a setting that is not Laue-equivalent to the grid's.
    The metric shows it: such a basis is moved to M @ U, U the proper {-1,0,1} change of basis that brings
    M.T M closest to the reference's shortest-first metric. Left alone unless M's own metric is more than
    SETTING_TOL off AND U at least halves the deviation, so frames already in the reference setting and
    pseudo-symmetric ties (|a+c| ~ |a|, where geometry cannot decide) are returned unchanged.

    Then the 2-fold sign flips (#225 review). A near-orthogonal triclinic or monoclinic frame can come back as
    (+-a, +-b, +-c): the same lengths, and cosines that differ from the reference's only in sign -- 0.4-0.5 % of
    max|G| at 89.5 deg, under SETTING_TOL. Only the reference's resolvable cosines count (an angle at least
    KC_FLIP_MIN_DEG from 90): among the proper flips that change one of them, the one whose resolvable cosines
    are closest to the reference's (summed absolute error) is taken if it is strictly closer than the identity. Cosines only, so a length or scale error in the reference cannot pull
    the choice; a flip the reference cannot resolve (an angle within KC_FLIP_MIN_DEG of 90) is never taken, and
    a clearly oblique frame never prefers one (a flip changes a large cosine's sign). A non-finite or singular
    basis is returned untouched."""
    return _ref_settings([M], Mc)[0]


# vec(G) -> the 6 unique entries of U^T G U for all 3480 U in one product: a shortlist for _closest_setting.
_IU = np.triu_indices(3)
_UPROJ = np.einsum('kji,klm->kimjl', _UNIMOD, _UNIMOD)[:, _IU[0], _IU[1]].reshape(-1, 9)


def _closest_setting(G, G0, nrm):
    """(k, dev[k]) for k = argmin over the 3480 U of max|U^T G U - G0| / nrm, EXACTLY as the full einsum gives it
    (same values, the first index among exact ties), at a fraction of its cost: one product over the 6 unique
    entries shortlists every U within a rounding margin of the minimum, and only those are evaluated with the
    original einsum (whose per-U arithmetic does not depend on how many U it is handed)."""
    approx = np.abs((_UPROJ @ G.ravel()).reshape(-1, 6) - G0[_IU]).max(1) / nrm
    c = np.flatnonzero(approx <= approx.min() + 1e-9)
    dev = np.abs(np.einsum('kji,jl,klm->kim', _UNIMOD[c], G, _UNIMOD[c]) - G0).max((1, 2)) / nrm
    j = int(np.argmin(dev))
    return int(c[j]), dev[j]


def _ref_settings(Ms, Mc):
    """_ref_setting for a batch of frames against one reference: the same result per frame, with the per-frame
    host work cut to one 3x3 product (the guard, the cosines and the sign-flip choice run on the whole batch)."""
    out = list(Ms)
    idx = [i for i, M in enumerate(Ms) if M is not None]
    if not idx:
        return out
    A = np.asarray(Mc, float)
    S = A[:, np.argsort(np.linalg.norm(A, axis=0))]            # the order _axes_from_cell anchors in
    G0 = S.T @ S; nrm = np.abs(G0).max()
    F = np.stack([np.asarray(Ms[i], float) for i in idx])
    good = np.isfinite(F).all((1, 2))
    if good.any():
        gi = np.flatnonzero(good)
        good[gi] = np.abs(np.linalg.det(F[gi])) > 1e-9 * np.prod(np.linalg.norm(F[gi], axis=1), axis=1)
    keep = [j for j in range(len(idx)) if good[j]]
    if not keep:
        return out
    Fk = [F[j] for j in keep]; Mk = [Ms[idx[j]] for j in keep]
    G = np.stack([f.T @ f for f in Fk])
    dev0 = np.abs(G - G0).max((1, 2)) / nrm
    for t in np.flatnonzero(dev0 > SETTING_TOL):              # the step of review r2 s4-01, unchanged
        k, dk = _closest_setting(G[t], G0, nrm)
        if dk < 0.5 * dev0[t]:
            Fk[t] = Fk[t] @ _UNIMOD[k]; Mk[t] = Fk[t]; G[t] = Fk[t].T @ Fk[t]
    cS = _cosines(S)
    res = 2.0 * np.abs(cS) >= FLIP_MIN * (1.0 - 1e-9)          # cosines a sign flip visibly changes (89.9 counts)
    if res.any():
        d = np.sqrt(np.diagonal(G, axis1=1, axis2=2))
        cM = np.stack([G[:, 0, 1] / (d[:, 0] * d[:, 1]), G[:, 0, 2] / (d[:, 0] * d[:, 2]),
                       G[:, 1, 2] / (d[:, 1] * d[:, 2])], 1)
        err = (np.abs(_FLIP_SIGN[None] * cM[:, None, :] - cS) * res).sum(2)   # unresolvable cosines cannot tie-break
        err[:, 1:][:, ~((_FLIP_SIGN[1:] < 0) & res).any(1)] = np.inf          # a flip that changes no resolvable one
        kf = np.argmin(err, 1)                                # identity first: a tie keeps the frame as is
        for t in np.flatnonzero((kf > 0) & (err[np.arange(len(kf)), kf] < err[:, 0])):
            Mk[t] = Fk[t] @ _FLIPS[kf[t]]
    for j, m in zip(keep, Mk):
        out[idx[j]] = m
    return out


def index_known_gpu_cell(q, Mc, topa=8, nc=None, laue=None):
    """GPU known-cell rescue against an ARBITRARY consensus cell Mc (3x3 real-space cols).

    topa (azimuths kept per anchor) and nc (anchor directions, default NC) set the search depth. The
    shipped rescue uses the defaults; the escalation arm runs topa=128, nc=32 (RESULTS_escalation.md's T2).
    Both must be integers >= 1 (ValueError otherwise, before any search)."""
    laue = _canonical_laue(laue)
    topa = _depth("topa", topa)
    nc = None if nc is None else _depth("nc", nc)
    L, c01, c02, c12, sgn = _axes_from_cell(Mc)
    q = clean_q(np.asarray(q, float))          # drop NaN/inf and zero-length rows: a NaN row loses the frame
    Q = torch.as_tensor(q, dtype=torch.float64, device=DEV)
    if len(Q) < 6:
        return None
    C = axis_candidates_t(Q, float(L[0]), nc=nc)            # anchor = shortest axis
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
    both = _both_hands(float(L[2]), c01, c02, c12, laue)
    if both:                                                # s4-01: the -v0 anchor's seed has the other hand
        a2s = torch.cat([a2s, _third_axis(a0s, a1s, float(L[2]), c02, c12, -sgn)])
        a0s = torch.cat([a0s, a0s]); a1s = torch.cat([a1s, a1s])
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
    if both and float(torch.linalg.det(best)) * sgn < 0:   # -M indexes the same spots, as hkl -> -hkl (a Friedel
        best = -best                                       # mate; -1 is in every Laue group): reference hand
    # GUARDED gate-matched polish: the winner was annealed tight (min_thr 0.02) and over-fits a few spots;
    # re-anneal it to the 0.15 gate tolerance so it indexes MORE spots (completeness). Accept the looser
    # refit ONLY if it indexes at least as many 0.15-inliers (reverts on drift -> never loses the lattice
    # or a >=10-refl frame; lifts the strict >=25%-of-spots gate).
    pol = anneal_batch_t(best[None], Q, thr0=0.30, contract=0.85, max_iter=10, min_thr=0.15)[0]
    Hp = Q @ pol; mp = int((torch.abs(Hp - torch.round(Hp)).amax(1) < 0.15).sum())
    poln = pol.cpu().numpy()
    if mp >= int(main[b]) and same_lattice(poln, np.asarray(Mc, float)):    # more spots AND still the known cell
        out = poln
    else:
        out = best.cpu().numpy()
    return _ref_setting(out, Mc) if both else out


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
        m = matched_strict(M, q)                # the published gate's matcher (glint_fast.GATE_TOL)
        f25 += m / len(q) >= GATE_FRAC; fN += m >= GATE_MIN
    if DEV == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"replica_GPU (known-cell)  device={DEV}  N={n}  {1e3*dt/n:.1f} ms/frame  ({n/dt:.0f} f/s)")
    print(f"  frac>=.25: {f25}/{n} ({100*f25//n}%)   >=10refl: {fN}/{n} ({100*fN//n}%)")
    print(f"  numpy replica baseline: ~127 ms/frame ; ffbidx ~4.4 ms/frame")
