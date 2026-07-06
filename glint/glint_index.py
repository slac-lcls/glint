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
from glint.lattice import buerger_reduce

PI = np.pi
DEV = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")

# M2 proximity-function form (climb + score). "" = default cos / cos^2 with HARD tol mask.
# Smooth forms (gauss/vonmises/softcos) carry their OWN falloff, so the hard mask is dropped
# -> they replace the sharp indicator with a smooth window (sigma/kappa = soft tol).
OBJFORM = os.environ.get("OBJFORM", "")
OBJSIG = float(os.environ.get("OBJSIG", "0.12"))            # wrapped-Gaussian / soft-window width
OBJKAP = float(os.environ.get("OBJKAP", "8.0"))             # von Mises concentration
LS_CAPK = float(os.environ.get("LS_CAPK", "0.5"))          # refine_vec_ls: step cap = LS_CAPK period (0.5=quarter)
LS_CAPQ = float(os.environ.get("LS_CAPQ", "1.0"))          # refine_vec_ls: cap keyed to this |psi| quantile (1=max)
LS_MOM = float(os.environ.get("LS_MOM", "0.0"))            # refine_vec_ls: momentum on the line-search step (0=pure)
BETA = float(os.environ.get("BETA", "0.7"))                # refine_vec_raar: RAAR feedback (beta=1 -> DR/HIO)
SO_WS = float(os.environ.get("SO_WS", "1.0"))              # refine_vec_so2d: support-error weight in L=e_d^2 - w_s e_s^2
WARM = int(os.environ.get("WARM", "0"))                    # refine_vec_so2d: RAAR warm-up iters before saddle (basin)
POLISH = int(os.environ.get("POLISH", "0"))               # raar/so2d/admm: final ER (alt-projection) cleanup steps
RHO = float(os.environ.get("RHO", "1.0"))                 # refine_vec_admm: dual penalty (scaled-ADMM)
RAAR_ANNEAL = int(os.environ.get("RAAR_ANNEAL", "0"))     # refine_vec_raar: soft->hard round-to-hkl schedule (0=off = hard from step 0).
                                                          # NEGATIVE (cxidb-120): hard-round baseline 88/120 beats every soft->hard schedule
                                                          # (79-84); the reflection feedback IS the escape, softening P_data just mushes it. Kept default-off.
RAAR_A0 = float(os.environ.get("RAAR_A0", "0.3"))         #   starting round hardness (0=identity/no pull, 1=hard round); ramps ->1 over steps
RAAR_TOLK = float(os.environ.get("RAAR_TOLK", "2.0"))     #   starting inlier-tol multiplier (wide window early); ramps ->1 over steps
BETA_SCHED = int(os.environ.get("BETA_SCHED", "0"))       # refine_vec_raar: ramp beta BETA_HI->BETA over steps = HIO/DR -> RAAR (0=off, fixed beta).
                                                          # NEGATIVE (cxidb-120): fixed beta=0.7 (88/120) beats every 1.0->beta schedule (81-86);
                                                          # 16 steps/seed is too short for beta-relaxation to pay (seed ensemble does the exploration). Default-off.
BETA_HI = float(os.environ.get("BETA_HI", "1.0"))         #   starting beta (1.0 = pure DR/HIO reflection averaging); ends at BETA (RAAR)


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


def refine_vec_cg(T, Q, w, qmax, steps=8, tol=0.18, sharp_last=None, mom=None):
    """M3 variant: nonlinear CONJUGATE-GRADIENT (Polak-Ribiere+) ascent of the M2 objective. Gradient-only
    (no Hessian -> gentler than refine_vec_newton, which basin-jumps on the multimodal cos-comb). The CG
    direction replaces the momentum average of refine_vec; the direction is normalized and driven by the
    SAME lr schedule so step control is identical -- only the direction differs. Question: does the
    conjugate direction reach the M2 maxima in fewer steps at equal rate? (M3 is compute-saturated, so
    fewer steps = proportional time.) `mom` accepted+ignored for a drop-in signature with refine_vec."""
    if sharp_last is None:
        sharp_last = max(1, steps // 3)
    step0 = 0.25 / qmax
    d = torch.zeros_like(T); g_prev = None
    for s in range(steps):
        lr = step0 * (1 - 0.7 * s / steps)
        _, g = objective(T, Q, w, tol, sharp=(s >= steps - sharp_last))
        if g_prev is None:
            d = g
        else:
            num = (g * (g - g_prev)).sum(1, keepdim=True)         # Polak-Ribiere
            den = (g_prev * g_prev).sum(1, keepdim=True) + 1e-12
            d = g + (num / den).clamp_min(0.0) * d                # PR+ (restart when beta<0)
        g_prev = g
        T = T + lr * d / (d.norm(dim=1, keepdim=True) + 1e-12)    # normalized dir (step control as refine_vec)
    return T


def refine_vec_bb(T, Q, w, qmax, steps=8, tol=0.18, sharp_last=None, mom=None):
    """M3 variant: safeguarded BARZILAI-BORWEIN ascent. Stays in the gradient direction (no basin-jump
    like Newton/CG) but the STEP SIZE is the BB1 estimate alpha = |s|^2 / -(s.y) (s=ΔT, y=Δg), which
    captures curvature scaling without a Hessian. First step = normalized GD; degenerate alpha (<=0 /
    non-finite, i.e. crossing a non-concave region) falls back to normalized GD; the step norm is capped
    to 2*step0 so it stays gentle on the multimodal cos-comb. Question: converge in fewer steps at equal rate?"""
    if sharp_last is None:
        sharp_last = max(1, steps // 3)
    step0 = 0.25 / qmax
    T_prev = None; g_prev = None
    for s in range(steps):
        _, g = objective(T, Q, w, tol, sharp=(s >= steps - sharp_last))
        gd = step0 * g / (g.norm(dim=1, keepdim=True) + 1e-12)       # safe normalized-GD step
        if T_prev is None:
            step = gd
        else:
            sv = T - T_prev; y = g - g_prev
            alpha = (sv * sv).sum(1, keepdim=True) / (-(sv * y).sum(1, keepdim=True) + 1e-12)   # BB1 (ascent)
            step = alpha * g
            sn = step.norm(dim=1, keepdim=True); cap = 2.0 * step0
            step = torch.where(sn > cap, step * (cap / (sn + 1e-12)), step)                     # gentle cap
            bad = ~torch.isfinite(alpha) | (alpha <= 0)
            step = torch.where(bad, gd, step)                                                    # fallback
        T_prev = T; g_prev = g
        T = T + step
    return T


def refine_vec_lm(T, Q, w, qmax, steps=8, tol=0.18, sharp_last=None, mu0=None):
    """M3 variant: LEVENBERG-MARQUARDT ascent (steepest -> Newton) with ADAPTIVE per-seed damping -- the
    safety valve plain refine_vec_newton (fixed mu) lacked. Step = (H - mu*I)^{-1} g; large mu = small
    steepest step (safe), small mu = Newton (fast). Accept the step only if the objective improves; on a
    good step SHRINK mu (toward Newton), on a bad step REJECT + GROW mu (retreat to steepest) -> no
    basin-jump. mu starts large (steepest) and self-tunes per seed."""
    if sharp_last is None:
        sharp_last = max(1, steps // 3)
    B = T.shape[0]
    eye = torch.eye(3, dtype=T.dtype, device=T.device)
    mu = torch.full((B, 1, 1), (2 * PI) ** 2 * (1.0 if mu0 is None else mu0), dtype=T.dtype, device=T.device)
    f, _ = objective(T, Q, w, tol, sharp=False)                      # (B,) accept metric (consistent)
    for s in range(steps):
        sharp = (s >= steps - sharp_last)
        proj = T @ Q.t()
        mask = (torch.abs(proj - torch.round(proj)) < tol).to(T.dtype)
        if sharp:
            sin2 = PI * torch.sin(2 * PI * proj); cos2 = (2 * PI) * PI * torch.cos(2 * PI * proj)
        else:
            sin2 = 2 * PI * torch.sin(2 * PI * proj); cos2 = (2 * PI) ** 2 * torch.cos(2 * PI * proj)
        wm = w.unsqueeze(0) * mask
        g = -((sin2 * wm) @ Q)                                       # = objective ascent grad
        H = -torch.einsum("bn,ni,nj->bij", cos2 * wm, Q, Q)          # Hessian of f (neg-def at max)
        step = torch.linalg.solve(H - mu * eye, g.unsqueeze(-1)).squeeze(-1)
        Tn = T - step
        fn, _ = objective(Tn, Q, w, tol, sharp=False)
        good = fn >= f                                               # (B,)
        T = torch.where(good[:, None], Tn, T)
        f = torch.where(good, fn, f)
        mu = torch.where(good[:, None, None], mu * 0.5, mu * 4.0).clamp(1e-2, 1e6)   # LM update
    return T


def refine_vec_ls(T, Q, w, qmax, steps=8, tol=0.18, sharp_last=None, mom=None, ls_iters=2):
    """M3 variant: gradient ascent with an EXACT 1-D Newton LINE SEARCH per outer step, at ~gradient cost.
    Along the ascent direction d, the cos objective is CLOSED FORM  S(a) = sum wm cos(phi + a*psi)  with
    phi = 2pi (T.q)  (base phase)  and  psi = 2pi (d.q)  (phase rate along d) -- so after the SINGLE matmul
    T@Q^T that gives phi, BOTH the gradient direction AND the line-search f/f'/f'' along d are matmul-free
    elementwise reductions (the fused f,f',f'' the idea calls for). a-Newton (a <- a - S'/S'') gives a
    parameter-free step tuned to the LOCAL curvature; a per-seed cap keeping max|a*psi| < pi/2 holds the
    step inside the current oscillation, so -- unlike plain Newton/CG which basin-jump -- it cannot overshoot
    into a wrong maximum, and an accept-guard zeroes any non-improving step. Hypothesis: the exact stride
    reaches the M2 maxima in FEWER outer steps than momentum-GD (each outer step is ONE matmul, as in GD),
    so ls-4 ~ gd-8 at ~half the matmuls. `mom` accepted+ignored for a drop-in signature with refine_vec."""
    if sharp_last is None:
        sharp_last = max(1, steps // 3)
    step0 = 0.25 / qmax
    twopi = 2 * PI
    vel = torch.zeros_like(T)                                       # LS_MOM>0: momentum on the line-search step
    for s in range(steps):
        sharp = (s >= steps - sharp_last)
        proj = T @ Q.t()                                            # (B,N)  -- the ONE matmul / outer step
        mask = (torch.abs(proj - torch.round(proj)) < tol).to(T.dtype)
        wm = w.unsqueeze(0) * mask                                  # mask frozen for this outer step
        phi = twopi * proj                                         # base phase (= line arg at a=0)
        sin_p = torch.sin(phi); cos_p = torch.cos(phi)
        cf = PI if sharp else twopi                                # cos^2(pi x) vs cos(2pi x) grad prefactor
        g = -((cf * sin_p) * wm) @ Q                               # (B,3) ascent gradient (matches objective())
        d = g / (g.norm(dim=1, keepdim=True) + 1e-12)              # normalized steepest direction
        psi = twopi * (d @ Q.t())                                 # (B,N) phase rate along d
        a = torch.zeros(T.shape[0], 1, dtype=T.dtype, device=T.device)
        pabs = psi.abs()
        psi_scale = pabs.amax(dim=1, keepdim=True) if LS_CAPQ >= 1.0 else \
            torch.quantile(pabs, LS_CAPQ, dim=1, keepdim=True)               # cap keyed to a |psi| quantile
        acap = (LS_CAPK * PI) / (psi_scale + 1e-12)                          # step cap = LS_CAPK*pi phase move
        for _ in range(ls_iters):                                  # exact 1-D Newton on the step a (matmul-free)
            arg = phi + a * psi
            Sp = -(wm * psi * torch.sin(arg)).sum(1, keepdim=True)              # S'(a)
            Spp = -(wm * psi * psi * torch.cos(arg)).sum(1, keepdim=True)       # S''(a)
            concave = Spp < -1e-12                                              # near a max -> Newton valid
            astep = torch.where(concave, -Sp / Spp.clamp_max(-1e-12),          # Newton toward S'=0 (S''<0)
                                torch.sign(Sp) * step0)                         # else a gentle uphill GD step
            a = (a + astep).clamp(-acap, acap)
        f1 = (wm * torch.cos(phi + a * psi)).sum(1, keepdim=True)               # accept-guard: improve or stay
        a = torch.where(f1 >= (wm * cos_p).sum(1, keepdim=True), a, torch.zeros_like(a))
        vel = LS_MOM * vel + a * d                                             # momentum coast (LS_MOM=0 -> pure LS)
        T = T + vel
    return T


def _psupport(Rd, Q, wm, eye, reg=1e-3):
    """P_support: weighted least-squares fit of a lattice vector to the (B,N) working projections Rd, then
    re-project. v = argmin_v sum wm (Rd - Q v)^2 via batched 3x3 normal equations; returns (u_s=Qv, v)."""
    A = torch.einsum("bn,ni,nj->bij", wm, Q, Q) + reg * eye                     # (B,3,3)
    rhs = torch.einsum("bn,ni->bi", wm * Rd, Q)                                 # (B,3)
    v = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)                    # (B,3)
    return v @ Q.t(), v


def _er_polish(u, Q, w0, eye, tol, k):
    """k error-reduction (alternating-projection) steps u <- P_support(P_data(u)) -- the clean-up the wild
    HIO/DR dynamics needs (its fixed points aren't clean projections). Returns the extracted feasible v."""
    v = None
    for _ in range(max(1, k)):
        r = torch.round(u); inl = (torch.abs(u - r) < tol).to(u.dtype)
        u, v = _psupport(torch.where(inl > 0, r, u), Q, w0 * inl, eye)
    return v


def refine_vec_raar(T, Q, w, qmax, steps=8, tol=0.18, sharp_last=None, mom=None, beta=None):
    """M3 variant: RAAR (relaxed averaged alternating reflections) -- INDEXING AS PHASE RETRIEVAL. Work in
    the per-peak projections u = q.v. Two constraint sets, both cheap: P_data = round the INLIER u_i to the
    nearest integer (the 'magnitude'/Miller-index constraint, elementwise, outliers left free); P_support =
    weighted least-squares refit v=Q^+u then u=Qv (project onto range(Q), the 'support' -- one batched 3x3
    solve, the anneal step). RAAR: u <- (beta/2)(R_s R_d + I)u + (1-beta) P_d u, R=2P-I. The reflections give
    HIO-style FEEDBACK that escapes the spurious basins plain ascent / error-reduction fall into; range(Q) is
    a FIXED linear subspace, so the support is maximally stable (the regime where saddle/DR methods work).
    beta via BETA env. Returns the feasible v fit to the confident integer assignments."""
    b_end = BETA if beta is None else beta
    b = b_end
    eye = torch.eye(3, dtype=T.dtype, device=T.device)
    w0 = w.unsqueeze(0)
    u = T @ Q.t()                                                              # (B,N) initial projections
    for s in range(steps):
        if BETA_SCHED:                                                         # HIO/DR (beta=BETA_HI, aggressive reflection) ->
            p = s / max(steps - 1, 1)                                          #   RAAR (beta=b_end, data-projection weight): explore
            b = BETA_HI * (1.0 - p) + b_end * p                                #   early to escape spurious basins, settle late
        r = torch.round(u)
        if RAAR_ANNEAL:                                                        # soft->hard round-to-hkl (deterministic annealing):
            p = s / max(steps - 1, 1)                                          #   the round constraint HARDENS as it ramps, co-active
            alpha = RAAR_A0 + (1.0 - RAAR_A0) * p                              #   with the RAAR reflection kick -- ambiguous peaks stay
            tol_s = tol * (1.0 + (RAAR_TOLK - 1.0) * (1.0 - p))                #   undecided until range(Q) resolves them, before the
            rr = alpha * r + (1.0 - alpha) * u                                 #   integer commitment locks a short-vector sublattice.
        else:
            tol_s = tol; rr = r
        inl = (torch.abs(u - r) < tol_s).to(u.dtype)
        wm = w0 * inl                                                          # weight only current inliers
        Pd = torch.where(inl > 0, rr, u)                                       # (soft-)round inliers, keep outliers
        Rd = 2 * Pd - u
        PsRd, _ = _psupport(Rd, Q, wm, eye)                                    # P_support(R_d u)
        RsRd = 2 * PsRd - Rd
        u = 0.5 * b * (RsRd + u) + (1 - b) * Pd                                # RAAR update (b=1 -> DR/HIO)
    if POLISH:                                                                 # HIO/DR needs an ER polish
        return _er_polish(u, Q, w0, eye, tol, POLISH)
    r = torch.round(u); inl = (torch.abs(u - r) < tol).to(u.dtype)            # extract feasible v from
    _, v = _psupport(r, Q, w0 * inl, eye)                                      # the confident integer hkl
    return v


def refine_vec_so2d(T, Q, w, qmax, steps=8, tol=0.18, sharp_last=None, mom=None, beta=None):
    """M3 variant: SO2D saddle-point optimizer -- RAAR whose feedback beta is chosen PER SEED PER STEP by an
    analytic inner solve of the saddle metric L = e_d^2 - w_s e_s^2 (data error minus support error), instead
    of a fixed beta. Along the RAAR direction u(b) = P_d u + b*Delta (Delta = M1 - P_d u, M1 the DR average),
    both e_d^2 and e_s^2 are quadratic in b (frozen hkl + the linear support projector), so the stationary
    b* = -(b_d - w_s b_s)/(c_d - w_s c_s) is closed-form (HIO fallback b=0.75 on a degenerate denominator).
    range(Q) is a FIXED linear subspace -> maximally stable support, the regime SO is built for. w_s via SO_WS."""
    eye = torch.eye(3, dtype=T.dtype, device=T.device)
    w0 = w.unsqueeze(0); ws = SO_WS
    u = T @ Q.t()
    for _ in range(WARM):                                                       # RAAR warm-up: form the basin
        r = torch.round(u); inl = (torch.abs(u - r) < tol).to(u.dtype); wm = w0 * inl   # (SO needs a stable
        Pd = torch.where(inl > 0, r, u); Rd = 2 * Pd - u                        #  iterate; raw seeds aren't)
        PsRd, _ = _psupport(Rd, Q, wm, eye)
        u = 0.5 * BETA * (2 * PsRd - Rd + u) + (1 - BETA) * Pd
    for s in range(steps):
        r = torch.round(u); inl = (torch.abs(u - r) < tol).to(u.dtype); wm = w0 * inl
        Pd = torch.where(inl > 0, r, u)
        Rd = 2 * Pd - u
        PsRd, _ = _psupport(Rd, Q, wm, eye)
        M1 = 0.5 * (2 * PsRd - Rd + u)                                          # DR average (RAAR beta=1 point)
        Delta = M1 - Pd                                                         # search direction from P_d u
        base_d = Pd - r                                                         # data residual at b=0 (0 on inliers)
        b_d = (wm * base_d * Delta).sum(1, keepdim=True)
        c_d = (wm * Delta * Delta).sum(1, keepdim=True)
        PsPd, _ = _psupport(Pd, Q, wm, eye); es_base = Pd - PsPd                # support residual at b=0
        PsDl, _ = _psupport(Delta, Q, wm, eye); es_del = Delta - PsDl           # its rate along Delta
        b_s = (wm * es_base * es_del).sum(1, keepdim=True)
        c_s = (wm * es_del * es_del).sum(1, keepdim=True)
        den = c_d - ws * c_s
        bstar = torch.where(den.abs() > 1e-9, -(b_d - ws * b_s) / den, torch.full_like(den, 0.75))
        u = Pd + bstar.clamp(-0.5, 2.5) * Delta                                # saddle step (HIO-range clamp)
    if POLISH:
        return _er_polish(u, Q, w0, eye, tol, POLISH)
    r = torch.round(u); inl = (torch.abs(u - r) < tol).to(u.dtype)
    _, v = _psupport(r, Q, w0 * inl, eye)
    return v


def refine_vec_admm(T, Q, w, qmax, steps=8, tol=0.18, sharp_last=None, mom=None, beta=None):
    """M3 variant: ADMM (scaled-dual splitting) for indexing-as-phase-retrieval. Split variables x (data)
    and z (support) with a dual u: x = P_data(z - u); z = P_support(x + u); u = u + rho*(x - z). For two
    indicator sets this is Douglas-Rachford on the dual, but the EXPLICIT multiplier + penalty rho give a
    differently-conditioned trajectory than fixed-beta RAAR. rho via RHO env; optional ER polish."""
    eye = torch.eye(3, dtype=T.dtype, device=T.device)
    w0 = w.unsqueeze(0); rho = RHO
    z = T @ Q.t()                                                              # support-side var (start feasible)
    dual = torch.zeros_like(z)
    for s in range(steps):
        a = z - dual                                                          # x = P_data(z - u)
        r = torch.round(a); inl = (torch.abs(a - r) < tol).to(a.dtype); wm = w0 * inl
        x = torch.where(inl > 0, r, a)
        z, _ = _psupport(x + dual, Q, wm, eye)                                # z = P_support(x + u)
        dual = dual + rho * (x - z)                                           # scaled dual update
    if POLISH:
        return _er_polish(z, Q, w0, eye, tol, POLISH)
    r = torch.round(z); inl = (torch.abs(z - r) < tol).to(z.dtype)
    _, v = _psupport(r, Q, w0 * inl, eye)
    return v


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
    from glint.lattice import cell_to_Ar
    from glint.multishot import same_lattice
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


# ---- M3 ADAPTIVE: step count keyed to the top-seed inlier RESIDUAL -------------
_ADAPT_LOG = []


def refine_vec_adapt(T, Q, w, qmax, steps=8, tol=0.18, sharp_last=25, mom=0.5,
                     patience=2, rel=0.03, kbest=64, min_steps=2):
    """M3 variant: same GD-momentum ascent as refine_vec, but EARLY-STOPS keyed to RESIDUAL -- the median
    inlier |q.v - round(q.v)| of the top-K seeds (by inlier count). When that stops decreasing, the good
    vectors have locked onto integer lattice positions and extra steps only churn the rest. ``steps`` caps
    the loop. Records steps used in ``_ADAPT_LOG``."""
    vel = torch.zeros_like(T); step0 = 0.25 / qmax
    prev = None; stall = 0; used = steps
    for s in range(steps):
        lr = step0 * (1 - 0.7 * s / steps)
        _, g = objective(T, Q, w, tol, sharp=(s >= steps - sharp_last))
        vel = mom * vel + g / (g.norm(dim=1, keepdim=True) + 1e-12)
        T = T + lr * vel
        if s + 1 >= min_steps:
            proj = T @ Q.t(); d = torch.abs(proj - torch.round(proj))
            inl = (d < tol).to(d.dtype); nin = inl.sum(1)
            seedres = (d * inl).sum(1) / (nin + 1e-9)
            k = min(kbest, int(nin.shape[0])); idx = torch.topk(nin, k).indices
            cur = float(seedres[idx].median())
            if prev is not None and cur >= prev * (1 - rel):
                stall += 1
                if stall >= patience:
                    used = s + 1; break
            else:
                stall = 0
            prev = cur
    _ADAPT_LOG.append(used)
    return T
