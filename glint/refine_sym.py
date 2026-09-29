"""Bravais-symmetry-CONSTRAINED prediction refinement.

GLINT's native ``index.refine`` fits an unconstrained reciprocal basis
``C = (G H^T)(H H^T)^{-1}`` -- all 9 entries free (a triclinic cell).  Under
realistic peak-position noise those extra DOF absorb noise into a spurious
lower-symmetry cell distortion (measured on real lyso: ``a != b`` by ~2.5%,
``gamma`` ~92.9 off 90), which couples into the orientation and drifts the
PREDICTED spot positions off the Bravais manifold -- the deficiency flagged at
``predict.py`` write_solution_file and the reason a native GLINT merge lags the
CrystFEL ``--fromfile`` handoff (whose prediction-refinement imposes lattice
symmetry).

``refine_bravais`` replaces the free 3x3 fit with a low-DOF Gauss-Newton / LM
fit over ``[rotation tangent (3) + the SYMMETRY-FREE cell params (1..6)]``, the
lattice symmetry built INTO the parameterization so ``a=b`` / angle=90/120 hold
to machine precision (a hard constraint, not a soft penalty): the cell is
REBUILT from ``cell_to_Ar`` each iterate, so it can never leave the manifold.

Convention (matches ``index.refine`` / ``predict.recip_from_M`` exactly)::

    M   = R @ Ar          (columns = rotated real axes,     A)
    C   = inv(M).T = R @ B (columns a*,b*,c*,               1/A)
    g_i = C @ h_i          (observed reciprocal node)
    h_i = round(inv(C) @ g_i) = round(M.T @ g_i)

Free params per Bravais system (standard setting; fixed entries substituted)::

    triclinic     (a,b,c,al,be,ga)          n_p = 6   (== unconstrained control)
    monoclinic    (a,b,c,be)  al=ga=90       n_p = 4   (unique axis b)
    orthorhombic  (a,b,c)     al=be=ga=90     n_p = 3
    hexagonal     (a,c)       b=a al=be=90 ga=120  n_p = 2  (unique axis c)
    trigonal      (a,al)      b=c=a be=ga=al   n_p = 2   (rhombohedral setting)
    tetragonal    (a,c)       b=a al=be=ga=90  n_p = 2  (unique axis c)
    cubic         (a)         b=c=a al=be=ga=90  n_p = 1

The rotation is retracted on SO(3) by a left exponential map so it stays a
proper rotation exactly; LM damping + a length/volume guard prevent divergence.

Default OFF everywhere it is wired in (``predict.integrate_cxi(sym_refine=...)``),
so nothing regresses if it underperforms.
"""
from __future__ import annotations

import numpy as np

from .lattice import cell_to_Ar, Ar_to_Br, cell_params

# free-parameter dimension per system
_NP = {"cubic": 1, "tetragonal": 2, "hexagonal": 2, "trigonal": 2,
       "orthorhombic": 3, "monoclinic": 4, "triclinic": 6}


# ----------------------------------------------------------------------------
# small SO(3) helpers
# ----------------------------------------------------------------------------
def _skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def _expmap(w):
    """Rodrigues: rotation matrix from an axis-angle vector w."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        K = _skew(w)
        return np.eye(3) + K + 0.5 * K @ K
    k = w / th
    K = _skew(k)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _polar(X):
    """Nearest proper rotation to X (polar factor, det>0)."""
    U, _, Vt = np.linalg.svd(X)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] = -U[:, -1]
        R = U @ Vt
    return R


# ----------------------------------------------------------------------------
# cell <-> free-param mapping (symmetry built in)
# ----------------------------------------------------------------------------
def _expand(p, system):
    """Free params p -> (a,b,c,alpha,beta,gamma) for cell_to_Ar."""
    if system == "cubic":
        a = p[0]; return (a, a, a, 90.0, 90.0, 90.0)
    if system == "tetragonal":
        a, c = p; return (a, a, c, 90.0, 90.0, 90.0)
    if system == "hexagonal":
        a, c = p; return (a, a, c, 90.0, 90.0, 120.0)
    if system == "trigonal":
        a, al = p; return (a, a, a, al, al, al)
    if system == "orthorhombic":
        a, b, c = p; return (a, b, c, 90.0, 90.0, 90.0)
    if system == "monoclinic":
        a, b, c, be = p; return (a, b, c, 90.0, be, 90.0)
    if system == "triclinic":
        return tuple(p)
    raise ValueError(f"unknown system {system}")


def _project(cell, system):
    """Project (a,b,c,al,be,ga) onto the Bravais manifold -> free params p0."""
    a, b, c, al, be, ga = cell
    if system == "cubic":
        return np.array([(a + b + c) / 3.0])
    if system == "tetragonal":
        return np.array([(a + b) / 2.0, c])
    if system == "hexagonal":
        return np.array([(a + b) / 2.0, c])
    if system == "trigonal":
        return np.array([(a + b + c) / 3.0, (al + be + ga) / 3.0])
    if system == "orthorhombic":
        return np.array([a, b, c])
    if system == "monoclinic":
        return np.array([a, b, c, be])
    if system == "triclinic":
        return np.array([a, b, c, al, be, ga])
    raise ValueError(f"unknown system {system}")


def _B_cell(p, system):
    """Reciprocal basis B (columns a*,b*,c*) rebuilt from free params -> ON the manifold."""
    return Ar_to_Br(cell_to_Ar(*_expand(p, system)))


# ----------------------------------------------------------------------------
# axis-setting standardization (unique axis where the parameterization expects it)
# ----------------------------------------------------------------------------
def _closest_pair(lens):
    """Indices (i,j) of the two most nearly equal of three lengths."""
    d = [(abs(lens[0] - lens[1]), 0, 1),
         (abs(lens[0] - lens[2]), 0, 2),
         (abs(lens[1] - lens[2]), 1, 2)]
    d.sort()
    return d[0][1], d[0][2]


def standardize_setting(M, system):
    """Reorder/relabel the real-space axes (columns of M) into the standard setting for
    ``system`` so the free-param constraints (a=b, unique axis, angles) act on the right
    axes.  Handedness preserved (proper rotation).  Idempotent when already standard."""
    M = np.asarray(M, float)
    lens = np.linalg.norm(M, axis=0)
    if system in ("cubic", "trigonal", "triclinic", "orthorhombic"):
        # cubic/trigonal: all axes equivalent.  triclinic: no constraint.  orthorhombic:
        # a,b,c all independently free (constraint = angles 90 only), so no axis is special
        # and reordering distinct axes would just relabel the orientation frame (not a symmetry).
        order = [0, 1, 2]
    elif system in ("tetragonal", "hexagonal"):
        i, j = _closest_pair(lens)          # a,b = the equal pair; c = odd one out -> unique axis c
        k = 3 - i - j
        order = [i, j, k]
    elif system == "monoclinic":
        cols = [M[:, i] / lens[i] for i in range(3)]
        cs = [abs(cols[0] @ cols[1]), abs(cols[0] @ cols[2]), abs(cols[1] @ cols[2])]
        pair = [(0, 1), (0, 2), (1, 2)][int(np.argmax(cs))]   # the oblique (a,c) pair
        i, k = pair
        j = 3 - i - k                        # the unique axis b
        order = [i, j, k]
    else:
        raise ValueError(f"unknown system {system}")
    Mo = M[:, order].copy()
    if np.linalg.det(Mo) < 0:
        Mo[:, 0] = -Mo[:, 0]                 # keep a proper (right-handed) basis
    return Mo


# ----------------------------------------------------------------------------
# the constrained refine
# ----------------------------------------------------------------------------
def _residual(R0, w, p, hkl, g, system):
    R = _expmap(w) @ R0
    C = R @ _B_cell(p, system)
    return (hkl @ C.T) - g            # (m,3) predicted - observed


def _lm_inner(R0, p, hkl, g, system, refine_cell, max_steps=8):
    """Levenberg-Marquardt on x=[w(3); p] with fixed hkl assignment. Rotation retracted
    on SO(3). Returns (R0, p, final_cost)."""
    n_p = _NP[system] if refine_cell else 0
    n = 3 + n_p
    w = np.zeros(3)
    r = _residual(R0, w, p, hkl, g, system).reshape(-1)
    cost = float(r @ r)
    mu = 1e-3
    for _ in range(max_steps):
        # finite-difference Jacobian (n params, m*3 residuals)
        m3 = r.size
        J = np.zeros((m3, n))
        for a in range(3):                        # rotation tangent
            dw = np.zeros(3); dw[a] = 1e-6
            rp = _residual(R0, w + dw, p, hkl, g, system).reshape(-1)
            rm = _residual(R0, w - dw, p, hkl, g, system).reshape(-1)
            J[:, a] = (rp - rm) / (2e-6)
        for b in range(n_p):                      # cell params
            eps = 1e-5 * max(abs(p[b]), 1.0)
            pp = p.copy(); pp[b] += eps
            pm = p.copy(); pm[b] -= eps
            rp = _residual(R0, w, pp, hkl, g, system).reshape(-1)
            rm = _residual(R0, w, pm, hkl, g, system).reshape(-1)
            J[:, 3 + b] = (rp - rm) / (2 * eps)
        JtJ = J.T @ J
        Jtr = J.T @ r
        diag = np.diag(np.diag(JtJ))
        accepted = False
        for _try in range(8):
            try:
                dx = np.linalg.solve(JtJ + mu * diag + 1e-15 * np.eye(n), -Jtr)
            except np.linalg.LinAlgError:
                mu *= 10.0; continue
            w_try = dx[:3]
            p_try = p.copy()
            if n_p:
                p_try = p + dx[3:]
                # length/volume guard: reject non-physical cells
                cell = _expand(p_try, system)
                if any(cell[i] <= 0 for i in range(3)):
                    mu *= 10.0; continue
                Ar = cell_to_Ar(*cell)
                if np.linalg.det(Ar) <= 0:
                    mu *= 10.0; continue
            r_try = _residual(R0, w_try, p_try, hkl, g, system).reshape(-1)
            cost_try = float(r_try @ r_try)
            if cost_try < cost:
                # accept: retract rotation, update p, reset w
                R0 = _expmap(w_try) @ R0
                p = p_try
                w = np.zeros(3)
                r = _residual(R0, w, p, hkl, g, system).reshape(-1)
                cost = cost_try
                mu = max(mu / 10.0, 1e-12)
                accepted = True
                break
            else:
                mu *= 10.0
        if not accepted or np.linalg.norm(dx) < 1e-9:
            break
    return R0, p, cost


def refine_bravais(g, M, system, tol_abs, n_iter=5, refine_cell=True,
                   min_inliers=6, standardize=True):
    """Bravais-constrained sibling of ``index.refine``.

    g        : (n,3) observed reciprocal peaks (1/A)
    M        : (3,3) initial real basis (columns = rotated real axes)
    system   : Bravais system name (cubic/tetragonal/orthorhombic/hexagonal/
               trigonal/monoclinic/triclinic)
    tol_abs  : absolute inlier tolerance in 1/A (as in index.refine)

    Returns (M, C, hkl, inliers) with the cell LOCKED to the Bravais manifold to
    machine precision.  Falls back to rotation-only refine when too few inliers
    constrain the cell.  Never leaves the manifold and guards against divergence.
    """
    g = np.asarray(g, float)
    if system == "triclinic":
        refine_cell = True  # 6 free params == unconstrained
    if standardize:
        M = standardize_setting(M, system)
    # initialise (R0, p0) from M: cell params are rotation-invariant, project onto manifold
    cell0 = cell_params(M)
    p = _project(cell0, system)
    Ar0 = cell_to_Ar(*_expand(p, system))
    R0 = _polar(M @ np.linalg.inv(Ar0))

    C = R0 @ _B_cell(p, system)
    hkl = np.rint(g @ np.linalg.inv(C).T)
    inliers = np.linalg.norm(g - hkl @ C.T, axis=1) < tol_abs

    for _ in range(n_iter):
        C = R0 @ _B_cell(p, system)
        Minv_t = np.linalg.inv(C).T                 # == M
        hkl = np.rint(g @ Minv_t)
        resid = np.linalg.norm(g - hkl @ C.T, axis=1)
        inliers = resid < tol_abs
        m = int(inliers.sum())
        if m < 4:
            break
        do_cell = refine_cell and (m >= min_inliers)
        R0, p, _cost = _lm_inner(R0, p, hkl[inliers], g[inliers], system, do_cell)

    C = R0 @ _B_cell(p, system)
    M_out = np.linalg.inv(C).T
    hkl = np.rint(g @ np.linalg.inv(C).T)
    inliers = np.linalg.norm(g - hkl @ C.T, axis=1) < tol_abs
    return M_out, C, hkl, inliers
