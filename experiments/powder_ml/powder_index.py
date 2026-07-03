"""Ab-initio powder-pattern autoindexing -- a small primitive, no crystallography-suite dependency.

Given the ring positions of a powder pattern (the peaks of ``radial.py``'s ``I(q)`` profile), find the
**unit cell + Miller indices** that explain them: classic autoindexing (ITO / TREOR / DICVOL / McMaille
territory), rebuilt as one dependency-free file that runs the same code on CPU (numpy) and GPU (cupy).

    from radial import RadialIntegrator
    from powder_index import index_powder, peaks_from_profile
    q, I   = RadialIntegrator(q_per_pixel).integrate(frame)      # powder profile
    qpk    = peaks_from_profile(q, I)                            # ring positions (physics q, A^-1)
    sols   = index_powder(qpk, units="q")                       # ranked unit-cell solutions
    print(sols[0])   # Solution(system='cubic', a=5.431, ..., M20=42.7, nindexed=18)

Design: the exhaustive search classic indexers run on one CPU core is phrased here as **batched array ops**
(vectorised over trial cells/indices), which keeps the code small and portable -- the same code dispatches to
cupy if you pass cupy arrays. NOTE, though, that unlike the pixel-parallel primitives in this repo
(``radial.py``, the peak finders), a powder pattern is *small* -- a few dozen lines, tiny metric solves -- so
this is a **latency-bound CPU workload**: the cupy path is bit-identical but *not* faster (measured ~60x
slower on an A100 for a 20-line pattern -- too many small kernel launches / host syncs to amortise). Run it on
the CPU; ``xp=`` exists for portability/consistency, not speed. A pattern indexes in well under a second.

  * **Seed-and-verify (TREOR/ITO-style)** -- ``1/d^2`` is *linear* in the reciprocal-metric components, so
    for cubic/tetragonal/hexagonal-trigonal/orthorhombic (1--3 free components) **and monoclinic** (4) we
    assign trial Miller indices to m-subsets of the first few observed lines, **batch-solve** the small
    metric systems, and verify every candidate cell against all the lines at once. Monoclinic cells are
    then reduced (2-D Gaussian reduction of the a-c plane) to the conventional setting. Fast and exhaustive.
  * **Triclinic (6 free components) -- simulated annealing.** Seed-and-verify's index combinatorics explode
    at m=6, and a plain random grid + refine does *not* converge (a signed 6-parameter lattice puts a
    reflection within tolerance of almost any line, so a nearest-residual cost is trivially minimised by
    making the lattice denser). The fix is a *directed* search (the McMaille idea done right): many Metropolis
    chains anneal the six cell parameters against the **de Wolff M20 objective** -- mean relative residual
    times the predicted-line count -- so density is penalised; a residual floor keeps that penalty active as
    the fit sharpens (so a supercell can't tie the true cell), candidates are polished by small-perturbation
    restarts, and integer-volume supercells are removed. Recovers general triclinic cells reliably (~20 s CPU,
    ``tri_*`` knobs); stochastic, and pseudo-symmetric cells (e.g. equal edges) remain a hard case.

The math.  Each ring gives ``Q = 1/d^2`` (A^-2).  A reflection ``(h,k,l)`` has

    1/d^2(hkl) = h^2 A11 + k^2 A22 + l^2 A33 + 2hk A12 + 2hl A13 + 2kl A23

where ``A = G*`` is the reciprocal metric tensor (``G* = G^-1``, ``G`` the direct metric from the cell).
Solutions are ranked by the **de Wolff figure of merit** ``M20 = Q20 / (2 <|dQ|> N20)`` (``Q20`` = the 20th
observed line, ``N20`` = number of calculable lines up to it); ``M20 > 10`` is the usual reliability mark.

Credit / lineage: P. M. de Wolff, *J. Appl. Cryst.* **1**, 108 (1968) (M20); J. W. Visser, *J. Appl. Cryst.*
**2**, 89 (1969) (ITO / zone indexing); P.-E. Werner *et al.*, *J. Appl. Cryst.* **18**, 367 (1985) (TREOR);
A. Boultif & D. Louer, *J. Appl. Cryst.* **24**, 987 (1991); **37**, 724 (2004) (DICVOL, successive
dichotomy); A. Le Bail, *Powder Diffraction* **19**, 249 (2004) (McMaille, Monte-Carlo). Benchmarked against
GSAS-II (B. H. Toby & R. B. Von Dreele, *J. Appl. Cryst.* **46**, 544 (2013)).

Authors: SLAC LCLS DRP team, with Claude (Anthropic).
"""
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Solution", "index_powder", "peaks_from_profile", "cell_to_metric", "d_spacings"]


def _xp(a):
    """numpy for a numpy array, cupy for a cupy array."""
    try:
        import cupy
        return cupy.get_array_module(a)
    except Exception:
        return np


# --------------------------------------------------------------------------------------------------
# Units: any input position -> Q = 1/d^2  (A^-2)
# --------------------------------------------------------------------------------------------------
def to_inv_d2(pos, units="q", wavelength=None):
    """Convert observed ring positions to ``Q = 1/d^2`` (A^-2), sorted ascending.

    units:
      * ``"q"``        -- physics q = 4*pi*sin(theta)/lambda (A^-1), as ``radial.py`` produces: d = 2*pi/q.
      * ``"d"``        -- d-spacing (A).
      * ``"twotheta"`` -- scattering angle 2*theta in **degrees**; needs ``wavelength`` (A).
      * ``"inv_d2"``   -- already 1/d^2 (A^-2).
    """
    pos = np.asarray(pos, dtype=np.float64)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    if units == "q":
        d = 2.0 * math.pi / pos
        Q = 1.0 / d ** 2
    elif units == "d":
        Q = 1.0 / pos ** 2
    elif units == "twotheta":
        if wavelength is None:
            raise ValueError("units='twotheta' requires wavelength=")
        th = np.deg2rad(pos) / 2.0
        d = wavelength / (2.0 * np.sin(th))
        Q = 1.0 / d ** 2
    elif units == "inv_d2":
        Q = pos.astype(np.float64)
    else:
        raise ValueError("units must be 'q', 'd', 'twotheta' or 'inv_d2'")
    return np.sort(Q)


# --------------------------------------------------------------------------------------------------
# Cell <-> metric tensor
# --------------------------------------------------------------------------------------------------
def cell_to_metric(a, b, c, alpha, beta, gamma):
    """Direct cell (angles in degrees) -> reciprocal metric tensor A = G* (3x3, A^-2)."""
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    G = np.array([[a * a, a * b * cg, a * c * cb],
                  [a * b * cg, b * b, b * c * ca],
                  [a * c * cb, b * c * ca, c * c]], dtype=np.float64)
    return np.linalg.inv(G)


def metric_to_cell(A):
    """Reciprocal metric A = G* -> direct cell (a,b,c,alpha,beta,gamma in degrees) + volume (A^3)."""
    G = np.linalg.inv(np.asarray(A, dtype=np.float64))
    a, b, c = (math.sqrt(max(G[i, i], 1e-30)) for i in range(3))
    al = math.degrees(math.acos(np.clip(G[1, 2] / (b * c), -1, 1)))
    be = math.degrees(math.acos(np.clip(G[0, 2] / (a * c), -1, 1)))
    ga = math.degrees(math.acos(np.clip(G[0, 1] / (a * b), -1, 1)))
    vol = math.sqrt(max(np.linalg.det(G), 1e-30))
    return a, b, c, al, be, ga, vol


def _reduce_monoclinic(a, b, c, alpha, beta, gamma):
    """Reduce a monoclinic cell to its conventional setting: shortest a,c in the (unique-b) a-c plane,
    obtuse beta.  Ab-initio indexing pins the *lattice* but often a non-reduced (a,c,beta) basis; a 2-D
    Gaussian (Lagrange) reduction of the a-c plane picks the conventional edges/angle."""
    u = np.array([a, 0.0])
    v = np.array([c * math.cos(math.radians(beta)), c * math.sin(math.radians(beta))])
    for _ in range(50):
        if v @ v < u @ u:
            u, v = v, u
        m = round((u @ v) / (u @ u))
        if m == 0:
            break
        v = v - m * u
    a2, c2 = math.hypot(*u), math.hypot(*v)
    ang = math.degrees(math.acos(np.clip((u @ v) / (a2 * c2), -1, 1)))
    if ang < 90:
        ang = 180.0 - ang                              # conventional obtuse beta
    return a2, b, c2, 90.0, ang, 90.0


def d_spacings(cell, hkl):
    """Cell (a,b,c,al,be,ga) and integer hkl array (n,3) -> d-spacings (A). Convenience for tests/sims."""
    A = cell_to_metric(*cell)
    hkl = np.asarray(hkl, dtype=np.float64)
    Q = np.einsum("ni,ij,nj->n", hkl, A, hkl)
    return 1.0 / np.sqrt(Q)


# --------------------------------------------------------------------------------------------------
# Per-system description: free-parameter count, design columns, cell reconstruction
# --------------------------------------------------------------------------------------------------
# The "design row" of a reflection gives the coefficient of each free metric parameter, so that
# Q(hkl) = design_row . params.  Signed hkl matters when there are cross terms (hex h*k; mono h*l; triclinic).
_SIGNED = {"hexagonal", "trigonal", "monoclinic", "triclinic"}


def _nparam(system):
    return {"cubic": 1, "tetragonal": 2, "hexagonal": 2, "trigonal": 2, "orthorhombic": 3,
            "monoclinic": 4, "triclinic": 6}[system]


def _design(system, H):
    """(n,3) integer hkl -> (n, nparam) design matrix of metric-parameter coefficients."""
    xp = _xp(H)
    h, k, l = H[:, 0], H[:, 1], H[:, 2]
    if system == "cubic":
        return (h * h + k * k + l * l).astype(xp.float64)[:, None]
    if system == "tetragonal":
        return xp.stack([h * h + k * k, l * l], axis=1).astype(xp.float64)
    if system in ("hexagonal", "trigonal"):
        return xp.stack([h * h + h * k + k * k, l * l], axis=1).astype(xp.float64)
    if system == "orthorhombic":
        return xp.stack([h * h, k * k, l * l], axis=1).astype(xp.float64)
    if system == "monoclinic":                                  # b-unique: free G* = A11,A22,A33,A13
        return xp.stack([h * h, k * k, l * l, 2.0 * h * l], axis=1).astype(xp.float64)
    if system == "triclinic":                                   # all six G* components
        return xp.stack([h * h, k * k, l * l, 2.0 * k * l, 2.0 * h * l, 2.0 * h * k],
                        axis=1).astype(xp.float64)
    raise NotImplementedError(system)


def _params_to_cell(system, p):
    """Free params -> (a,b,c,alpha,beta,gamma, volume). p is a length-nparam sequence (host floats)."""
    p = [float(x) for x in p]
    if system == "cubic":
        a = 1.0 / math.sqrt(p[0]);  return a, a, a, 90., 90., 90., a ** 3
    if system == "tetragonal":
        a = 1.0 / math.sqrt(p[0]); c = 1.0 / math.sqrt(p[1]); return a, a, c, 90., 90., 90., a * a * c
    if system in ("hexagonal", "trigonal"):
        a = math.sqrt(4.0 / (3.0 * p[0])); c = 1.0 / math.sqrt(p[1])
        return a, a, c, 90., 90., 120., (math.sqrt(3) / 2.0) * a * a * c
    if system == "orthorhombic":
        a, b, c = (1.0 / math.sqrt(x) for x in p[:3]); return a, b, c, 90., 90., 90., a * b * c
    if system == "monoclinic":                                  # comps = A11,A22,A33,A13
        A = np.array([[p[0], 0, p[3]], [0, p[1], 0], [p[3], 0, p[2]]], float)
        return metric_to_cell(A)
    if system == "triclinic":                                   # comps = A11,A22,A33,A23,A13,A12
        A = np.array([[p[0], p[5], p[4]], [p[5], p[1], p[3]], [p[4], p[3], p[2]]], float)
        return metric_to_cell(A)
    raise NotImplementedError(system)


# lattice-centering reflection conditions (True = allowed)
def _centering_ok(H, centering):
    xp = _xp(H)
    h, k, l = H[:, 0], H[:, 1], H[:, 2]
    if centering == "P":
        return xp.ones(h.shape, bool)
    if centering == "I":
        return ((h + k + l) % 2) == 0
    if centering == "F":
        return ((h % 2) == (k % 2)) & ((k % 2) == (l % 2))
    if centering == "A":
        return ((k + l) % 2) == 0
    if centering == "B":
        return ((h + l) % 2) == 0
    if centering == "C":
        return ((h + k) % 2) == 0
    if centering == "R":                              # obverse setting, hexagonal axes
        return ((-h + k + l) % 3) == 0
    raise ValueError("unknown centering " + centering)


def _gen_hkl(system, imax, centering, xp):
    """All lattice-allowed reflections in an index box, as an (n,3) int array (origin removed).

    Non-signed systems (cubic/tet/ortho) fold to the non-negative octant since Q is even in each index;
    signed systems (hex/mono/triclinic) use a Friedel-unique half of the full box (first nonzero index > 0)
    because a cross term makes Q depend on the relative signs.
    """
    if system in _SIGNED:
        rng = xp.arange(-imax, imax + 1)
        H = xp.stack(xp.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
        h, k, l = H[:, 0], H[:, 1], H[:, 2]
        friedel = (h > 0) | ((h == 0) & (k > 0)) | ((h == 0) & (k == 0) & (l > 0))
        H = H[friedel]
    else:
        hh = xp.arange(0, imax + 1)
        H = xp.stack(xp.meshgrid(hh, hh, hh, indexing="ij"), -1).reshape(-1, 3)
        H = H[xp.any(H != 0, axis=1)]
    H = H[_centering_ok(H, centering)]
    return H.astype(xp.int64)


# --------------------------------------------------------------------------------------------------
# Figure of merit (de Wolff M20) -- computed exactly for the few final solutions
# --------------------------------------------------------------------------------------------------
def m20_fom(Qobs, system, params, centering, imax, nlines=20, dq_floor_rel=1e-4):
    """de Wolff M_N (default N=20) for a candidate cell, plus the indexed-hkl assignment.

    M_N = Q_N / (2 <|dQ|> N_N):  Q_N = Nth observed 1/d^2, <|dQ|> = mean |Q_obs - Q_calc| over the first N
    indexed lines, N_N = number of distinct calculable lines with Q_calc <= Q_N.  Returns (M, ncalc, hkl,
    dQmean) where hkl is the nearest reflection for each observed line (host numpy).

    ``<|dQ|>`` is floored at ``dq_floor_rel * Q_N`` (a realistic relative precision): with perfect
    synthetic data ``<|dQ|>`` -> 0 and M_N would blow up, which lets a denser wrong cell over-fit past the
    true one.  Flooring makes ``N_N`` (the count of predicted-but-unobserved lines) the real discriminator,
    as de Wolff intended -- the correct cell predicts no extra lines (``N_N`` = number of observed lines).
    """
    Qobs = np.asarray(Qobs, np.float64)
    N = min(nlines, len(Qobs))
    QN = Qobs[N - 1]
    H = np.asarray(_gen_hkl(system, imax, centering, np), np.int64)
    Qc = (np.asarray(_design(system, H)) @ np.asarray(params, np.float64))
    keep = Qc > 0
    H, Qc = H[keep], Qc[keep]
    order = np.argsort(Qc); Qc, H = Qc[order], H[order]
    # distinct calculable lines up to Q_N (multiplicity-merged by rounding)
    Qc_le = Qc[Qc <= QN * (1.0 + 1e-6)]
    if Qc_le.size:
        uniq = np.unique(np.round(Qc_le / (QN * 1e-4)).astype(np.int64))
        NN = max(int(uniq.size), 1)
    else:
        NN = 1
    # nearest calc line for each observed line
    idx = np.searchsorted(Qc, Qobs)
    idx = np.clip(idx, 1, len(Qc) - 1)
    left = np.abs(Qobs - Qc[idx - 1]); right = np.abs(Qobs - Qc[idx])
    near = np.where(left <= right, idx - 1, idx)
    dQ = np.abs(Qobs - Qc[near])
    dQmean = float(np.mean(dQ[:N])) if N else float("inf")
    dQeff = max(dQmean, dq_floor_rel * QN)                    # floor: perfect fit -> N_N discriminates
    M = QN / (2.0 * dQeff * NN)
    return float(M), NN, H[near], dQmean


# --------------------------------------------------------------------------------------------------
# Engine A -- seed-and-verify for the high-symmetry systems
# --------------------------------------------------------------------------------------------------
@dataclass
class Solution:
    system: str
    centering: str
    a: float; b: float; c: float
    alpha: float; beta: float; gamma: float
    volume: float
    M20: float
    nindexed: int
    nlines: int
    params: tuple = field(default=(), repr=False)
    hkl: object = field(default=None, repr=False)

    @property
    def cell(self):
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    def __repr__(self):
        return ("Solution(system=%r, centering=%r, a=%.4f b=%.4f c=%.4f al=%.2f be=%.2f ga=%.2f, "
                "V=%.1f, M20=%.1f, indexed=%d/%d)" % (
                    self.system, self.centering, self.a, self.b, self.c, self.alpha, self.beta,
                    self.gamma, self.volume, self.M20, self.nindexed, self.nlines))


def _seed_and_verify(Qobs, system, centering, xp, amin, amax, reltol, seed_smax, imax, max_unindexed,
                     seed_lines=None, seed_chunk=4_000_000):
    """Return a list of (params_tuple, n_indexed) candidate cells for one system+centering.

    Seeds are assigned to *m*-subsets of the first ``seed_lines`` observed lines (not just the first m):
    for anisotropic cells the reflections that pin the short axis are not among the lowest-Q lines, so a
    first-m-only seeding misses them.  All resulting trial cells are then verified against every line.
    """
    from itertools import combinations
    m = _nparam(system)
    Q = xp.asarray(Qobs, xp.float64)
    nobs = int(Q.size)
    if nobs < m:
        return []
    K = seed_lines if seed_lines is not None else max(m + 3, 6)
    K = min(K, nobs)
    # verification reflection set + its design matrix (built once)
    Hv = _gen_hkl(system, imax, centering, xp)
    Dv = _design(system, Hv)                                   # (Ncand, m)
    # seed candidate coeff rows = the small-index reflections
    smag = Hv[:, 0] ** 2 + Hv[:, 1] ** 2 + Hv[:, 2] ** 2
    Ds = xp.unique(Dv[smag <= seed_smax], axis=0)              # (S, m)
    S = int(Ds.shape[0])
    if S == 0:
        return []
    grids = xp.meshgrid(*[xp.arange(S)] * m, indexing="ij")
    combo = xp.stack([g.ravel() for g in grids], axis=1)       # (S^m, m) candidate assignments
    Mmat = Ds[combo]                                           # (C, m, m)
    det = xp.linalg.det(Mmat)
    Mmat = Mmat[xp.abs(det) > 1e-9]                            # well-conditioned only
    if Mmat.shape[0] == 0:
        return []
    Amax = 1.0 / amin ** 2; Amin = 1.0 / amax ** 2
    ndiag = 3 if m >= 4 else m                                 # off-diagonal metric terms may be negative
    Plist = []
    for lines in combinations(range(K), m):                   # which observed lines seed the system
        rhs = Q[list(lines)]                                   # (m,)
        b = xp.broadcast_to(rhs, (Mmat.shape[0], m))[..., None]
        P = xp.linalg.solve(Mmat, b)[..., 0]                  # (C, m) trial params
        dg = P[:, :ndiag]
        ok = xp.all(dg > 0, axis=1) & xp.all(dg >= Amin * 0.25, axis=1) & xp.all(dg <= Amax * 4.0, axis=1)
        P = P[ok]
        if P.shape[0]:
            Plist.append(P)
    if not Plist:
        return []
    P = xp.concatenate(Plist, axis=0)
    # approximate dedup of identical trials to cut the verify cost
    key = xp.round(P * 1e6).astype(xp.int64)
    P = P[xp.unique(key, axis=0, return_index=True)[1]]
    # verify: how many observed lines does each trial index?  (chunk trials to bound memory)
    Ncand = int(Hv.shape[0])
    counts = xp.empty(P.shape[0], xp.int64)
    step = max(1, seed_chunk // max(1, Ncand * nobs))
    for s in range(0, P.shape[0], step):
        e = min(P.shape[0], s + step)
        Qc = P[s:e] @ Dv.T                                    # (t, Ncand)
        diff = xp.abs(Qc[:, None, :] - Q[None, :, None])      # (t, nobs, Ncand)
        counts[s:e] = (diff <= (reltol * Q)[None, :, None]).any(axis=2).sum(axis=1)
    need = nobs - max_unindexed
    winners = xp.where(counts >= need)[0]
    Pw = P[winners]; cw = counts[winners]
    Pw = np.asarray(Pw.get() if hasattr(Pw, "get") else Pw, np.float64)   # -> host
    cw = np.asarray(cw.get() if hasattr(cw, "get") else cw, np.int64)
    # many seed subsets/index combos give the SAME cell -> dedup before the (costly) refine pass
    if Pw.shape[0]:
        _, uniq = np.unique(np.round(Pw, 6), axis=0, return_index=True)
        Pw, cw = Pw[uniq], cw[uniq]
    return [(tuple(float(x) for x in Pw[i]), int(cw[i])) for i in range(Pw.shape[0])]


# --------------------------------------------------------------------------------------------------
# Triclinic -- simulated annealing (directed search; the McMaille idea done right)
# --------------------------------------------------------------------------------------------------
def _cells_to_free_tri(a, b, c, al, be, ga):
    """Vectorised triclinic cells -> the 6 free reciprocal-metric components (host numpy)."""
    ca, cb, cg = np.cos(np.radians(al)), np.cos(np.radians(be)), np.cos(np.radians(ga))
    n = len(a); G = np.empty((n, 3, 3))
    G[:, 0, 0] = a * a; G[:, 1, 1] = b * b; G[:, 2, 2] = c * c
    G[:, 0, 1] = G[:, 1, 0] = a * b * cg
    G[:, 0, 2] = G[:, 2, 0] = a * c * cb
    G[:, 1, 2] = G[:, 2, 1] = b * c * ca
    A = np.linalg.inv(G)
    return np.stack([A[:, 0, 0], A[:, 1, 1], A[:, 2, 2], A[:, 1, 2], A[:, 0, 2], A[:, 0, 1]], 1)


def _cell_to_vectors(cell):
    a, b, c, al, be, ga = cell
    ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga)); sg = math.sin(math.radians(ga))
    cx = c * cb; cy = c * (ca - cb * cg) / sg; cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([[a, 0, 0], [b * cg, b * sg, 0], [cx, cy, cz]], float)


def _vectors_to_cell(V):
    a, b, c = (float(np.linalg.norm(v)) for v in V)
    al = math.degrees(math.acos(np.clip(V[1] @ V[2] / (b * c), -1, 1)))
    be = math.degrees(math.acos(np.clip(V[0] @ V[2] / (a * c), -1, 1)))
    ga = math.degrees(math.acos(np.clip(V[0] @ V[1] / (a * b), -1, 1)))
    vol = abs(float(np.linalg.det(V)))
    return a, b, c, al, be, ga, vol


def _reduce_cell_3d(cell):
    """Gaussian (Buerger) reduction of the direct basis -> shortest-vector triclinic cell for reporting.
    (Not a unique Niggli cell -- edge lengths are canonical, the angle set is one valid representative.)"""
    V = _cell_to_vectors(cell)
    for _ in range(100):
        changed = False
        for i in range(3):
            for j in range(3):
                if i != j:
                    m = round((V[i] @ V[j]) / (V[j] @ V[j]))
                    if m != 0:
                        V[i] = V[i] - m * V[j]; changed = True
        V = V[np.argsort((V * V).sum(1))]
        if not changed:
            break
    return _vectors_to_cell(V)[:6]


def _index_anneal(Qobs, centering, amin, amax, ang_range, imax, reltol, max_unindexed,
                  nchain, nstep, restarts, seed, T0=0.06, Tend=5e-4, se0=0.7, sa0=5.0):
    """Recover a triclinic cell by simulated annealing over the 6 cell parameters (host numpy).

    Each of ``nchain`` Metropolis chains walks the cell space with geometric cooling; the cost is the de
    Wolff M20 objective -- mean relative line residual times the number of predicted lines up to Q20 -- so
    the trivial "make the lattice denser" minimum is penalised (that was the failure of a plain residual
    cost / random grid).  ``restarts`` independent seeds are pooled.  Pseudo-symmetric cells (e.g. equal
    edges) are a known hard regime and may not solve.  Returns [(free-metric-comps, n_indexed), ...].
    """
    Q = np.asarray(Qobs, np.float64); nobs = Q.size; Q20 = Q.max()
    Hv = np.asarray(_gen_hkl("triclinic", imax, centering, np)); Dv = np.asarray(_design("triclinic", Hv))
    need = nobs - max_unindexed

    def cost(a, b, c, al, be, ga):
        Qc = _cells_to_free_tri(a, b, c, al, be, ga) @ Dv.T           # (nchain, Ncand)
        nd = np.abs(Qc[:, None, :] - Q[None, :, None]).min(2)        # nearest residual per line
        resid = (nd / Q[None, :]).mean(1)
        NN = ((Qc > 0) & (Qc <= Q20 * (1 + 1e-6))).sum(1)           # predicted-line-count density penalty
        # (resid + floor)*NN, not resid*NN: the floor keeps the density term active as resid->0, so a
        # supercell (same residual, ~2x the predicted lines) is penalised and the reduced cell wins.
        return (resid + 1e-3) * np.maximum(NN, nobs)

    pool = []
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        a = rng.uniform(amin, amax, nchain); b = rng.uniform(amin, amax, nchain)
        c = rng.uniform(amin, amax, nchain)
        al = rng.uniform(*ang_range, nchain); be = rng.uniform(*ang_range, nchain)
        ga = rng.uniform(*ang_range, nchain)
        c0 = cost(a, b, c, al, be, ga)
        for step in range(nstep):
            T = T0 * (Tend / T0) ** (step / nstep); f = (T / T0) ** 0.5
            na = np.clip(a + rng.normal(0, se0 * f, nchain), amin, amax)
            nb = np.clip(b + rng.normal(0, se0 * f, nchain), amin, amax)
            nc = np.clip(c + rng.normal(0, se0 * f, nchain), amin, amax)
            nal = np.clip(al + rng.normal(0, sa0 * f, nchain), *ang_range)
            nbe = np.clip(be + rng.normal(0, sa0 * f, nchain), *ang_range)
            nga = np.clip(ga + rng.normal(0, sa0 * f, nchain), *ang_range)
            cn = cost(na, nb, nc, nal, nbe, nga)
            acc = (cn < c0) | (rng.random(nchain) < np.exp(np.clip(-(cn - c0) / max(T, 1e-9), -700, 0)))
            a = np.where(acc, na, a); b = np.where(acc, nb, b); c = np.where(acc, nc, c)
            al = np.where(acc, nal, al); be = np.where(acc, nbe, be); ga = np.where(acc, nga, ga)
            c0 = np.where(acc, cn, c0)
        order = np.argsort(c0)[:40]
        comps = _cells_to_free_tri(a[order], b[order], c[order], al[order], be[order], ga[order])
        pool.extend(comps.tolist())
    # refine the best distinct candidates, keep those that index enough lines
    out = []; seen = set(); rngp = np.random.default_rng(seed + 997)
    for comp in pool:
        ref = _refine_params(Qobs, "triclinic", centering, tuple(comp), imax, reltol)
        if ref is None:
            continue
        # polish: a few tiny-perturbation restarts, keep the best M20 -- refinement can settle ~0.2% off in
        # the dense triclinic reflection set (wrong nearest-hkl), which tanks M20 and lets a supercell win;
        # a restart from a small perturbation reliably drops into the exact basin.
        bestM = m20_fom(Qobs, "triclinic", ref, centering, imax)[0]
        for _ in range(6):
            pert = tuple(x * (1 + rngp.normal(0, 0.004)) for x in ref)
            r2 = _refine_params(Qobs, "triclinic", centering, pert, imax, reltol)
            if r2 is not None:
                m2 = m20_fom(Qobs, "triclinic", r2, centering, imax)[0]
                if m2 > bestM:
                    ref, bestM = r2, m2
        key = tuple(np.round(np.asarray(ref) * 1e4).astype(np.int64))
        if key in seen:
            continue
        seen.add(key)
        D = np.asarray(_design("triclinic", np.asarray(m20_fom(Qobs, "triclinic", ref, centering, imax)[2])))
        nidx = int(np.sum(np.abs(D @ np.asarray(ref) - Q) <= reltol * Q))
        if nidx >= need:
            out.append((ref, nidx))
    return out


# --------------------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------------------
_HIGHSYM = ("cubic", "tetragonal", "hexagonal", "orthorhombic")
# Engine A (seed-and-verify) also covers monoclinic (4 linear metric params, signed reflections).
_SEEDABLE = _HIGHSYM + ("monoclinic",)


def _system_centerings(system, centerings):
    """Restrict the requested centerings to those meaningful for a crystal system."""
    if system == "triclinic":
        return ("P",)
    if system == "monoclinic":
        return tuple(c for c in centerings if c in ("P", "C", "I", "A"))
    if system in _SIGNED:                                      # hexagonal/trigonal: P and R only
        return tuple(c for c in centerings if c in ("P", "R"))
    return tuple(c for c in centerings if c != "R")            # cubic/tet/ortho: no R


def index_powder(pos, units="q", wavelength=None,
                 systems=_HIGHSYM, centerings=("P", "I", "F", "C", "R"),
                 amin=2.0, amax=60.0, reltol=0.01, max_unindexed=0,
                 seed_smax=6, imax=6, min_m20=3.0, ntop=10, xp=None, seed_lines=None,
                 tri_nchain=3000, tri_nstep=200, tri_restarts=3, tri_seed=0, tri_angle=(72.0, 108.0),
                 tri_imax=5):
    """Autoindex a powder pattern -> ranked list of ``Solution`` (best M20 first).

    pos       : observed ring positions (see ``units``).
    units     : "q" (physics q, default), "d", "twotheta" (+wavelength) or "inv_d2".
    systems   : crystal systems to try -- cubic/tetragonal/hexagonal/orthorhombic/monoclinic via
                seed-and-verify; "triclinic" via simulated annealing (``tri_*`` knobs; stochastic, slower,
                and pseudo-symmetric cells are a known hard case).
    centerings: lattice centerings to try (subset of P,I,F,A,B,C,R). R is only meaningful for hex/trigonal.
    amin,amax : allowed cell-edge range (A).
    reltol    : a line is "indexed" if |Q_calc-Q_obs| <= reltol*Q_obs (Q=1/d^2; ~2x the d-spacing tol).
    max_unindexed : how many observed lines a solution may leave unindexed (impurity/error).
    seed_smax : max h^2+k^2+l^2 for the reflections tried on the seed lines.
    imax      : index-box half-width for verification/M20 (raise for very large cells).
    min_m20   : discard solutions below this de Wolff M20.
    ntop      : keep at most this many (deduped) solutions.
    xp        : force a backend (numpy/cupy); default = numpy (pass cupy arrays / cupy module for GPU).
    """
    Qobs = to_inv_d2(pos, units=units, wavelength=wavelength)
    if xp is None:
        xp = np
    cands = []                                                 # (system, centering, params, nindexed)
    for system in systems:
        if system == "triclinic":
            # 6 free metric params: seed-and-verify combinatorics explode and random grid+refine does not
            # converge, so triclinic uses a directed search -- simulated annealing on the M20 objective.
            for params, nidx in _index_anneal(Qobs, "P", amin, amax, tri_angle, tri_imax, reltol,
                                              max_unindexed, tri_nchain, tri_nstep, tri_restarts, tri_seed):
                cands.append((system, "P", params, nidx))
            continue
        if system not in _SEEDABLE:
            raise ValueError("unknown crystal system: %r" % system)
        for cen in _system_centerings(system, centerings):
            for params, nidx in _seed_and_verify(Qobs, system, cen, xp, amin, amax, reltol,
                                                  seed_smax, imax, max_unindexed, seed_lines):
                cands.append((system, cen, params, nidx))
    # score each candidate exactly (M20) on the host, refine, dedup
    sols = []
    for system, cen, params, nidx in cands:
        params = _refine_params(Qobs, system, cen, params, imax, reltol)
        if params is None:
            continue
        a, b, c, al, be, ga, vol = _params_to_cell(system, params)
        if system == "monoclinic":
            a, b, c, al, be, ga = _reduce_monoclinic(a, b, c, al, be, ga)
        elif system == "triclinic":
            a, b, c, al, be, ga = _reduce_cell_3d((a, b, c, al, be, ga))
        if not (amin <= min(a, b, c) and max(a, b, c) <= amax):
            continue
        M, ncalc, hkl, dqm = m20_fom(Qobs, system, params, cen, imax)
        if not math.isfinite(M) or M < min_m20:
            continue
        sols.append(Solution(system, cen, a, b, c, al, be, ga, vol, M, int(nidx), len(Qobs),
                             tuple(params), hkl))
    return _dedup(sols)[:ntop]


def _A_from_params(system, p):
    """Free metric params -> full 3x3 reciprocal metric A = G* (for validity checks / low-symmetry)."""
    p = np.asarray(p, np.float64)
    if system == "monoclinic":
        return np.array([[p[0], 0, p[3]], [0, p[1], 0], [p[3], 0, p[2]]], float)
    if system == "triclinic":
        return np.array([[p[0], p[5], p[4]], [p[5], p[1], p[3]], [p[4], p[3], p[2]]], float)
    return None


def _params_valid(system, p):
    """A trial cell is valid iff the diagonal metric terms are positive and (low-sym) A is pos-definite."""
    p = np.asarray(p, np.float64)
    if not np.all(np.isfinite(p)):
        return False
    ndiag = 3 if system in ("monoclinic", "triclinic") else p.size
    if np.any(p[:ndiag] <= 0):
        return False
    A = _A_from_params(system, p)
    if A is not None and np.linalg.eigvalsh(A).min() <= 1e-12:
        return False
    return True


def _refine_params(Qobs, system, centering, params, imax, reltol):
    """Least-squares refine the metric params on the current index assignment (a couple of passes)."""
    Qobs = np.asarray(Qobs, np.float64)
    p = np.asarray(params, np.float64)
    for _ in range(4):
        _, _, hkl, _ = m20_fom(Qobs, system, p, centering, imax)
        D = np.asarray(_design(system, np.asarray(hkl, np.int64)), np.float64)   # (nobs, m)
        # keep only lines actually indexed within tol (robust to impurity lines)
        good = np.abs(D @ p - Qobs) <= reltol * Qobs
        if good.sum() < p.size:
            return None
        pnew, *_ = np.linalg.lstsq(D[good], Qobs[good], rcond=None)
        if not _params_valid(system, pnew):
            return None
        done = np.allclose(pnew, p, rtol=1e-8)
        p = pnew
        if done:
            break
    return tuple(float(x) for x in p)


_MULT = {"P": 1, "I": 2, "F": 4, "A": 2, "B": 2, "C": 2, "R": 3}
_SYMRANK = {"cubic": 6, "hexagonal": 5, "trigonal": 4, "tetragonal": 3,
            "orthorhombic": 2, "monoclinic": 1, "triclinic": 0}


def _prim_volume(sol):
    return sol.volume / _MULT.get(sol.centering, 1)


def _rank_key(s):
    # best M20 first; ties (perfect synthetic fits pin the M20 floor) -> highest symmetry, then larger cell
    return (-round(s.M20, 4), -_SYMRANK[s.system], -s.volume)


def _dedup(sols, vrtol=8e-3):
    """Collapse solutions that describe the same lattice, keeping the highest-symmetry representative.

    A tetragonal lattice also indexes as a C-centred orthorhombic supercell, etc.; such settings share the
    same *primitive* cell volume and index the same lines.  We merge solutions with equal primitive volume
    and keep the highest crystal-symmetry member (ties -> smaller/conventional cell, then higher M20).
    (Distinct lattices -- even with commensurate volumes -- are kept separate; the ``_rank_key`` symmetry
    tie-break puts the conventional high-symmetry cell on top when several pseudo-cells fit equally.)
    """
    kept = []
    for s in sorted(sols, key=_rank_key):
        pv = _prim_volume(s)
        rep = None
        for i, k in enumerate(kept):
            if abs(pv - _prim_volume(k)) <= vrtol * max(pv, 1.0) and abs(s.nindexed - k.nindexed) <= 1:
                rep = i; break
        if rep is None:
            kept.append(s); continue
        k = kept[rep]
        if (_SYMRANK[s.system], -s.volume, s.M20) > (_SYMRANK[k.system], -k.volume, k.M20):
            kept[rep] = s
    kept.sort(key=_rank_key)
    # supercell removal: if a kept cell's volume is ~an integer multiple of a smaller same-system cell that
    # indexes as many lines, it's a supercell of the true lattice -> drop it (both index the observed lines).
    final = []
    for s in sorted(kept, key=lambda x: x.volume):
        sup = False
        for k in final:
            if k.system == s.system and s.nindexed <= k.nindexed:
                r = s.volume / max(k.volume, 1e-9)
                if abs(r - round(r)) <= 0.03 * round(r) and 2 <= round(r) <= 8:
                    sup = True; break
        if not sup:
            final.append(s)
    final.sort(key=_rank_key)
    return final


# --------------------------------------------------------------------------------------------------
# Peak picker: radial profile -> ring positions (bridge from radial.py)
# --------------------------------------------------------------------------------------------------
def peaks_from_profile(q, I, min_prominence=None, min_distance=1, max_peaks=None):
    """Pick ring positions from a 1-D profile ``I(q)`` via prominence (scipy.signal.find_peaks).

    Returns the ``q`` positions (same units as the input ``q``), strongest-first when ``max_peaks`` is set,
    else in ascending q.  ``min_prominence`` defaults to a robust fraction of the signal spread.
    """
    from scipy.signal import find_peaks
    q = np.asarray(q, np.float64); I = np.asarray(I, np.float64)
    finite = np.isfinite(I)
    q, I = q[finite], I[finite]
    if min_prominence is None:
        med = np.median(I)
        mad = np.median(np.abs(I - med)) + 1e-12
        min_prominence = 3.0 * 1.4826 * mad
    idx, props = find_peaks(I, prominence=min_prominence, distance=min_distance)
    if max_peaks is not None and idx.size > max_peaks:
        idx = idx[np.argsort(props["prominences"])[::-1][:max_peaks]]
    return np.sort(q[idx])
