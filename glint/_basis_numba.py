"""Numba-JIT'd basis search -- the hot loop of the indexer.

search_basis was thousands of tiny numpy calls per shot (C(topk,3) triplets x 5
refine iterations x small matmuls/inv/round), so it was dispatch-overhead-bound and
became ~90% of the GPU-fronted indexer's time. Here the whole triplet enumeration +
refine + parsimony selection is ONE nopython function with hand-rolled 3x3 algebra
(closed-form det/inverse beat np.linalg.inv's LAPACK call for a 3x3) and explicit
spot loops (no temporaries). prange parallelizes the outer triplet over cores.

Math mirrors index.refine exactly: C = M^{-T} (reciprocal basis), h_s = round(M^T g_s),
predicted g = C h, inliers within tol_abs, then C <- (G H^T)(H H^T)^{-1} refit, 5x.
Selection = parsimony (smallest |det M| among bases indexing >= min_inlier_frac).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _det3(A):
    return (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
            - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
            + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))


@njit(cache=True)
def _inv3(A):
    d = _det3(A)
    B = np.empty((3, 3))
    B[0, 0] = (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) / d
    B[0, 1] = (A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]) / d
    B[0, 2] = (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) / d
    B[1, 0] = (A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]) / d
    B[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) / d
    B[1, 2] = (A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]) / d
    B[2, 0] = (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]) / d
    B[2, 1] = (A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]) / d
    B[2, 2] = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) / d
    return B, d


@njit(cache=True)
def _refine(M0, g, tol_abs, n_iter):
    """Return (M, ninl, |det M|, ok). C=M^{-T}; alternate assign<->LSQ refit."""
    Minv, dM = _inv3(M0)
    if dM == 0.0:
        return M0, 0, 1e30, False
    C = Minv.T.copy()                                   # C = M^{-T}
    n = g.shape[0]
    M = M0.copy()
    ninl = 0
    for _ in range(n_iter):
        Cinv, dC = _inv3(C)
        if dC == 0.0:
            return M, 0, 1e30, False
        M = Cinv.T                                      # real basis, = inv(C).T
        HHt = np.zeros((3, 3))
        GHt = np.zeros((3, 3))
        ninl = 0
        for s in range(n):
            g0, g1, g2 = g[s, 0], g[s, 1], g[s, 2]
            h0 = round(g0 * M[0, 0] + g1 * M[1, 0] + g2 * M[2, 0])   # h = round(M^T g)
            h1 = round(g0 * M[0, 1] + g1 * M[1, 1] + g2 * M[2, 1])
            h2 = round(g0 * M[0, 2] + g1 * M[1, 2] + g2 * M[2, 2])
            p0 = C[0, 0] * h0 + C[0, 1] * h1 + C[0, 2] * h2          # pred = C h
            p1 = C[1, 0] * h0 + C[1, 1] * h1 + C[1, 2] * h2
            p2 = C[2, 0] * h0 + C[2, 1] * h1 + C[2, 2] * h2
            r0, r1, r2 = g0 - p0, g1 - p1, g2 - p2
            if r0 * r0 + r1 * r1 + r2 * r2 < tol_abs * tol_abs:
                ninl += 1
                HHt[0, 0] += h0 * h0; HHt[0, 1] += h0 * h1; HHt[0, 2] += h0 * h2
                HHt[1, 0] += h1 * h0; HHt[1, 1] += h1 * h1; HHt[1, 2] += h1 * h2
                HHt[2, 0] += h2 * h0; HHt[2, 1] += h2 * h1; HHt[2, 2] += h2 * h2
                GHt[0, 0] += g0 * h0; GHt[0, 1] += g0 * h1; GHt[0, 2] += g0 * h2
                GHt[1, 0] += g1 * h0; GHt[1, 1] += g1 * h1; GHt[1, 2] += g1 * h2
                GHt[2, 0] += g2 * h0; GHt[2, 1] += g2 * h1; GHt[2, 2] += g2 * h2
        if ninl < 3 or abs(_det3(HHt)) < 1e-9:
            break
        HHt_inv, _ = _inv3(HHt)
        Cn = GHt @ HHt_inv
        if abs(_det3(Cn)) < 1e-12:
            break
        C = Cn
    Cinv, dC = _inv3(C)
    if dC == 0.0:
        return M, ninl, 1e30, False
    M = Cinv.T
    return M, ninl, abs(_det3(M)), (ninl >= 3)


@njit(cache=True, parallel=True)
def search_real_njit(cands, g, tol_abs, eps, min_inlier_frac, n_iter):
    """Parsimony-best basis over all column-triplets of `cands`. Returns
    (best_M (3,3), best_det, found). prange over the outer candidate index; each
    thread keeps its own best, reduced after."""
    nc = cands.shape[0]
    n = g.shape[0]
    thresh = min_inlier_frac * n
    nrm = np.empty(nc)
    for i in range(nc):
        nrm[i] = np.sqrt(cands[i, 0]**2 + cands[i, 1]**2 + cands[i, 2]**2)

    best_det = np.full(nc, 1e30)
    best_M = np.zeros((nc, 3, 3))
    for i in prange(nc):
        for j in range(i + 1, nc):
            for k in range(j + 1, nc):
                M0 = np.empty((3, 3))
                for r in range(3):
                    M0[r, 0] = cands[i, r]; M0[r, 1] = cands[j, r]; M0[r, 2] = cands[k, r]
                scale = nrm[i] * nrm[j] * nrm[k]
                if scale <= 0.0 or abs(_det3(M0)) < eps * scale:
                    continue
                M, ninl, detM, ok = _refine(M0, g, tol_abs, n_iter)
                if ok and ninl >= thresh and detM < best_det[i]:
                    best_det[i] = detM
                    best_M[i] = M

    bi = 0
    for i in range(1, nc):
        if best_det[i] < best_det[bi]:
            bi = i
    return best_M[bi], best_det[bi], best_det[bi] < 1e30
