"""Joint multi-shot cell recovery from the pooled, rotation-invariant length spectrum.

For shot s, every spot satisfies |q^s_i|^2 = h^s_i^T M h^s_i (rotation drops out), so
pooling |q|^2 over MANY shots gives a powder-like spectrum: the discrete set of allowed
reflection squared-lengths {h^T M h}. The shared metric tensor M is fit to that
spectrum with NO per-shot indexing -- so it works at extreme sparsity (n<=12) where
single-shot bootstrap stalls, and it's cheap (no FFT/triplet search per shot).

v1 fits an orthorhombic (diagonal M -> A*,B*,C*) cell; the off-diagonal terms for the
general triclinic metric come from pooled within-shot dot products q_i.q_j (TODO).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d


def pooled_q2(shots):
    """All |q|^2 over all shots (the rotation-invariant powder sample)."""
    return np.concatenate([np.sum(np.asarray(s.g, float) ** 2, axis=1) for s in shots])


def spectrum_peaks(q2, qmax, nbins=3000, smooth=2.0, rel_thresh=0.1):
    """Peaks of the pooled |q|^2 histogram = allowed reflection squared-lengths."""
    hist, edges = np.histogram(q2, bins=nbins, range=(0, qmax ** 2))
    centers = 0.5 * (edges[:-1] + edges[1:])
    h = gaussian_filter1d(hist.astype(float), smooth)
    mx = maximum_filter1d(h, 5)
    is_peak = (h == mx) & (h > rel_thresh * h.max()) & (centers > 1e-6)
    return centers[is_peak], h[is_peak]


def fit_orthorhombic(peaks, n_gen=10, K=15, hmax=6, rtol=0.025, w=0.7):
    """Fit (A*,B*,C*) = (|a*|^2,|b*|^2,|c*|^2) so {A*h^2+B*k^2+C*l^2} reproduces peaks.

    Powder-style figure of merit on the lowest K (well-separated) lines: score =
    (#low lines matched) - w * (#predicted low lines with NO observed peak). The
    over-prediction penalty is what rules out too-large cells (and the coincident-axis
    cubic case that a "3 smallest peaks" heuristic gets wrong). Search generator
    triplets from the smallest n_gen peaks.
    """
    peaks = np.sort(np.asarray(peaks, float))
    if len(peaks) < 4:
        return None
    low = peaks[:K]
    pmax = low[-1]
    cand = peaks[:n_gen]
    hs = np.arange(0, hmax + 1)
    HKL = np.array(np.meshgrid(hs, hs, hs, indexing="ij")).reshape(3, -1).T
    HKL = HKL[np.any(HKL != 0, axis=1)]
    h2, k2, l2 = (HKL ** 2).T

    best = None  # (score, (A,B,C))
    for ia in range(len(cand)):
        for ib in range(ia, len(cand)):
            for ic in range(ib, len(cand)):
                A, B, C = cand[ia], cand[ib], cand[ic]
                pred = np.unique((A * h2 + B * k2 + C * l2))
                pred = pred[pred <= pmax * 1.02]
                if len(pred) == 0:
                    continue
                matched = int((np.abs(low[:, None] - pred[None, :]).min(axis=1)
                               < rtol * low).sum())
                extra = int((np.abs(pred[:, None] - peaks[None, :]).min(axis=1)
                             >= rtol * pred).sum())          # over-predicted lines
                score = matched - w * extra
                if best is None or score > best[0]:
                    best = (score, (A, B, C))
    return np.array(best[1]) if best else None


def cell_from_diag(ABC):
    """Orthorhombic cell edges (a,b,c) from (A*,B*,C*) = (1/a^2, 1/b^2, 1/c^2)."""
    return np.sort(1.0 / np.sqrt(np.asarray(ABC, float)))[::-1]


def fit_orthorhombic_topk(peaks, topk=6, n_gen=10, K=15, hmax=6, rtol=0.025, w=0.7):
    """Like fit_orthorhombic but return the top-k candidate (A*,B*,C*) by powder FOM."""
    peaks = np.sort(np.asarray(peaks, float))
    if len(peaks) < 4:
        return []
    low, pmax, cand = peaks[:K], peaks[:K][-1], peaks[:n_gen]
    hs = np.arange(0, hmax + 1)
    HKL = np.array(np.meshgrid(hs, hs, hs, indexing="ij")).reshape(3, -1).T
    HKL = HKL[np.any(HKL != 0, axis=1)]
    h2, k2, l2 = (HKL ** 2).T
    scored = []
    for ia in range(len(cand)):
        for ib in range(ia, len(cand)):
            for ic in range(ib, len(cand)):
                A, B, C = cand[ia], cand[ib], cand[ic]
                pred = np.unique(A * h2 + B * k2 + C * l2)
                pred = pred[pred <= pmax * 1.02]
                if len(pred) == 0:
                    continue
                matched = int((np.abs(low[:, None] - pred[None, :]).min(axis=1)
                               < rtol * low).sum())
                extra = int((np.abs(pred[:, None] - peaks[None, :]).min(axis=1)
                             >= rtol * pred).sum())
                scored.append((matched - w * extra, (A, B, C)))
    scored.sort(key=lambda t: -t[0])
    out, seen = [], []
    for _, abc in scored:                       # dedupe near-identical triplets
        if not any(np.allclose(abc, s, rtol=0.01) for s in seen):
            out.append(np.array(abc)); seen.append(abc)
        if len(out) >= topk:
            break
    return out


def short_pair_triples(shots, qmax, cut=0.4):
    """Pooled within-shot (|qi|^2, |qj|^2, qi.qj) for pairs of LOW-order spots."""
    lim2 = (cut * qmax) ** 2
    rows = []
    for s in shots:
        g = np.asarray(s.g, float)
        l2 = np.sum(g ** 2, axis=1)
        sh = np.where(l2 <= lim2)[0]
        for a in range(len(sh)):
            for b in range(a + 1, len(sh)):
                i, j = sh[a], sh[b]
                rows.append((l2[i], l2[j], g[i] @ g[j]))
                rows.append((l2[j], l2[i], g[i] @ g[j]))
    return np.array(rows) if rows else np.zeros((0, 3))


def _pred_triples(ABC, qmax, cut=0.4, hmax=8):
    """Predicted (|h|^2_M, |h'|^2_M, h.M.h') for low-order integer vectors of cell ABC."""
    M = np.diag(np.asarray(ABC, float))
    r = np.arange(-hmax, hmax + 1)
    H = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, axis=1)]
    l2 = np.einsum("ni,ij,nj->n", H, M, H)
    H = H[l2 <= (cut * qmax) ** 2]
    l2 = np.einsum("ni,ij,nj->n", H, M, H)
    G = H @ M @ H.T
    li = np.repeat(l2, len(l2))
    lj = np.tile(l2, len(l2))
    return np.column_stack([li, lj, G.ravel()])


def fit_joint(shots, qmax, topk=6, nbins=6000, smooth=1.5, tol_frac=0.015):
    """Powder fit -> angular re-rank. The (|qi|,|qj|,qi.qj) triples break the powder
    geometric ambiguity (e.g. cubic a vs (a,2a,2a)): the twin can't make equal-shortest-
    length pairs at 90deg (dot 0). Returns the cell (a,b,c)."""
    from scipy.spatial import cKDTree
    pk, _ = spectrum_peaks(pooled_q2(shots), qmax, nbins=nbins, smooth=smooth)
    cands = fit_orthorhombic_topk(pk, topk=topk)
    if not cands:
        return None
    obs = short_pair_triples(shots, qmax)
    if len(obs) < 10:
        return cell_from_diag(cands[0])
    tol = tol_frac * qmax ** 2
    best = None
    for ABC in cands:
        pred = _pred_triples(ABC, qmax)
        d = cKDTree(pred).query(obs)[0]
        score = float((d < tol).mean())          # fraction of observed triples explained
        if best is None or score > best[0]:
            best = (score, ABC)
    return cell_from_diag(best[1])
