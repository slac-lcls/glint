"""End-to-end indexer: peak cloud -> (M, hkl).

Pipeline:
    g_i  -->  fft_volume  -->  find_peaks  -->  search_basis  -->  refine
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .lattice import buerger_reduce
from .localize import localize_peaks
from .peakfind import find_peaks_batch, find_peaks_batch_coarse, find_peaks_classical
from .seed import difference_vector_seeds, projection_axis_seeds
from .transform import estimate_grid_n, fft_volume, fft_volume_batch

try:
    from ._basis_numba import search_real_njit
    _HAVE_NUMBA = True
except Exception:                                   # numba absent -> pure-Python path
    _HAVE_NUMBA = False


@dataclass
class IndexResult:
    M: np.ndarray | None          # (3,3) recovered real basis (cols), or None
    C: np.ndarray | None          # (3,3) recovered reciprocal basis = M^{-T}
    hkl: np.ndarray | None        # (n,3) assigned integer indices
    inliers: np.ndarray | None    # (n,) bool, spots explained within tol
    n_indexed: int
    candidates: np.ndarray | None = None  # raw peak vectors (for inspection)


def _triplets(cands, eps):
    """Yield non-degenerate column-triplets (as 3x3 matrices) from candidate vecs."""
    for i, j, k in combinations(range(len(cands)), 3):
        cols = [cands[i], cands[j], cands[k]]
        M = np.column_stack(cols)
        scale = np.prod([np.linalg.norm(c) for c in cols])
        if scale > 0 and abs(np.linalg.det(M)) >= eps * scale:
            yield M


def _reconstruct(M, g, tol_abs):
    """(M, C, hkl, inliers) for a chosen real basis M -- one cheap assignment."""
    C = np.linalg.inv(M).T
    hkl = np.rint(g @ M)
    inliers = np.linalg.norm(g - hkl @ C.T, axis=1) < tol_abs
    return M, C, hkl, inliers


def search_basis(real_cands, recip_seeds, g, tol_abs, n_real=8, n_seed=8,
                 eps=0.1, min_inlier_frac=0.5, select="excess", gate_frac=0.4):
    """Robust basis selection from two independent candidate sources.

    Real-space |F(x)| peaks (real_cands) give basis hypotheses M0 directly;
    difference-vector seeds (recip_seeds, reciprocal) give C0 -> M0 = inv(C0).T.
    Each hypothesis is refined under the ABSOLUTE tolerance; the surviving cells are
    ranked by `select`:

      "excess" (default): max EXCESS COVERAGE = (fraction of spots indexed at a tight
        tol) - (chance fraction p = min(1, V * 4/3 pi tol^3) that a random spot lands
        within tol of a lattice of that density). This is a parameter-free likelihood
        ratio: it penalizes a fake SMALL cell (low coverage at tight tol) AND an
        over-fit LARGE cell (its dense reciprocal lattice makes p -> 1, so its high
        raw coverage is "free") simultaneously. Each cell is Buerger-reduced first so
        non-primitive 'true axes + a diagonal' variants collapse onto the true cell.
        On real cxidb lysozyme this lifts the blind rate 10% -> 28% over parsimony.

      "parsimony": among cells indexing >= min_inlier_frac of spots, the SMALLEST
        volume. Cheap (JIT'd search_real_njit when no recip seeds) but picks a fake
        sub-cell whenever one exists -- the historical rate bottleneck.

    Fail safe: returns None ("could not index") rather than a wrong lattice when no
    cell clears the gate.
    """
    if select == "parsimony" and _HAVE_NUMBA and len(recip_seeds) < 3:
        cands = np.ascontiguousarray(real_cands[:n_real], dtype=np.float64)
        if len(cands) < 3:
            return None
        g = np.ascontiguousarray(g, dtype=np.float64)
        M, _det, found = search_real_njit(cands, g, float(tol_abs), float(eps),
                                          float(min_inlier_frac), 5)
        if not found:
            return None
        return _reconstruct(M, g, tol_abs)
    return _search_basis_py(real_cands, recip_seeds, g, tol_abs, n_real, n_seed,
                            eps, min_inlier_frac, select, gate_frac)


def _search_basis_py(real_cands, recip_seeds, g, tol_abs, n_real=8, n_seed=8,
                     eps=0.1, min_inlier_frac=0.5, select="excess", gate_frac=0.4):
    """Pure-Python basis search; `select` in {"excess","parsimony"}."""
    hypotheses = list(_triplets(real_cands[:n_real], eps))
    hypotheses += [np.linalg.inv(C0).T for C0 in _triplets(recip_seeds[:n_seed], eps)]
    n_obs = len(g)

    if select == "excess":
        # Tight scoring tol (0.6x the fit tol) is where chance-coverage discriminates;
        # at the loose fit tol every big cell scores ~1 (p -> 1) and the signal dies.
        stol = 0.6 * tol_abs
        cap = (4.0 / 3.0) * np.pi * stol ** 3
        best = None; best_s = None
        for M0 in hypotheses:
            M, C, hkl, inliers = refine(g, M0, tol_abs)
            if C is None or inliers is None or inliers.sum() < 3:
                continue
            Mr = buerger_reduce(M)
            Cr = np.linalg.inv(Mr).T
            resid = np.linalg.norm(g - np.rint(g @ Mr) @ Cr.T, axis=1)
            cov = float((resid < stol).mean())
            if cov < gate_frac:
                continue
            score = cov - min(1.0, abs(np.linalg.det(Mr)) * cap)
            if best_s is None or score > best_s:
                best_s = score
                inl = resid < tol_abs
                best = (Mr, Cr, np.rint(g @ Mr), inl)
        return best

    accepted = []  # (real_volume, M, C, hkl, inliers)  -- parsimony
    for M0 in hypotheses:
        M, C, hkl, inliers = refine(g, M0, tol_abs)
        if C is None or inliers.sum() < min_inlier_frac * n_obs:
            continue
        accepted.append((abs(np.linalg.det(M)), M, C, hkl, inliers))
    if not accepted:
        return None
    accepted.sort(key=lambda r: r[0])
    _, M, C, hkl, inliers = accepted[0]
    return M, C, hkl, inliers


def refine(g, M, tol_abs, n_iter=5):
    """Alternate hkl assignment <-> least-squares basis fit; return (M, C, hkl, inliers).

    tol_abs is an absolute inlier tolerance in reciprocal units (1/A). The LSQ refit
    uses the normal equations C = (G H^T)(H H^T)^{-1} -- 3x3 solves only, no SVD
    (np.linalg.matrix_rank / pinv were the BLAS-thrash hot spots).
    """
    C = np.linalg.inv(M).T
    inliers = np.ones(len(g), bool)
    hkl = None
    for _ in range(n_iter):
        hkl = np.rint(g @ np.linalg.inv(C).T)          # h = round(C^{-1} g)
        resid = np.linalg.norm(g - hkl @ C.T, axis=1)
        inliers = resid < tol_abs
        if inliers.sum() < 3:
            break
        G, H = g[inliers].T, hkl[inliers].T            # 3xm
        HHt = H @ H.T                                  # 3x3
        if abs(np.linalg.det(HHt)) < 1e-9:             # rank-deficient assignment
            break
        C_new = (G @ H.T) @ np.linalg.inv(HHt)
        if abs(np.linalg.det(C_new)) < 1e-12:
            break
        C = C_new
        M = np.linalg.inv(C).T
    return M, C, hkl, inliers


def _index_at(g, qmax, n, tol_abs, peakfinder, min_len, topk, localize,
              seed_diff, min_inlier_frac, gpu=False, seed_proj=True,
              proj_max_peaks=350):
    """One indexing attempt at a fixed grid size n. Returns IndexResult.

    seed_proj=True default: enrich the candidate pool with projection-slice
    (MOSFLM/DPS) direct-axis seeds on sparse/moderate frames (P <= proj_max_peaks).
    The FFT |F(x)| peaks alone miss the long lyso axes on sparse frames (the
    candidate CEILING, not selection, then caps the rate); the periodicity readout
    recovers them, lifting real-cxidb solved 16% -> ~28% with the excess selector.

    seed_diff=False default: difference-vector (Patterson) seeding proved neutral on
    synthetic data. localize=False default: refine()'s LSQ fit removes grid quant.
    """
    vol, x = fft_volume(g, qmax, n=n, gpu=gpu)
    vecs, amps = peakfinder(vol, x, g, qmax, min_len=min_len)
    real_cands = np.asarray(vecs[:topk], float).reshape(-1, 3)
    if localize and len(real_cands):
        real_cands = localize_peaks(g, real_cands, min_len=min_len)
    if seed_proj and 3 <= len(g) <= proj_max_peaks:
        proj = projection_axis_seeds(g, topk=topk)
        if len(proj):
            real_cands = np.vstack([real_cands, proj]) if len(real_cands) else proj
    recip_seeds = difference_vector_seeds(g, qmax) if seed_diff else np.zeros((0, 3))
    if len(real_cands) < 3 and len(recip_seeds) < 3:
        return IndexResult(None, None, None, None, 0, vecs)
    out = search_basis(real_cands, recip_seeds, g, tol_abs, n_real=len(real_cands),
                       min_inlier_frac=min_inlier_frac)
    if out is None:
        return IndexResult(None, None, None, None, 0, vecs)
    M, C, hkl, inliers = out
    return IndexResult(M, C, hkl, inliers, int(inliers.sum()), vecs)


def detect_batch(gs, qmaxs, n="auto", min_len=3.0, gpu=True, dtype="float32",
                 n_max=256, coarse=True):
    """The batched GPU detect stage shared by the blind and known-cell indexers:
    deposit + cuFFT + peakfind over the (B,n,n,n) stack. Returns (peaks, n) where
    peaks is a per-shot list of (vecs, amps). All shots share one grid n (batched
    cuFFT needs a common shape); n="auto" sizes it to the largest cell in the batch.

    Coarse-to-fine peakfind (max-pool -> top-K on (n/4)^3 -> fine refine) is ~9x
    faster on the GPU than the full-n^3 argpartition, with identical accuracy
    (verified A100). Falls back to dense automatically when n%pool != 0.
    """
    if n == "auto":
        n = min(n_max, max(estimate_grid_n(g, q) for g, q in zip(gs, qmaxs)))
    vols, xs = fft_volume_batch(gs, qmaxs, n=n, gpu=gpu, dtype=dtype)
    pf = find_peaks_batch_coarse if coarse else find_peaks_batch
    return pf(vols, xs, min_len=min_len), n


def index_shots_batch(gs, qmaxs, n="auto", min_len=3.0, topk=12, tol_frac=0.02,
                      min_inlier_frac=0.7, gpu=True, dtype="float32", n_max=256,
                      coarse=True):
    """Index B shots together: ONE batched GPU detect stage, then the small
    per-shot basis search on CPU. Returns list[IndexResult].

    The detect stage runs entirely on the GPU at batched rate; only
    search_basis/refine -- tiny 3x3 linear algebra -- stay per-shot on the CPU.
    No grid escalation here (that's a per-shot fallback); a shot that needs a bigger
    grid is re-run with index_shot.
    """
    peaks, n = detect_batch(gs, qmaxs, n=n, min_len=min_len, gpu=gpu, dtype=dtype,
                            n_max=n_max, coarse=coarse)
    results = []
    for g, q, (vecs, _amps) in zip(gs, qmaxs, peaks):
        g = np.asarray(g, float)
        tol_abs = tol_frac * q
        real_cands = vecs[:topk]
        if len(real_cands) < 3:
            results.append(IndexResult(None, None, None, None, 0, vecs))
            continue
        out = search_basis(real_cands, np.zeros((0, 3)), g, tol_abs, n_real=topk,
                           min_inlier_frac=min_inlier_frac)
        if out is None:
            results.append(IndexResult(None, None, None, None, 0, vecs))
        else:
            M, C, hkl, inliers = out
            results.append(IndexResult(M, C, hkl, inliers, int(inliers.sum()), vecs))
    return results


def index_shot(g, qmax, n="auto", peakfinder=find_peaks_classical, min_len=3.0,
               topk=12, localize=False, tol_frac=0.02, seed_diff=False,
               min_inlier_frac=0.7, n_max=256, gpu=False, seed_proj=True):
    # n="auto": size the grid to the longest cell axis (estimate_grid_n). A fixed
    # grid spans |x| <= n/(4 qmax) ~ 96 A at n=128, so large cells fall off it. The
    # longest axis often has no observed adjacent pair, so the estimate can still be
    # short -> escalate to n_max once before giving up.
    # gpu=True runs the deposit+FFT+peakfind on cupy/cuFFT (falls back to numpy if
    # cupy is absent); the small basis search stays on CPU either way.
    # seed_proj=True adds projection-slice long-axis candidates on sparse frames.
    g = np.asarray(g, float)
    tol_abs = tol_frac * qmax
    grid_sizes = sorted({estimate_grid_n(g, qmax), n_max}) if n == "auto" else [n]
    res = None
    for nn in grid_sizes:
        res = _index_at(g, qmax, nn, tol_abs, peakfinder, min_len, topk,
                        localize, seed_diff, min_inlier_frac, gpu=gpu,
                        seed_proj=seed_proj)
        if res.M is not None:
            return res
    return res
