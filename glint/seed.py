"""Difference-vector (Patterson) candidate seeding.

Pairwise differences d = g_i - g_j are reciprocal-lattice vectors. Spurious spots
produce *scattered* differences that don't reinforce, so density in the difference
cloud robustly picks out true reciprocal basis vectors (a*, b*, c* and combos) even
when spurious spots have wrecked the |F(x)| peak ranking. This is dirax's principle,
Fourier-dual to the real-space FFT peaks in transform.fft_volume.

We histogram the difference cloud onto a reciprocal-space grid, smooth, and reuse
peakfind.find_peaks_classical to pull out the densest voxels = candidate reciprocal
basis vectors. These are an independent candidate source for search_basis.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .peakfind import find_peaks_classical


def difference_vector_seeds(g, qmax, max_seeds=8, n=96, smooth=1.0, min_len=None):
    """Return up to max_seeds candidate *reciprocal* basis vectors (vote-ranked)."""
    g = np.asarray(g, float)
    N = len(g)
    if N < 3:
        return np.zeros((0, 3))

    iu, ju = np.triu_indices(N, k=1)
    d = g[iu] - g[ju]
    d = np.vstack([d, -d])                       # symmetric cloud
    d = d[np.linalg.norm(d, axis=1) <= qmax]
    if len(d) < 3:
        return np.zeros((0, 3))

    dq = 2 * qmax / n
    hist = np.zeros((n, n, n))
    idx = np.floor((d + qmax) / dq).astype(int)
    idx = idx[np.all((idx >= 0) & (idx < n), axis=1)]
    np.add.at(hist, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    if smooth > 0:
        hist = gaussian_filter(hist, smooth, mode="constant")

    coords = -qmax + (np.arange(n) + 0.5) * dq    # voxel-center coordinates
    if min_len is None:
        min_len = 1.5 * dq                        # skip the origin pileup
    vecs, amps = find_peaks_classical(hist, coords, min_len=min_len)
    return vecs[:max_seeds]


def _fib_hemisphere(D):
    i = np.arange(D); phi = np.pi * (3 - np.sqrt(5)) * i
    z = 1.0 - (i + 0.5) / D
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def projection_axis_seeds(g, n_dir=1200, s_min=25.0, s_max=130.0, ds=0.5,
                          topk=12, rel=0.7, dchunk=256):
    """Projection-slice (MOSFLM/DPS) *direct*-axis candidates L*d_hat.

    By the projection-slice theorem, projecting the peak cloud onto a direction d_hat
    and reading its 1D periodicity is a ray of the 3D |F(x)| along x = s*d_hat -- but
    scored as PERIODICITY (spots at s=a,2a,3a reinforce), which survives sparsity and
    digitization smear that bury the long-axis FFT peak. For each Fibonacci-sphere
    direction we form rho(s) = |sum_p exp(2 pi i s (g_p . d_hat))| over real-space
    periods s in [s_min,s_max] and take the FUNDAMENTAL (smallest local max above
    rel*max) as the axis length. These complement the FFT |F(x)| peaks in the pool;
    they are what lifts the candidate ceiling on sparse frames (FFT alone misses the
    long axes). Cost is O(n_dir * P * n_s); call only on sparse/moderate frames.
    """
    g = np.asarray(g, float)
    if len(g) < 3:
        return np.zeros((0, 3))
    dirs = _fib_hemisphere(n_dir)
    S = np.arange(s_min, s_max, ds)
    out = []
    for a in range(0, n_dir, dchunk):
        Dc = dirs[a:a + dchunk]                      # (dc,3)
        t = g @ Dc.T                                 # (P,dc) projections
        ph = 2 * np.pi * (S[:, None, None] * t[None])  # (ns,P,dc)
        rho = np.abs(np.exp(1j * ph).sum(1))         # (ns,dc) periodicity strength
        rmax = rho.max(0)                            # (dc,)
        loc = (rho[1:-1] > rho[:-2]) & (rho[1:-1] > rho[2:]) & (rho[1:-1] > rel * rmax[None])
        for j in range(Dc.shape[0]):
            si = np.nonzero(loc[:, j])[0]
            if len(si):
                out.append((rmax[j], S[si[0] + 1] * Dc[j]))
    if not out:
        return np.zeros((0, 3))
    out.sort(key=lambda c: -c[0])
    return np.array([c[1] for c in out[:topk]])
