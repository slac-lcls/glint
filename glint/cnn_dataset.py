"""Volume + heatmap dataset for the 3D-CNN peakfinder (PeakNet-style).

Pure numpy so it runs and is testable anywhere; torch wraps it on the fly in
cnn.py / train_cnn.py. For each simulated shot we render the FFT volume at a FIXED
grid and a target heatmap with Gaussian blobs at the true direct-lattice vectors
(integer combinations of the ground-truth basis M within the grid).

v1 scope: cells whose axes fit the fixed grid (|x| <= n/(4 qmax)); larger cells are
handled at inference by resampling (CNNPeakFinder) or by the feature re-ranker +
escalation path. Default n=96, qmax=1/3 -> grid spans ~72 A, so cell_hi defaults 64.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .lattice import random_cell
from .simulate import simulate_shot
from .transform import fft_volume


def true_lattice_vectors(M, xmax, min_len, hmax=8):
    """Direct-lattice vectors M @ h (h integer, nonzero) with min_len <= |.| <= xmax."""
    r = np.arange(-hmax, hmax + 1)
    H = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, axis=1)]
    V = (M @ H.T).T
    L = np.linalg.norm(V, axis=1)
    return V[(L >= min_len) & (L <= xmax)]


def render_heatmap(M, xcoords, n, min_len=4.0, blob=1.5):
    """Target peakness volume: Gaussian blobs at the true lattice vectors."""
    V = true_lattice_vectors(M, xcoords.max(), min_len)
    hm = np.zeros((n, n, n), np.float32)
    dx, x0 = xcoords[1] - xcoords[0], xcoords[0]
    idx = np.rint((V - x0) / dx).astype(int)
    idx = idx[np.all((idx >= 0) & (idx < n), axis=1)]
    if len(idx):
        np.add.at(hm, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    if blob > 0:
        hm = gaussian_filter(hm, blob, mode="constant")
        m = hm.max()
        if m > 0:
            hm /= m
    return hm


def make_cnn_sample(rng, n=96, symmetry="random", cell_lo=30.0, cell_hi=64.0,
                    spot_range=(20, 90), sigma_range=(0.0, 0.0025),
                    spur_range=(0.0, 0.3), min_len=4.0, dmin=3.0):
    """One (volume, heatmap) pair, both (n,n,n) float32. Volume is peak-normalized.

    dmin sets the resolution limit -> qmax = 1/dmin -> real-space grid half-extent
    xmax = n/(4 qmax) = n*dmin/4. The longest axis must satisfy cell_hi < xmax, else
    it aliases off-grid: for cell_hi=82 at n=96 use dmin>=4.0 (xmax>=96). The default
    dmin=3.0 keeps the original 30-64A regime (xmax=72) byte-compatible.
    """
    cell = random_cell(rng, symmetry, lo=cell_lo, hi=cell_hi)
    nt = int(rng.integers(spot_range[0], spot_range[1] + 1))
    ps = rng.uniform(*sigma_range)
    fs = rng.uniform(*spur_range)
    shot = simulate_shot(rng=rng, cell=cell, n_target=nt, pos_sigma=ps,
                         frac_spurious=fs, dmin=dmin)
    vol, x = fft_volume(shot.g, shot.meta["qmax"], n=n)
    vol = (vol / max(vol.max(), 1e-9)).astype(np.float32)
    hm = render_heatmap(shot.M, x, n, min_len=min_len)
    return vol, hm


def make_cnn_dataset(n_samples, n=96, rng=None, **kw):
    """Materialize a small dataset (V, Y), shapes (n_samples,1,n,n,n). For tests only
    -- training uses cnn.VolumeHeatmapDataset which generates on the fly (a full
    materialized set at n=96 is ~GBs)."""
    rng = np.random.default_rng() if rng is None else rng
    V = np.empty((n_samples, 1, n, n, n), np.float32)
    Y = np.empty((n_samples, 1, n, n, n), np.float32)
    for i in range(n_samples):
        V[i, 0], Y[i, 0] = make_cnn_sample(rng, n=n, **kw)
    return V, Y
