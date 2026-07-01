"""Training-data generator for the learned peakfinder.

For many simulated shots (random orientation + noise) we run the classical volume
peakfinder, compute per-peak features, and label each candidate peak as a true
direct-lattice vector or not (ground truth M is known). Output is a flat
(features, labels) table for a per-peak classifier.

Reusable for the future 3D-CNN: swap the per-peak labelling for a rendered target
heatmap over the volume; the simulator + ground truth are identical.
"""

from __future__ import annotations

import numpy as np

from .features import peak_features
from .lattice import random_cell
from .peakfind import find_peaks_classical
from .simulate import simulate_shot
from .transform import estimate_grid_n, fft_volume

DEFAULT_CELL = (78.0, 78.0, 38.0, 90.0, 90.0, 120.0)


def label_true_axis(M, X, tol=0.12):
    """x is a true real-space lattice vector iff M^{-1} x is a nonzero integer."""
    n = np.linalg.solve(M, np.atleast_2d(X).T).T          # (K,3) fractional coords
    is_int = np.max(np.abs(n - np.rint(n)), axis=1) < tol
    nonzero = np.linalg.norm(np.rint(n), axis=1) > 0
    return is_int & nonzero


def make_dataset(n_shots=400, topk=30, rng=None, n="auto", symmetry=None,
                 spot_range=(20, 90), sigma_range=(0.0, 0.0025),
                 spur_range=(0.0, 0.3)):
    """Labeled (features, true-axis?) table.

    symmetry: None -> the fixed DEFAULT_CELL (single-cell, hexagonal); a symmetry
    name or "random" -> draw a fresh random_cell per shot (cross-cell training).
    """
    rng = np.random.default_rng() if rng is None else rng
    Xs, ys = [], []
    for _ in range(n_shots):
        nt = int(rng.integers(spot_range[0], spot_range[1] + 1))
        ps = rng.uniform(*sigma_range)
        fs = rng.uniform(*spur_range)
        cell = random_cell(rng, symmetry) if symmetry else DEFAULT_CELL
        shot = simulate_shot(rng=rng, cell=cell, n_target=nt, pos_sigma=ps,
                             frac_spurious=fs)
        qmax = shot.meta["qmax"]
        nn = estimate_grid_n(shot.g, qmax) if n == "auto" else n
        vol, x = fft_volume(shot.g, qmax, n=nn)
        vecs, amps = find_peaks_classical(vol, x, min_len=3.0)
        vecs = vecs[:topk]
        if len(vecs) < 3:
            continue
        Xs.append(peak_features(shot.g, vecs, qmax))
        ys.append(label_true_axis(shot.M, vecs))
    return np.vstack(Xs), np.concatenate(ys)
