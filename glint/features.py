"""Per-peak features for the learned peakfinder.

Given the observed reciprocal spots g and a candidate real-space vector x (a peak
of |F(x)|), we describe how "lattice-vector-like" x is, using grid-free direct-FT
quantities. The discriminating signals: a true direct-lattice vector makes every
spot phase g_i.x land on an integer (high fringe amplitude + tight integer
residuals), the peak is sharp, and its harmonics behave like a lattice (2x is also
a node; x/2 is NOT, for a primitive vector). Spurious-induced peaks fail these.

These features need g (not just the volume), so the learned peakfinder is called
with g and qmax -- see index.index_shot / peakfind.find_peaks_classical signatures.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["amp", "inlier_frac", "med_resid", "len", "sharp", "amp_2x", "amp_half"]


def _absF(g, X):
    """|F(x)| = |sum_i exp(2 pi i g_i . x)| for each row x of X (K,3)."""
    phase = 2 * np.pi * (np.atleast_2d(X) @ g.T)        # (K, N)
    return np.abs(np.exp(1j * phase).sum(axis=1))        # (K,)


def peak_features(g, X, qmax):
    """Feature matrix (K, len(FEATURE_NAMES)) for candidate real-space vectors X."""
    g = np.asarray(g, float)
    X = np.atleast_2d(np.asarray(X, float))
    N = len(g)

    amp = _absF(g, X) / N                                # normalized fringe score
    proj = X @ g.T                                       # (K, N) = g_i . x
    resid = proj - np.rint(proj)
    inlier_frac = (np.abs(resid) < 0.1).mean(axis=1)
    med_resid = np.median(np.abs(resid), axis=1)
    length = np.linalg.norm(X, axis=1)

    # sharpness: amplitude drop under small real-space perturbations (~1 A)
    d = 1.0
    perturb = np.array([[d, 0, 0], [-d, 0, 0], [0, d, 0],
                        [0, -d, 0], [0, 0, d], [0, 0, -d]], float)
    nbr = np.zeros(len(X))
    for p in perturb:
        nbr += _absF(g, X + p) / N
    nbr /= len(perturb)
    sharp = amp / (nbr + 1e-6)

    amp_2x = _absF(g, 2 * X) / N                          # harmonic (node) support
    amp_half = _absF(g, 0.5 * X) / N                      # high => x is non-primitive

    return np.column_stack([amp, inlier_frac, med_resid, length, sharp, amp_2x, amp_half])
