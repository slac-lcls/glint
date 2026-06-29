"""Grid-free peak localization by direct (nonuniform) Fourier transform.

The gridded FFT pins a candidate axis only to +-dx/2; for sparse clouds we can
do better essentially for free. The transform is analytic at *arbitrary* x:

    F(x)   = sum_i exp(2 pi i g_i . x)
    |F|^2  = C^2 + S^2,   C = sum cos(phi_i), S = sum sin(phi_i),  phi_i = 2pi g_i.x
    grad|F|^2 = 4 pi sum_i g_i (S cos phi_i - C sin phi_i)

|F(x)|^2 is the Fourier transform of the difference-vector (Patterson) cloud
{g_i - g_j}, so this is also dirax's difference-vector criterion in Fourier form.
We maximize |F|^2 locally (BFGS, analytic gradient) from each grid-peak seed to
recover the exact lattice vector with no grid-quantization floor.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def F2_grad(g, x):
    """Return (|F(x)|^2, grad_x |F(x)|^2) for a single query point x (shape (3,))."""
    phi = 2 * np.pi * (g @ x)
    c, s = np.cos(phi), np.sin(phi)
    C, S = c.sum(), s.sum()
    F2 = C * C + S * S
    grad = 4 * np.pi * (g * (S * c - C * s)[:, None]).sum(axis=0)
    return F2, grad


def localize_peak(g, x0, min_len=2.0, maxiter=60):
    """Maximize |F(x)|^2 from seed x0; return the refined lattice vector.

    Falls back to x0 if the optimizer wanders to the origin (the global DC max)
    or fails to improve the fringe score.
    """
    def neg(x):
        F2, grad = F2_grad(g, x)
        return -F2, -grad

    res = minimize(neg, np.asarray(x0, float), jac=True, method="BFGS",
                   options=dict(maxiter=maxiter))
    x = res.x
    if np.linalg.norm(x) < min_len:
        return np.asarray(x0, float)
    if F2_grad(g, x)[0] < F2_grad(g, np.asarray(x0, float))[0]:
        return np.asarray(x0, float)
    return x


def localize_peaks(g, X, min_len=2.0):
    """Localize each row of X (coarse grid peaks) -> exact lattice vectors."""
    g = np.asarray(g, float)
    return np.array([localize_peak(g, x0, min_len=min_len) for x0 in X])
