"""Array backend: run the FFT-volume hot path (deposit, FFT, maximum_filter) on
numpy (CPU) or cupy (GPU) through ONE code path.

The deposit becomes a scatter-add (np.add.at / cupyx.scatter_add) of 8 trilinear
weights per spot -- the same accumulate-with-repeated-indices pattern as the
droplet/Overlapc photon kernels -- so the former Python triple-loop is gone on
CPU and the work maps straight onto the GPU. cupy.fft.fftn -> cuFFT and
cupyx.scipy.ndimage.maximum_filter cover the rest. If cupy is absent, gpu=True
silently falls back to numpy, so the same code is correct everywhere.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as _cp
    _HAVE_CUPY = True
except Exception:                                   # no cupy / no GPU
    _cp = None
    _HAVE_CUPY = False


def cupy_available() -> bool:
    return _HAVE_CUPY


def array_module(gpu: bool = False):
    """Return (xp, on_gpu). gpu=True falls back to numpy when cupy is unavailable."""
    if gpu and _HAVE_CUPY:
        return _cp, True
    return np, False


def module_of(a):
    """numpy or cupy -- whichever owns array `a`."""
    if _HAVE_CUPY and isinstance(a, _cp.ndarray):
        return _cp
    return np


def scatter_add(a, idx, vals):
    """In-place a[idx] += vals with repeated-index accumulation, either backend."""
    if module_of(a) is np:
        np.add.at(a, idx, vals)
    else:
        import cupyx
        cupyx.scatter_add(a, idx, vals)


def fft_mag_centered(rho):
    """|fftshift(fftn(ifftshift(rho)))| -- the indexing volume's magnitude.

    CPU: scipy.fft (pocketfft) with workers=-1 is ~2x numpy.fft (which is
    single-threaded). GPU: cupy.fft -> cuFFT. (Follow-up: rho is real so |F| is
    centrosymmetric -> rfftn computes a half-spectrum for another ~5x, at the cost
    of mirror reconstruction.)
    """
    if module_of(rho) is np:
        try:
            import scipy.fft as sfft
            return np.abs(sfft.fftshift(sfft.fftn(sfft.ifftshift(rho), workers=-1)))
        except Exception:
            return np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(rho))))
    xp = module_of(rho)
    return xp.abs(xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(rho))))


def maximum_filter(vol, size):
    """n-D maximum filter on the backend that owns `vol`."""
    if module_of(vol) is np:
        from scipy.ndimage import maximum_filter as mf
    else:
        from cupyx.scipy.ndimage import maximum_filter as mf
    return mf(vol, size=size, mode="constant")


def asnumpy(a):
    """Bring an array back to host (no-op for numpy)."""
    if _HAVE_CUPY and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)


def sync():
    """Block until queued GPU work finishes -- for honest timing only."""
    if _HAVE_CUPY:
        _cp.cuda.Stream.null.synchronize()
