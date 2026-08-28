"""Fast, portable azimuthal (radial) integration -- a small primitive, no pyFAI dependency.

Build the sparse integration matrix ``M`` once from the detector geometry; then every frame is a single
CSR matrix-vector product::

    I(q) = (M @ signal) / (M @ norm)

``M`` is the same object pyFAI builds (a CSR look-up table); all the accuracy lives in how ``M`` is built,
and the per-frame cost never changes -- ideal for anything that integrates repeatedly (peak finders, live
monitors, powder-indexing / Rietveld front ends). The identical code runs on the CPU (numpy/scipy) or the
GPU (cupy/cuSPARSE) -- it dispatches on the array type of the inputs.

Two pixel-splitting schemes (``split=``):
  * ``"linear"`` (default): each pixel is a POINT at its centre, split between its two neighbouring bins by
    (1-frac, frac). Cheapest ``M`` (2 nnz/pixel); gives unbiased, sub-bin PEAK POSITIONS.
  * ``"area"``: each pixel is a POLYGON that spans a q-range; its unit weight is spread over every bin the
    range overlaps, by overlap fraction (pyFAI "bbox"-style area splitting). Correct profile SHAPE for
    oversampled bins / high tilt / large pixels. The per-pixel q half-width is taken from the gradient of
    the q field (so only q-per-pixel is needed), or you can pass ``q_halfwidth`` (e.g. pyFAI's ``deltaQ``).

Corrections (solid angle, polarisation, flat field) go in via the per-pixel ``norm`` argument, exactly as
pyFAI applies them -- I(q) = sum(w*signal) / sum(w*norm). NOTE pyFAI's ``integrate1d`` applies the
solid-angle (and polarisation) correction BY DEFAULT, so to match a pyFAI profile pass the same ``norm``
(e.g. ``ai.solidAngleArray(shape)``) or turn pyFAI's corrections off.

Performance (A100, 16 MB frame, ~1449 bins): CSR matvec ~0.1-0.2 ms -- cuSPARSE ``csrmv`` is memory-bound-
optimal, ~20-50x faster than a weighted ``bincount``. Validated against pyFAI's OpenCL full-split to
<0.05% RMS on a real geometry (see ``test_radial.py``), at ~4.7x the kernel speed (~0.2 vs ~1.0 ms;
cuSPARSE vs OpenCL, and pyFAI also computes error bars + corrections each call). Batch many frames with
``integrate_batch`` (one sparse-dense matmul). Keep frames on the GPU -- for streaming from the host the
PCIe transfer (~1.2 ms/16 MB) dominates the compute.

Origin: the "sparse matrix as accumarray" radial-averaging technique (build a sparse operator once, then a
matvec per frame) goes back to S. Marchesini's 2013 GPU code *"GPU sparse, accumarray, non-uniform grid"*
(MATLAB Central File Exchange #44423; ``gcsparse`` COO/CSR via NVIDIA cusp, with a NUFFT companion for the
non-uniform-grid interpolation). It was reprototyped in cupy by the SLAC LCLS DRP team in the external
``slac-lcls/drp-benchmarks`` repository (``radial_integration/testing2.py``, ``index2avg_op``; not vendored here); this module generalises it into a reusable CPU/GPU
primitive with pixel-splitting, corrections (``norm``) and batching.

The existing LCLS CPU integrator, smalldata_tools ``ana_funcs/azimuthalBinning.py``, does the same
precompute-assignment-once / reduce-per-frame pattern via ``np.digitize`` + ``np.bincount`` (nearest-bin,
no splitting) with full 2-D q-phi caking and geometry/polarisation corrections + MPI; this is essentially
its GPU sparse-matvec form (a 1-nnz/pixel accumarray, ``M@img`` == its bincount) plus pixel-splitting, with
its per-pixel ``correction`` (geom x pol) mapping onto ``norm=``.

Credit: the sparse-LUT + pixel-split idea follows pyFAI -- J. Kieffer et al.; Ashiotis, Deschildre, Nawaz,
Wright, Karkoulis, Picca & Kieffer, "The fast azimuthal integration Python library: pyFAI", J. Appl. Cryst.
48, 510-519 (2015). pyFAI's exact 2-D polygon splitting, error propagation and OpenCL back end are theirs
-- use pyFAI when you need those.

Authors: SLAC LCLS DRP team, with Claude (Anthropic).
"""
import numpy as np

__all__ = ["RadialIntegrator", "radial_average"]


def _xp(a):
    """numpy for a numpy array, cupy for a cupy array."""
    try:
        import cupy
        return cupy.get_array_module(a)
    except Exception:
        return np


class RadialIntegrator:
    """Precompute the sparse integration matrix ``M`` for one geometry; reuse it per frame.

    Parameters
    ----------
    q_per_pixel : 2-D array (H, W)
        The radial coordinate (q, 2theta, r, ... -- any scalar) at each pixel centre.
    nbin : int, optional
        Number of output bins (default ~sqrt(npix)/2).
    qmin, qmax : float, optional
        Bin range (default the data min/max).
    mask : bool array, optional
        True = keep the pixel.
    norm : array, optional
        Per-pixel normalisation (solid angle x polarisation x flat); default 1.
    split : {"linear", "area"}
        Pixel-splitting scheme (see module docstring).
    q_halfwidth : array, optional
        Per-pixel q half-extent for ``split="area"`` (default: estimated from the q-field gradient).
    """

    def __init__(self, q_per_pixel, nbin=None, qmin=None, qmax=None, mask=None, norm=None,
                 split="linear", q_halfwidth=None):
        xp = _xp(q_per_pixel); self._xp = xp
        q2d = q_per_pixel
        q = q2d.ravel().astype(xp.float64); npix = q.size
        self.qmin = float(q.min() if qmin is None else qmin)
        self.qmax = float(q.max() if qmax is None else qmax)
        self.nbin = NB = int(nbin if nbin is not None else np.sqrt(npix) / 2)
        self.dq = (self.qmax - self.qmin) / NB
        self.q = self.qmin + (xp.arange(NB) + 0.5) * self.dq       # bin centres
        okmask = None if mask is None else mask.ravel().astype(bool)
        if split == "linear":
            rows, cols, vals = self._linear(q, okmask)
        elif split == "area":
            rows, cols, vals = self._area(q2d, q, okmask, q_halfwidth)
        else:
            raise ValueError("split must be 'linear' or 'area'")
        sp = self._sparse_mod()
        self.M = sp.coo_matrix((vals, (rows, cols)), shape=(NB, npix)).tocsr()
        n = xp.ones(npix, xp.float64) if norm is None else norm.ravel().astype(xp.float64)
        self.den = self.M @ n
        self.den = xp.where(self.den == 0, xp.nan, self.den)
        self.split = split

    def _sparse_mod(self):
        if self._xp is np:
            import scipy.sparse as sp
        else:
            import cupyx.scipy.sparse as sp
        return sp

    def _linear(self, q, okmask):
        xp = self._xp; NB = self.nbin
        t = (q - self.qmin) / self.dq - 0.5                       # split about bin CENTRES
        b0 = xp.floor(t).astype(xp.int64); frac = t - b0
        ok = xp.ones(q.size, bool) if okmask is None else okmask.copy()
        ok0 = ok & (b0 >= 0) & (b0 < NB); ok1 = ok & (b0 + 1 >= 0) & (b0 + 1 < NB)
        rows = xp.concatenate([xp.where(ok0, b0, 0), xp.where(ok1, b0 + 1, 0)])
        cols = xp.concatenate([xp.arange(q.size), xp.arange(q.size)])
        vals = xp.concatenate([xp.where(ok0, 1.0 - frac, 0.0), xp.where(ok1, frac, 0.0)])
        return rows, cols, vals

    def _area(self, q2d, q, okmask, q_halfwidth):
        xp = self._xp; NB = self.nbin
        if q_halfwidth is not None:
            hw = q_halfwidth.ravel().astype(xp.float64)
        else:
            g0, g1 = xp.gradient(q2d.astype(xp.float64))
            hw = 0.5 * (xp.abs(g0) + xp.abs(g1)).ravel()          # L1 half-width to a square-pixel corner
        hw = xp.maximum(hw, self.dq * 1e-4)
        qlo = xp.clip(q - hw, self.qmin, self.qmax)
        qhi = xp.clip(q + hw, self.qmin, self.qmax)
        jlo = xp.clip(xp.floor((qlo - self.qmin) / self.dq).astype(xp.int64), 0, NB - 1)
        jhi = xp.clip(xp.floor((qhi - self.qmin) / self.dq).astype(xp.int64), 0, NB - 1)
        ok = xp.ones(q.size, bool) if okmask is None else okmask.copy()
        ok &= (qhi > qlo)
        counts = xp.where(ok, jhi - jlo + 1, 0).astype(xp.int64)
        total = int(counts.sum())
        csum = xp.cumsum(counts); ranges = xp.arange(total)       # ragged-range expansion, cupy-portable
        cols = xp.searchsorted(csum, ranges, side="right")        #   (cupy.repeat rejects array repeats)
        within = ranges - (csum - counts)[cols]
        rows = jlo[cols] + within
        elo = self.qmin + rows * self.dq; ehi = elo + self.dq
        ov = xp.clip(xp.minimum(qhi[cols], ehi) - xp.maximum(qlo[cols], elo), 0.0, None)
        vals = ov / (2.0 * hw)[cols]                             # normalise by UNCLIPPED width so a pixel
        return rows, cols, vals                                 # straddling qmin/qmax contributes <1 (like linear)

    def _M2(self):
        """Squared-weight operator M^2 (same sparsity, values squared), built once, for error propagation."""
        if getattr(self, "_M2cache", None) is None:
            M2 = self.M.copy(); M2.data = M2.data ** 2
            self._M2cache = M2
        return self._M2cache

    def _Mc(self):
        """Complex fused operator ``Mc = M + i M^2`` (complex64), built once. Because ``M`` and ``M^2`` share
        sparsity and (Poisson) input, one complex matvec ``Mc @ s`` returns both numerators in a single pass:
        ``real = sum(w*s)`` (average) and ``imag = sum(w^2*s)`` (error). ~1.6x vs the two real matvecs on GPU
        in float32 (measured H100); float64 is only ~1.1x (complex arithmetic offsets the traffic saving)."""
        if getattr(self, "_Mccache", None) is None:
            sp = self._sparse_mod(); xp = self._xp
            d = self.M.data.astype(xp.float32)
            data = (d + 1j * (d * d)).astype(xp.complex64)
            self._Mccache = sp.csr_matrix((data, self.M.indices.copy(), self.M.indptr.copy()),
                                          shape=self.M.shape)
        return self._Mccache

    def integrate(self, image, errors=False, variance=None, fused=False):
        """image (H, W) -> (q_centres, I(q)). One CSR matvec.

        ``errors=True`` also returns the per-bin propagated 1-sigma error (like pyFAI's error models), for
        one extra matvec: for the weighted mean I = sum(w*s)/sum(w*n), sigma_I = sqrt(M^2 @ var) / den, where
        ``var`` is the per-pixel signal variance (default = the signal itself, i.e. Poisson counts; pass
        ``variance=`` for a Gaussian/known-variance model). Returns (q, I, sigma) when errors=True.

        ``fused=True`` (GPU + Poisson only) computes I and sigma from a SINGLE complex matvec ``Mc = M + i M^2``
        in float32 -- ~1.6x faster on the error path by reading the shared sparsity/input once, at float32
        precision. Ignored on CPU or when a ``variance=`` is given; the default float64 two-matvec path is
        unchanged and bit-identical.
        """
        xp = self._xp
        if fused and errors and variance is None and xp is not np:
            z = self._Mc() @ image.ravel().astype(xp.complex64)
            return self.q, z.real / self.den, xp.sqrt(xp.maximum(z.imag, 0.0)) / self.den
        s = image.ravel().astype(xp.float64)
        I = (self.M @ s) / self.den
        if not errors:
            return self.q, I
        var = s if variance is None else variance.ravel().astype(xp.float64)
        sigma = xp.sqrt(self._M2() @ var) / self.den
        return self.q, I, sigma

    def integrate_batch(self, images, dtype=None):
        """(B, H, W) or (B, npix) -> (q_centres, I[B, nbin]) in one sparse-dense matmul (SpMM).

        Amortises the per-call overhead over the batch -- the fast path for many frames. Keep the frames
        on the GPU; the host->device transfer dominates otherwise. SpMM (this) vs a loop of SpMVs crosses
        over around B~=8-16 on an A100 (below that a per-frame ``integrate`` loop is faster; SpMM wins
        ~1.8x at B>=32). Note ``X.T`` of a C-contiguous ``X`` is F-contiguous, which is the layout cuSPARSE
        csrmm prefers -- ~2x faster than a C-order dense RHS. (CUDA graphs can't help: cuSPARSE refuses
        stream capture; multi-stream doesn't either -- one SpMV already saturates memory bandwidth.)
        """
        X = images.reshape(images.shape[0], -1).astype(self.M.dtype if dtype is None else dtype)
        return self.q, (self.M @ X.T / self.den[:, None]).T


# back-compat alias (older code imports RadialLUT)
RadialLUT = RadialIntegrator


def radial_average(q_per_pixel, image, nbin=None, **kw):
    """One-shot convenience: build the integrator and integrate a single frame -> (q, I)."""
    return RadialIntegrator(q_per_pixel, nbin=nbin, **kw).integrate(image)
