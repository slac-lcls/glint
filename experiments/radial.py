"""Self-contained azimuthal (radial) integration -- no pyFAI dependency, CPU (numpy) or GPU (cupy) via the
same code. Keeps the two things that make pyFAI good on this problem:

  * FRACTIONAL PIXEL-SPLITTING (each pixel spread over its two neighbouring q-bins by 1-frac / frac)
    -> sub-bin-accurate ring positions, the property that matters for indexing (vs nearest-bin aliasing).
  * the reduction as a precomputed sparse (nbin x npix) CSR MATRIX-VECTOR PRODUCT, I = (M@signal)/(M@norm).
    This is the FAST path on both CPU (scipy, ~5x faster than bincount) and GPU (cupy/cuSPARSE): benchmarked
    on an A100 (16 MB image, radial_bench.py), CSR matvec = 0.13 ms vs a weighted `bincount` = 6.3 ms (~50x
    -- bincount contends on the hot outer bins; a generic csrmv does NOT suffer the load imbalance I first
    guessed, it is memory-bound-optimal). A custom privatized-shared-memory kernel (radial_bench.py) matches
    CSR and also beats bincount ~40x, but CSR is simplest and fastest here. M and the denominator are built
    ONCE; the per-frame cost is one sparse matvec.

Algorithm (numerator/denominator form, as pyFAI): I(bin) = sum_i w_i s_i / sum_i w_i n_i, w = split weight,
s = signal, n = per-pixel normalisation (solid angle x polarisation x flat; default 1). Geometry (q per
pixel) is upstream -- feed q_per_pixel from the detector geometry (poni/wavelength) or the GLINT geom bridge.

Credit: this reimplements the core idea of pyFAI (J. Kieffer et al.; Ashiotis, Deschildre, Nawaz, Wright,
Karkoulis, Picca & Kieffer, "The fast azimuthal integration Python library: pyFAI", J. Appl. Cryst. 48,
510-519, 2015). We keep only the sparse-LUT + split idea; the sophisticated 2-D pixel-area splitting and
the OpenCL back end are pyFAI's -- use pyFAI if you need those.
"""
import numpy as np


def _xp(a):
    try:
        import cupy
        return cupy.get_array_module(a)
    except Exception:
        return np


class RadialLUT:
    """Precomputed sparse (nbin x npix) integration matrix M for one detector geometry: each pixel contributes
    to its two neighbouring q-bins with split weights (1-frac, frac). Build once; per frame I = (M@signal)/
    (M@norm). The reduction is a CSR sparse matrix-vector product -- the FAST path on both CPU (scipy) and
    GPU (cupy), ~50x faster than a weighted `bincount` on GPU (which contends on hot outer bins). Same code
    numpy/cupy."""
    def __init__(self, q_per_pixel, nbin=None, qmin=None, qmax=None, mask=None, norm=None):
        xp = _xp(q_per_pixel); self._xp = xp
        q = q_per_pixel.ravel().astype(xp.float64); npix = q.size
        self.qmin = float(q.min() if qmin is None else qmin)
        self.qmax = float(q.max() if qmax is None else qmax)
        self.nbin = int(nbin if nbin is not None else np.sqrt(npix) / 2)
        NB = self.nbin
        self.dq = (self.qmax - self.qmin) / NB
        self.q = self.qmin + (xp.arange(NB) + 0.5) * self.dq           # bin centres
        t = (q - self.qmin) / self.dq - 0.5                           # split about CENTRES (not edges)
        b0 = xp.floor(t).astype(xp.int64); frac = t - b0
        ok = xp.ones(npix, bool)
        if mask is not None:
            ok &= mask.ravel().astype(bool)
        ok0 = ok & (b0 >= 0) & (b0 < NB)                              # in-range for the lo/hi bin
        ok1 = ok & (b0 + 1 >= 0) & (b0 + 1 < NB)
        rows = xp.concatenate([xp.where(ok0, b0, 0), xp.where(ok1, b0 + 1, 0)])
        cols = xp.concatenate([xp.arange(npix), xp.arange(npix)])
        vals = xp.concatenate([xp.where(ok0, 1.0 - frac, 0.0), xp.where(ok1, frac, 0.0)])
        if xp is np:
            import scipy.sparse as sp
        else:
            import cupyx.scipy.sparse as sp
        self.M = sp.coo_matrix((vals, (rows, cols)), shape=(NB, npix)).tocsr()   # the integration operator
        n = xp.ones(npix, xp.float64) if norm is None else norm.ravel().astype(xp.float64)
        self.den = self.M @ n                                        # normalisation sum per bin (static)
        self.den = xp.where(self.den == 0, xp.nan, self.den)

    def integrate(self, image):
        """image -> (q_centres, I(q)). One CSR matvec; same code on numpy or cupy arrays."""
        return self.q, (self.M @ image.ravel().astype(self._xp.float64)) / self.den

    def integrate_batch(self, images, dtype=None):
        """Stack of B frames -> I(q) for all of them in ONE sparse-dense matmul (SpMM), amortising the
        per-call overhead -- the fast path for many radial averages. `images`: (B,H,W) or (B,npix).
        Returns (q_centres, I) with I shape (B, nbin). Keep the frames on the GPU (transfer dominates)."""
        xp = self._xp
        X = images.reshape(images.shape[0], -1)
        X = X.astype(self.M.dtype if dtype is None else dtype)
        num = self.M @ X.T                                       # (nbin, npix)@(npix, B) -> (nbin, B)
        return self.q, (num / self.den[:, None]).T              # (B, nbin)


# ------------------------------------------------------------------ self-test (numpy) ------------------
if __name__ == "__main__":
    # synthetic detector: N x N, q = radius; place sharp rings at chosen q, integrate, check sub-bin accuracy
    N = 512
    yy, xx = np.mgrid[-N // 2:N // 2, -N // 2:N // 2]
    r = np.sqrt(xx ** 2 + yy ** 2)
    qpp = r * (2.0 / N)                                                 # arbitrary q scale, qmax ~ sqrt(2)
    true_q = np.array([0.30, 0.55, 0.812, 1.05])                       # rings, some between bin centres
    img = np.zeros((N, N))
    for tq in true_q:
        img += np.exp(-((qpp - tq) ** 2) / (2 * 0.004 ** 2))           # thin gaussian rings

    lut = RadialLUT(qpp, nbin=300, qmin=0.0, qmax=1.4)
    q, I = lut.integrate(img)

    # brute-force reference (nearest-bin average) for validation of the reduction
    b = np.clip(((qpp.ravel()) / (1.4 / 300)).astype(int), 0, 299)
    ref_num = np.bincount(b, img.ravel(), minlength=300)
    ref_den = np.bincount(b, minlength=300).astype(float)
    ref = ref_num / np.where(ref_den == 0, np.nan, ref_den)

    # recover ring positions: parabolic peak fit around each local max near a true ring
    def peak(qc, Ic, q0):
        j = np.nanargmin(np.abs(qc - q0))
        j = j - 3 + np.nanargmax(Ic[j - 3:j + 4])
        d = (Ic[j - 1] - Ic[j + 1]) / (2 * (Ic[j - 1] - 2 * Ic[j] + Ic[j + 1]) + 1e-12)
        return qc[j] + d * (qc[1] - qc[0])

    print("ring   true      split-LUT   (err)      nearest-ref (err)")
    for tq in true_q:
        ps = peak(q, I, tq); pr = peak(q, ref, tq)
        print(f"      {tq:6.4f}   {ps:8.5f}  {1000*(ps-tq):+5.1f}m   {pr:8.5f}  {1000*(pr-tq):+5.1f}m")
    finite = np.isfinite(I) & np.isfinite(ref)
    print("\nmean |split - nearest| where both defined:", float(np.nanmean(np.abs(I - ref)[finite])))
    print("LUT: %d bins, %d pixels, per-frame = one CSR sparse matvec (M has %d nnz)" % (lut.nbin, N * N, lut.M.nnz))
