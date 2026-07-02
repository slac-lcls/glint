"""Self-contained azimuthal (radial) integration -- no pyFAI dependency, CPU (numpy) or GPU (cupy) via the
same code. Keeps the two things that make pyFAI fast+accurate on this problem, without cuSPARSE:

  * FRACTIONAL PIXEL-SPLITTING (each pixel spread over its two neighbouring q-bins by 1-frac / frac)
    -> sub-bin-accurate ring positions, the property that matters for indexing (vs nearest-bin aliasing).
  * a LOAD-BALANCED reduction via `bincount` (num & denom sums per bin) instead of a (bins x pixels) CSR
    matrix-vector product. Radial bins have hugely uneven pixel counts (pixels/bin grows ~linearly with
    radius), which wrecks a generic cuSPARSE `csrmv` (row-per-bin load imbalance + scattered gather);
    `bincount` sidesteps both. The per-pixel bin/weight LUT and the denominator are precomputed ONCE, so
    the per-frame cost is two weighted bincounts.

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
    """Precomputed per-pixel bin/weight table for one detector geometry. Build once, reuse per frame."""
    def __init__(self, q_per_pixel, nbin=None, qmin=None, qmax=None, mask=None, norm=None):
        xp = _xp(q_per_pixel)
        q = q_per_pixel.ravel().astype(xp.float64)
        self.qmin = float(q.min() if qmin is None else qmin)
        self.qmax = float(q.max() if qmax is None else qmax)
        self.nbin = int(nbin if nbin is not None else np.sqrt(q.size) / 2)
        self.dq = (self.qmax - self.qmin) / self.nbin
        self.q = self.qmin + (xp.arange(self.nbin) + 0.5) * self.dq     # bin centres
        t = (q - self.qmin) / self.dq - 0.5                            # split about CENTRES (not edges)
        b0 = xp.floor(t).astype(xp.int64)
        frac = t - b0
        ok = xp.ones(q.size, bool)
        if mask is not None:
            ok &= mask.ravel().astype(bool)
        wlo = xp.where(ok, 1.0 - frac, 0.0)                             # -> bin b0
        whi = xp.where(ok, frac, 0.0)                                   # -> bin b0+1 (pixel splitting)
        NB = self.nbin
        self._b0 = xp.clip(b0, -1, NB)                                  # out-of-range -> overflow bins
        self._b1 = xp.clip(b0 + 1, -1, NB)
        self._b0 = xp.where((self._b0 < 0), NB, self._b0)               # fold -1 into the NB overflow slot
        self._b1 = xp.where((self._b1 < 0), NB, self._b1)
        self._wlo, self._whi = wlo, whi
        self._xp = xp
        n = xp.ones(q.size, xp.float64) if norm is None else norm.ravel().astype(xp.float64)
        self.den = self._reduce(n)                                      # normalisation sum per bin (static)
        self.den = xp.where(self.den == 0, xp.nan, self.den)

    def _reduce(self, vals):
        xp = self._xp; NB = self.nbin
        r = (xp.bincount(self._b0, self._wlo * vals, minlength=NB + 1)
             + xp.bincount(self._b1, self._whi * vals, minlength=NB + 1))
        return r[:NB]

    def integrate(self, image):
        """image -> (q_centres, I(q)). Same code on numpy or cupy arrays."""
        num = self._reduce(image.ravel().astype(self._xp.float64))
        return self.q, num / self.den


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
    print("LUT: %d bins, %d pixels, per-frame = 2 weighted bincounts" % (lut.nbin, N * N))
