"""Self-contained azimuthal (radial) integration as a portable, fast primitive -- no pyFAI dependency,
CPU (numpy) or GPU (cupy) via the same code. Build the sparse integration matrix M once; every frame is
then one CSR matvec  I = (M @ signal) / (M @ norm)  (or M @ stacked-frames for a batch). All the accuracy
lives in how M is built; the per-frame cost never changes -- ideal for tools that integrate repeatedly
(peakfinders, live monitors, Rietveld front ends).

Two pixel-splitting schemes (choose with `split=`):
  * "linear"  (default): each pixel is a POINT at its centre, split between its two neighbouring bins by
    (1-frac, frac). Unbiased sub-bin PEAK POSITIONS -- enough for indexing; cheapest M (2 nnz/pixel).
  * "area"   (pyFAI "bbox"-style): each pixel is a POLYGON that covers a q-range; its unit weight is spread
    over every bin that range overlaps, by overlap fraction. Correct profile SHAPE (multi-bin pixels near
    the centre / at high tilt) -- what you want for Rietveld or accurate intensities. The per-pixel q half-
    width is taken from the gradient of the q field (so only q-per-pixel is needed), or pass `q_halfwidth`.

Reduction = sparse CSR matvec: benchmarked (radial_bench.py, A100) at ~0.1-0.2 ms / 16-MB frame -- cuSPARSE
csrmv is memory-bound-optimal; ~20-50x faster than a weighted bincount. Batch many frames with
`integrate_batch` (one SpMM). Credit: reimplements the core idea of pyFAI (Kieffer et al.; Ashiotis,
Deschildre, Nawaz, Wright, Karkoulis, Picca & Kieffer, J. Appl. Cryst. 48, 510-519, 2015) -- the sparse-LUT
+ split idea only; pyFAI's exact 2-D polygon splitting and OpenCL back end are theirs.
"""
import numpy as np


def _xp(a):
    try:
        import cupy
        return cupy.get_array_module(a)
    except Exception:
        return np


class RadialLUT:
    def __init__(self, q_per_pixel, nbin=None, qmin=None, qmax=None, mask=None, norm=None,
                 split="linear", q_halfwidth=None):
        xp = _xp(q_per_pixel); self._xp = xp
        q2d = q_per_pixel                                           # keep 2-D shape (needed for "area")
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
        if xp is np:
            import scipy.sparse as sp
        else:
            import cupyx.scipy.sparse as sp
        self.M = sp.coo_matrix((vals, (rows, cols)), shape=(NB, npix)).tocsr()   # the integration operator
        n = xp.ones(npix, xp.float64) if norm is None else norm.ravel().astype(xp.float64)
        self.den = self.M @ n
        self.den = xp.where(self.den == 0, xp.nan, self.den)
        self.split = split

    # ---- linear (point-at-centre) split: 2 nnz/pixel -----------------------------------------------
    def _linear(self, q, okmask):
        xp = self._xp; NB = self.nbin
        t = (q - self.qmin) / self.dq - 0.5                       # split about CENTRES
        b0 = xp.floor(t).astype(xp.int64); frac = t - b0
        ok = xp.ones(q.size, bool) if okmask is None else okmask.copy()
        ok0 = ok & (b0 >= 0) & (b0 < NB); ok1 = ok & (b0 + 1 >= 0) & (b0 + 1 < NB)
        rows = xp.concatenate([xp.where(ok0, b0, 0), xp.where(ok1, b0 + 1, 0)])
        cols = xp.concatenate([xp.arange(q.size), xp.arange(q.size)])
        vals = xp.concatenate([xp.where(ok0, 1.0 - frac, 0.0), xp.where(ok1, frac, 0.0)])
        return rows, cols, vals

    # ---- area ("bbox") split: pixel covers [qlo,qhi], spread by overlap fraction -> variable nnz ----
    def _area(self, q2d, q, okmask, q_halfwidth):
        xp = self._xp; NB = self.nbin
        if q_halfwidth is not None:                               # per-pixel q half-extent supplied
            hw = q_halfwidth.ravel().astype(xp.float64)
        else:                                                     # estimate from the q-field gradient
            g0, g1 = xp.gradient(q2d.astype(xp.float64))
            hw = 0.5 * (xp.abs(g0) + xp.abs(g1)).ravel()          # L1 half-width to a square-pixel corner
        hw = xp.maximum(hw, self.dq * 1e-4)                       # avoid a zero-width pixel
        qlo = xp.clip(q - hw, self.qmin, self.qmax)
        qhi = xp.clip(q + hw, self.qmin, self.qmax)
        jlo = xp.clip(xp.floor((qlo - self.qmin) / self.dq).astype(xp.int64), 0, NB - 1)
        jhi = xp.clip(xp.floor((qhi - self.qmin) / self.dq).astype(xp.int64), 0, NB - 1)
        ok = xp.ones(q.size, bool) if okmask is None else okmask.copy()
        ok &= (qhi > qlo)
        counts = xp.where(ok, jhi - jlo + 1, 0).astype(xp.int64)  # bins each pixel touches
        total = int(counts.sum())
        starts = xp.cumsum(counts) - counts                      # ragged-range expansion:
        cols = xp.repeat(xp.arange(q.size), counts)              #   pixel index per nnz
        within = xp.arange(total) - xp.repeat(starts, counts)    #   0,1,.. within each pixel's run
        rows = xp.repeat(jlo, counts) + within                   #   bin index per nnz
        elo = self.qmin + rows * self.dq; ehi = elo + self.dq    # this bin's edges
        a = xp.repeat(qlo, counts); b = xp.repeat(qhi, counts)
        ov = xp.clip(xp.minimum(b, ehi) - xp.maximum(a, elo), 0.0, None)   # [qlo,qhi] ∩ bin length
        vals = ov / xp.repeat(qhi - qlo, counts)                 # fraction of the pixel's q-range
        return rows, cols, vals

    def integrate(self, image):
        """image -> (q_centres, I(q)). One CSR matvec; same code on numpy or cupy."""
        return self.q, (self.M @ image.ravel().astype(self._xp.float64)) / self.den

    def integrate_batch(self, images, dtype=None):
        """(B,H,W) or (B,npix) -> (q_centres, I[B,nbin]) in ONE sparse-dense matmul (amortises per-call
        overhead -- the fast path for many frames; keep them on the GPU, transfer dominates)."""
        xp = self._xp
        X = images.reshape(images.shape[0], -1).astype(self.M.dtype if dtype is None else dtype)
        return self.q, (self.M @ X.T / self.den[:, None]).T


# ------------------------------------------------------------------ self-test (numpy) ------------------
if __name__ == "__main__":
    N = 512
    yy, xx = np.mgrid[-N // 2:N // 2, -N // 2:N // 2]
    r = np.sqrt(xx ** 2 + yy ** 2)
    qpp = r * (2.0 / N)
    true_q = np.array([0.30, 0.55, 0.812, 1.05])
    img = np.zeros((N, N))
    for tq in true_q:
        img += np.exp(-((qpp - tq) ** 2) / (2 * 0.006 ** 2))
    nbin = 300

    lin = RadialLUT(qpp, nbin=nbin, qmin=0.0, qmax=1.4, split="linear")
    area = RadialLUT(qpp, nbin=nbin, qmin=0.0, qmax=1.4, split="area")
    for name, lut in (("linear", lin), ("area", area)):
        colsum = np.asarray(lut.M.sum(0)).ravel()                # each pixel's weights should sum to 1
        print(f"{name:6s}: nnz={lut.M.nnz:>8d} ({lut.M.nnz/ (N*N):.2f}/pixel)  "
              f"col-sum(min/mean/max)={colsum.min():.3f}/{colsum.mean():.3f}/{colsum.max():.3f}")

    # ground truth = fine 4x4 supersample of each pixel (point-split the subpixels)
    s = 4
    ys, xs = (np.mgrid[0:N * s, 0:N * s] + 0.5) / s - N / 2
    rs = np.sqrt(xs ** 2 + ys ** 2) * (2.0 / N)
    imgs = np.repeat(np.repeat(img, s, 0), s, 1)
    ref = RadialLUT(rs, nbin=nbin, qmin=0.0, qmax=1.4, split="linear").integrate(imgs)[1]

    ql, Il = lin.integrate(img); qa, Ia = area.integrate(img)
    m = np.isfinite(ref) & np.isfinite(Il) & np.isfinite(Ia)
    nrm = np.nanmax(ref[m])
    print(f"\nprofile RMS vs 4x4-supersampled truth:  linear={np.sqrt(np.mean(((Il-ref)/nrm)[m]**2))*1e3:.2f}e-3"
          f"   area={np.sqrt(np.mean(((Ia-ref)/nrm)[m]**2))*1e3:.2f}e-3")

    def peak(qc, Ic, q0):
        j = int(np.nanargmin(np.abs(qc - q0))); j = j - 3 + int(np.nanargmax(Ic[j - 3:j + 4]))
        d = (Ic[j - 1] - Ic[j + 1]) / (2 * (Ic[j - 1] - 2 * Ic[j] + Ic[j + 1]) + 1e-12)
        return qc[j] + d * (qc[1] - qc[0])
    print("ring   true     linear(err)    area(err)   [milli-q]")
    for tq in true_q:
        print(f"      {tq:5.3f}   {1000*(peak(ql,Il,tq)-tq):+6.2f}      {1000*(peak(qa,Ia,tq)-tq):+6.2f}")
