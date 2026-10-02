"""GPU/CPU peakfinder9 -- LOCAL-background Bragg peak finder (cupy, numpy fallback).

peakfinder9 (Y. Gevorkov, CFEL; in CrystFEL via T. A. White; ported to single-threaded Python at European
XFEL by D. Hammer for calNG) differs from peakfinder8 in *where the background comes from*: not a radial
ring model, but a LOCAL square window around each candidate. It accepts a peak with a hierarchy of tests --
the biggest pixel must stand over its local neighbourhood, be a local maximum, be significantly over the
LOCAL background noise, and the whole summed peak must be significantly over that noise.

The local background is the natural GPU primitive here: the per-pixel mean/sigma over the *border ring* of a
(2r+1) window is a difference of two box sums, and a box sum is one separable ``uniform_filter`` (a summed-
area table), so mu(p), sigma(p) come out in O(1)/pixel for the whole frame::

    ring_sum(p)  = box_sum_{2r+1}(I) - box_sum_{2r-1}(I)        # the outer border layer only
    ring_n(p)    = box_sum_{2r+1}(good) - box_sum_{2r-1}(good)  # good-pixel count in that ring (handles masks)
    mu(p)        = ring_sum / ring_n
    sigma(p)     = sqrt( ring_sq/ring_n - mu^2 )               # ring_sq from box sums of I^2

(border ring, not the full window, so the peak's own bright pixels don't inflate its background). Then the
per-pixel SNR ``(I-mu)/sigma`` drives a connected-component label + the SAME gathered scatter-add reduction
as peakfinder8 (bincount over the labelled pixels): peak mass Sum(I-mu), its noise sqrt(Sum sigma^2), npix,
centroid, and whether the component contains a valid local maximum. One code path runs on the GPU
(cupy/cupyx.scipy.ndimage) or CPU (numpy/scipy.ndimage) -- it dispatches on the array type.

This complements ``peakfinder8.py`` (radial-ring background, a sparse matvec): same class shape, same
reduction and ``find_stream`` machinery, but a box-filter local background instead of the radial operator M.

Credit: the peakfinder9 ALGORITHM is Y. Gevorkov (master thesis, CFEL/DESY Hamburg), integrated into CrystFEL
by T. A. White; European XFEL has a single-threaded-Python PORT of it (D. Hammer, calNG). This file is a
fused-CUDA implementation of that algorithm. Its closest psana sibling is M. Dubrovin's (SLAC/LCLS, psalgos)
adaptive-SNR ``peaks_adaptive`` = ``peak_finder_v3r3`` -- an INDEPENDENT LCLS implementation of the same
local-window family (peakfinder9 agrees with it but isn't bit-identical). The psalgos DUAL-THRESHOLD finder
``peaks_droplet`` = ``peak_finder_v4r3`` is matched by peakfinder_v4.py instead. peakfinder8 for reference:
A. Barty et al., "Cheetah", J. Appl. Cryst. 47, 1118-1131 (2014).
Authors of this GPU version: SLAC LCLS DRP team, with Claude (Anthropic).
"""
import numpy as np

__all__ = ["peakfinder9", "PeakFinder9"]


def _xp(a):
    try:
        import cupy
        return cupy.get_array_module(a)
    except Exception:
        return np


def _ndimage(xp):
    if xp is np:
        import scipy.ndimage as ndi
    else:
        import cupyx.scipy.ndimage as ndi
    return ndi


def _scatter_max(xp, rows, values, n):
    """Per-group max via scatter (portable): np.maximum.at (CPU) / cupyx.scatter_max (GPU)."""
    out = xp.zeros(int(n))
    if xp is np:
        np.maximum.at(out, rows, values)
    else:
        import cupyx; cupyx.scatter_max(out, rows, values)
    return out


# A pixel takes part in the ring, the local-max test and `valid` only if the mask keeps it AND its value
# is a finite number -- the kernel twin of PeakFinder9._usable. Two comparisons rather than isfinite():
# both are false for NaN and for +-inf, and the kernel needs no math header under NVRTC.
_PF9_SRC = "#define GLINT_FINITE(v) ((v) <= 3.402823466e+38f && (v) >= -3.402823466e+38f)\n" + r"""
        extern "C" __global__ void pf9_stats(const float* I, const float* good, int H, int W, int r, int lmr,
            float min_snr_big, float min_snr_peak, float min_sig, float min_pon,
            float* snr, float* sub, float* var, unsigned char* cand, unsigned char* ismax){
          int idx = blockIdx.x*blockDim.x + threadIdx.x; if(idx >= H*W) return;
          int yy = idx / W, xx = idx % W; float Ic = I[idx];
          double s=0.0, sq=0.0, nn=0.0;                       // border ring (Chebyshev distance == r)
          // g = 0 for a masked or non-finite pixel, and then v = 0 too: g*v is NaN for v = NaN or inf.
          for(int dy=-r; dy<=r; ++dy){
            int y=yy+dy; if(y<0||y>=H) continue;
            if(dy==-r || dy==r){
              for(int dx=-r; dx<=r; ++dx){ int x=xx+dx; if(x<0||x>=W) continue;
                int j=y*W+x; double g=(GLINT_FINITE(I[j]) ? good[j] : 0.0f), v=(g>0 ? (double)I[j] : 0.0);
                s+=g*v; sq+=g*v*v; nn+=g; }
            } else {
              int x=xx-r; if(x>=0){ int j=y*W+x; double g=(GLINT_FINITE(I[j]) ? good[j] : 0.0f), v=(g>0 ? (double)I[j] : 0.0);
                s+=g*v; sq+=g*v*v; nn+=g; }
              x=xx+r; if(x<W){ int j=y*W+x; double g=(GLINT_FINITE(I[j]) ? good[j] : 0.0f), v=(g>0 ? (double)I[j] : 0.0);
                s+=g*v; sq+=g*v*v; nn+=g; }
            }
          }
          float mu = nn>0.5 ? (float)(s/nn) : 0.0f;
          float vv = nn>0.5 ? (float)(sq/nn - (double)mu*mu) : 0.0f; if(vv<0.0f) vv=0.0f;
          float sg = sqrtf(vv); int valid = (good[idx]>0.5f) && GLINT_FINITE(Ic) && (nn>0.5) && (sg>min_sig);
          float sb = Ic - mu; float sn = valid ? sb/(sg+1e-12f) : 0.0f;
          int islm = 1;                 // local max over the GOOD, finite pixels of the (2*lmr+1) neighbourhood
          for(int dy=-lmr; dy<=lmr && islm; ++dy){ int y=yy+dy; if(y<0||y>=H) continue;
            for(int dx=-lmr; dx<=lmr; ++dx){ int x=xx+dx; if(x<0||x>=W) continue;
              int j=y*W+x; if(good[j]>0.5f && GLINT_FINITE(I[j]) && I[j] > Ic){ islm=0; break; } } }
          snr[idx]=sn; sub[idx]= sb>0.0f? sb:0.0f; var[idx]=vv;
          cand[idx]  = (valid && sn>min_snr_peak) ? 1 : 0;
          ismax[idx] = (valid && islm && sn>min_snr_big && sb>min_pon) ? 1 : 0;
        }"""


_PF9 = None
def _pf9_kernel():
    """One fused fp32 kernel for the whole pf9 front end: per pixel, sum its 8r border ring for the local
    background mu/sigma, test its (2*lmr+1) neighbourhood for a local maximum, and emit snr, (I-bg), sigma^2,
    the 'peak pixel' flag (cand) and the 'valid maximum' flag (ismax). Collapses ~6 uniform_filter passes +
    maximum_filter + ~a dozen elementwise kernels into a single launch (the pf8-style fusion, for pf9).
    The source is ``_PF9_SRC``, kept apart so it can be compiled and checked off a GPU."""
    global _PF9
    if _PF9 is None:
        import cupy
        _PF9 = cupy.RawKernel(_PF9_SRC, "pf9_stats")
    return _PF9


class PeakFinder9:
    """Local-window peakfinder9. Unlike peakfinder8 it needs NO geometry/q -- only the image shape (+ an
    optional bad-pixel mask). Build once, then ``.find(image)`` per frame; ``.find_stream(frames)`` streams a
    batch with the transfer hidden behind compute.

    Parameters (CrystFEL peakfinder9 names in brackets):
      window_radius            [local-bg-radius]      radius r of the local background ring.
      min_snr_biggest_pix      [min-snr-biggest-pix]  the peak's max pixel must exceed the local bg by this
                                                       many sigma.
      min_snr_peak_pix         [min-snr-peak-pix]     pixels this many sigma over bg are 'peak' pixels.
      min_snr_whole_peak       [-]                    the integrated peak Sum(I-bg) must exceed this many
                                                       sigma of the summed background noise sqrt(Sum sigma^2).
      min_sig                  [min-sig]              floor on the local background sigma (guards flat regions).
      min_peak_over_neighbour  [min-peak-over-neighbour]  ADU floor: max pixel must be this far over local bg.
      local_max_radius, min_pix, max_pix, dtype.
    """

    def __init__(self, mask=None, *, shape=None, window_radius=4, min_snr_biggest_pix=7.0,
                 min_snr_peak_pix=6.0, min_snr_whole_peak=8.0, min_sig=0.0,
                 min_peak_over_neighbour=0.0, local_max_radius=1, min_pix=1, max_pix=200, dtype=None):
        if mask is not None:
            xp = _xp(mask); good = mask.astype(bool)
        elif shape is not None:
            xp = np; good = np.ones(shape, bool)
        else:
            raise ValueError("pass either mask (2-D bool good-pixel array) or shape=(H, W)")
        self._xp = xp; self._ndi = _ndimage(xp)
        self.dt = xp.float64 if dtype is None else dtype
        self.good = good; self._goodf = good.astype(self.dt)
        self.H, self.W = good.shape
        self.yy, self.xx = (g.astype(xp.float64) for g in xp.mgrid[0:self.H, 0:self.W])
        self.r = int(window_radius)
        self.p = dict(min_snr_biggest_pix=min_snr_biggest_pix, min_snr_peak_pix=min_snr_peak_pix,
                      min_snr_whole_peak=min_snr_whole_peak, min_sig=min_sig,
                      min_peak_over_neighbour=min_peak_over_neighbour,
                      local_max_radius=int(local_max_radius), min_pix=min_pix, max_pix=max_pix)
        # fp32 GPU fast path: one fused kernel for the whole front end (else portable uniform_filter path)
        self._fused_gpu = False
        if xp is not np and self.dt == xp.float32:
            npix = self.H * self.W
            self._pf9 = _pf9_kernel(); self._blk = 256; self._grid = (npix + 255) // 256
            self._goodf_flat = xp.ascontiguousarray(self._goodf.ravel())
            self._snr = xp.empty(npix, xp.float32); self._sub = xp.empty(npix, xp.float32)
            self._var = xp.empty(npix, xp.float32)
            self._cand = xp.empty(npix, xp.uint8); self._ismax = xp.empty(npix, xp.uint8)
            self._fused_gpu = True

    def _usable(self, I):
        """Per-frame usable pixels: the mask keeps them AND their value is finite. A NaN or inf pixel is
        treated as a bad pixel whatever the mask says. On a finite frame this is exactly `self.good`."""
        return self.good & self._xp.isfinite(I)

    def _ring_bg(self, I, good=None):
        """Per-pixel local background mu, sigma from the border ring of a (2r+1) window (box-sum differences,
        mask-aware). O(1)/pixel via separable uniform_filter. `good` defaults to ``_usable(I)``."""
        xp = self._xp; ndi = self._ndi; r = self.r
        g = self._usable(I) if good is None else good
        goodf = g.astype(self.dt)
        w, wi = 2 * r + 1, 2 * r - 1
        bs = lambda a, s: ndi.uniform_filter(a, size=s, mode="constant") * float(s * s)   # box SUM
        # where(), never I * goodf: NaN*0 and inf*0 are NaN, and the running box sum then carries one
        # masked NaN across the rest of the frame (see peakfinder_v4._ring_bg). Bit-identical on finite frames.
        Ig = xp.where(g, I, 0.0); Ig2 = xp.where(g, I * I, 0.0)
        ring_sum = bs(Ig, w) - bs(Ig, wi)
        ring_sq = bs(Ig2, w) - bs(Ig2, wi)
        ring_n = bs(goodf, w) - bs(goodf, wi)
        nz = ring_n > 0.5
        den = xp.where(nz, ring_n, 1.0)
        mu = ring_sum / den
        sig = xp.sqrt(xp.clip(ring_sq / den - mu * mu, 0.0, None))
        return mu, sig, nz

    def _stats_portable(self, I):
        """Front-end via portable ops (numpy/cupy uniform_filter): -> snr, sub(=clip(I-bg)), var(=sigma^2),
        cand (peak pixels), ismax (valid maxima). Masked and non-finite pixels take no part: not the ring,
        not `valid`, not the local-max test (a masked hot pixel must not unseed the peak beside it)."""
        xp = self._xp; ndi = self._ndi; p = self.p
        g = self._usable(I)
        mu, sig, nz = self._ring_bg(I, g); sub = I - mu
        valid = g & nz & (sig > p["min_sig"])
        snr = xp.where(valid, sub / (sig + 1e-12), 0.0)
        lm = 2 * p["local_max_radius"] + 1
        ismax = valid & (I >= ndi.maximum_filter(xp.where(g, I, -xp.inf), size=lm)) \
            & (snr > p["min_snr_biggest_pix"]) & (sub > p["min_peak_over_neighbour"])
        cand = valid & (snr > p["min_snr_peak_pix"])
        return snr, xp.clip(sub, 0.0, None), sig * sig, cand, ismax

    def _stats_gpu(self, I):
        """Front-end via the single fused fp32 kernel -> same five arrays (as views of persistent buffers)."""
        xp = self._xp; H, W = self.H, self.W; p = self.p
        Ic = xp.ascontiguousarray(I.ravel(), dtype=xp.float32)
        self._pf9((self._grid,), (self._blk,),
                  (Ic, self._goodf_flat, np.int32(H), np.int32(W), np.int32(self.r),
                   np.int32(p["local_max_radius"]), np.float32(p["min_snr_biggest_pix"]),
                   np.float32(p["min_snr_peak_pix"]), np.float32(p["min_sig"]),
                   np.float32(p["min_peak_over_neighbour"]),
                   self._snr, self._sub, self._var, self._cand, self._ismax))
        return (self._snr.reshape(H, W), self._sub.reshape(H, W), self._var.reshape(H, W),
                self._cand.reshape(H, W), self._ismax.reshape(H, W))

    def find(self, image):
        xp = self._xp; ndi = self._ndi; H, W = self.H, self.W; p = self.p
        I = image.astype(self.dt)
        snr, sub, var, cand, ismax = self._stats_gpu(I) if self._fused_gpu else self._stats_portable(I)
        lbl, n = ndi.label(cand)
        if n == 0:
            z = xp.zeros(0)
            return {"x": z, "y": z, "intensity": z, "snr": z, "npix": z.astype(xp.int64)}
        n = int(n)
        # per-peak stats via the gathered scatter-add (same as peakfinder8), touching only labelled pixels
        lab = lbl.ravel(); pix = xp.nonzero(lab)[0]; rows = lab[pix] - 1
        iv = sub.ravel()[pix]                                        # (I-bg) per peak pixel (already clipped >=0)
        mass = xp.bincount(rows, weights=iv, minlength=n)           # integrated peak signal
        sumvar = xp.bincount(rows, weights=var.ravel()[pix], minlength=n)   # summed background variance
        npix = xp.bincount(rows, minlength=n).astype(xp.float64)
        cm = xp.clip(mass, 1e-12, None)
        cy = xp.bincount(rows, weights=iv * self.yy.ravel()[pix], minlength=n) / cm
        cx = xp.bincount(rows, weights=iv * self.xx.ravel()[pix], minlength=n) / cm
        whole_snr = mass / xp.sqrt(xp.clip(sumvar, 1e-12, None))     # SNR of the integrated peak
        has_max = _scatter_max(xp, rows, ismax.ravel()[pix].astype(xp.float64), n)
        sel = (has_max > 0) & (npix >= p["min_pix"]) & (npix <= p["max_pix"]) \
            & (whole_snr > p["min_snr_whole_peak"])
        return {"x": cx[sel], "y": cy[sel], "intensity": mass[sel], "snr": whole_snr[sel],
                "npix": npix[sel].astype(xp.int64)}

    def find_stream(self, frames):
        """Batch of host frames with the H2D transfer hidden behind compute (pinned + double-buffer), like
        PeakFinder8.find_stream. frames: (N, H, W) host ndarray -> list of per-frame peak dicts."""
        xp = self._xp
        frames = np.asarray(frames)
        if frames.ndim == 2:
            frames = frames[None]
        N, H, W = frames.shape
        if xp is np:
            return [self.find(frames[i]) for i in range(N)]
        import cupy
        dt = self.dt
        hp = cupy.cuda.alloc_pinned_memory(N * H * W * xp.dtype(dt).itemsize)
        host = np.frombuffer(hp, dt, N * H * W).reshape(N, H, W); host[:] = frames
        cs = cupy.cuda.Stream(non_blocking=True); comp = cupy.cuda.get_current_stream()
        gbuf = [cupy.empty((H, W), dt), cupy.empty((H, W), dt)]; ev = [cupy.cuda.Event(), cupy.cuda.Event()]
        with cs:
            gbuf[0].set(host[0]); ev[0].record(cs)
        out = []
        for i in range(N):
            s = i % 2
            if i + 1 < N:
                with cs:
                    gbuf[(i + 1) % 2].set(host[i + 1]); ev[(i + 1) % 2].record(cs)
            comp.wait_event(ev[s]); out.append(self.find(gbuf[s]))
        return out


def peakfinder9(image, mask=None, **kw):
    """One-shot convenience: build a PeakFinder9 (from the image shape / mask) and find peaks in one frame.
    For many frames build ``PeakFinder9`` once and reuse ``.find()`` / ``.find_stream()``."""
    if mask is None:
        mask = (image == image)                 # all-good, same array module as image
    return PeakFinder9(mask, **kw).find(image)
