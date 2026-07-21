"""GPU/CPU adaptive DUAL-THRESHOLD peak finder -- the psana psalgos family (M. Dubrovin, SLAC/LCLS).

This is the *adaptive dual-threshold* cousin of peakfinder9: same adaptive LOCAL background (per-pixel mu/sigma
from a border ring, so no geometry/q needed), but instead of peakfinder9's 4-condition hierarchy it uses a
hysteresis dual threshold -- the region-growing idea behind psana's ``peak_finder_v4r3`` / droplet finders:

  * SEED  a peak where the local SNR clears the HIGH threshold (and the pixel is a local maximum),
  * GROW  it over connected pixels whose SNR clears the LOW threshold,
  * KEEP  components that contain a seed and whose integrated SNR (son = sum(I-bg)/sqrt(sum sigma^2)) and
    pixel count pass the cuts.

Everything reuses the peakfinder8/9 machinery: one fused CUDA kernel does the border-ring mu/sigma + local-max
+ the two thresholds in a single pass (snr, I-bg, sigma^2, grow-flag, seed-flag), then ndimage.label groups the
GROW pixels and the SAME gathered scatter-add reduction gives the per-peak stats. One code path runs on the GPU
(cupy) or CPU (numpy/scipy).

NOTE this is *modelled on* the Dubrovin/psalgos DUAL-THRESHOLD finder ``peaks_droplet`` = ``peak_finder_v4r3``
(a GPU re-implementation of the idea), not a port of the psalgos C++; it agrees with
``psalgos.pypsalgos.peaks_droplet`` on peaks but is not bit-identical. (The adaptive-SNR sibling
``peaks_adaptive`` = ``peak_finder_v3r3`` is matched by peakfinder9.py instead.)
Credit: dual-threshold local-background peak finding -- M. Dubrovin (SLAC/LCLS), psalgos
(github.com/lcls-psana/psalgos), ``peak_finder_v4r3`` / ``peaks_droplet``; local-window lineage shared with
peakfinder9 (Gevorkov/CFEL). Authors of this GPU version: SLAC LCLS DRP team, with Claude (Anthropic).
"""
import os

import numpy as np

__all__ = ["peakfinder_v4", "PeakFinderV4"]


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
    out = xp.zeros(int(n))
    if xp is np:
        np.maximum.at(out, rows, values)
    else:
        import cupyx; cupyx.scatter_max(out, rows, values)
    return out


# Ring-accumulator precision. The ring carries only ~32 samples, so fp64 is far more precision than
# the statistic needs, and fp64 is half rate on A100 (and ~1/32-1/64 on fp32-strong cards) -- which
# matters because peakfind is the largest single stage of the device-resident pipeline.
#
# DO NOT simply change `double` to `float` here. The variance is sq/nn - mu*mu, a difference of two
# O(mu^2) quantities recovering an O(sigma^2) one; at a realistic pedestal (mu~2000 ADU, sigma~20)
# that burns ~4 of fp32's ~7 digits. MEASURED on an A100: a naive fp32 leaves `grow` bit-identical at
# 10 and 500 ADU but flips 1/3/5/7/16 pixels at 2000-8000 ADU, growing with pedestal and frame size --
# i.e. it would pass a synthetic low-background test and then silently perturb peak extents (and so
# centroids) on real detector data, while never looking broken (`seed` is unaffected, being far from
# its threshold).
#
# PFV4_FP32=1 selects the SHIFTED-DATA form instead: subtract a constant K taken from the ring's own
# corner before accumulating, so the sums carry O(sigma) rather than O(mu). Algebraically identical,
# and measured bit-identical to fp64 in `grow` AND `seed` at every pedestal (10/500/2000/8000 ADU) and
# size (1024/2048/4096 square) tested. 1.26x on the kernel; ~5% end-to-end, since the kernel is
# memory-bound (~80 loads/pixel) rather than fp64-ALU-bound -- the load count, not the precision, is
# the remaining lever. Default stays fp64 until it has run on real detector frames.
FP32_RING = os.environ.get("PFV4_FP32", "0") == "1"

_PFV4 = {}
def _pfv4_kernel(fp32=None):
    """One fused fp32 kernel: per pixel, border-ring mu/sigma + (2lmr+1) local-max, then the two SNR
    thresholds -> snr, (I-bg), sigma^2, grow-flag (snr>thr_low), seed-flag (snr>thr_high & local max).

    `fp32` (default: the PFV4_FP32 env knob) switches the ring accumulator to the shifted fp32 form."""
    fp32 = FP32_RING if fp32 is None else bool(fp32)
    if fp32 not in _PFV4:
        import cupy
        acc, val = ("float", "(I[j]-K)") if fp32 else ("double", "I[j]")
        shift = "int ky=max(yy-r,0), kx=max(xx-r,0); float K = I[ky*W+kx];" if fp32 else ""
        unshift = " + K" if fp32 else ""
        _PFV4[fp32] = cupy.RawKernel(r"""
        extern "C" __global__ void pfv4_stats(const float* I, const float* good, int H, int W, int r, int lmr,
            float thr_low, float thr_high, float min_sig,
            float* snr, float* sub, float* var, unsigned char* grow, unsigned char* seed){
          int idx = blockIdx.x*blockDim.x + threadIdx.x; if(idx >= H*W) return;
          int yy = idx / W, xx = idx % W; float Ic = I[idx];
          __ACC__ s=0, sq=0, nn=0;                            // border ring (Chebyshev distance == r)
          __SHIFT__
          for(int dy=-r; dy<=r; ++dy){
            int y=yy+dy; if(y<0||y>=H) continue;
            if(dy==-r || dy==r){
              for(int dx=-r; dx<=r; ++dx){ int x=xx+dx; if(x<0||x>=W) continue;
                int j=y*W+x; __ACC__ g=good[j], v=__VAL__; s+=g*v; sq+=g*v*v; nn+=g; }
            } else {
              int x=xx-r; if(x>=0){ int j=y*W+x; __ACC__ g=good[j], v=__VAL__; s+=g*v; sq+=g*v*v; nn+=g; }
              x=xx+r; if(x<W){ int j=y*W+x; __ACC__ g=good[j], v=__VAL__; s+=g*v; sq+=g*v*v; nn+=g; }
            }
          }
          __ACC__ mbar = nn>0.5 ? s/nn : 0;
          float mu = nn>0.5 ? (float)(mbar__UNSHIFT__) : 0.0f;
          float vv = nn>0.5 ? (float)(sq/nn - mbar*mbar) : 0.0f; if(vv<0.0f) vv=0.0f;
          float sg = sqrtf(vv); int valid = (good[idx]>0.5f) && (nn>0.5) && (sg>min_sig);
          float sb = Ic - mu; float sn = valid ? sb/(sg+1e-12f) : 0.0f;
          int islm = 1;                                        // local maximum over the (2*lmr+1) neighbourhood
          for(int dy=-lmr; dy<=lmr && islm; ++dy){ int y=yy+dy; if(y<0||y>=H) continue;
            for(int dx=-lmr; dx<=lmr; ++dx){ int x=xx+dx; if(x<0||x>=W) continue;
              if(I[y*W+x] > Ic){ islm=0; break; } } }
          snr[idx]=sn; sub[idx]= sb>0.0f? sb:0.0f; var[idx]=vv;
          grow[idx] = (valid && sn > thr_low) ? 1 : 0;
          seed[idx] = (valid && sn > thr_high && islm) ? 1 : 0;
        }""".replace("__ACC__", acc).replace("__VAL__", val)
             .replace("__SHIFT__", shift).replace("__UNSHIFT__", unshift), "pfv4_stats")
    return _PFV4[fp32]


class PeakFinderV4:
    """Adaptive dual-threshold peak finder (psana psalgos family). No geometry/q -- only the image shape (+ mask).

    Parameters (psalgos ``peaks_adaptive`` names in brackets):
      window_radius     [r0/dr]        radius of the local-background ring.
      thr_low           [nsigm]        SNR to GROW a peak (a pixel joins if snr > thr_low).
      thr_high          [-]           SNR to SEED a peak (a component is kept only if it contains snr > thr_high).
      son_min           [son_min]      the integrated peak SNR sum(I-bg)/sqrt(sum sigma^2) must exceed this.
      min_sig, local_max_radius, min_pix, max_pix, dtype.
    """

    def __init__(self, mask=None, *, shape=None, window_radius=4, thr_low=5.0, thr_high=8.0, son_min=8.0,
                 min_sig=0.0, local_max_radius=1, min_pix=1, max_pix=200, dtype=None):
        if mask is not None:
            xp = _xp(mask); good = mask.astype(bool)
        elif shape is not None:
            xp = np; good = np.ones(shape, bool)
        else:
            raise ValueError("pass either mask (2-D bool) or shape=(H, W)")
        self._xp = xp; self._ndi = _ndimage(xp)
        self.dt = xp.float64 if dtype is None else dtype
        self.good = good; self._goodf = good.astype(self.dt)
        self.H, self.W = good.shape
        self.yy, self.xx = (g.astype(xp.float64) for g in xp.mgrid[0:self.H, 0:self.W])
        self.r = int(window_radius)
        self.p = dict(thr_low=thr_low, thr_high=thr_high, son_min=son_min, min_sig=min_sig,
                      local_max_radius=int(local_max_radius), min_pix=min_pix, max_pix=max_pix)
        self._fused_gpu = False
        if xp is not np and self.dt == xp.float32:
            npix = self.H * self.W
            self._pfv4 = _pfv4_kernel(); self._blk = 256; self._grid = (npix + 255) // 256
            self._goodf_flat = xp.ascontiguousarray(self._goodf.ravel())
            self._snr = xp.empty(npix, xp.float32); self._sub = xp.empty(npix, xp.float32)
            self._var = xp.empty(npix, xp.float32)
            self._grow = xp.empty(npix, xp.uint8); self._seed = xp.empty(npix, xp.uint8)
            self._fused_gpu = True

    def _ring_bg(self, I):
        """Portable adaptive local mu/sigma from the border ring (box-filter differences), CPU/fp64 path."""
        xp = self._xp; ndi = self._ndi; r = self.r; goodf = self._goodf
        w, wi = 2 * r + 1, 2 * r - 1
        bs = lambda a, s: ndi.uniform_filter(a, size=s, mode="constant") * float(s * s)
        Ig = I * goodf
        ring_sum = bs(Ig, w) - bs(Ig, wi); ring_sq = bs(Ig * I, w) - bs(Ig * I, wi)
        ring_n = bs(goodf, w) - bs(goodf, wi); nz = ring_n > 0.5
        den = xp.where(nz, ring_n, 1.0); mu = ring_sum / den
        sig = xp.sqrt(xp.clip(ring_sq / den - mu * mu, 0.0, None))
        return mu, sig, nz

    def _stats(self, I):
        """-> snr, sub(=clip(I-bg)), var(=sigma^2), grow (dual-thresh low), seed (dual-thresh high + local max)."""
        xp = self._xp; ndi = self._ndi; p = self.p; H, W = self.H, self.W
        if self._fused_gpu:
            Ic = xp.ascontiguousarray(I.ravel(), dtype=xp.float32)
            self._pfv4((self._grid,), (self._blk,),
                       (Ic, self._goodf_flat, np.int32(H), np.int32(W), np.int32(self.r),
                        np.int32(p["local_max_radius"]), np.float32(p["thr_low"]), np.float32(p["thr_high"]),
                        np.float32(p["min_sig"]), self._snr, self._sub, self._var, self._grow, self._seed))
            return (self._snr.reshape(H, W), self._sub.reshape(H, W), self._var.reshape(H, W),
                    self._grow.reshape(H, W), self._seed.reshape(H, W))
        mu, sig, nz = self._ring_bg(I); sub = I - mu
        valid = self.good & nz & (sig > p["min_sig"])
        snr = xp.where(valid, sub / (sig + 1e-12), 0.0)
        lm = 2 * p["local_max_radius"] + 1
        islm = I >= ndi.maximum_filter(I, size=lm)
        grow = valid & (snr > p["thr_low"])
        seed = valid & (snr > p["thr_high"]) & islm
        return snr, xp.clip(sub, 0.0, None), sig * sig, grow, seed

    def find(self, image):
        xp = self._xp; ndi = self._ndi; H, W = self.H, self.W; p = self.p
        I = image.astype(self.dt)
        snr, sub, var, grow, seed = self._stats(I)
        lbl, n = ndi.label(grow)                                    # connected regions above the LOW threshold
        if n == 0:
            z = xp.zeros(0)
            return {"x": z, "y": z, "intensity": z, "snr": z, "npix": z.astype(xp.int64)}
        n = int(n)
        lab = lbl.ravel(); pix = xp.nonzero(lab)[0]; rows = lab[pix] - 1
        iv = sub.ravel()[pix]
        mass = xp.bincount(rows, weights=iv, minlength=n)
        sumvar = xp.bincount(rows, weights=var.ravel()[pix], minlength=n)
        npix = xp.bincount(rows, minlength=n).astype(xp.float64)
        cm = xp.clip(mass, 1e-12, None)
        cy = xp.bincount(rows, weights=iv * self.yy.ravel()[pix], minlength=n) / cm
        cx = xp.bincount(rows, weights=iv * self.xx.ravel()[pix], minlength=n) / cm
        son = mass / xp.sqrt(xp.clip(sumvar, 1e-12, None))          # integrated peak SNR
        has_seed = _scatter_max(xp, rows, seed.ravel()[pix].astype(xp.float64), n)   # HIGH-threshold hysteresis
        sel = (has_seed > 0) & (npix >= p["min_pix"]) & (npix <= p["max_pix"]) & (son > p["son_min"])
        return {"x": cx[sel], "y": cy[sel], "intensity": mass[sel], "snr": son[sel],
                "npix": npix[sel].astype(xp.int64)}

    def find_stream(self, frames):
        """Batch of host frames with the H2D transfer hidden behind compute (pinned + double-buffer)."""
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


def peakfinder_v4(image, mask=None, **kw):
    """One-shot: build a PeakFinderV4 from the image shape/mask and find peaks in one frame."""
    if mask is None:
        mask = (image == image)
    return PeakFinderV4(mask, **kw).find(image)
