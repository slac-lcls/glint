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


_PFV4 = None
def _pfv4_kernel():
    """One fused fp32 kernel: per pixel, border-ring mu/sigma + (2lmr+1) local-max, then the two SNR
    thresholds -> snr, (I-bg), sigma^2, grow-flag (snr>thr_low), seed-flag (snr>thr_high & local max)."""
    global _PFV4
    if _PFV4 is None:
        import cupy
        _PFV4 = cupy.RawKernel(r"""
        extern "C" __global__ void pfv4_stats(const float* I, const float* good, int H, int W, int r, int lmr,
            float thr_low, float thr_high, float min_sig,
            float* snr, float* sub, float* var, unsigned char* grow, unsigned char* seed){
          int idx = blockIdx.x*blockDim.x + threadIdx.x; if(idx >= H*W) return;
          int yy = idx / W, xx = idx % W; float Ic = I[idx];
          double s=0.0, sq=0.0, nn=0.0;                       // border ring (Chebyshev distance == r)
          for(int dy=-r; dy<=r; ++dy){
            int y=yy+dy; if(y<0||y>=H) continue;
            if(dy==-r || dy==r){
              for(int dx=-r; dx<=r; ++dx){ int x=xx+dx; if(x<0||x>=W) continue;
                int j=y*W+x; double g=good[j], v=I[j]; s+=g*v; sq+=g*v*v; nn+=g; }
            } else {
              int x=xx-r; if(x>=0){ int j=y*W+x; double g=good[j], v=I[j]; s+=g*v; sq+=g*v*v; nn+=g; }
              x=xx+r; if(x<W){ int j=y*W+x; double g=good[j], v=I[j]; s+=g*v; sq+=g*v*v; nn+=g; }
            }
          }
          float mu = nn>0.5 ? (float)(s/nn) : 0.0f;
          float vv = nn>0.5 ? (float)(sq/nn - (double)mu*mu) : 0.0f; if(vv<0.0f) vv=0.0f;
          float sg = sqrtf(vv); int valid = (good[idx]>0.5f) && (nn>0.5) && (sg>min_sig);
          float sb = Ic - mu; float sn = valid ? sb/(sg+1e-12f) : 0.0f;
          int islm = 1;                                        // local maximum over the (2*lmr+1) neighbourhood
          for(int dy=-lmr; dy<=lmr && islm; ++dy){ int y=yy+dy; if(y<0||y>=H) continue;
            for(int dx=-lmr; dx<=lmr; ++dx){ int x=xx+dx; if(x<0||x>=W) continue;
              if(I[y*W+x] > Ic){ islm=0; break; } } }
          snr[idx]=sn; sub[idx]= sb>0.0f? sb:0.0f; var[idx]=vv;
          grow[idx] = (valid && sn > thr_low) ? 1 : 0;
          seed[idx] = (valid && sn > thr_high && islm) ? 1 : 0;
        }""", "pfv4_stats")
    return _PFV4


_REDUCE = None
def _reduce_kernel():
    """One pass over the label image, replacing nonzero + 5 bincount + scatter_max + 5 gathers.

    The eager chain walked the same `rows` six times, each pass its own kernel, after compacting the
    label image with nonzero() and gathering sub/var/seed/yy/xx through it. This does all of it in a
    single pass: ~99% of pixels read their 4-byte label and leave, only the survivors do atomics.

    Two things fall out of working per pixel rather than per compacted row. The centroid coordinates
    are just idx/W and idx%W, so the two full-frame float64 mgrid arrays -- 134 MB EACH at 4096^2 --
    are neither gathered nor allocated. And sub/var/seed are read at their natural index, so those
    gathers disappear as well.

    Measured bit-identical to the bincount chain (every find() field, exactly zero deviation) and
    reproducible run to run, even though atomics fix no summation order. That is not luck: `sub` and
    `var` are float32, hence exact in float64, and idx/W is a small integer, so every accumuland is
    exact and each label's sum needs ~24 + log2(max npix) + the intensity dynamic range in bits, well
    inside float64's 53. Order-independence follows from exactness. It would stop holding for a peak
    whose partial sums span more than ~2^29 in magnitude, which max_pix=200 makes unreachable here."""
    global _REDUCE
    if _REDUCE is None:
        import cupy
        _REDUCE = cupy.RawKernel(r"""
        extern "C" __global__ void pfv4_reduce(const int* lab, const float* sub, const float* var,
            const unsigned char* seed, int H, int W,
            double* mass, double* sumvar, double* cyn, double* cxn, int* npix, unsigned char* hasseed){
          int idx = blockIdx.x*blockDim.x + threadIdx.x; if(idx >= H*W) return;
          int L = lab[idx]; if(L == 0) return;        // the ~99% leave here, having read 4 bytes
          int rr = L - 1;
          double iv = (double)sub[idx];
          atomicAdd(&mass[rr], iv);
          atomicAdd(&sumvar[rr], (double)var[idx]);
          atomicAdd(&cyn[rr], iv * (double)(idx / W));
          atomicAdd(&cxn[rr], iv * (double)(idx % W));
          atomicAdd(&npix[rr], 1);
          if(seed[idx]) hasseed[rr] = 1;              // racing writers all store the same 1
        }""", "pfv4_reduce")
    return _REDUCE


_CCL = None
def _ccl_module():
    """Sync-free 4-connectivity connected-components labeling (atomic union-find, Playne-Komura).

    Replaces ndi.label(grow) + the int(n) per-frame D2H sync + the ``if n == 0`` host branch. Five
    fixed-schedule kernels over grid=(HW+255)//256, blk=256 (data-independent launch => no host
    readback of the component count anywhere on the hot path):

      K1 init    : par[p] = p if grow[p] else -1                 (self-root; -1 = background)
      K2 merge   : unite right (p+1) and down (p+W) ONLY         (4-CONN -- NO diagonal p+W+-1)
      K3 flatten : par[p] = findr(par,p)                          (every fg pixel -> its root)
      K4 claim   : root pixels take a DENSE id via one global atomicAdd(counter) -> lab[root]=1..K
      K5 write   : lab[p] = grow[p] ? lab[par[p]] : 0            (0=bg, dense 1..K -> feeds pfv4_reduce)

    Dense labels are 1..K int32 exactly (0=bg), so pfv4_reduce's rr=L-1 indexes the size-C reduce
    arrays in range. Overflow (K > C) is flagged in overflow[0] and the pixel dropped (lab=0); the
    flag is read ONCE at end of run, never per frame. Dense numbering is atomic-order nondeterministic
    run-to-run -- CORRECT: the invariant is the PARTITION, not the numbering (the oracle sort neutralizes it)."""
    global _CCL
    if _CCL is None:
        import cupy
        _CCL = cupy.RawModule(code=r"""
        extern "C" {

        // read-only find: NO writes -> safe under concurrent merge. Terminates because union-by-min
        // keeps every edge pointing to a strictly smaller id (par[non-root] < id), so p decreases each hop.
        __device__ int ccl_find_ro(const int* par, int p){ while(par[p]!=p) p=par[p]; return p; }

        __device__ void ccl_unite(int* par, int a, int b){    // textbook Komura union-by-min, read-only find
          while(true){ a=ccl_find_ro(par,a); b=ccl_find_ro(par,b); if(a==b) return;
            int hi=a>b?a:b, lo=a<b?a:b;
            if(atomicCAS(&par[hi],hi,lo)==hi) return; } }      // attach larger root under smaller; spin on failure

        __global__ void ccl_init(const unsigned char* grow, int* par, int N){
          int p = blockIdx.x*blockDim.x + threadIdx.x; if(p>=N) return;
          par[p] = grow[p] ? p : -1; }

        __global__ void ccl_merge(const unsigned char* grow, int* par, int H, int W){
          int p = blockIdx.x*blockDim.x + threadIdx.x; int N=H*W; if(p>=N || !grow[p]) return;
          int y=p/W, x=p%W;
          if(x+1<W && grow[p+1]) ccl_unite(par,p,p+1);         // right
          if(y+1<H && grow[p+W]) ccl_unite(par,p,p+W); }       // down  (4-CONN: right+down only)

        // flatten: read-only find, then write ONLY this pixel's own par[p]. Write targets are disjoint
        // across threads (each writes par[p_orig]) => no clobbering. (A compressing find that writes
        // par of OTHER pixels races here: a neighbour's path-halving overwrites this pixel's finalized
        // root with a non-root intermediate, and flatten has no retry loop to repair it.)
        __global__ void ccl_flatten(const unsigned char* grow, int* par, int N){
          int p = blockIdx.x*blockDim.x + threadIdx.x; if(p>=N || !grow[p]) return;
          par[p] = ccl_find_ro(par,p); }

        __global__ void ccl_claim(const unsigned char* grow, const int* par, int* lab,
            int* counter, int* overflow, int N, int C){
          int p = blockIdx.x*blockDim.x + threadIdx.x; if(p>=N || !grow[p]) return;
          if(par[p]==p){                                       // roots only
            int d = atomicAdd(counter,1) + 1;                  // dense id 1..K
            if(d > C){ atomicExch(overflow,1); lab[p]=0; }     // overflow: drop + flag (no per-frame read)
            else lab[p]=d; } }                                 // stash dense id AT the root pixel

        __global__ void ccl_write(const unsigned char* grow, const int* par, int* lab, int N){
          int p = blockIdx.x*blockDim.x + threadIdx.x; if(p>=N) return;
          lab[p] = grow[p] ? lab[par[p]] : 0; }                // 0=bg, dense 1..K -> pfv4_reduce unchanged

        }
        """, options=("--std=c++11",))
    return _CCL


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
        self.r = int(window_radius)
        self.p = dict(thr_low=thr_low, thr_high=thr_high, son_min=son_min, min_sig=min_sig,
                      local_max_radius=int(local_max_radius), min_pix=min_pix, max_pix=max_pix)
        self._fused_gpu = False
        # Stage-1 sync-free CCL: env selects the default path; find(..., _force_ccl=) overrides per call
        # (mandatory for single-process A/B on the SAME finder + frame). C = fixed reduce capacity.
        self._use_ccl = os.environ.get("GLINT_PF_CCL") == "1"
        self._ccl_C = 65536
        if xp is not np and self.dt == xp.float32:
            npix = self.H * self.W
            self._pfv4 = _pfv4_kernel(); self._blk = 256; self._grid = (npix + 255) // 256
            self._reduce = _reduce_kernel()
            self._goodf_flat = xp.ascontiguousarray(self._goodf.ravel())
            self._snr = xp.empty(npix, xp.float32); self._sub = xp.empty(npix, xp.float32)
            self._var = xp.empty(npix, xp.float32)
            self._grow = xp.empty(npix, xp.uint8); self._seed = xp.empty(npix, xp.uint8)
            self._fused_gpu = True
            # CCL device state (all device-resident; nothing here is read back on the hot path)
            mod = _ccl_module()
            self._ccl_init = mod.get_function("ccl_init")
            self._ccl_merge = mod.get_function("ccl_merge")
            self._ccl_flatten = mod.get_function("ccl_flatten")
            self._ccl_claim = mod.get_function("ccl_claim")
            self._ccl_write = mod.get_function("ccl_write")
            self._ccl_par = xp.empty(npix, xp.int32)
            self._ccl_lab = xp.empty(npix, xp.int32)
            self._ccl_counter = xp.zeros(1, xp.int32)
            self._ccl_overflow = xp.zeros(1, xp.int32)   # reset once here; read once at end of run
        # Only the eager reduction gathers centroids through coordinate arrays; the fused kernel
        # derives them from the pixel index. Two float64 frames is 268 MB at 4096^2, so do not
        # allocate them where nothing reads them.
        if not self._fused_gpu:
            self.yy, self.xx = (g.astype(xp.float64) for g in xp.mgrid[0:self.H, 0:self.W])

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

    def find(self, image, _force_ccl=None):
        xp = self._xp; ndi = self._ndi; H, W = self.H, self.W; p = self.p
        I = image.astype(self.dt)
        snr, sub, var, grow, seed = self._stats(I)
        use_ccl = self._use_ccl if _force_ccl is None else _force_ccl
        if self._fused_gpu and use_ccl:
            # --- Stage-1 SYNC-FREE PATH: GPU CCL at fixed capacity C, no host readback of the count ---
            N = H * W; C = self._ccl_C
            grow_flat = grow.ravel()                               # C-contiguous view of self._grow (no copy)
            par = self._ccl_par; lab = self._ccl_lab
            self._ccl_counter[...] = 0                             # device fill on compute stream (no host stall)
            self._ccl_init((self._grid,), (self._blk,), (grow_flat, par, np.int32(N)))
            self._ccl_merge((self._grid,), (self._blk,), (grow_flat, par, np.int32(H), np.int32(W)))
            self._ccl_flatten((self._grid,), (self._blk,), (grow_flat, par, np.int32(N)))
            self._ccl_claim((self._grid,), (self._blk,),
                            (grow_flat, par, lab, self._ccl_counter, self._ccl_overflow,
                             np.int32(N), np.int32(C)))
            self._ccl_write((self._grid,), (self._blk,), (grow_flat, par, lab, np.int32(N)))
            mass = xp.zeros(C, xp.float64); sumvar = xp.zeros(C, xp.float64)
            cyn = xp.zeros(C, xp.float64);  cxn = xp.zeros(C, xp.float64)
            npix_i = xp.zeros(C, xp.int32); hs = xp.zeros(C, xp.uint8)
            self._reduce((self._grid,), (self._blk,),
                         (lab, sub.ravel(), var.ravel(), seed.ravel(), np.int32(H), np.int32(W),
                          mass, sumvar, cyn, cxn, npix_i, hs))            # lab already int32 -> no cast/copy
            npix = npix_i.astype(xp.float64)
            cm = xp.clip(mass, 1e-12, None)
            cy, cx = cyn / cm, cxn / cm
            has_seed = hs
            son = mass / xp.sqrt(xp.clip(sumvar, 1e-12, None))
            sel = (has_seed > 0) & (npix >= p["min_pix"]) & (npix <= p["max_pix"]) & (son > p["son_min"])
            return {"x": cx[sel], "y": cy[sel], "intensity": mass[sel], "snr": son[sel],
                    "npix": npix[sel].astype(xp.int64)}
        lbl, n = ndi.label(grow)                                    # connected regions above the LOW threshold
        if n == 0:
            z = xp.zeros(0)
            return {"x": z, "y": z, "intensity": z, "snr": z, "npix": z.astype(xp.int64)}
        n = int(n)
        if self._fused_gpu:
            mass = xp.zeros(n, xp.float64); sumvar = xp.zeros(n, xp.float64)
            cyn = xp.zeros(n, xp.float64);  cxn = xp.zeros(n, xp.float64)
            npix_i = xp.zeros(n, xp.int32); hs = xp.zeros(n, xp.uint8)
            self._reduce((self._grid,), (self._blk,),
                         (xp.ascontiguousarray(lbl.ravel(), dtype=xp.int32),
                          sub.ravel(), var.ravel(), seed.ravel(), np.int32(H), np.int32(W),
                          mass, sumvar, cyn, cxn, npix_i, hs))
            npix = npix_i.astype(xp.float64)
            cm = xp.clip(mass, 1e-12, None)
            cy, cx = cyn / cm, cxn / cm
            has_seed = hs
        else:
            lab = lbl.ravel(); pix = xp.nonzero(lab)[0]; rows = lab[pix] - 1
            iv = sub.ravel()[pix]
            mass = xp.bincount(rows, weights=iv, minlength=n)
            sumvar = xp.bincount(rows, weights=var.ravel()[pix], minlength=n)
            npix = xp.bincount(rows, minlength=n).astype(xp.float64)
            cm = xp.clip(mass, 1e-12, None)
            cy = xp.bincount(rows, weights=iv * self.yy.ravel()[pix], minlength=n) / cm
            cx = xp.bincount(rows, weights=iv * self.xx.ravel()[pix], minlength=n) / cm
            has_seed = _scatter_max(xp, rows, seed.ravel()[pix].astype(xp.float64), n)  # HIGH-thresh hysteresis
        son = mass / xp.sqrt(xp.clip(sumvar, 1e-12, None))          # integrated peak SNR
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
