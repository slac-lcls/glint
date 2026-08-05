"""GPU peakfinder8-style Bragg peak finder (cupy, numpy fallback), built on the radial primitive.

peakfinder8 (A. Barty et al., "Cheetah", J. Appl. Cryst. 47, 1118-1131, 2014; also CrystFEL) finds Bragg
peaks by a per-ring SNR test: model a radial background -- mean mu(r) and std sigma(r) per resolution ring
-- flag pixels with (I - mu)/sigma above a threshold, and group connected flagged pixels into peaks. The
radial background IS the radial primitive: with the sparse CSR operator M (radial.py),

    mu    = (M @ I)   / (M @ 1)
    sigma = sqrt( (M @ I^2)/(M @ 1) - mu^2 )        # two matvecs, same M

and the ring stats are interpolated back onto pixels with M^T (M^T @ mu). The ring statistics are recomputed
with the found peaks excluded (iterative), as in peakfinder8.

Both reductions are accumarrays, but of two different shapes:
  * radial background: the operator M is FIXED for the geometry, built once, reused every frame, over ALL
    pixels and many rings -- a precomputed CSR matvec (radial.py) is exactly right.
  * per-peak stats: the groups (connected-component labels) change every frame and cover only a sparse
    handful of pixels, so the fast form is a GATHERED scatter-add -- gather the labelled pixels once, then
    ``bincount`` the sums and ``scatter_max`` the peak SNR. It never builds a per-frame matrix and never
    materialises a full-image weight array. Measured on an A100 (weight formation included), flat ~1 ms
    from 512^2 to a real 3000^2 detector and 16..2000 peaks, vs a sparse-SpMM form 4 ms (dragged down by
    the 288 MB dense RHS it must form) and ndimage.sum_labels/maximum 3..93 ms (a scale trap -- it only
    looks good on a tiny toy image). The one step that is genuinely not a reduction, connected-component
    labelling, stays ndimage.label. One code path runs on the GPU (cupy) or CPU (numpy/scipy).

Performance: ~5 ms/frame on a real 3000^2 CSPAD frame in fp32 (A100), ~2x faster than pyFAI's OpenCL
OCL_PeakFinder (~10.5 ms) at the same recovered peaks, and ~1.9x throughput again via find_stream.

Credit: peakfinder8 algorithm -- A. Barty et al., "Cheetah", J. Appl. Cryst. 47, 1118-1131 (2014); also
CrystFEL (T. A. White et al.). The accumarray idea (build a sparse group-reduce, or scatter-add to groups)
follows S. Marchesini's 2013 GPU code (MATLAB Central #44423, SLAC LCLS DRP team's testing2.py), used
directly as the radial-background operator via radial.py. Reference and benchmark: pyFAI's OpenCL
OCL_PeakFinder / sparsify-Bragg (Kieffer et al., J. Appl. Cryst. 48, 510, 2015). Authors: SLAC LCLS DRP
team, with Claude (Anthropic).
"""
import numpy as np

try:                                    # vendored into the glint package
    from .radial import RadialIntegrator
except ImportError:                     # or loaded by file path, bypassing the package
    from radial import RadialIntegrator

__all__ = ["peakfinder8", "PeakFinder8"]


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


_FWD = None
def _fwd_kernel():
    """Custom fp32 FORWARD (replaces 3 cuSPARSE csrmv + 2 elementwise, ~1.9x). Reads M once in the
    per-pixel split form, fuses I*keep / I^2*keep, and accumulates d0,s1,s2 by a pixel-parallel SCATTER.
    Global atomics on the ~1500 hot ring bins are too contended (3x slower than cuSPARSE), so privatise:
    each block sums into fp32 SHARED bins (fast shared atomics), then flushes ONCE to the fp64 global
    accumulators -- fp64 output, and MORE precise than an fp32 cuSPARSE sum (block partials touch ~10
    pixels/ring, not the whole 30k-pixel ring). Needs 3*nring*4 bytes of shared."""
    global _FWD
    if _FWD is None:
        import cupy
        _FWD = cupy.RawKernel(r"""
        extern "C" __global__ void pf_fwd3(const int* r0, const int* r1, const float* w0, const float* w1,
            const float* I, const float* keep, int npix, int nring, double* d0, double* s1, double* s2){
          extern __shared__ float sh[]; float* sd0=sh; float* ss1=sh+nring; float* ss2=sh+2*nring;
          for(int i=threadIdx.x;i<nring;i+=blockDim.x){sd0[i]=0.f;ss1[i]=0.f;ss2[i]=0.f;} __syncthreads();
          for(int p=blockIdx.x*blockDim.x+threadIdx.x;p<npix;p+=blockDim.x*gridDim.x){
            float kp=keep[p]; if(kp==0.f) continue;
            float ip=I[p]; float v0=w0[p]*kp, v1=w1[p]*kp; int a=r0[p], b=r1[p];
            atomicAdd(&sd0[a],v0);atomicAdd(&ss1[a],v0*ip);atomicAdd(&ss2[a],v0*ip*ip);
            atomicAdd(&sd0[b],v1);atomicAdd(&ss1[b],v1*ip);atomicAdd(&ss2[b],v1*ip*ip);
          } __syncthreads();
          for(int i=threadIdx.x;i<nring;i+=blockDim.x){
            if(sd0[i]!=0.f)atomicAdd(&d0[i],(double)sd0[i]);
            if(ss1[i]!=0.f)atomicAdd(&s1[i],(double)ss1[i]);
            if(ss2[i]!=0.f)atomicAdd(&s2[i],(double)ss2[i]);
          }}""", "pf_fwd3")
    return _FWD


_RING = None
def _ring_kernel():
    """Fold the per-ring combine den/mu/e2/sig into ONE ElementwiseKernel (fp64 sums in -> mu,sig in the
    loop dtype T out): mu = s1/d0, sig = sqrt(max(s2/d0 - mu^2, 0)), guarding d0<=0. Replaces ~7 elementwise
    launches per iteration; the fp64 math keeps the variance safe in the fp32 path."""
    global _RING
    if _RING is None:
        import cupy
        _RING = cupy.ElementwiseKernel(
            "float64 d0, float64 s1, float64 s2", "T mu, T sig",
            "double den = d0 > 0.0 ? d0 : 1.0;"
            "double m = d0 > 0.0 ? s1 / den : 0.0;"
            "double e2 = d0 > 0.0 ? s2 / den : 0.0;"
            "double v = e2 - m * m; if (v < 0.0) v = 0.0;"
            "mu = (T)m; sig = (T)sqrt(v);", "pf_ring")
    return _RING


_BWD = None
def _bwd_kernel():
    """One fused ElementwiseKernel for the whole peakfinder8 BACKWARD half: with the linear split each
    pixel maps to <=2 rings (r0,r1,w0,w1), so bg=M.T@mu is a 2-ring gather -- fuse bg, sg, snr, threshold
    and the keep-update into a single pixel kernel (replaces 2 SpMV + ~4 full-image passes; ~1.7x on the
    loop, bit-identical). Generic dtype ``T`` so it serves both fp64 and fp32."""
    global _BWD
    if _BWD is None:
        import cupy
        _BWD = cupy.ElementwiseKernel(
            "int32 r0, int32 r1, T w0, T w1, T img, uint8 good, raw T mu, raw T sig, T thr",
            "T snr, T keep, T bg",
            "T b = w0 * mu[r0] + w1 * mu[r1];"
            "T sg = w0 * sig[r0] + w1 * sig[r1];"
            "T s = good ? (img - b) / (sg + (T)1e-6) : (T)0;"
            "snr = s; bg = b; keep = (good && !(s > thr)) ? (T)1 : (T)0;",
            "pf_bwd")
    return _BWD


class PeakFinder8:
    """Build the sparse radial operator ``M`` ONCE for a geometry, then call ``.find(image)`` per frame.
    ``M`` (and the pixel grids) depend only on the geometry, so the per-frame cost is dominated by the
    iterative radial-background matvecs (``M @ .`` forward, ``M.T @ .`` back). On a real 3000^2 CSPAD frame
    (A100): ~9-13 ms/frame, of which the background loop is ~9 ms and the label+reduction ~1.5 ms.

    Two measured tuning notes baked in here: (1) the three forward stats per iteration are three separate
    SpMVs, NOT one SpMM over a stacked RHS -- cupyx ``csrmm`` is ~2.7x SLOWER than looping ``csrmv`` for
    this tall-skinny operator; (2) ``dtype=cupy.float32`` runs the loop in fp32, ~1.35x faster (13.3->9.9 ms)
    and validated to give BIT-IDENTICAL peaks on real CSPAD data (default fp64 for safety on high-count
    detectors, where the E[I^2]-E[I]^2 variance can lose fp32 precision). Parameters as in ``peakfinder8``.
    """

    def __init__(self, q_per_pixel, mask=None, *, nbin=None, thr_snr=5.0, min_snr=5.0,
                 min_pix=2, max_pix=200, r_min=0.0, n_iter=3, dtype=None, graph=False, smooth=0):
        xp = _xp(q_per_pixel); self._xp = xp; self._ndi = _ndimage(xp)
        self.dt = xp.float64 if dtype is None else dtype   # fp32 ~25% faster background loop (verify recall)
        self._smooth = int(smooth)                          # opt-in (>1): uniform_filter1d the radial mu/sig
        # profile each iter -- denoises the background where rings are UNDER-SAMPLED/noisy (few px/ring). No
        # recall gain on well-sampled synthetic frames in our tests (misses are CC-merge/edge, not bg noise),
        # and a wide window over a steep radial falloff adds false peaks -- try it on real noisy backgrounds.
        # Runs eager (disables the CUDA-graph fast path).
        self.H, self.W = q_per_pixel.shape
        q = q_per_pixel.astype(xp.float64)
        self.good = (xp.ones((self.H, self.W), bool) if mask is None else mask.astype(bool)) & (q >= r_min)
        self.M = RadialIntegrator(q, nbin=nbin, mask=self.good).M.astype(self.dt)   # built ONCE
        self._good_f = self.good.ravel().astype(self.dt)
        self.yy, self.xx = (g.astype(xp.float64) for g in xp.mgrid[0:self.H, 0:self.W])  # fp64 for centroids
        self.p = dict(thr_snr=thr_snr, min_snr=min_snr, min_pix=min_pix, max_pix=max_pix, n_iter=n_iter)
        self._fused = False                                # GPU-only fused backward (linear split <=2 rings/px)
        if xp is not np:
            csc = self.M.tocsc(); ip = csc.indptr; ind = csc.indices; dat = csc.data
            cnt = xp.diff(ip)
            if int(cnt.max()) <= 2:                        # each pixel hits <=2 rings -> 2-ring gather is exact
                npix = self.M.shape[1]; base = ip[:-1]; h0 = cnt >= 1; h1 = cnt >= 2
                self._r0 = xp.zeros(npix, xp.int32); self._r1 = xp.zeros(npix, xp.int32)
                self._w0 = xp.zeros(npix, self.dt); self._w1 = xp.zeros(npix, self.dt)
                self._r0[h0] = ind[base[h0]].astype(xp.int32); self._w0[h0] = dat[base[h0]]
                self._r1[h1] = ind[(base + 1)[h1]].astype(xp.int32); self._w1[h1] = dat[(base + 1)[h1]]
                self._good_u8 = self.good.ravel().astype(xp.uint8)
                self._bwd = _bwd_kernel(); self._ring = _ring_kernel(); self._fused = True
                self._mu = xp.empty(self.M.shape[0], self.dt)    # ring-combine out buffers (T inferred from these)
                self._sig = xp.empty(self.M.shape[0], self.dt)
        # custom fp32 forward (shared-privatised scatter, ~1.9x): only when fp32 + shared fits (3*nring*4 B)
        self._fwd_fast = False
        if self._fused and self.dt == xp.float32 and 3 * self.M.shape[0] * 4 <= 45000:
            self._nring = self.M.shape[0]; self._npix = self.M.shape[1]
            self._fwd = _fwd_kernel(); self._fwd_grid = 512; self._fwd_blk = 256
            self._fwd_sh = 3 * self._nring * 4
            self._d0 = xp.zeros(self._nring); self._s1 = xp.zeros(self._nring); self._s2 = xp.zeros(self._nring)
            self._fwd_fast = True
        # opt-in CUDA graph of the (now cuSPARSE-free) fp32 background loop -- kills per-kernel launch
        # overhead. Fixed ~0.03 ms/frame, so it pays for SMALL frames/panels (1.3x at 256^2) and is
        # negligible at 3000^2 (1.02x). Needs the fp32 fast path + preallocated buffers (captured once).
        self._graph_want = False; self._graph = None
        if graph and self._fwd_fast and self._smooth <= 1:      # smoothing runs eager (not in the captured loop)
            self._Ibuf = xp.zeros(self._npix, self.dt); self._keepb = xp.empty(self._npix, self.dt)
            self._snr = xp.empty(self._npix, self.dt); self._bg = xp.empty(self._npix, self.dt)
            self._graph_want = True                        # (_mu/_sig ring buffers allocated above)

    def _bg_eager(self, image):
        """Iterate the radial background; return (snr[H,W], bg[npix]). Forward = fp32 scatter kernel or 3
        cuSPARSE csrmv; ring combine + backward = one fused kernel each on GPU, elementwise + M.T on CPU."""
        xp = self._xp; ndi = self._ndi; M = self.M; H, W = self.H, self.W; p = self.p; thr = p["thr_snr"]; f64 = xp.float64
        I = image.astype(self.dt); Iflat = I.ravel(); keep = self._good_f.copy(); snr = bg = snrf = None
        for _ in range(p["n_iter"]):
            if self._fwd_fast:
                d0, s1, s2 = self._d0, self._s1, self._s2
                d0.fill(0); s1.fill(0); s2.fill(0)
                self._fwd((self._fwd_grid,), (self._fwd_blk,),
                          (self._r0, self._r1, self._w0, self._w1, Iflat, keep,
                           np.int32(self._npix), np.int32(self._nring), d0, s1, s2), shared_mem=self._fwd_sh)
            else:
                d0 = (M @ keep).astype(f64)
                s1 = (M @ (Iflat * keep)).astype(f64)
                s2 = (M @ (Iflat * Iflat * keep)).astype(f64)
            if self._fused:
                self._ring(d0, s1, s2, self._mu, self._sig)   # out-args (T inferred from mu/sig)
                mu, sig = self._mu, self._sig
                if self._smooth > 1:                          # denoise the 1-D radial background profile
                    mu = ndi.uniform_filter1d(mu, self._smooth, mode="nearest")
                    sig = ndi.uniform_filter1d(sig, self._smooth, mode="nearest")
                snrf, keep, bg = self._bwd(self._r0, self._r1, self._w0, self._w1, Iflat,
                                           self._good_u8, mu, sig, thr)
            else:
                den = xp.where(d0 <= 0, xp.nan, d0)
                mu = xp.nan_to_num(s1 / den); e2 = xp.nan_to_num(s2 / den)
                sig = xp.sqrt(xp.clip(e2 - mu * mu, 0.0, None)).astype(self.dt); mu = mu.astype(self.dt)
                if self._smooth > 1:
                    mu = ndi.uniform_filter1d(mu, self._smooth, mode="nearest")
                    sig = ndi.uniform_filter1d(sig, self._smooth, mode="nearest")
                bgw = (M.T @ mu).reshape(H, W); sg = (M.T @ sig).reshape(H, W)
                snr = xp.where(self.good, (I - bgw) / (sg + 1e-6), 0.0); bg = bgw.ravel()
                keep = (self.good.ravel() & ~(snr.ravel() > thr)).astype(self.dt)
        return (snrf.reshape(H, W) if self._fused else snr), bg

    def _loop_pre(self):
        """The fp32 background loop over PREALLOCATED buffers (out-args) -- the capturable form."""
        thr = self.p["thr_snr"]; self._keepb[:] = self._good_f
        for _ in range(self.p["n_iter"]):
            d0, s1, s2 = self._d0, self._s1, self._s2
            d0.fill(0); s1.fill(0); s2.fill(0)
            self._fwd((self._fwd_grid,), (self._fwd_blk,),
                      (self._r0, self._r1, self._w0, self._w1, self._Ibuf, self._keepb,
                       np.int32(self._npix), np.int32(self._nring), d0, s1, s2), shared_mem=self._fwd_sh)
            self._ring(d0, s1, s2, self._mu, self._sig)
            self._bwd(self._r0, self._r1, self._w0, self._w1, self._Ibuf, self._good_u8,
                      self._mu, self._sig, thr, self._snr, self._keepb, self._bg)

    def _bg_graph(self, image):
        """Same loop via a CUDA graph captured once (cuSPARSE-free, so capturable): copy the frame into the
        fixed input buffer, replay. Returns (snr[H,W], bg[npix]) as views of the persistent buffers."""
        import cupy
        self._Ibuf[:] = image.astype(self.dt).ravel()
        if self._graph is None:
            st = cupy.cuda.Stream(non_blocking=True)
            with st:
                self._loop_pre(); st.synchronize()
                st.begin_capture(); self._loop_pre(); self._graph = st.end_capture()
            self._gstream = st
        self._graph.launch(self._gstream); self._gstream.synchronize()
        return self._snr.reshape(self.H, self.W), self._bg

    def find(self, image):
        xp = self._xp; ndi = self._ndi; H, W = self.H, self.W; p = self.p
        thr = p["thr_snr"]; f64 = xp.float64
        snr, bg = self._bg_graph(image) if self._graph_want else self._bg_eager(image)
        cand = snr > thr
        isub = xp.clip(image.astype(f64) - bg.reshape(H, W).astype(f64), 0.0, None)  # fp64 for centroids
        lbl, n = ndi.label(cand)                                    # connected-component labelling (not a reduce)
        if n == 0:
            z = xp.zeros(0)
            return {"x": z, "y": z, "intensity": z, "snr": z, "npix": z.astype(xp.int64)}
        n = int(n)
        # per-peak stats: GATHERED accumarray -- touch only the labelled pixels (flat ~1 ms at any detector
        # size; a full-image ndimage/SpMM reduce is 50-90x slower at real 3000^2 scale -- see module doc)
        lab = lbl.ravel(); pix = xp.nonzero(lab)[0]; rows = lab[pix] - 1
        iv = isub.ravel()[pix]
        inten = xp.bincount(rows, weights=iv, minlength=n)
        ci = xp.clip(inten, 1e-12, None)
        npix = xp.bincount(rows, minlength=n).astype(xp.float64)
        cy = xp.bincount(rows, weights=iv * self.yy.ravel()[pix], minlength=n) / ci
        cx = xp.bincount(rows, weights=iv * self.xx.ravel()[pix], minlength=n) / ci
        maxsnr = _scatter_max(xp, rows, snr.ravel()[pix], n)        # per-peak max SNR (not a linear reduce)
        sel = (npix >= p["min_pix"]) & (npix <= p["max_pix"]) & (maxsnr >= p["min_snr"])
        return {"x": cx[sel], "y": cy[sel], "intensity": inten[sel], "snr": maxsnr[sel], "npix": npix[sel].astype(xp.int64)}

    def find_stream(self, frames):
        """Process a batch of HOST frames, hiding each frame's host->device transfer behind the previous
        frame's compute. ``frames``: (N, H, W) host ndarray. Returns a list of the per-frame peak dicts.

        The transfer is a real cost that single-frame timing hides: a 36 MB frame is ~5.2 ms pageable /
        ~1.9 ms pinned, vs ~5 ms compute. This pins the frames once (fast async H2D) and double-buffers on
        a copy stream -- frame i+1 copies while frame i computes -- so throughput jumps ~1.9x (e.g. 92 ->
        178 f/s at 3000^2 on an A100) by overlapping the (pinned) transfer with compute. CPU falls back to
        a plain loop. NOTE: pins the whole batch, so for very large N (or an unbounded live stream) feed
        frames that already live in a pinned ring buffer instead and reuse ``.find()`` with your own
        double-buffering; here N should be a manageable chunk.
        """
        xp = self._xp
        frames = np.asarray(frames)
        if frames.ndim == 2:
            frames = frames[None]
        N, H, W = frames.shape
        if xp is np:
            return [self.find(frames[i]) for i in range(N)]
        import cupy
        dt = self.dt
        hp = cupy.cuda.alloc_pinned_memory(N * H * W * xp.dtype(dt).itemsize)   # pin once (amortised)
        host = np.frombuffer(hp, dt, N * H * W).reshape(N, H, W); host[:] = frames
        cs = cupy.cuda.Stream(non_blocking=True)                # copy stream (overlaps the compute stream)
        comp = cupy.cuda.get_current_stream()
        gbuf = [cupy.empty((H, W), dt), cupy.empty((H, W), dt)] # 2 device buffers (double-buffer)
        ev = [cupy.cuda.Event(), cupy.cuda.Event()]
        with cs:
            gbuf[0].set(host[0]); ev[0].record(cs)              # prime frame 0
        out = []
        for i in range(N):
            s = i % 2
            if i + 1 < N:                                       # kick async H2D of i+1 (overlaps compute of i)
                with cs:
                    gbuf[(i + 1) % 2].set(host[i + 1]); ev[(i + 1) % 2].record(cs)
            comp.wait_event(ev[s])                              # compute waits until frame i has landed
            out.append(self.find(gbuf[s]))                      # find() syncs -> gbuf[s] free to reuse next round
        return out


def peakfinder8(image, q_per_pixel, mask=None, **kw):
    """One-shot convenience: build a PeakFinder8 for this geometry and find peaks in one frame.
    For many frames of one geometry, build ``PeakFinder8`` once and reuse ``.find()`` (much faster).
    Returns {x, y, intensity, snr, npix} (x=col, y=row intensity-weighted centroids)."""
    return PeakFinder8(q_per_pixel, mask, **kw).find(image)
