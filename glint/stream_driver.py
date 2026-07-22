"""Device-resident streaming driver: peakfind -> index -> integrate -> running merge, with the
detector frame staying on the GPU for the whole chain.

Why a driver is needed at all. Each stage is fast on its own (index 0.26 ms/frame, integrate
0.33 ms/frame), but they have opposite batching requirements: indexing wants MANY frames at once
(one thread-block per frame, so the batch sets GPU occupancy -- B>=64 saturates an A100), while
integration is per-frame and needs THAT frame's pixels. Uploading the frame twice would cost ~3-5 ms
for a 16 Mpix frame, an order of magnitude more than either kernel. So the driver holds a ring of B
frames resident on the device, indexes them as one batch, and then integrates each against the
pixels that are still sitting there.

Merge statistics accumulate incrementally. CC1/2, CC* and Rsplit are scale-invariant, so the global
`gmean` in the batch scaling cancels and only the frame-local mean is needed -- which is knowable on
arrival. That makes the whole thing a running per-(asu-hkl, half-set, snr-bucket) sum of w, w*v and
count, i.e. O(#unique) memory with one-frame buffering. Half-sets are frame parity.

Completeness is computed here (the batch merge_stats.py does not have it): the theoretical unique
count to `dmin` is enumerated once from the reference cell, so `stats()["completeness"]` is the
live "we have enough data, stop collecting" signal.

Numbers reported are on an arbitrary common intensity scale (the batch path's global gmean is
dropped as it cancels); this affects nothing that is reported, all of which are ratios.
"""
import numpy as np

try:
    import cupy as cp
    _HAVE_CP = True
except Exception:                                            # pragma: no cover - CPU-only env
    cp = None
    _HAVE_CP = False

from glint.lute_bridge import peaks_to_q
from glint.predict import (predict_spots, integrate_spots, recip_from_M, _canonical_axes,
                           _hkl_grid, project_q)
from glint.peakfinder_v4 import PeakFinderV4
import glint.replica_gpu_batch as rgb


# ------------------------------------------------------------- orientation-invariant hkl grid ----
# Fused predict gate: matmul(g@R) + Ewald/qmax test + atomic compaction, one kernel + one D2H
# (replaces the ~6 cupy ops + full-Nhkl qg/qn2/exc intermediates). Survivors written unordered;
# the caller argsorts by hkl index to restore grid order, so the result matches the eager path.
_GATE_SRC = r"""
extern "C" __global__ void predict_gate(const double* g, int nhkl, const double* R, double wave,
                                         double qmax2, double tol, double* out, int* counter, int cap){
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= nhkl) return;
  double gx=g[3*i], gy=g[3*i+1], gz=g[3*i+2];
  double qx=gx*R[0]+gy*R[3]+gz*R[6];
  double qy=gx*R[1]+gy*R[4]+gz*R[7];
  double qz=gx*R[2]+gy*R[5]+gz*R[8];
  double qn2=qx*qx+qy*qy+qz*qz;
  double exc=qz+0.5*wave*qn2;
  if (qn2<=qmax2 && fabs(exc)<tol){
    int p=atomicAdd(counter,1);
    if (p<cap){ out[6*p]=(double)i; out[6*p+1]=qx; out[6*p+2]=qy; out[6*p+3]=qz; out[6*p+4]=qn2; out[6*p+5]=exc; }
  }
}"""
_GATE_KERNEL = cp.RawKernel(_GATE_SRC, "predict_gate") if _HAVE_CP else None


class HKLGrid:
    """The candidate hkl set, hoisted out of the per-frame path.

    predict_spots rebuilds `_hkl_grid` on every call -- a full 3-D meshgrid (~10^5-10^6 points) plus
    a norm cut. But the kept set is `|hkl @ R| <= qmax`, and R differs between frames only by a
    ROTATION, which preserves norms: the set is orientation-INVARIANT. It depends only on the cell
    and dmin, so it is built once here and reused for every frame. Measured on the sim: predict
    drops from ~19.8 ms to ~1 ms/frame.

    The grid is built with a small margin and the exact |q| <= qmax cut is re-applied per frame, so
    a per-frame refined cell whose axes differ slightly from the reference cannot clip the edge.
    """

    def __init__(self, Mc, dmin, margin=1.02, gpu=True):
        self.qmax = 1.0 / float(dmin)
        R = recip_from_M(np.asarray(Mc, float))
        self.g, _ = _hkl_grid(R, self.qmax * margin)
        # the two gates below are elementwise over ~10^5 hkl -- run them on the device and bring
        # back only the few hundred survivors, which is what the (already vectorised) host-side
        # projection then works on.
        self.gpu = bool(gpu) and _HAVE_CP
        self._gg = cp.asarray(self.g.astype(np.float64)) if self.gpu else None
        if self.gpu:
            self._ggr = cp.ascontiguousarray(self._gg.ravel())
            self._gout = cp.empty((self.g.shape[0], 6), cp.float64)
            self._gcnt = cp.zeros(1, cp.int32)

    def predict(self, M_or_R, panels, clen_m, wavelength_A, tol=0.006, is_recip=False):
        """Same result as predict_spots(..., dmin=self.dmin, tol=tol) with the grid reused."""
        R = np.asarray(M_or_R, float) if is_recip else recip_from_M(M_or_R)
        if self.gpu:
            # fused kernel: matmul + Ewald/qmax gate + compaction in one pass, then ONE D2H
            nhkl = self.g.shape[0]; self._gcnt[0] = 0
            Rf = cp.ascontiguousarray(cp.asarray(R).ravel())
            tpb = 256; blocks = (nhkl + tpb - 1) // tpb
            _GATE_KERNEL((blocks,), (tpb,),
                         (self._ggr, np.int32(nhkl), Rf, np.float64(wavelength_A),
                          np.float64(self.qmax * self.qmax), np.float64(tol),
                          self._gout.ravel(), self._gcnt, np.int32(nhkl)))
            n = int(self._gcnt[0])
            C = cp.asnumpy(self._gout[:n])
            C = C[np.argsort(C[:, 0], kind="stable")]         # restore hkl-grid order
            idx = C[:, 0].astype(np.int64)
            hkl = self.g[idx]
            q = C[:, 1:4]; qn2 = C[:, 4]; exc = C[:, 5]
        else:
            q = self.g @ R
            qn2 = np.einsum("ij,ij->i", q, q)
            keep = qn2 <= self.qmax * self.qmax              # exact cut, per frame
            hkl, q, qn2 = self.g[keep], q[keep], qn2[keep]
            exc = q[:, 2] + 0.5 * wavelength_A * qn2
            near = np.abs(exc) < tol
            hkl, q, exc, qn2 = hkl[near], q[near], exc[near], qn2[near]
        fs, ss, pan = project_q(q, panels, clen_m, wavelength_A)
        on = pan >= 0
        out = np.zeros(int(on.sum()), dtype=[("h", int), ("k", int), ("l", int),
                                             ("fs", float), ("ss", float), ("panel", int),
                                             ("exc", float), ("res", float)])
        out["h"], out["k"], out["l"] = hkl[on, 0], hkl[on, 1], hkl[on, 2]
        out["fs"], out["ss"], out["panel"] = fs[on], ss[on], pan[on]
        out["exc"] = exc[on]; out["res"] = 1.0 / np.sqrt(qn2[on])
        return out


# --------------------------------------------------------------- asymmetric unit (4/mmm) --------
def laue_ops_4mmm():
    """16 Laue 4/mmm operators (4-fold c, 2-fold a, inversion). Same generators as merge_stats.py."""
    gens = [np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            -np.eye(3, dtype=int)]
    G = [np.eye(3, dtype=int)]
    ch = True
    while ch:
        ch = False
        for g in list(G):
            for s in gens:
                h = (s @ g).astype(int)
                if not any(np.array_equal(h, x) for x in G):
                    G.append(h); ch = True
    return G


def canon(hkl, ops):
    """Canonical ASU representative: the symmetry equivalent maximising a radix key."""
    eqs = np.stack([hkl @ op.T for op in ops])
    key = eqs[:, :, 0] * 10 ** 8 + eqs[:, :, 1] * 10 ** 4 + eqs[:, :, 2]
    return eqs[key.argmax(0), np.arange(eqs.shape[1])]


def _asu_key(hkl, ops):
    c = canon(np.asarray(hkl, int), ops)
    return c[:, 0].astype(np.int64) * 10 ** 8 + c[:, 1].astype(np.int64) * 10 ** 4 + c[:, 2].astype(np.int64)


def theoretical_unique(Mc, dmin, ops):
    """Number of unique reflections to `dmin` for the reference cell -- the completeness denominator."""
    R = recip_from_M(np.asarray(Mc, float))
    qmax = 1.0 / dmin
    n = np.linalg.norm(R, axis=1)
    H, K, L = (int(np.ceil(qmax / x)) + 1 for x in n)
    g = np.mgrid[-H:H + 1, -K:K + 1, -L:L + 1].reshape(3, -1).T.astype(int)
    g = g[np.any(g != 0, axis=1)]
    q = g @ R
    g = g[np.linalg.norm(q, axis=1) <= qmax]
    return int(np.unique(_asu_key(g, ops)).size)


# ------------------------------------------------------------------- merge accumulator ----------
class MergeAccumulator:
    """Running Monte-Carlo merge: per (asu key, half-set, snr bucket) sums of w, w*v and count.

    Purely additive, so frames fold in as they arrive. Snr buckets are nested (bucket j holds
    measurements whose I/sigma falls in [thr_j, thr_{j+1})), so a threshold sweep is a suffix sum.
    """

    def __init__(self, snr_bins=(0.0, 1.0, 2.0, 3.0, 5.0), ops=None):
        self.thr = np.asarray(snr_bins, float)
        self.nb = len(self.thr)
        self.ops = ops if ops is not None else laue_ops_4mmm()
        self._row = {}
        cap = 4096
        self.sw = np.zeros((cap, 2, self.nb)); self.swv = np.zeros((cap, 2, self.nb))
        self.cnt = np.zeros((cap, 2, self.nb))
        self.n_rows = 0
        self.n_meas = 0
        self.n_frames = 0

    def _grow(self, need):
        cap = self.sw.shape[0]
        if need <= cap:
            return
        new = max(2 * cap, need)
        for a in ("sw", "swv", "cnt"):
            old = getattr(self, a)
            z = np.zeros((new,) + old.shape[1:])
            z[:self.n_rows] = old[:self.n_rows]
            setattr(self, a, z)

    def add_frame(self, hkl, I, sigma, frame_index):
        """Fold one indexed+integrated frame in. hkl (n,3) int; I, sigma (n,) float."""
        I = np.asarray(I, float); sigma = np.maximum(np.asarray(sigma, float), 1e-3)
        good = np.isfinite(I) & np.isfinite(sigma)
        I, sigma, hkl = I[good], sigma[good], np.asarray(hkl, int)[good]
        if I.size == 0:
            return
        # per-frame scale to the frame mean; the batch path's global gmean cancels in every ratio
        scale = 1.0
        if I.size > 5 and I.mean() > 0:
            scale = 1.0 / I.mean()
        v = I * scale
        w = 1.0 / sigma ** 2
        # bucket j holds measurements passing thr[j] but not thr[j+1], so summing j>=J reproduces
        # the batch selection `snr > thr[J]` EXACTLY. side="left" makes it strictly-greater, and
        # bucket -1 (snr <= thr[0], e.g. negative intensities) is DROPPED rather than folded into 0.
        b = np.searchsorted(self.thr, I / sigma, side="left") - 1
        keep = b >= 0
        if not keep.any():
            self.n_frames += 1
            return
        v, w, b = v[keep], w[keep], b[keep]
        hkl = hkl[keep]
        half = frame_index & 1
        keys = _asu_key(hkl, self.ops)
        # Map keys -> rows with one dict pass over PYTHON ints (.tolist() avoids boxing a numpy
        # scalar per element), then accumulate with np.add.at. The old per-element
        # `self.sw[r, half, bb] += w` was ~1 us of numpy fancy-indexing EACH, which dominated.
        row = self._row; nr = self.n_rows
        rows = np.empty(len(keys), np.int64)
        for i, kk in enumerate(keys.tolist()):
            r = row.get(kk)
            if r is None:
                r = row[kk] = nr; nr += 1
            rows[i] = r
        if nr > self.n_rows:
            self._grow(nr); self.n_rows = nr
        np.add.at(self.sw, (rows, half, b), w)
        np.add.at(self.swv, (rows, half, b), w * v)
        np.add.at(self.cnt, (rows, half, b), 1.0)
        self.n_meas += int(keep.sum())
        self.n_frames += 1

    def stats(self, thr=0.0, n_theoretical=None):
        """Figures of merit from the running sums, at an I/sigma floor."""
        j = int(np.searchsorted(self.thr, thr, side="left"))
        r = slice(0, self.n_rows)
        sw = self.sw[r, :, j:].sum(2); swv = self.swv[r, :, j:].sum(2); cnt = self.cnt[r, :, j:].sum(2)
        tot = cnt.sum(1)
        obs = tot > 0
        out = {"frames": self.n_frames, "measurements": int(self.n_meas),
               "unique": int(obs.sum()),
               "redundancy": float(tot[obs].mean()) if obs.any() else 0.0,
               "cc_half": float("nan"), "cc_star": float("nan"), "rsplit": float("nan"),
               "common": 0, "completeness": float("nan")}
        if n_theoretical:
            out["completeness"] = 100.0 * out["unique"] / n_theoretical
        both = (sw[:, 0] > 0) & (sw[:, 1] > 0)
        out["common"] = int(both.sum())
        if both.sum() >= 10:
            a = swv[both, 0] / sw[both, 0]; b = swv[both, 1] / sw[both, 1]
            cc = float(np.corrcoef(a, b)[0, 1])
            out["cc_half"] = cc
            out["cc_star"] = float(np.sqrt(2 * cc / (1 + cc))) if cc > 0 else float("nan")
            denom = 0.5 * np.sum(a + b)
            out["rsplit"] = float((1 / np.sqrt(2)) * np.sum(np.abs(a - b)) / denom) if denom else float("nan")
        return out


# ------------------------------------------------------------------------- the driver -----------
class StreamDriver:
    """Streaming index+integrate with the frame resident on the device.

    push() one frame at a time; the driver uploads it into a preallocated device ring slot,
    peak-finds it ON DEVICE, and queues the reciprocal vectors. When B frames are queued it indexes
    them as one batch and integrates each against its still-resident pixels, then folds the
    intensities into the running merge. Call flush() at the end of a run, then stats().
    """

    def __init__(self, Mc, panels, clen_m, wavelength_A, shape, dtype=np.uint16, mask=None,
                 B=64, dmin=2.0, tol=0.002, half=3, gap=2, ring=3, min_peaks=6,
                 snr_bins=(0.0, 1.0, 2.0, 3.0, 5.0), pf_kw=None, use_gpu=True):
        if use_gpu and not _HAVE_CP:
            raise RuntimeError("cupy required for the device-resident path")
        self.gpu = bool(use_gpu)
        self.Mc = np.asarray(Mc, float)
        self.panels, self.clen_m, self.wavelength_A = panels, float(clen_m), float(wavelength_A)
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.B, self.dmin, self.tol = int(B), float(dmin), float(tol)
        self.half, self.gap, self.ring_w, self.min_peaks = half, gap, ring, int(min_peaks)

        xp = cp if self.gpu else np
        if mask is None:
            m = xp.ones(self.shape, bool)
        else:
            m = xp.asarray(mask).astype(bool)
        # built ONCE and reused; dtype=float32 is what enables the fused GPU stats kernel
        kw = dict(dtype=xp.float32) if self.gpu else {}
        kw.update(pf_kw or {})
        self.finder = PeakFinderV4(m, **kw)

        # preallocated resident ring -- no per-frame device allocation
        self._ring = [xp.zeros(self.shape, self.dtype) for _ in range(self.B)]
        self._q = [None] * self.B
        self._n = 0
        self._frame_no = 0

        self.grid = HKLGrid(self.Mc, self.dmin, gpu=self.gpu)   # built once, reused every frame
        self.ops = laue_ops_4mmm()
        self.acc = MergeAccumulator(snr_bins, self.ops)
        self.n_theoretical = theoretical_unique(self.Mc, self.dmin, self.ops)
        self.n_pushed = self.n_indexed = self.n_integrated = 0

    # ------------------------------------------------------------------ ingest ------------------
    def push(self, frame):
        """Ingest one detector frame (host numpy or already-device array)."""
        slot = self._n
        xp = cp if self.gpu else np
        self._ring[slot][...] = xp.asarray(frame)            # single H2D into the resident slot
        pk = self.finder.find(self._ring[slot])              # peak-find ON DEVICE, no readback of pixels
        fs = pk["x"]; ss = pk["y"]
        if self.gpu:
            fs = cp.asnumpy(fs); ss = cp.asnumpy(ss)         # peaks are tiny; pixels stay put
        q = None
        if fs.size >= self.min_peaks:
            qq = peaks_to_q(fs, ss, self.panels, self.clen_m, self.wavelength_A)
            qq = qq[np.isfinite(qq).all(1)]                  # peaks_to_q returns NaN rows off-panel
            if len(qq) >= self.min_peaks:
                q = qq
        self._q[slot] = q
        self._n += 1
        self.n_pushed += 1
        if self._n == self.B:
            self.flush()

    # ------------------------------------------------------------------ batch -------------------
    def flush(self):
        """Index the resident batch, integrate each frame against its still-resident pixels."""
        if self._n == 0:
            return
        slots = [i for i in range(self._n) if self._q[i] is not None]
        if slots:
            qs = [self._q[i] for i in slots]
            Ms = rgb.index_fused(qs, self.Mc, B=max(len(qs), 1))
            for i, M in zip(slots, Ms):
                if M is None or abs(np.linalg.det(np.asarray(M, float))) < 1.0:
                    continue
                self.n_indexed += 1
                Mcan = _canonical_axes(np.asarray(M, float))   # cross-frame consistent hkl setting
                pred = self.grid.predict(Mcan, self.panels, self.clen_m, self.wavelength_A,
                                         tol=self.tol)
                if len(pred) == 0:
                    continue
                if self.gpu:
                    from glint.fused_integrate import integrate_fused
                    I, sig, _, _ = integrate_fused(self._ring[i], pred,
                                                   half=self.half, gap=self.gap, ring=self.ring_w)
                else:
                    I, sig, _, _ = integrate_spots(self._ring[i], pred,
                                                   half=self.half, gap=self.gap, ring=self.ring_w)
                hkl = np.stack([pred["h"], pred["k"], pred["l"]], 1)
                keep = I != 0.0                               # off-frame boxes integrate to exactly 0
                if keep.any():
                    self.acc.add_frame(hkl[keep], I[keep], sig[keep], self._frame_no)
                    self.n_integrated += 1
                    self._frame_no += 1
        self._n = 0
        self._q = [None] * self.B

    def stats(self, thr=0.0):
        s = self.acc.stats(thr=thr, n_theoretical=self.n_theoretical)
        s.update(pushed=self.n_pushed, indexed=self.n_indexed, integrated=self.n_integrated,
                 theoretical_unique=self.n_theoretical)
        return s
