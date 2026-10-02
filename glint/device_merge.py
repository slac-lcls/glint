"""Device-resident ASU merge: the running scatter-add moved onto the GPU.

The host MergeAccumulator (stream_driver.py) folds each frame's measurements into a running
per-(asu-key, half-set, snr-bucket) sum with a Python dict row-lookup plus three np.add.at calls --
~0.51 ms/frame, one of the larger remaining host stages. This module relocates that scatter to the
device WITHOUT changing a single accumulated bit.

Bit-identity is the hard gate. np.add.at(buf, idx, w) adds each w[i] into a persistent buffer one at
a time in i-order, and the buffer persists across frames, so for a given (key,half,bucket) cell the
host result is a STRICTLY LEFT-TO-RIGHT float64 sum over the measurements in their global arrival
order. float64 addition is non-associative for the accumulands here (w = 1/sigma^2, w*v are general
rounded doubles, not exact integers -- so the peakfinder's "exactness => order-independence" trick
does NOT transfer; a float64 atomicAdd would reorder additions and drift by a few ULP per cell).

The only order-FAITHFUL device design is therefore: stage every measurement's (composite-key, w, w*v)
in persistent device buffers as it arrives (a free async append, no dict, no scatter), then DRAIN
ONCE at the end -- stable-sort by composite key so each cell's members sit contiguously in arrival
order, and a one-thread-per-segment sequential reduce replays the exact host addition sequence. Same
rounding at every step => bit-for-bit identical sw/swv, and integer-exact cnt/measurements/unique.

The accumulands (scale, v, w, w*v, snr bucket, asu key) are computed host-side with the SAME numpy
ops as the host path -- notably scale = 1/I.mean() from the host's frame_scale (numpy's pairwise
sum, and the same refuse-unmeasured-frames gate), which a cupy reduction would round differently --
so the staged values are identical before the sum even starts. Only the scatter/reduce moves to the
GPU.
"""
import os
import numpy as np

from glint.merge_scale import MERGE_SNR_BINS, frame_scale

try:
    import cupy as cp
    _HAVE_CP = True
except Exception:                                            # pragma: no cover - CPU-only env
    cp = None
    _HAVE_CP = False


# One thread per sorted segment. Each thread sums its segment's w and w*v STRICTLY left-to-right
# (i.e. in arrival order, because the sort is stable), seeded from 0.0 -- exactly what np.add.at
# does into the persistent host buffer. Counts are integer-exact regardless of order.
_SEGREDUCE_SRC = r"""
extern "C" __global__ void seg_reduce(
    const long long* starts, int nseg, long long m,
    const double* w, const double* wv,
    double* seg_sw, double* seg_swv, long long* seg_cnt){
  int s = blockIdx.x*blockDim.x + threadIdx.x; if (s >= nseg) return;
  long long a = starts[s];
  long long b = (s+1 < nseg) ? starts[s+1] : m;
  double sw = 0.0, swv = 0.0;
  for (long long i = a; i < b; i++){ sw += w[i]; swv += wv[i]; }
  seg_sw[s] = sw; seg_swv[s] = swv; seg_cnt[s] = b - a;
}
"""
_SEGREDUCE = cp.RawKernel(_SEGREDUCE_SRC, "seg_reduce") if _HAVE_CP else None


class MergeAccumulatorDevice:
    """Drop-in for MergeAccumulator with the scatter-add on the GPU (deferred, order-faithful).

    add_frame stages measurements into persistent device buffers; stats() drains them once via a
    stable sort + segmented sequential reduce and then reuses MergeAccumulator.stats verbatim on the
    resulting (n_unique, 2, nb) sums. Bit-identical to the host merge on the same measurements.
    """
    n_refused = 0                                            # frames not merged (frame_scale is None)

    def __init__(self, snr_bins=MERGE_SNR_BINS, ops=None, cap=1 << 16):
        if not _HAVE_CP:
            raise RuntimeError("cupy required for the device-resident merge")
        from glint.stream_driver import laue_ops_4mmm, _asu_key
        self._asu_key = _asu_key
        self.thr = np.asarray(snr_bins, float)
        self.nb = len(self.thr)
        # ops=None keeps the historical 4/mmm for direct construction only: StreamDriver always
        # passes the operator set it resolved from laue/ops/stream_symmetry (glint#180).
        self.ops = ops if ops is not None else laue_ops_4mmm()
        self.n_meas = 0
        self.n_frames = 0
        self.n_refused = 0                                   # same rule and counter as the host path
        self._m = 0                                          # staged measurement count
        # Staging is on the HOST: appending a few hundred rows per frame is a cheap vectorised copy,
        # whereas a per-frame H2D of these arrays is latency-bound and (measured) LOSES to the host
        # np.add.at. The whole staged buffer is shipped to the GPU ONCE at drain, where the sort +
        # segmented reduce (the actual scatter-add work) runs on the device.
        self.h_comp = np.empty(cap, np.int64)
        self.h_key = np.empty(cap, np.int64)
        self.h_hb = np.empty(cap, np.int32)                  # half*nb + bucket
        self.h_w = np.empty(cap, np.float64)
        self.h_wv = np.empty(cap, np.float64)
        self._dirty = True
        # filled by _drain, consumed by stats (same layout/dtype as the host buffers)
        self.sw = self.swv = self.cnt = None
        self.n_rows = 0
        self._uk = None

    def _ensure(self, need):
        cap = self.h_comp.size
        if self._m + need <= cap:
            return
        new = max(2 * cap, self._m + need)
        for a in ("h_comp", "h_key", "h_hb", "h_w", "h_wv"):
            old = getattr(self, a)
            z = np.empty(new, old.dtype)
            z[:self._m] = old[:self._m]
            setattr(self, a, z)

    def add_frame(self, hkl, I, sigma, frame_index):
        """Stage one indexed+integrated frame. Accumulands are computed EXACTLY as the host path."""
        I = np.asarray(I, float); sigma = np.maximum(np.asarray(sigma, float), 1e-3)
        good = np.isfinite(I) & np.isfinite(sigma)
        I, sigma, hkl = I[good], sigma[good], np.asarray(hkl, int)[good]
        if I.size == 0:
            return
        scale = frame_scale(I)                               # host's rule: same gate, same scale
        if scale is None:
            self.n_refused += 1; self.n_frames += 1
            return
        v = I * scale
        w = 1.0 / sigma ** 2
        b = np.searchsorted(self.thr, I / sigma, side="left") - 1
        keep = b >= 0
        if not keep.any():
            self.n_frames += 1
            return
        v, w, b = v[keep], w[keep], b[keep]
        hkl = hkl[keep]
        half = frame_index & 1
        keys = self._asu_key(hkl, self.ops)                  # int64, same canon() as host
        wv = w * v                                           # exactly the np.add.at(swv, .., w*v) operand
        hb = (half * self.nb + b).astype(np.int64)           # segment discriminator, in [0, 2*nb)
        comp = keys * (2 * self.nb) + hb                     # injective (key, half, bucket) key
        n = len(comp); self._ensure(n); s = self._m
        self.h_comp[s:s + n] = comp
        self.h_key[s:s + n] = keys
        self.h_hb[s:s + n] = hb
        self.h_w[s:s + n] = w
        self.h_wv[s:s + n] = wv
        self._m += n
        self.n_meas += int(keep.sum())
        self.n_frames += 1
        self._dirty = True

    def _drain(self):
        """Ship the staged records to the GPU once, sort by composite key (arrival-stable), reduce."""
        if not self._dirty:
            return
        m = self._m
        if m == 0:
            self.sw = np.zeros((0, 2, self.nb)); self.swv = np.zeros((0, 2, self.nb))
            self.cnt = np.zeros((0, 2, self.nb)); self.n_rows = 0; self._uk = np.zeros(0, np.int64)
            self._dirty = False
            return
        comp = cp.asarray(self.h_comp[:m])                   # single H2D of the whole staged buffer
        d_w = cp.asarray(self.h_w[:m]); d_wv = cp.asarray(self.h_wv[:m])
        d_key = cp.asarray(self.h_key[:m]); d_hb = cp.asarray(self.h_hb[:m])
        # stable order: primary = composite key, secondary = arrival index (forces arrival order for
        # within-cell ties regardless of the underlying sort's stability). lexsort's LAST key is primary.
        order = cp.lexsort(cp.stack([cp.arange(m, dtype=cp.int64), comp]))
        comp_s = comp[order]
        w_s = d_w[order]; wv_s = d_wv[order]
        key_s = d_key[order]; hb_s = d_hb[order]
        # segment starts = first index of each run of equal composite in the sorted array
        flag = cp.ones(m, cp.bool_)
        flag[1:] = comp_s[1:] != comp_s[:-1]
        starts = cp.where(flag)[0].astype(cp.int64)
        nseg = int(starts.size)
        seg_sw = cp.empty(nseg, cp.float64); seg_swv = cp.empty(nseg, cp.float64)
        seg_cnt = cp.empty(nseg, cp.int64)
        tpb = 128; blocks = (nseg + tpb - 1) // tpb
        _SEGREDUCE((blocks,), (tpb,),
                   (starts, np.int32(nseg), np.int64(m), w_s, wv_s, seg_sw, seg_swv, seg_cnt))
        seg_key = cp.asnumpy(key_s[starts]); seg_hb = cp.asnumpy(hb_s[starts])
        seg_sw = cp.asnumpy(seg_sw); seg_swv = cp.asnumpy(seg_swv)
        seg_cnt = cp.asnumpy(seg_cnt).astype(float)
        # Row order must match the host's, which assigns each asu key a row at its FIRST APPEARANCE
        # in the arrival stream (not sorted-key order). stats() reduces cc_half/rsplit over the rows
        # with np.corrcoef / np.sum, which are order-sensitive at the last ULP -- so replay the host
        # row order to keep even the FOM aggregates bit-identical, not just the per-cell sums.
        akey = self.h_key[:m]                                # arrival order (host staging)
        uk_sorted, first_idx = np.unique(akey, return_index=True)
        uk = uk_sorted[np.argsort(first_idx)]                # unique keys in first-appearance order
        rowof = {int(k): r for r, k in enumerate(uk)}
        inv = np.fromiter((rowof[int(k)] for k in seg_key), dtype=int, count=len(seg_key))
        nr = int(uk.size)
        sw = np.zeros((nr, 2, self.nb)); swv = np.zeros((nr, 2, self.nb)); cnt = np.zeros((nr, 2, self.nb))
        half = (seg_hb // self.nb).astype(int); bb = (seg_hb % self.nb).astype(int)
        sw[inv, half, bb] = seg_sw; swv[inv, half, bb] = seg_swv; cnt[inv, half, bb] = seg_cnt
        self.sw = sw; self.swv = swv; self.cnt = cnt; self.n_rows = nr; self._uk = uk
        self._dirty = False

    def stats(self, thr=-np.inf, n_theoretical=None):
        from glint.stream_driver import MergeAccumulator
        self._drain()
        return MergeAccumulator.stats(self, thr=thr, n_theoretical=n_theoretical)

    def per_key_sums(self):
        """(keys, sw, swv, cnt) per asu key summed over halves+buckets -- for the bit-identity gate."""
        self._drain()
        return (np.asarray(self._uk), self.sw.copy(), self.swv.copy(), self.cnt.copy())
