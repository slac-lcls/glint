"""Device-resident streaming driver: peakfind -> index -> integrate -> running merge, with the
detector frame staying on the GPU for the whole chain.

Blind or known-cell. Constructed with a cell (Mc) it runs known-cell from frame 0. Constructed with
Mc=None it starts BLIND: it indexes the first frames one at a time (~26 ms/frame), accumulates a
running-histogram cross-frame consensus (glint.running_consensus), and the instant the cell LOCKS it
builds the hkl grid and switches to the batched known-cell path (~0.26 ms/frame) for the rest of the
run -- so a live run needs no cell handed in, only a ~150 ms discovery transient.

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
import os
from collections import deque
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
from glint.running_consensus import RunningConsensus
from glint.multishot import same_lattice
from glint.multilattice import deflate_peaks
try:
    import glint.replica_gpu_batch as rgb                     # the q-only batch indexer (needs torch)
except Exception:                                            # pragma: no cover - CPU-only unit env (no torch)
    rgb = None                                               # batch/rescue indexers are injected as fakes there


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

    def add_frame(self, hkl, I, sigma, frame_index, values=None, weights=None):
        """Fold one indexed+integrated frame in. hkl (n,3) int; I, sigma (n,) float.

        Default (values=weights=None): per-frame 1/mean(I) scale + inverse-variance
        weight -- the original behaviour, BIT-IDENTICAL. If ``values`` AND ``weights``
        are supplied (the partiality path, ``glint.partiality.PartialityScaler``) they
        are used as the merged value v and weight w directly (v = I/(G*p),
        w = p^2/sigma^2); the I/sigma snr bucketing is unchanged."""
        I = np.asarray(I, float); sigma = np.maximum(np.asarray(sigma, float), 1e-3)
        good = np.isfinite(I) & np.isfinite(sigma)
        _part = values is not None and weights is not None
        if _part:
            values = np.asarray(values, float); weights = np.asarray(weights, float)
            good = good & np.isfinite(values) & np.isfinite(weights)
        I, sigma, hkl = I[good], sigma[good], np.asarray(hkl, int)[good]
        if I.size == 0:
            return
        if _part:
            v = values[good]
            w = weights[good]
        else:
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

    def merged_by_key(self, thr=0.0):
        """Merged intensity per asu key (both half-sets and all snr buckets >= thr
        combined). Returns {asu_key(int): I_merged(float)} -- used to score R_vs_truth
        against a KNOWN I_full on the synthetic experiment."""
        j = int(np.searchsorted(self.thr, thr, side="left"))
        sw = self.sw[:self.n_rows, :, j:].sum((1, 2))
        swv = self.swv[:self.n_rows, :, j:].sum((1, 2))
        out = {}
        for k, r in self._row.items():
            if sw[r] > 0:
                out[int(k)] = float(swv[r] / sw[r])
        return out

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
def _conventional_tetragonal(M):
    """Permute a tetragonal cell's columns so the unique (4-fold) axis is c, matching laue_ops_4mmm.

    Buerger reduction orders axes by length, so the short 4-fold axis of a cell like 79/79/38 can land
    in column a. The two most-equal-length columns are taken as a,b; the length outlier becomes c.
    Handedness is preserved (negate one column if the permutation flipped the determinant sign)."""
    M = np.asarray(M, float)
    L = np.linalg.norm(M, axis=0)
    i, j, k = min([(0, 1, 2), (0, 2, 1), (1, 2, 0)],
                  key=lambda p: abs(L[p[0]] - L[p[1]]) / max(L[p[0]], L[p[1]]))
    P = M[:, [i, j, k]].copy()
    if np.linalg.det(P) < 0:
        P[:, 0] = -P[:, 0]
    return P


class StreamDriver:
    """Streaming index+integrate with the frame resident on the device.

    push() one frame at a time; the driver uploads it into a preallocated device ring slot,
    peak-finds it ON DEVICE, and queues the reciprocal vectors. When B frames are queued it indexes
    them as one batch and integrates each against its still-resident pixels, then folds the
    intensities into the running merge. Call flush() at the end of a run, then stats().
    """

    def __init__(self, Mc, panels, clen_m, wavelength_A, shape, dtype=np.uint16, mask=None,
                 B=64, dmin=2.0, tol=0.002, half=3, gap=2, ring=3, min_peaks=6,
                 snr_bins=(0.0, 1.0, 2.0, 3.0, 5.0), pf_kw=None, use_gpu=True,
                 lock_support=3, lock_gap=2, adaptive_gap=True, warmup_nbest=3,
                 adaptive_relock=False, min_inliers=0, warm_topk=16, warm_floor=1,
                 double_hit=False, geom_refine=False, geom_refine_kw=None,
                 rescue_buffer=0, fanout=None, alias_gate=None,
                 lock_probe=False, probe_null=64, lock_min_z=None, warmup_rescue=False):
        if use_gpu and not _HAVE_CP:
            raise RuntimeError("cupy required for the device-resident path")
        self.gpu = bool(use_gpu)
        self.panels, self.clen_m, self.wavelength_A = panels, float(clen_m), float(wavelength_A)
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.B, self.dmin, self.tol = int(B), float(dmin), float(tol)
        self.half, self.gap, self.ring_w, self.min_peaks = half, gap, ring, int(min_peaks)
        self.warmup_nbest = int(warmup_nbest)
        self.warm_topk, self.warm_floor = int(warm_topk), int(warm_floor)                              # warmup_batch triage
        self.double_hit = bool(double_hit); self.n_double = 0                                          # deflate-and-reindex 2nd lattice
        if self.double_hit:
            from glint.glint_fast import index_blind_nbest
            self._dh_index = index_blind_nbest

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
        self._pk = [None] * self.B                              # observed (fs,ss) peaks per slot (geom refine)
        self._n = 0
        self._frame_no = 0

        self.ops = laue_ops_4mmm()                              # cell-independent (Laue group)
        self.snr_bins = snr_bins
        # GLINT_DEVICE_MERGE=1 relocates the running scatter-add onto the GPU (deferred,
        # order-faithful, bit-identical to the host merge). Host path stays the default for A/B.
        # (grid + n_theoretical are built by _lock / the blind-check below, once Mc is known.)
        if os.environ.get("GLINT_DEVICE_MERGE") == "1" and self.gpu:
            from glint.device_merge import MergeAccumulatorDevice
            self.acc = MergeAccumulatorDevice(snr_bins, self.ops)
        else:
            self.acc = MergeAccumulator(snr_bins, self.ops)
        self.n_pushed = self.n_indexed = self.n_integrated = 0

        # Live geometry refinement (opt-in, DIAGNOSTIC-only): pool per-frame predicted-vs-observed
        # residuals into a running (clen, beam-shift) correction, reported in stats()["geom_correction"].
        # Default off keeps the integrate path bit-identical; feeding the correction BACK into indexing
        # is the flagged next step. See glint.geom_refine.
        self.geom_refine = bool(geom_refine)
        self._grefiner = None
        if self.geom_refine:
            from glint.geom_refine import GeomRefiner
            self._grefiner = GeomRefiner(self.panels, self.clen_m, self.wavelength_A,
                                         **(geom_refine_kw or {}))

        # Adaptive re-lock (opt-in): keep a blind watchdog on the frames that miss every active cell,
        # and add a new cell when one recurs there -- so a mid-run SAMPLE CHANGE (or a mixture) is
        # detected and adapted to. Default off keeps the single-cell path bit-identical. NOTE: this
        # handles a DISCRETE new cell; a continuous cell CREEP (thermal expansion of the same crystal)
        # should be absorbed by widening `tol`/refining the active cell, NOT re-locked as a new one --
        # the miss-gate `min_inliers` + fit window `tol` are that boundary.
        self.adaptive_relock = bool(adaptive_relock)
        self.min_inliers = int(min_inliers) if min_inliers else self.min_peaks
        self.extra = []; self._watch = None; self.n_relock = 0
        if self.adaptive_relock and not hasattr(self, "_blind_index"):
            try:
                from glint.glint_fast import index_blind_nbest
                self._blind_index = index_blind_nbest
            except Exception:                                # pragma: no cover - CPU-only unit env (no torch)
                self._blind_index = None                     # tests inject a fake _blind_index via the seam

        # Miss-buffer retroactive rescue (opt-in, INDEX-ONLY). Frames that fit NO active cell are the
        # "rose/unindexed" bars during a sample change; today they are dropped once the batch flushes.
        # With rescue_buffer>0 (needs adaptive_relock) their q-vectors are buffered, and when the
        # watchdog LOCKS a new cell they are re-indexed against it -- recovering the INDEXING rate of
        # the pre-lock misses. q-only => tiny (no raw-pixel ring); integrate/merge rescue is a deferred
        # later layer. `fanout` batches the watchdog's blind indexing across workers (default = serial
        # single-GPU loop, BIT-IDENTICAL to today); opt-in glint.warmup_batch.mpi_fanout detects the
        # change in ~one blind-frame-time. Both seams (`_fanout`, `_known_index`) let CPU tests inject
        # fakes with no GPU, mirroring the `_blind_index` seam.
        self.n_rescued = 0
        self._missbuf = (deque(maxlen=int(rescue_buffer))
                         if self.adaptive_relock and rescue_buffer > 0 else None)
        self._known_index = rgb.index_fused if rgb is not None else None   # q-only batch indexer (test seam)
        self._fanout = fanout or (lambda Q, k: [self._blind_index(q, k) for q in Q])
        # Opt-in lock-time alias gate (glint.alias_gate.AliasGate): a deterministic Occam-tightness
        # confirmation over the leader's small-index derivative lattices, run on the observed q of the
        # voting frames just before a relock commits. Default None => never runs => bit-identical.
        self._alias_gate = alias_gate
        # Opt-in lock-quality probe (glint.spurious_meter.null_margin): on each relock, measure how far
        # the new cell's overlap sits above the random-orientation floor on its supporting frames -- a
        # live "is this lock resting on real signal" z. lock_min_z (if set) refuses a too-weak lock.
        # Default lock_probe=False => never runs => bit-identical.
        self._lock_probe = bool(lock_probe)
        self._probe_null = int(probe_null)
        self._lock_min_z = lock_min_z
        self.lock_z = None                               # z of the most recent relock (None until one fires)

        # Warm-up rescue (opt-in, INDEX-ONLY): _push_blind's per-frame q-vectors are ordinarily
        # discarded after casting their consensus vote -- slot 0 is scratch, "the frame is not kept"
        # (below). With warmup_rescue=True they're retained in a small list and, the instant the cell
        # locks, re-indexed against it via the same q-only known-cell path _missbuf uses -- recovering
        # the handful of frames (median ~6, per the paper) otherwise permanently sacrificed to
        # discovery. No pixels were ever kept for these frames (slot 0 is overwritten every warm-up
        # push), so this is index-only like _missbuf's rescue, not a full integration. Default off
        # keeps warm-up bit-identical to before.
        self.warmup_rescue = bool(warmup_rescue)
        self._warmup_buf = [] if self.warmup_rescue else None
        self.n_warmup_rescued = 0
        # Watchdog individual rescue (bundled into adaptive_relock, no separate flag): _watchdog
        # already blind-indexes every missed frame to pool votes toward NEW-cell detection -- that
        # compute is spent regardless. Checking each frame's own N-best candidates against the
        # ALREADY-active cell(s) first is nearly free on top of it, and rescues a same-cell miss
        # immediately instead of only ever detecting a genuinely different cell (see _watchdog).
        self.n_watchdog_rescued = 0

        # Blind warm-up: with Mc=None the driver has no cell yet, so it indexes the first frames
        # blind (~26 ms/frame) one at a time, accumulating cross-frame consensus; when the running
        # histogram LOCKS the cell it builds the hkl grid and drops into the batched known-cell path
        # below (~0.26 ms/frame). Warm-up frames are spent on discovery (not integrated) -- ~6-10
        # frames, negligible for completeness. See glint.running_consensus.
        self._blind = Mc is None
        self.n_warmup = 0; self.locked_after = None; self.consensus_support = None
        if self._blind:
            self.Mc = None; self.grid = None; self.n_theoretical = None
            self._rc = RunningConsensus(min_support=lock_support, gap=lock_gap, adaptive=adaptive_gap)
            from glint.glint_fast import index_blind_nbest      # torch; imported only in blind mode
            self._blind_index = index_blind_nbest
        else:
            self._lock(np.asarray(Mc, float))

    def _lock(self, Mc, support=None, standardize=False):
        """Fix the cell: build the hkl grid + theoretical-unique count, and leave blind mode.

        standardize: put the unique (4-fold) axis in c so laue_ops_4mmm / theoretical_unique are
        counted in the conventional setting. Needed for a consensus-locked cell (Buerger reduction
        orders axes by length, so the short 4-fold axis can land in a/b); a user-supplied known cell
        is taken as authoritative and left as-is."""
        Mc = np.asarray(Mc, float)
        if standardize:
            Mc = _conventional_tetragonal(Mc)
        if self._blind:
            self.locked_after = self.n_pushed; self.consensus_support = support
        self.Mc = Mc
        self.grid = HKLGrid(self.Mc, self.dmin, gpu=self.gpu)   # built once, reused every frame
        self.n_theoretical = theoretical_unique(self.Mc, self.dmin, self.ops)
        self._blind = False
        if self._warmup_buf:                                    # retroactive index-only rescue (see __init__)
            qs, self._warmup_buf = self._warmup_buf, []
            Ms = self._known_index(qs, self.Mc, B=max(len(qs), 1))
            self.n_warmup_rescued += sum(1 for q, M in zip(qs, Ms)
                                         if M is not None and abs(np.linalg.det(np.asarray(M, float))) >= 1.0
                                         and self._inliers(q, np.asarray(M, float)) >= self.min_inliers)

    # ------------------------------------------------------------------ ingest ------------------
    def _push_blind(self, frame):
        """Warm-up ingest: peak-find + blind-index this frame, feed the running consensus, lock on fire.

        Runs per frame (not batched) so the cell can lock at the first few frames; slot 0 is scratch
        (the frame is not kept -- there is no cell to integrate against yet)."""
        xp = cp if self.gpu else np
        self.n_pushed += 1
        self._ring[0][...] = xp.asarray(frame)
        pk = self.finder.find(self._ring[0]); fs = pk["x"]; ss = pk["y"]
        if self.gpu:
            fs = cp.asnumpy(fs); ss = cp.asnumpy(ss)
        if fs.size >= self.min_peaks:
            qq = peaks_to_q(fs, ss, self.panels, self.clen_m, self.wavelength_A)
            qq = qq[np.isfinite(qq).all(1)]
            if len(qq) >= self.min_peaks:
                nb = self._blind_index(qq, self.warmup_nbest)   # N-best candidate cells for this frame
                self._rc.add_frame([c for c, _ in nb])
                self.n_warmup += 1
                if self._warmup_buf is not None:
                    self._warmup_buf.append(qq)                 # retained for post-lock rescue (see __init__)
        Mc, sup, _ = self._rc.verdict()
        if Mc is not None:
            self._lock(Mc, sup, standardize=True)               # -> known-cell batched path from here

    def warmup_batch(self, frames, fanout=None):
        """Parallel, PEAK-triaged blind warm-up over a buffered startup stack (blind mode only).

        Instead of grinding the warm-up frames one at a time (`_push_blind`), peak-find the whole
        stack, rank events by Bragg-peak count, blind-index only the top `warm_topk` -- fanned across
        workers via `fanout` (callable(list_q, nbest) -> list of N-best lists; default = serial
        single-GPU loop) -- and pool their N-best into ONE consensus round. Locks the cell (fast
        known-cell path) when consensus fires; returns True. Frames not picked are not consumed here;
        push() them afterwards to integrate.

        Triage is by the peak-finder's peak COUNT (spatial-background ring finder), not a temporal
        MAD z-count: on a liquid jet the MAD score is dominated by shot-varying water/jet scatter
        (calibrated on cxic0415 r0100 -- median ~87k z>4 px/event, no hit/blank separation), whereas
        the ring finder's spatial background is jet-robust. Peak-find is ~100x cheaper than a blind
        index, so scoring every frame is effectively free.

        `frames`: (B,H,W) host/device stack. See glint.warmup_batch for the CPU-testable core.
        """
        if not self._blind:
            return True                                          # already locked -- nothing to warm up
        from glint.warmup_batch import triage_order, warmup_consensus
        xp = cp if self.gpu else np
        counts, qmap = [], []
        for fr in frames:                                        # cheap peak-find every frame
            pk = self.finder.find(xp.asarray(fr)); fs, ss = pk["x"], pk["y"]
            if self.gpu:
                fs, ss = cp.asnumpy(fs), cp.asnumpy(ss)
            q = None
            if fs.size >= self.min_peaks:
                q = peaks_to_q(fs, ss, self.panels, self.clen_m, self.wavelength_A)
                q = q[np.isfinite(q).all(1)]
                q = q if len(q) >= self.min_peaks else None
            counts.append(len(q) if q is not None else 0)
            qmap.append(q)
        picks = triage_order(counts, self.warm_topk, self.warm_floor)   # rank by peak count; skip low-signal
        qs = [qmap[k] for k in picks if qmap[k] is not None]
        self.n_pushed += len(frames); self.n_warmup += len(qs)
        if self._warmup_buf is not None:
            self._warmup_buf.extend(qs)                          # retained for post-lock rescue (see __init__)
        Mc, sup = warmup_consensus(qs, self._blind_index, self._rc, self.warmup_nbest, fanout)
        if Mc is not None:
            self.locked_after = self.n_pushed
            self._lock(Mc, sup, standardize=True)
            return True
        return False

    def push(self, frame):
        """Ingest one detector frame (host numpy or already-device array)."""
        if self._blind:
            self._push_blind(frame); return
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
        if self.geom_refine:
            self._pk[slot] = np.stack([fs, ss], 1) if fs.size else None
        self._n += 1
        self.n_pushed += 1
        if self._n == self.B:
            self.flush()

    # ------------------------------------------------------------------ batch -------------------
    def _inliers(self, q, M):
        """# of q peaks near-integer in cell M (hkl = q @ M) -- the 'does this frame fit this cell' test."""
        hf = np.asarray(q, float) @ M
        return int((np.abs(hf - np.round(hf)).max(1) < 0.15).sum())

    def _integrate_one(self, i, M, grid, acc):
        """Canonicalize + predict + integrate slot i under an ALREADY-ACCEPTED matrix M into acc.
        Split out of _index_integrate so _watchdog's individual rescue can integrate a validated
        blind candidate directly, without re-registering it through known-cell (which could just
        miss again for the same reason the frame was flagged in the first place)."""
        self.n_indexed += 1
        Mcan = _canonical_axes(M)                            # cross-frame consistent hkl setting
        if self.double_hit:                                 # deflate-and-reindex: a 2nd crystal in this shot?
            resid = deflate_peaks(self._q[i], Mcan)
            if len(resid) >= self.min_peaks:
                nb2 = self._dh_index(resid, 1)
                if nb2:
                    M2 = np.asarray(nb2[0][0], float)
                    # a 2nd crystal = the deflated residual re-indexes to a valid lattice with enough
                    # inliers. NOT gated on a different CELL -- SFX double-hits are usually two crystals
                    # of the SAME protein at different orientations. Deflation removed lattice-1's peaks,
                    # so the residual-inlier test already rejects merely re-finding lattice 1.
                    if abs(np.linalg.det(M2)) >= 1.0 and self._inliers(resid, M2) >= self.min_peaks:
                        self.n_double += 1
        pred = grid.predict(Mcan, self.panels, self.clen_m, self.wavelength_A, tol=self.tol)
        if len(pred) == 0:
            return
        if self._grefiner is not None and self._pk[i] is not None:
            self._grefiner.add_frame(recip_from_M(Mcan), self._pk[i], pred)
        if self.gpu:
            from glint.fused_integrate import integrate_fused
            I, sig, _, _ = integrate_fused(self._ring[i], pred, half=self.half, gap=self.gap, ring=self.ring_w)
        else:
            I, sig, _, _ = integrate_spots(self._ring[i], pred, half=self.half, gap=self.gap, ring=self.ring_w)
        hkl = np.stack([pred["h"], pred["k"], pred["l"]], 1)
        keep = I != 0.0                                     # off-frame boxes integrate to exactly 0
        if keep.any():
            acc.add_frame(hkl[keep], I[keep], sig[keep], self._frame_no)
            self.n_integrated += 1; self._frame_no += 1

    def _index_integrate(self, slots, Mc, grid, acc, gate):
        """Index `slots` against Mc, integrate the fits into `acc`. With gate=True a frame with fewer than
        min_inliers is a MISS (returned, not integrated) so it can be tried against another cell / the
        watchdog. Returns the missed slots (empty when gate=False -- the single-cell path is unchanged)."""
        qs = [self._q[i] for i in slots]
        Ms = rgb.index_fused(qs, Mc, B=max(len(qs), 1))
        missed = []
        for i, M in zip(slots, Ms):
            M = np.asarray(M, float) if M is not None else None
            if M is None or abs(np.linalg.det(M)) < 1.0 or (gate and self._inliers(self._q[i], M) < self.min_inliers):
                if gate:
                    missed.append(i)
                continue
            self._integrate_one(i, M, grid, acc)
        return missed

    def _all_cells(self):
        return [self.Mc] + [e["Mc"] for e in self.extra]

    def _watchdog(self, missed):
        """Blind-index the frames that fit no active cell. Each frame's OWN N-best candidates are
        checked directly against every ALREADY-active cell first and integrated immediately if one
        matches -- this fan-out is already blind-indexing every missed frame to pool votes toward
        new-cell detection below, so checking each result against what we already know costs nothing
        extra, and it rescues a same-cell miss right away instead of only ever detecting a genuinely
        different cell. Uses the blind candidate directly (via _integrate_one) rather than
        re-registering through known-cell, which could just miss again for the same reason the frame
        was flagged. Only frames no active cell explains feed the pooled consensus below; a new cell
        is added when one RECURS there (sample change). Aliases from ordinary failures scatter and
        never accumulate, so this does not thrash on junk -- the same specificity that refuses
        non-crystals."""
        if self._watch is None:
            self._watch = RunningConsensus(min_support=3, gap=2, adaptive=False)
        cells = self._all_cells()
        still_missed = []
        # Fan the independent blind indexes across workers (default `_fanout` is the serial
        # single-GPU loop, so the votes -- and the lock -- are BIT-IDENTICAL to the old per-frame
        # loop; RunningConsensus is a histogram, add_frame order does not change the verdict.
        # Opt-in mpi_fanout distributes the frames across GPUs).
        for i, nb in zip(missed, self._fanout([self._q[i] for i in missed], self.warmup_nbest)):
            q = self._q[i]
            rescued = False
            for c, _ in nb:
                c = np.asarray(c, float)
                if abs(np.linalg.det(c)) < 1.0 or self._inliers(q, c) < self.min_inliers:
                    continue
                for k, Mk in enumerate(cells):
                    if same_lattice(c, Mk):
                        grid = self.grid if k == 0 else self.extra[k - 1]["grid"]
                        acc = self.acc if k == 0 else self.extra[k - 1]["acc"]
                        self._integrate_one(i, c, grid, acc)
                        self.n_watchdog_rescued += 1
                        rescued = True
                        break
                if rescued:
                    break
            if rescued:
                continue
            self._watch.add_frame([c for c, _ in nb])
            still_missed.append(i)
        missed = still_missed
        Mn = self._watch.verdict()[0]
        if Mn is None:
            return
        Mn = _conventional_tetragonal(np.asarray(Mn, float))
        if self._alias_gate is not None:
            # Deterministic single-lock confirmation. Score the gate on the ONE missed frame that best fits
            # the voted cell -- the missed frames are at DIFFERENT orientations, so a pooled cloud has no
            # common lattice fit and tightness would be noise; a single well-fitting frame is one
            # orientation, which is what coverage/occupancy need. Returns the leader (confirmed), a tighter
            # derivative lattice (adopt mode), or None (refuse -> do NOT reset self._watch, so the histogram
            # keeps accumulating for a later, cleaner lock).
            cand_q, best_ni = None, -1
            for i in missed:
                q = self._q[i]
                if q is None:
                    continue
                ni = self._inliers(q, Mn)
                if ni > best_ni:
                    best_ni, cand_q = ni, q
            if cand_q is not None:
                Mg = self._alias_gate.confirm(Mn, cand_q)
                if Mg is None:
                    return
                Mn = _conventional_tetragonal(np.asarray(Mg, float))
        if any(same_lattice(Mn, Mc) for Mc in self._all_cells()):
            return
        lock_z = None
        if self._lock_probe and self._known_index is not None:
            # "Does this lock rest on real signal?" -- index the just-voted frames against the new cell
            # with the SAME q-only indexer the rescue/gate use (convention-safe: it returns a per-frame
            # q @ M = hkl matrix, which is what _inliers/null_margin expect -- the stored voted cell is a
            # reduced-cell reference, a different convention). Take the best-fitting supporting frame and
            # score its overlap against the random-orientation floor (glint.spurious_meter.null_margin).
            # Seeded by n_relock for a reproducible number. lock_min_z (if set) refuses a too-weak lock
            # WITHOUT resetting self._watch, so votes keep accumulating for a cleaner later lock.
            from glint.spurious_meter import null_margin
            qs = [self._q[i] for i in missed if self._q[i] is not None]
            Ms = self._known_index(qs, Mn, B=max(len(qs), 1)) if qs else []
            best_q, best_M, best_ni = None, None, -1
            for q, Mi in zip(qs, Ms):
                if Mi is None:
                    continue
                ni = self._inliers(q, Mi)
                if ni > best_ni:
                    best_q, best_M, best_ni = q, Mi, ni
            lock_z = (null_margin(best_q, best_M, n_null=self._probe_null,
                                  rng=np.random.default_rng(self.n_relock))["z"]
                      if best_M is not None and best_ni >= self.min_inliers else 0.0)
            self.lock_z = lock_z
            if self._lock_min_z is not None and lock_z < self._lock_min_z:
                return
        self.extra.append(dict(Mc=Mn, grid=HKLGrid(Mn, self.dmin, gpu=self.gpu),
                               nth=theoretical_unique(Mn, self.dmin, self.ops),
                               acc=MergeAccumulator(self.snr_bins, self.ops), lock_z=lock_z))
        self.n_relock += 1
        if self._missbuf:                                       # retroactive INDEX-ONLY rescue of buffered misses
            # Re-index the buffered pre-lock misses against the newly locked cell Mn with the same
            # q-only fast-path indexer the batch uses (rgb.index_fused: q in, [M or None] out, no
            # pixels). Selection-misses (the recurring new cell) pass the same _inliers gate; blank/
            # spurious "generation" misses are correctly refused. No pixels, no integrate -- that is
            # the deferred merge-rescue layer.
            qs = [q for _, q in self._missbuf]
            Ms = self._known_index(qs, Mn, B=max(len(qs), 1))
            self.n_rescued += sum(1 for q, M in zip(qs, Ms)
                                  if M is not None and abs(np.linalg.det(np.asarray(M, float))) >= 1.0
                                  and self._inliers(q, M) >= self.min_inliers)
            self._missbuf.clear()
        self._watch = RunningConsensus(min_support=3, gap=2, adaptive=False)    # reset for the next change

    def flush(self):
        """Index the resident batch, integrate each frame against its still-resident pixels."""
        if self._blind or self._n == 0:                         # nothing to integrate without a cell
            return
        slots = [i for i in range(self._n) if self._q[i] is not None]
        if slots and not self.adaptive_relock:
            self._index_integrate(slots, self.Mc, self.grid, self.acc, gate=False)   # single-cell path (unchanged)
        elif slots:
            remaining = self._index_integrate(slots, self.Mc, self.grid, self.acc, gate=True)
            for e in self.extra:                                # try each additional active cell in turn
                if not remaining:
                    break
                remaining = self._index_integrate(remaining, e["Mc"], e["grid"], e["acc"], gate=True)
            if remaining:                                       # fit no active cell -> blind watchdog
                if self._missbuf is not None:                   # buffer q-only for retroactive rescue on lock
                    self._missbuf.extend((self.n_pushed, self._q[i].copy()) for i in remaining)
                self._watchdog(remaining)
        self._n = 0
        self._q = [None] * self.B
        self._pk = [None] * self.B

    def stats(self, thr=0.0):
        if self._blind:                                         # not yet locked -- warm-up in progress
            _, sup, lead = self._rc.verdict()
            return dict(locked=False, pushed=self.n_pushed, warmup_indexed=self.n_warmup,
                        consensus_support=sup, consensus_lead=lead)
        s = self.acc.stats(thr=thr, n_theoretical=self.n_theoretical)
        s.update(locked=True, locked_after=self.locked_after, consensus_support=self.consensus_support,
                 pushed=self.n_pushed, indexed=self.n_indexed, integrated=self.n_integrated,
                 theoretical_unique=self.n_theoretical)
        if self.warmup_rescue:
            s["n_warmup_rescued"] = self.n_warmup_rescued         # warm-up frames recovered the instant the cell locked
        if self.adaptive_relock:                                # adaptive: report the active cell set
            s["n_cells"] = 1 + len(self.extra); s["n_relock"] = self.n_relock
            s["n_rescued"] = self.n_rescued                      # buffered pre-lock misses recovered on relock
            s["n_watchdog_rescued"] = self.n_watchdog_rescued    # same-cell misses the watchdog rescued individually
            if self._lock_probe:
                s["lock_z"] = self.lock_z                        # signal-strength z of the most recent relock
            s["extra_cells"] = [dict(axes=list(np.linalg.norm(e["Mc"], axis=0).round(1)),
                                     **e["acc"].stats(thr=thr, n_theoretical=e["nth"])) for e in self.extra]
        if self.double_hit:
            s["n_double"] = self.n_double
            s["double_hit_rate"] = self.n_double / max(self.n_indexed, 1)
        if self._grefiner is not None:
            s["geom_correction"] = self._grefiner.correction()
        return s
