"""Device-resident streaming driver: peakfind -> index -> integrate -> running merge, with the
detector frame staying on the GPU for the whole chain.

Blind or known-cell. Constructed with a cell (Mc) it runs known-cell from frame 0. Constructed with
Mc=None it starts BLIND: it indexes the first frames one at a time (~26 ms/frame), accumulates a
running-histogram cross-frame consensus (glint.running_consensus), and the instant the cell LOCKS it
builds the hkl grid and switches to the batched known-cell path (~0.21 ms/frame at the driver's
default B=64; 0.17 at B=120) for the rest of the
run -- so a live run needs no cell handed in, only a ~150 ms discovery transient.

Why a driver is needed at all. Each stage is fast on its own (index 0.21 ms/frame at the default
B=64 -- 0.17 at B=120 -- integrate
0.33 ms/frame), but they have opposite batching requirements: indexing wants MANY frames at once
(the anneal and refine kernels take one thread-block per frame, so the batch sets GPU occupancy;
obj splits its candidates across blocks since #165 but that did not remove the need to batch).
B=64 is the driver default and is NOT a saturation point: B=120 measures 26% faster (0.214 ->
0.170 ms/frame fp64), so the batch is a latency/throughput trade, not a knee. Integration is
per-frame and needs THAT frame's pixels. Uploading the frame twice would cost ~3-5 ms
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
import warnings
from collections import Counter, OrderedDict, deque
import numpy as np

try:
    import cupy as cp
    _HAVE_CP = True
except Exception:                                            # pragma: no cover - CPU-only env
    cp = None
    _HAVE_CP = False

from glint.lattice import LENGTH_ORDER_LAUE, UNIQUE_C_LAUE, cell_to_Ar, standardize_axes
from glint.lute_bridge import peaks_to_q
from glint.predict import (predict_spots, integrate_spots, recip_from_M, _canonical_axes,
                           _hkl_grid, project_q)
from glint.peakfinder_v4 import PeakFinderV4
from glint.running_consensus import RunningConsensus
from glint.multishot import same_lattice
from glint.multilattice import deflate_peaks
from glint.retry_cascade import DEFAULT_NBEST, arm_blind_nbest, arm_known_perframe
from glint.spurious_meter import HKL_TOL as _SPURIOUS_HKL_TOL

# _inliers' near-integer window on q @ M -- the tolerance THE live gate (_fits) counts peaks at.
# spurious_meter.HKL_TOL declares itself "== stream_driver._inliers near-integer window"; that
# equality was a comment until now, so check it where either side would break it. All of
# spurious_meter's lattice meters are calibrated against this exact window.
HKL_TOL = 0.15
# An explicit raise, not `assert`: python -O strips asserts, and an invariant that vanishes under
# optimization is a comment with extra steps -- the exact defect this check exists to fix
# (Copilot review of #170).
if HKL_TOL != _SPURIOUS_HKL_TOL:
    raise ImportError(
        "stream_driver.HKL_TOL and spurious_meter.HKL_TOL have drifted apart "
        f"({HKL_TOL} vs {_SPURIOUS_HKL_TOL}); the live gate and the spurious meters must count "
        "inliers at the same window")

# The live gate's chance floor for StreamDriver(null_floor=...) (opt-in): n_inl >= a*n + b + c*sqrt(n), as
# (a, b, c). Fitted on cxidb-17 (480 frames, peakfinder8 peaks, the locked lysozyme cell, the shipped
# index_fused depth, HKL_TOL 0.15) at the 99th percentile of 32 azimuth-scrambled copies per frame --
# experiments/live_gate_null.py, experiments/RESULTS_live_gate_null.md. Not portable as is: re-fit for
# another detector, peak finder, cell family or search depth.
NULL_FLOOR_CXIDB17 = (0.0224, 5.32, 1.211)

try:
    import glint.replica_gpu_batch as rgb                     # the q-only batch indexer (needs torch)
except Exception:                                            # pragma: no cover - CPU-only unit env (no torch)
    rgb = None                                               # batch/rescue indexers are injected as fakes there


# ------------------------------------------------------------- orientation-invariant hkl grid ----
# Fused predict gate: matmul(g@R) + Ewald/qmax test + atomic compaction, one kernel + one D2H
# (replaces the ~6 cupy ops + full-Nhkl qg/qn2/exc intermediates). Survivors written unordered;
# the caller argsorts by hkl index to restore grid order, so the result matches the eager path.
_GATE_SRC = r"""
// R arrives as nine SCALAR arguments, not a device pointer. It is 72 bytes, but shipping it as an
// array cost 0.045 ms/frame (10.6% of predict) -- cp.asarray allocates a fresh device array every
// call, then ascontiguousarray/ravel may allocate again, and a pageable source forces the driver to
// stage through its own pinned buffer. As kernel arguments the values ride in constant memory and are
// broadcast-read by every thread, which is also strictly better than the global load they replace.
extern "C" __global__ void predict_gate(const double* g, int nhkl,
                                         double R0, double R1, double R2, double R3, double R4,
                                         double R5, double R6, double R7, double R8, double wave,
                                         double qmax2, double tol, double* out, int* counter, int cap){
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= nhkl) return;
  double gx=g[3*i], gy=g[3*i+1], gz=g[3*i+2];
  double qx=gx*R0+gy*R3+gz*R6;
  double qy=gx*R1+gy*R4+gz*R7;
  double qz=gx*R2+gy*R5+gz*R8;
  double qn2=qx*qx+qy*qy+qz*qz;
  double exc=qz+0.5*wave*qn2;
  if (qn2<=qmax2 && fabs(exc)<tol){
    int p=atomicAdd(counter,1);
    if (p<cap){ out[6*p]=(double)i; out[6*p+1]=qx; out[6*p+2]=qy; out[6*p+3]=qz; out[6*p+4]=qn2; out[6*p+5]=exc; }
  }
}"""

# Gate WITH the detector projection fused in. The gate above emits every Ewald survivor anywhere in
# reciprocal space; the host then ran project_q over all of them and discarded the ones that miss.
# On a node owning a few ASICs of a ~25-panel detector that is most of them -- 195 survivors for 94
# real on-panel reflections -- and project_q was 40% of predict.
#
# The thread already holds qx,qy,qz in registers and the panel geometry is frame-INVARIANT (uploaded
# once), so projecting here costs ~20 flops in a kernel that is launch-bound, and buys three things
# at once: project_q leaves the host entirely, the cull becomes EXACT rather than a conservative
# cone, and every emitted row is on-panel so the host's boolean masking over eight struct fields
# disappears. NOTE this is not the known-negative "on-device project_q", which was a SEPARATE kernel
# and lost 3.5x to launch overhead; fusing into a kernel already running adds no launch.
#
# The arithmetic mirrors project_q's evaluation order (predict.py) so the results agree to ~1e-13.
_GATE_PROJ_SRC = r"""
extern "C" __global__ void gate_project(const double* g, int nhkl,
    double R0,double R1,double R2,double R3,double R4,double R5,double R6,double R7,double R8,
    double wave, double qmax2, double tol, const double* pg, int npan,
    double* out, int* counter, int cap){
  int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= nhkl) return;
  double gx=g[3*i], gy=g[3*i+1], gz=g[3*i+2];
  double qx=gx*R0+gy*R3+gz*R6, qy=gx*R1+gy*R4+gz*R7, qz=gx*R2+gy*R5+gz*R8;
  double qn2=qx*qx+qy*qy+qz*qz;
  double exc=qz+0.5*wave*qn2;
  if (!(qn2<=qmax2 && fabs(exc)<tol)) return;
  double sx=wave*qx, sy=wave*qy, sz=wave*qz+1.0;          // s_hat = wave*q + zhat, then normalise
  double nrm=sqrt(sx*sx+sy*sy+sz*sz);
  sx/=nrm; sy/=nrm; sz/=nrm;
  if (!(sz > 1e-6)) return;                                // forward-scattered only
  for (int p=0;p<npan;++p){                                // first panel that catches it wins
    const double* P = pg + p*12;                           // Zp,res,inv(A)[4],cx,cy,min_fs,max_fs,min_ss,max_ss
    double t = P[0]/sz;
    double rx = P[1]*(sx*t) - P[6], ry = P[1]*(sy*t) - P[7];
    double f = P[8] + (P[2]*rx + P[3]*ry);
    double s = P[10] + (P[4]*rx + P[5]*ry);
    if (f>=P[8] && f<=P[9] && s>=P[10] && s<=P[11]){
      int k=atomicAdd(counter,1);
      if (k<cap){ out[6*k]=(double)i; out[6*k+1]=f; out[6*k+2]=s;
                  out[6*k+3]=(double)p; out[6*k+4]=exc; out[6*k+5]=1.0/sqrt(qn2); }
      return;
    }
  }
}"""
_GATE_KERNEL = cp.RawKernel(_GATE_SRC, "predict_gate") if _HAVE_CP else None
_GATE_PROJ_KERNEL = cp.RawKernel(_GATE_PROJ_SRC, "gate_project") if _HAVE_CP else None


def _panel_geom(panels, clen_m):
    """Flatten panel geometry for the fused gate+project kernel: 12 doubles per panel, in the order
    the kernel indexes them -- Zp, res, inv(A) as 4, corner (cx, cy), then the fs/ss bounds.

    Everything here is frame-INVARIANT, which is the whole point: project_q rebuilt `A`, its 2x2
    inverse and the corner vector on EVERY call. Uploaded once, the kernel reads it from global memory
    (a handful of doubles, broadcast across the block) and the host never touches it again.
    Returns None when the geometry cannot be projected (a panel at or behind the sample).
    """
    rows = []
    for p in panels:
        Zp = clen_m + p.get("coffset", 0.0)
        if Zp <= 0:
            return None
        A = np.array([[p["fs"][0], p["ss"][0]], [p["fs"][1], p["ss"][1]]], float)
        if abs(np.linalg.det(A)) < 1e-12:
            return None                                   # degenerate fs/ss basis
        Ai = np.linalg.inv(A)
        rows.append([Zp, float(p["res"]), Ai[0, 0], Ai[0, 1], Ai[1, 0], Ai[1, 1],
                     float(p["cx"]), float(p["cy"]),
                     float(p["min_fs"]), float(p["max_fs"]),
                     float(p["min_ss"]), float(p["max_ss"])])
    return np.asarray(rows, float).ravel() if rows else None


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

    def __init__(self, Mc, dmin, margin=1.02, gpu=True,
                 panels=None, clen_m=None, wavelength_A=None):
        self.Mc_ref = np.asarray(Mc, float).copy()   # the setting this grid's hkl box was sized in
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
        # Fused gate+project geometry, uploaded once (see _panel_geom / _GATE_PROJ_SRC). predict()
        # checks panel IDENTITY before using it, so a caller passing different panels falls back to
        # the plain gate + host project_q rather than getting a silently wrong answer.
        self._pan_panels = panels
        self._pan_geom = None
        self._npan = 0
        if panels is not None and clen_m is not None and self.gpu:
            pg = _panel_geom(panels, float(clen_m))
            if pg is not None:
                self._pan_geom = cp.asarray(pg)
                self._npan = len(panels)
        # PINNED staging for the two device->host reads. Both move trivial amounts (4 B for the
        # counter, n*48 B for the payload) but cost 0.032 and 0.031 ms/frame through CuPy: `int(arr[0])`
        # builds a 0-d array and round-trips it through asnumpy, and cp.asnumpy allocates a fresh
        # pageable host array which the driver must then stage. Raw memcpyAsync into preallocated
        # PINNED memory skips all of that. The payload buffer grows to a high-water mark rather than
        # being sized for the whole grid -- pinned memory is a scarce, global resource on a DAQ node,
        # and after the panel cone the survivor count is a few hundred, not ~1e5.
        self._pin = None
        if self.gpu:
            self._pin_cnt_mem = cp.cuda.alloc_pinned_memory(4)
            self._pin_cnt = np.frombuffer(self._pin_cnt_mem, np.int32, 1)
            self._grow_pin(4096)

    def _grow_pin(self, nrows):
        """(Re)allocate the pinned payload buffer to hold at least `nrows` survivors."""
        nrows = min(int(nrows), int(self.g.shape[0]))
        self._pin_mem = cp.cuda.alloc_pinned_memory(nrows * 6 * 8)
        self._pin = np.frombuffer(self._pin_mem, np.float64, nrows * 6).reshape(nrows, 6)

    def predict(self, M_or_R, panels, clen_m, wavelength_A, tol=0.006, is_recip=False):
        """Same result as predict_spots(..., dmin=self.dmin, tol=tol) with the grid reused."""
        R = np.asarray(M_or_R, float) if is_recip else recip_from_M(M_or_R)
        if self.gpu:
            # fused kernel: matmul + Ewald/qmax gate + compaction in one pass, then ONE D2H
            nhkl = self.g.shape[0]
            st = cp.cuda.get_current_stream()
            D2H = cp.cuda.runtime.memcpyDeviceToHost
            # counter reset without a CuPy scalar assignment (which was 0.015 ms/frame of API)
            cp.cuda.runtime.memsetAsync(self._gcnt.data.ptr, 0, 4, st.ptr)
            Rr = np.ascontiguousarray(R, np.float64).ravel()
            tpb = 256; blocks = (nhkl + tpb - 1) // tpb
            f8 = np.float64
            # fused path only when the geometry was built for THESE panels
            fused = self._pan_geom is not None and panels is self._pan_panels
            args = (self._ggr, np.int32(nhkl),
                    f8(Rr[0]), f8(Rr[1]), f8(Rr[2]), f8(Rr[3]), f8(Rr[4]),
                    f8(Rr[5]), f8(Rr[6]), f8(Rr[7]), f8(Rr[8]), f8(wavelength_A),
                    f8(self.qmax * self.qmax), f8(tol))
            if fused:
                _GATE_PROJ_KERNEL((blocks,), (tpb,), args + (
                    self._pan_geom, np.int32(self._npan),
                    self._gout.ravel(), self._gcnt, np.int32(nhkl)))
            else:
                _GATE_KERNEL((blocks,), (tpb,), args + (
                    self._gout.ravel(), self._gcnt, np.int32(nhkl)))
            cp.cuda.runtime.memcpyAsync(self._pin_cnt.ctypes.data, self._gcnt.data.ptr, 4, D2H, st.ptr)
            st.synchronize()
            n = int(self._pin_cnt[0])
            if n > len(self._pin):
                self._grow_pin(max(2 * len(self._pin), n))
            if n:
                cp.cuda.runtime.memcpyAsync(self._pin.ctypes.data, self._gout.data.ptr,
                                            n * 6 * 8, D2H, st.ptr)
                st.synchronize()
            C = self._pin[:n].copy()          # copy out of the reused pinned buffer
            C = C[np.argsort(C[:, 0], kind="stable")]         # restore hkl-grid order
            idx = C[:, 0].astype(np.int64)
            hkl = self.g[idx]
            if fused:
                # every row is already on-panel and carries its own fs/ss/panel/res, so the host does
                # no projection and no boolean masking -- just relabel the columns into the record.
                out = np.zeros(n, dtype=[("h", int), ("k", int), ("l", int), ("fs", float),
                                         ("ss", float), ("panel", int), ("exc", float),
                                         ("res", float)])
                out["h"], out["k"], out["l"] = hkl[:, 0], hkl[:, 1], hkl[:, 2]
                out["fs"], out["ss"] = C[:, 1], C[:, 2]
                out["panel"] = C[:, 3].astype(int)
                out["exc"], out["res"] = C[:, 4], C[:, 5]
                return out
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


# ------------------------------------------------------------ asymmetric unit (Laue classes) -----
# Operators act on Miller indices as integer 3x3 matrices, hkl' = op @ hkl (canon() applies them as
# `hkl @ op.T`). A point-group operation with rotation part W (on fractional coordinates) acts on
# hkl as W^-T; since the set {W^-T} over a group is the set of transposes of that group, generating
# from the TRANSPOSED rotation parts gives the right hkl group. Settings are the conventional ones
# -- unique axis c for the tetragonal/trigonal/hexagonal classes, b for monoclinic (2/m_ua* name
# the other two), hexagonal axes for the trigonal classes unless the name ends in _R.
_I3 = np.eye(3, dtype=int)
_INV = -_I3
_R4_C = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])           # 4-fold about c
_R2_A = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])          # 2-fold about a
_R2_B = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])          # 2-fold about b
_R2_C = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])          # 2-fold about c
_R3_C_HEX = np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]])      # 3-fold about c, hexagonal axes
_R6_C_HEX = np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 1]])       # 6-fold about c, hexagonal axes
_R2_A_HEX = np.array([[1, 0, 0], [-1, -1, 0], [0, 0, -1]])     # 2-fold about a, hexagonal axes (-3m1)
_R2_AB_HEX = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])    # 2-fold about [1-10], hexagonal (-31m)
_R3_111 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])          # 3-fold about [111] (cubic / rhombohedral)
_R2_AB_R = _R2_AB_HEX                                          # [1-10] 2-fold reads the same in rhombohedral axes

# Rotation generators per Laue class; the inversion is appended to every entry by laue_ops(). The
# 4/mmm entry is (4-fold c, 2-fold a) in that order so laue_ops_4mmm() returns the list it always
# has, element for element (pinned by experiments/test_laue_ops.py against a frozen copy).
_LAUE_GENERATORS = {
    "-1": (),
    "2/m_uaa": (_R2_A,), "2/m_uab": (_R2_B,), "2/m_uac": (_R2_C,),
    "mmm": (_R2_A, _R2_B),
    "4/m": (_R4_C,), "4/mmm": (_R4_C, _R2_A),
    "-3": (_R3_C_HEX,), "-3m1": (_R3_C_HEX, _R2_A_HEX), "-31m": (_R3_C_HEX, _R2_AB_HEX),
    "-3_R": (_R3_111,), "-3m_R": (_R3_111, _R2_AB_R),
    "6/m": (_R6_C_HEX,), "6/mmm": (_R6_C_HEX, _R2_A_HEX),
    "m-3": (_R3_111, _R2_C), "m-3m": (_R3_111, _R4_C),
}
# Short names -> the setting they mean. "-3m" is the -3m1 setting (2-folds along a, b, a+b: R-3m,
# P-3m1, P321), which is the common one; a P312-type crystal must ask for "-31m" explicitly.
_LAUE_ALIASES = {
    "2/m": "2/m_uab", "-3m": "-3m1",
    "-3_H": "-3", "-3m1_H": "-3m1", "-31m_H": "-31m", "-3m_H": "-3m1",
    "m3": "m-3", "m3m": "m-3m",
}
# The eleven Laue classes by their usual names -- what a CLI `choices=` should offer.
LAUE_CLASSES = ("-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m")
# Classes whose operators need the 4-fold axis in c, i.e. the ones _conventional_tetragonal serves.
TETRAGONAL_LAUE = ("4/m", "4/mmm")
# Lattice type (CrystFEL stream vocabulary, plus "trigonal") -> holohedry, the Laue class ASSUMED when
# only the lattice is known. A crystal of lower symmetry on the same lattice (4/m on tetragonal, -3 or
# -3m on hexagonal P, 6/m, m-3) has to name its class through `laue=` -- the lattice cannot tell.
_HOLOHEDRY = {
    "triclinic": "-1", "monoclinic": "2/m", "orthorhombic": "mmm", "tetragonal": "4/mmm",
    "trigonal": "-3m", "rhombohedral": "-3m_R", "hexagonal": "6/mmm", "cubic": "m-3m",
}


def _close_group(gens):
    """Close a generator list under multiplication (breadth-first, first-seen order)."""
    G = [_I3.copy()]
    ch = True
    while ch:
        ch = False
        for g in list(G):
            for s in gens:
                h = (s @ g).astype(int)
                if not any(np.array_equal(h, x) for x in G):
                    G.append(h); ch = True
    return G


def laue_name(name):
    """Canonical registry key for a Laue-class name (aliases resolved); ValueError if unknown."""
    key = _LAUE_ALIASES.get(str(name).strip(), str(name).strip())
    if key not in _LAUE_GENERATORS:
        raise ValueError(f"unknown Laue class {name!r}; known: {', '.join(LAUE_CLASSES)} "
                         f"(settings: {', '.join(sorted(set(_LAUE_GENERATORS) - set(LAUE_CLASSES)))})")
    return key


def laue_ops(name):
    """Operator list (integer 3x3, inversion included) for a Laue class, for MergeAccumulator /
    theoretical_unique / _asu_key. Orders: -1 2, 2/m 4, mmm 8, 4/m 8, 4/mmm 16, -3 6, -3m 12,
    6/m 12, 6/mmm 24, m-3 24, m-3m 48."""
    return _close_group(list(_LAUE_GENERATORS[laue_name(name)]) + [_INV])


def laue_ops_4mmm():
    """16 Laue 4/mmm operators (4-fold c, 2-fold a, inversion). Same generators as merge_stats.py."""
    return laue_ops("4/mmm")


def _op_set(ops):
    """Operator list -> a hashable set, so two Laue classes can be compared as groups."""
    return {tuple(np.asarray(o, int).ravel()) for o in ops}


def _is_subgroup(sub, sup):
    """Is every operator of `sub` present in `sup`? This is the test for whether a Laue class is a
    LEGITIMATE lower-symmetry choice on a lattice whose holohedry is `sup` -- 4/m under 4/mmm, -3 or
    -3m under 6/mmm, m-3 under m-3m, 2/m_uab under mmm -- as opposed to a genuine contradiction (a
    monoclinic unique axis the header does not name, or a class from another crystal system). An
    equality test would flag every one of the lower classes `laue=` exists to express (glint#186
    review), since the header carries the lattice and never the point group."""
    return _op_set(sub) <= _op_set(sup)


def laue_from_symmetry(sym):
    """Laue class implied by a CrystFEL-style symmetry record {lattice_type, centering, unique_axis}
    -- the one StreamDriver(stream_symmetry=...) stamps on every chunk -- or None when it names no
    lattice_type. Assumes the HOLOHEDRY (see _HOLOHEDRY): the header carries the lattice, not the
    point group, so this is the most the record can support; a lower class is `laue=`'s job.
    Monoclinic honours unique_axis (a/b/c; '*' or absent means b). The c-unique classes warn when the
    record says the unique axis is elsewhere, because the operator set assumes c and cannot follow."""
    sym = dict(sym or {})
    lt = sym.get("lattice_type")
    if lt is None:
        return None
    lt = str(lt).strip().lower()
    if lt not in _HOLOHEDRY:
        raise ValueError(f"unknown lattice_type {lt!r}; known: {', '.join(_HOLOHEDRY)}")
    ua = str(sym.get("unique_axis") or "*").strip().lower()
    if lt == "monoclinic":
        if ua in ("a", "b", "c"):
            return f"2/m_ua{ua}"
        if ua != "*":
            raise ValueError(f"unknown monoclinic unique_axis {ua!r}; known: *, a, b, c")
    if lt in ("tetragonal", "trigonal", "hexagonal") and ua not in ("*", "c"):
        warnings.warn(f"lattice_type {lt} with unique_axis {ua!r}: the {_HOLOHEDRY[lt]} operator set "
                      "assumes the unique axis in c; the live merge will use c (glint#180)")
    return _HOLOHEDRY[lt]


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
def _relabel_like(M, ref):
    """Relabel a frame's axes into the REFERENCE's column order.

    The known-cell indexer anchors on the shortest axis and returns its columns shortest-first --
    `replica_gpu._axes_from_cell` sorts the reference's columns by length and the returned basis is
    (short, mid, long) in the frame's orientation, whatever order the reference itself was written in
    (glint#186 review). For the classes whose standardizer is the identity (triclinic, the monoclinic
    settings, rhombohedral, cubic) that leaves the frame in a different labelling from the grid and
    the operators: a monoclinic reference with its unique axis in b, handed back shortest-first, has
    that axis wherever its length puts it.

    Undoing the sort is exact, because it IS the permutation the indexer applied: column k of the
    returned basis is the reference's column `argsort(lengths)[k]`. That premise is the ENGINE's to
    keep, not this function's: it only permutes columns, so a basis in another setting of the same
    lattice -- or a mirrored, non-lattice basis with the reference metric -- passes through unrepaired.
    Before review r2 s4-01 the known-cell engines returned such bases on ~half of triclinic and most
    monoclinic frames; replica_gpu._both_hands and _ref_setting now return the reference setting
    (experiments/test_kc_setting.py checks it against ground truth). Length ties are broken the same
    way at both ends (`kind="stable"`), so the round trip is the identity when the reference is
    already shortest-first."""
    M = np.asarray(M, float)
    order = np.argsort(np.linalg.norm(np.asarray(ref, float), axis=0), kind="stable")
    out = np.empty_like(M)
    out[:, order] = M
    return out


def _conventional_tetragonal(M):
    """Permute a tetragonal cell's columns so the unique (4-fold) axis is c, matching laue_ops_4mmm.

    Buerger reduction orders axes by length, so the short 4-fold axis of a cell like 79/79/38 can land
    in column a. The two equal-length columns are taken as a,b; the length outlier becomes c.
    Handedness is preserved (negate one column if the permutation flipped the determinant sign).

    A thin wrapper over ``glint.lattice.standardize_axes`` -- the SAME function ``_canonical_axes``
    applies to every accepted frame in ``_integrate_one``, so the reference cell the ``HKLGrid`` and
    the 4/mmm operators are built on and the frames predicted against it can no longer land in
    different settings (glint#181: for c > a cells the frame's 4-fold ended up in b and the grid
    missed ~19% of its predictions). Cells with no equal pair fall back to (long, long, short)."""
    return standardize_axes(M, laue="4/mmm")


class _PixelStore:
    """Detector frames kept past their batch for a LATER rescue (StreamDriver's rescue_pixels): the
    warm-up frames until the cell locks, the buffered misses until a relock. Keyed by arrival index.

    `cap` frames are preallocated, like the ring, so the memory is reserved when the driver is built
    and nothing is allocated per frame. When the store is full the OLDEST frame is evicted; its rescue
    then stays index-only, as it is without the store."""

    def __init__(self, cap, shape, dtype, xp):
        self.xp = xp
        self._buf = [xp.zeros(shape, dtype) for _ in range(int(cap))]
        self._free = list(range(len(self._buf)))
        self._at = OrderedDict()                      # ev -> (buffer index, src, pkq), oldest first
        self.n_evicted = 0

    def __len__(self):
        return len(self._at)

    def put(self, ev, frame, src=None, pkq=None):
        """Copy `frame` in (device to device for a ring slot, one upload for a host array)."""
        if ev in self._at:
            self.release([ev])
        if not self._free:
            _, (b, _, _) = self._at.popitem(last=False)
            self._free.append(b)
            self.n_evicted += 1
        b = self._free.pop()
        self._buf[b][...] = self.xp.asarray(frame)
        self._at[ev] = (b, src, pkq)

    def get(self, ev):
        """(frame, src, pkq) for arrival index `ev`, or None if it was never kept or was evicted."""
        e = self._at.get(ev)
        return None if e is None else (self._buf[e[0]], e[1], e[2])

    def release(self, evs):
        for ev in evs:
            e = self._at.pop(ev, None)
            if e is not None:
                self._free.append(e[0])


# Depth tiers for the adaptive effort policy: (topa, nc, full_grid, ms per frame). The costs are the known-cell
# depth sweep on the cxidb-17 480 (one A100, B=120, exp/batched-escalation): 308 / 341 / 365 / 379 frames indexed at
# 0.15 / 0.27 / 0.40 / 0.94 ms per frame. Tier 0 is the shipped search (replica_gpu's topa=8, nc=NC=16, adaptive
# azimuth grid); the last tier is also the depth of the deep search on the misses.
EFFORT_TIERS = ((8, 16, False, 0.15), (32, 16, False, 0.27), (32, 32, False, 0.40), (128, 32, True, 0.94))


class _EffortPolicy:
    """Effort that follows the hit rate (StreamDriver's `effort=`). The GPU time each hit may have is
    1000 * n_gpu / (rate_hz * hit_rate) ms per hit, so the veto count sets the search depth. It is spent in two
    steps, both decided from the recent history at flush boundaries only, so a run is reproducible from its log:
      1. the fast path runs EVERY hit at the deepest tier whose per-frame cost fits the budget;
      2. if what is left also covers the misses' deep searches -- the miss fraction of the recent flushes times
         (1 + k_null) searches at the top tier -- the misses of a flush get the top-tier search with the
         azimuth-scrambled null of glint.retry_cascade.escalate_batch (glint#211). Only below the top tier: at the
         top the deep search would repeat the fast path's own search, and the null can only reject. k_null
         defaults to 32, the setting glint#211 measured 0 null accepts at (a lattice-free frame beats all k_null
         exchangeable copies with probability <= 1/(k_null + 1)); at 32 copies the deep search is affordable only
         while misses are rare, which is the point.
    The hit rate is estimated from the last `hit_window` frames pushed while locked (a frame with too few peaks to
    index is a blank), the miss fraction from the last `miss_window` flushes. The FIRST decision ignores the window
    and uses the conservative estimates (`hit_prior` hits per frame, every frame a miss), so a fresh driver's first
    flush runs at the depth the prior affords (tier 0 with the default prior of 1); later decisions use the window.
    With more than one active cell (adaptive relock) every search may be repeated per cell -- best-fit assignment
    searches every frame against every cell, first-fit tries the remaining frames on each further cell, and the
    deep search runs per cell -- so the per-frame costs are multiplied by the number of active cells before they
    are compared with the budget. A log entry's `at` is the arrival index of the first frame the decision applied
    to (the first frame of the flush it was taken for); it stays in force until the next entry.
    `overhead_ms` is the per-hit GPU time that is not search -- integration when the driver integrates (about
    0.33 ms per frame on the fused integrator, see the class docstring), nothing for an index-only stream -- and is
    taken off the budget before the tier is chosen; `n_gpu` is the share of a GPU left to the indexer after peak
    finding. The tier costs are per-frame costs AT THE BATCH SIZE THEY WERE MEASURED (the defaults: B=120 on an
    A100); at a smaller B the same search costs more per frame (index_fused: B=16 is 0.58 ms against B=120's 0.17
    at tier 0), so give `tiers=` measured at the B you run."""

    KEYS = ("rate_hz", "n_gpu", "k_null", "round_copies", "every", "hit_window", "miss_window", "hit_prior",
            "tiers", "seed", "deep_budget", "overhead_ms")

    def __init__(self, cfg):
        if not isinstance(cfg, dict):
            raise ValueError(f"effort must be a dict of settings (rate_hz, n_gpu, ...), got {type(cfg).__name__}")
        unknown = sorted(set(cfg) - set(self.KEYS))
        if unknown:
            raise ValueError(f"effort: unknown setting(s) {unknown}; known: {list(self.KEYS)}")
        if "rate_hz" not in cfg:
            raise ValueError("effort needs rate_hz (frames per second the driver has to keep up with)")
        self.rate_hz = self._pos("rate_hz", cfg["rate_hz"])
        self.n_gpu = self._pos("n_gpu", cfg.get("n_gpu", 1.0))
        self.k_null = self._int("k_null", cfg.get("k_null", 32))     # glint#211's measured setting (0 null accepts)
        self.round_copies = self._int("round_copies", cfg.get("round_copies", 8))
        self.every = self._int("every", cfg.get("every", 1))
        self.hit_window = self._int("hit_window", cfg.get("hit_window", 256))
        self.miss_window = self._int("miss_window", cfg.get("miss_window", 4))
        hp = cfg.get("hit_prior", 1.0)
        if isinstance(hp, bool) or not isinstance(hp, (int, float, np.integer, np.floating)) or not 0.0 < hp <= 1.0:
            raise ValueError(f"effort: hit_prior must be in (0, 1], got {hp!r}")
        self.hit_prior = float(hp)
        self.seed = self._int("seed", cfg.get("seed", 20260926))
        self.deep_budget = self._int("deep_budget", cfg.get("deep_budget", 12000))
        oh = cfg.get("overhead_ms", 0.0)
        if isinstance(oh, bool) or not isinstance(oh, (int, float, np.integer, np.floating)) or not oh >= 0:
            raise ValueError(f"effort: overhead_ms must be a number >= 0, got {oh!r}")
        self.overhead_ms = float(oh)
        try:
            tiers = [(self._int("topa", t[0]), self._int("nc", t[1]), bool(t[2]), float(t[3]))
                     for t in cfg.get("tiers", EFFORT_TIERS)]
        except (TypeError, IndexError):
            raise ValueError("effort: tiers must be a sequence of (topa, nc, full_grid, ms_per_frame)") from None
        if not tiers or any(t[3] <= 0 for t in tiers) or any(b[3] <= a[3] for a, b in zip(tiers, tiers[1:])):
            raise ValueError("effort: tiers need strictly increasing positive ms per frame, shallowest first")
        self.tiers = tuple(tiers)
        self.hits = deque(maxlen=self.hit_window)          # 1 per pushed frame that had a q, 0 per blank
        self.flushes = deque(maxlen=self.miss_window)      # (frames, fast-path misses) per flush
        self.tier = None; self.deep = False
        self.budget = self.hit_est = self.miss_frac = None
        self.n_flush = 0; self.log = []
        self.fast_by_tier = Counter(); self.n_tried = self.n_accepted = self.n_searches = 0

    @staticmethod
    def _int(name, v):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1:
            raise ValueError(f"effort: {name} must be an integer >= 1, got {v!r}")
        return int(v)

    @staticmethod
    def _pos(name, v):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not v > 0:
            raise ValueError(f"effort: {name} must be a number > 0, got {v!r}")
        return float(v)

    def seen(self, hit):
        self.hits.append(1 if hit else 0)

    def flushed(self, n, n_miss):
        self.flushes.append((int(n), int(n_miss)))
        self.fast_by_tier[self.tier] += int(n)

    def decide(self, at, n_cells=1):
        """Re-choose (tier, deep) from the recent history; True when either changed. Called at a flush boundary,
        before that flush's fast path, every `every` flushes. `at` = arrival index of the first frame in that flush;
        `n_cells` = active cells the searches are repeated for (see the class docstring)."""
        first = self.n_flush <= 1                            # the first decision: the conservative prior, not the window
        self.hit_est = self.hit_prior if first or not self.hits else max(float(np.mean(self.hits)), 0.02)
        nf = sum(n for n, _ in self.flushes)
        self.miss_frac = 1.0 if first or not nf else sum(m for _, m in self.flushes) / nf
        self.budget = 1000.0 * self.n_gpu / (self.rate_hz * self.hit_est) - self.overhead_ms   # what the SEARCH may spend
        n_cells = max(int(n_cells), 1)
        tier = 0
        for i, t in enumerate(self.tiers):
            if t[3] * n_cells <= self.budget:
                tier = i
        top = len(self.tiers) - 1
        deep = tier < top and (self.tiers[tier][3] + self.miss_frac * (1 + self.k_null) * self.tiers[top][3]) * n_cells <= self.budget
        changed = (tier, deep) != (self.tier, self.deep)
        self.tier, self.deep = tier, deep
        if changed:
            self.log.append(dict(at=int(at), tier=tier, deep=bool(deep), budget_ms=round(self.budget, 4),
                                 hit_est=round(self.hit_est, 4), miss_frac=round(self.miss_frac, 4), n_cells=n_cells))
        return changed

    def depth_kw(self):
        topa, nc, full_grid, _ = self.tiers[self.tier if self.tier is not None else 0]
        return dict(topa=topa, nc=nc, full_grid=full_grid)

    def summary(self):
        return dict(tier=self.tier, deep=self.deep, budget_ms=self.budget, hit_est=self.hit_est,
                    miss_frac=self.miss_frac, n_changes=len(self.log), log=[dict(e) for e in self.log],
                    fast_by_tier={int(k): int(v) for k, v in sorted(self.fast_by_tier.items())},
                    n_deep_tried=self.n_tried, n_escalated=self.n_accepted, deep_searches=self.n_searches)


class StreamDriver:
    """Streaming index+integrate with the frame resident on the device.

    push() one frame at a time; the driver uploads it into a preallocated device ring slot,
    peak-finds it ON DEVICE, and queues the reciprocal vectors. When B frames are queued it indexes
    them as one batch and integrates each against its still-resident pixels, then folds the
    intensities into the running merge. Call flush() at the end of a run, then stats().

    Ring slots: by default every pushed frame takes one, including a frame whose peak list is too
    short to index, so at a 10 % hit rate a B=120 ring holds about 12 frames worth indexing and the
    batch runs at small-batch speed. hits_only=True gives such a frame's slot straight back (it has
    nothing to integrate; its `blank` record is emitted as before), so B counts hits: the ring's
    memory goes to frames that need it, and the batch is the size the engine was timed at.

    Merge symmetry: `laue` names the Laue class the running merge (completeness, CC*, Rsplit and
    the theoretical-unique denominator) is accumulated under -- one of LAUE_CLASSES or a setting such
    as "2/m_uac" / "-31m" (see laue_ops) -- or `ops` supplies the operator list outright. Left unset
    it follows `stream_symmetry`'s lattice_type (its holohedry; laue_from_symmetry), and without
    that it is "4/mmm", the historical default. Blind locks are put in the tetragonal conventional
    setting (4-fold axis in c) only for the tetragonal classes; other classes are merged in the
    setting the cell arrives in, so a known cell must be given in the setting its class assumes.

    Adaptive effort: `effort=dict(rate_hz=..., n_gpu=1)` makes the known-cell search depth follow the hit rate
    (GPU time per hit = n_gpu / (rate x hit rate)) and spends what is left on deep, chance-controlled searches of
    the misses; decisions are taken at flush boundaries and logged (stats()["effort"], "effort" events). Off by
    default. See _EffortPolicy.
    """

    def __init__(self, Mc, panels, clen_m, wavelength_A, shape, dtype=np.uint16, mask=None,
                 B=64, dmin=2.0, tol=0.002, half=3, gap=2, ring=3, min_peaks=6,
                 snr_bins=(0.0, 1.0, 2.0, 3.0, 5.0), pf_kw=None, use_gpu=True,
                 lock_support=3, lock_gap=2, adaptive_gap=True, warmup_nbest=3,
                 adaptive_relock=False, min_inliers=0, min_inlier_frac=0.15,
                 warm_topk=32, warm_floor=1,   # 16 refused real MFX data; see warmup_batch()
                 double_hit=False, geom_refine=False, geom_refine_kw=None,
                 rescue_buffer=0, fanout=None, alias_gate=None,
                 lock_probe=False, probe_null=64, lock_min_z=None, warmup_rescue=False,
                 qc_frac_threshold=None, stream_out=None, stream_geom_text=None,
                 stream_image="glint.cxi", stream_symmetry=None, stream_peaks=None,
                 # APPENDED, not inserted next to the other lock_* options where they belong
                 # by topic: this constructor is not keyword-only, so adding a parameter anywhere
                 # but the end silently rebinds every positional argument after it.
                 lock_frac=0.02, lock_lead=1.5, lock_pool_switch=72,
                 retry_cascade=False, retry_nbest=None, bg_mode="clipmean",
                 laue=None, ops=None,
                 # glint#199: named cell registry + per-frame event trace + peaks-in ingest (all opt-in)
                 roster=None, events=False, on_event=None, cell_window=200,
                 # which active cell takes a frame that fits more than one: "first" (the cascade, the
                 # published behaviour) or "best" (a challenger takes it from the first-fit cell when it
                 # explains at least max(assign_margin, assign_margin_frac * n_peaks) more peaks) --
                 # see _index_best_fit
                 assign="first", assign_margin=8, assign_margin_frac=0.05,
                 # per-lattice scoring of double hits (off by default) -- see _per_lattice_rescue
                 per_lattice=False, per_lattice_below=0.25,
                 # ring slots for hits only, and pixels kept for the retroactive rescues (both off by
                 # default) -- see the class docstring and the warm-up rescue block below
                 hits_only=False, rescue_pixels=0,
                 # adaptive effort (off by default): the search depth follows the hit rate -- see _EffortPolicy
                 effort=None,
                 # a per-peak-count chance floor on the live gate (off by default) -- see min_inlier_frac
                 null_floor=None):
        if use_gpu and not _HAVE_CP:
            raise RuntimeError("cupy required for the device-resident path")
        self.gpu = bool(use_gpu)
        self.panels, self.clen_m, self.wavelength_A = panels, float(clen_m), float(wavelength_A)
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.B, self.dmin, self.tol = int(B), float(dmin), float(tol)
        self.half, self.gap, self.ring_w, self.min_peaks = half, gap, ring, int(min_peaks)
        # Annulus background estimator, handed to whichever integrator runs below. Without it the
        # streaming path could not reproduce pre-glint#131 intensities that the offline path can,
        # which is the asymmetry #130 was about in the other direction.
        from glint.predict import BG_MODES as _BG_MODES
        if bg_mode not in _BG_MODES:
            raise ValueError(f"bg_mode must be one of {_BG_MODES}, got {bg_mode!r}")
        self.bg_mode = bg_mode
        self.warmup_nbest = int(warmup_nbest)
        self.warm_topk, self.warm_floor = int(warm_topk), int(warm_floor)                              # warmup_batch triage
        self.double_hit = bool(double_hit); self.n_double = 0                                          # deflate-and-reindex 2nd lattice
        if self.double_hit:
            from glint.glint_fast import index_blind_nbest
            self._dh_index = index_blind_nbest
            # The bare rule (valid cell + >= min_peaks residual inliers) is a peak-count artifact on
            # peak-rich frames -- 95% real vs 93% azimuth-scrambled on mfxl1038923 r0278, and 96% vs
            # 96% on r0058 (jobs 34409542, 34468402) -- so n_double counts the GATED rule instead
            # (same cell AND >= 15 deg from lattice 1; multilattice.second_lattice_verdict), which
            # measured 3.6% and 4.8% on those two runs against a 0% scrambled floor (job 39151575; the
            # basis-dependent angle before glint#207 read 8.7% and 8.9%). n_double_raw
            # keeps the old count for comparison, and a 1-in-16 azimuth-scramble null measures the
            # gated rule's own false-accept floor live, on the run in front of it.
            self.n_double_raw = 0
            self._dh_n = 0; self.n_dh_null_tested = 0; self.n_dh_null_acc = 0
            self._dh_rng = np.random.default_rng(0xD0B13)

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
        self._idx = [0] * self.B                                # global arrival index per slot (stream provenance)
        self._pkq = [None] * self.B                             # observed peaks aligned with _q (stream_peaks)
        self._haspix = [True] * self.B                          # False = push_peaks/push_q slot: index-only, no pixels to integrate
        self._src = [None] * self.B                             # (image, event) the caller says this slot came from, for the .stream
        self._inl_alt = [None] * self.B                         # assign="best": inlier count under EVERY active cell, for the event
        self._pl = [None] * self.B                              # per_lattice: the frame's second lattice, when one was found
        self._pl_v = [None] * self.B                            # ... and the (verdict, residual) the rescue already computed
        self._rej = {}                                          # per_lattice: best REJECTED registration per slot, this flush
        self._n = 0
        self._frame_no = 0
        self.stream_symmetry = dict(stream_symmetry or {})

        # Merge symmetry (glint#180). This used to be `laue_ops_4mmm()` unconditionally, so any
        # non-tetragonal cell had its completeness/CC*/Rsplit merged under 4/mmm while the stream
        # header (stream_symmetry) could say otherwise. Resolution, first match wins:
        #   ops              an explicit operator list, used verbatim (laue is then only a label;
        #                    None unless given, so no tetragonal standardization is applied);
        #   laue             a Laue-class name (laue_ops); aliases/settings resolve to laue_name();
        #   stream_symmetry  the holohedry of its lattice_type (laue_from_symmetry), so the header an
        #                    offline merger reads and the live merge cannot disagree;
        #   "4/mmm"          the historical default -- bit-identical to before for every caller that
        #                    passes neither.
        # An explicit laue that contradicts stream_symmetry WARNS rather than fails: the operator is
        # the one who knows which of the two is wrong for this sample.
        if ops is not None:
            # Explicit operators win OUTRIGHT, so the header is never consulted: deriving from it
            # first meant an unknown lattice_type or a malformed monoclinic unique_axis could still
            # raise on a path whose operators make the header's symmetry irrelevant (glint#186
            # review). `laue` is then only a reporting label -- see _standardize.
            self.ops = [np.asarray(o, int) for o in ops]
            self.laue = laue_name(laue) if laue is not None else None
            self._ops_explicit = True
            # With explicit ops the label only REPORTS (see _standardize): letting it steer the double-hit
            # gate would make an optional label change verdicts for the same operators (glint#207 review).
            self._laue_known = False
            # A label that does not describe the operators would make stats()["laue"] misreport the
            # class the numbers were merged under (glint#186 review), so it has to agree with them.
            if self.laue is not None and _op_set(self.ops) != _op_set(laue_ops(self.laue)):
                raise ValueError(
                    f"laue={laue!r} does not describe the operators given in ops "
                    f"({len(self.ops)} of them against {len(laue_ops(self.laue))} for "
                    f"{self.laue}): stats() reports this label as the class the merge used, so it "
                    f"cannot name a different one. Pass ops alone to leave the class unlabelled.")
        else:
            derived = laue_from_symmetry(stream_symmetry)              # None without a lattice_type
            self.laue = laue_name(laue if laue is not None else (derived or "4/mmm"))
            self.ops = laue_ops(self.laue)
            self._ops_explicit = False
            # The 4/mmm fallback is a merge default, not knowledge of this sample's class: the double-hit
            # gate (misorientation_deg) gets a Laue class only when the caller or the stream header gave
            # one, and infers the symmetry from the cells otherwise (glint#207).
            self._laue_known = laue is not None or derived is not None
            # A LOWER class on the same lattice is legitimate and is the reason `laue=` exists: the
            # header names the lattice, not the point group, so 4/m on tetragonal, -3/-3m on
            # hexagonal P and m-3 on cubic are all valid and must not be reported as contradictions
            # (glint#186 review). The test is therefore subgroup, not equality -- what remains a
            # real conflict is a class the header's holohedry does not contain, such as a monoclinic
            # unique axis the record puts elsewhere, or a class from another crystal system.
            if (laue is not None and derived is not None
                    and not _is_subgroup(self.ops, laue_ops(derived))):
                warnings.warn(f"laue={laue!r} is not a subgroup of the holohedry implied by "
                              f"stream_symmetry lattice_type="
                              f"{stream_symmetry.get('lattice_type')!r} ({laue_name(derived)}): the "
                              f"live merge uses {laue_name(laue)}, the stream header the other "
                              f"(glint#180)")
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
        self.n_gate_rejected = 0            # refused at ingest with no other cell to try (see _index_integrate)

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

        # Offline-merge handoff (opt-in): append a CrystFEL .stream chunk per integrated frame, so the
        # merge can happen LATER, off the DAQ, from a file -- rather than only as the in-process
        # MergeAccumulator, which a downstream merger cannot see. Every chunk is stamped with the state
        # that was in effect FOR THAT FRAME (geometry correction, which active cell, lock generation,
        # per-frame completeness), because a run-level summary would tell an offline merger to apply a
        # late-run correction to early-run data. Default None keeps the integrate path untouched.
        self.stream_out = stream_out
        self._writer = None
        if stream_out is not None:
            from glint.predict import StreamWriter
            names = [p.get("name", f"p{k}") for k, p in enumerate(self.panels)]
            self._writer = StreamWriter(
                stream_out, geom_text=stream_geom_text, panel_name=names[0], panel_names=names,
                photon_eV=(12398.419843320026 / self.wavelength_A), clen_m=self.clen_m)
        self.stream_image = str(stream_image)
        # Emit the OBSERVED peak list per chunk: None (off) | "flagged" | "all".
        # Why it matters: a chunk otherwise carries only the PREDICTED reflections, computed under the
        # orientation the driver chose -- so it cannot rescue a frame whose orientation WAS the
        # problem. An offline pass can see from qc_frac_threshold WHICH frames to retry but has
        # nothing to retry WITH. Shipping the peaks makes a flagged chunk self-contained for
        # re-indexing, which is worth a measured 20 of 42 strict-gate failures on the cxidb set
        # (14 recoverable by offline GLINT's blind/N-best/known-cell arsenal, 6 more by xgandalf or
        # ffbidx). "flagged" is the cheap default choice: ~15 kB per flagged frame, and only the
        # ~1-in-3 that are flagged; "all" makes every chunk a standalone CrystFEL record.
        if stream_peaks not in (None, "flagged", "all"):
            raise ValueError("stream_peaks must be None, 'flagged' or 'all'")
        self.stream_peaks = stream_peaks
        if stream_peaks == "flagged" and qc_frac_threshold is None:
            raise ValueError("stream_peaks='flagged' needs qc_frac_threshold set to define 'flagged'")
        # GeomRefiner's (dfs, dss) is a shift in the panel's own DATA-ARRAY basis (added to the local
        # fs/ss that project_q produced); CrystFEL's predict_refine/det_shift is a shift in LAB x/y, mm.
        # project_q solves  res*X_xy - corner = [fs_xy ss_xy] @ [lf, ls]  (predict.py), so the lab-frame
        # shift is dfs*fs_vec + dss*ss_vec, scaled by 1/res (m per px) and 1000 (m -> mm). Using the
        # identity here instead would silently ROTATE the correction on any panel whose fs/ss are not
        # +x/+y -- routine on CSPAD/Jungfrau/epix quadrants. `res` is pixels per metre.
        p0 = self.panels[0]
        mm_px = 1000.0 / float(p0["res"])
        self._shift_basis = np.array([[p0["fs"][0], p0["ss"][0]],
                                      [p0["fs"][1], p0["ss"][1]]], float) * mm_px

        # Adaptive re-lock (opt-in): keep a blind watchdog on the frames that miss every active cell,
        # and add a new cell when one recurs there -- so a mid-run SAMPLE CHANGE (or a mixture) is
        # detected and adapted to. Default off keeps the single-cell path bit-identical. NOTE: this
        # handles a DISCRETE new cell; a continuous cell CREEP (thermal expansion of the same crystal)
        # should be absorbed by widening `tol`/refining the active cell, NOT re-locked as a new one --
        # the miss-gate `min_inliers` + fit window `tol` are that boundary.
        self.adaptive_relock = bool(adaptive_relock)
        self.min_inliers = int(min_inliers) if min_inliers else self.min_peaks
        # min_inliers alone is a COUNT, and a count cannot say "this frame fits this cell" when frames
        # differ in peak count: chance agreement scales with the number of peaks. The fit test is a
        # per-peak box of half-width tol=0.15 in each of h,k,l, so a random direction lands inside it
        # with probability (2*0.15)^3 = 2.7%; at ~1340 peaks that is ~36 inliers expected from noise
        # alone, and a WRONG cell was measured at a median 84 (6.2%) -- clearing any count gate, while
        # a genuinely-fitting sparse frame with 60 peaks may total only ~40. A peak-rich wrong frame
        # therefore outscores a peak-sparse right one. The FRACTION is what separates them: measured
        # 0.647 for right-cell frames vs 0.062 for wrong-cell on the same data.
        # Both bars now apply (see _fits), mirroring the research gate's own `frac >= 0.25 AND
        # count >= 10` structure: the count keeps a 3-of-4-peak frame from passing on 75%, the
        # fraction keeps a peak-rich impostor from passing on chance.
        #
        # DEFAULT 0.15 is a CELL-DISCRIMINATION floor, not a completeness bar -- judging completeness
        # stays qc_frac_threshold's annotation-only job, per the DRP call to send a low-completeness
        # registration downstream WITH a flag rather than drop it. Chosen from two curves measured
        # together: cost on the real 120-frame cxidb set (gate isolated over the 115 post-lock frames,
        # 73 of which clear the strict research bar), and wrong-cell rejection on a synthetic run
        # where the driver is locked to a cell the frames do not have:
        #     frac              0.00  0.10  0.15  0.20  0.25  0.35
        #     real: accepted     115   115   111    98    73    55
        #     real: good lost      0     0     0     0     0    18
        #     wrong-cell refused   0    10    16    16    16     -    (of 16)
        # 0.15 is the knee: the smallest value tested that refuses EVERY wrong-cell frame, costing 4
        # of 115 real frames, none of which clear the research bar. Below it discrimination is partial
        # (0.10 refuses 10/16) because index_fused OPTIMISES the fit rather than returning a
        # chance-level one, so a wrong-cell frame lands above the 2.7% random rate. Above it the cost
        # climbs with no further benefit. Caveat: the wrong-cell column is n=16 on one synthetic pair,
        # so read 0.15-vs-0.10 as indicative, not as a sharp threshold. Set 0.0 for the pure count gate.
        #
        # MEASURED SINCE (28 Sep 2026, experiments/live_gate_null.py, RESULTS_live_gate_null.md): the case
        # a stream actually presents is not that one. A frame with NO lattice, searched with the RIGHT cell,
        # passes this gate often, because the known-cell search keeps the best of ~10^4 orientations. On
        # the cxidb-17 480 at the shipped depth, azimuth-scrambled copies (|q| and q_z kept, lattice gone)
        # pass it 211/480 times (44%), and the real lysozyme frames searched with the Proteinase K cell,
        # a real wrong cell, pass 204/480. The chance accepts sit at the SPARSE end: 96% of null frames
        # with < 60 peaks pass, 2.5% at 130-200 peaks, none above 200 -- there the fraction does its job,
        # below it the 10-count is at chance. null_floor (below) is the opt-in fix.
        self.min_inlier_frac = float(min_inlier_frac)
        # CHANCE FLOOR (opt-in). null_floor=(a, b) or (a, b, c) adds a third bar to the live gate: the
        # registration must explain n_inl >= a*n + b + c*sqrt(n) of the frame's n peaks. The shape follows
        # the null: the best of K orientations of a Binomial(n, p) count sits near n*p + sqrt(2 n p ln K),
        # so a straight line (c = 0, the form of the 16 Sep per-event calibration) over-rejects sparse
        # frames and under-rejects the middle. NULL_FLOOR_CXIDB17 is fitted at the 99th percentile of
        # 15,360 scrambled copies of the 480; with it, on frames and copies not used in the fit:
        #     gate                  null accepts (480 held-out)   real accepted   strict-gate frames kept
        #     live (10, 0.15)           211  (44%)                    447              327 of 327
        #     live + floor                2  (0.4%)                   335              305 of 327
        # (0.63% of the fit copies; median 0.66%, max 1.03% on held-out FRAMES in a 20x2 frame-split
        # cross-validation; 0.5-1.2% in every peak-count band below 200 peaks, 0 above.) The 22 strict
        # frames it refuses all have 38-60 peaks and 10-16 inliers, where the null's 99th percentile is
        # 14-17: the published xgandalf arm finds the same lattice within 2 deg on 0 of them, against 23
        # of the 27 sparse strict frames the floor keeps. Of the 112 frames the live gate accepts and the
        # floor refuses, xgandalf confirms 2. End to end (the published 480 arm, adaptive_relock, plus
        # the 480 scrambled): chance accepts 234 -> 5, strict 331 -> 332, because 23 real frames the
        # batched pass had accepted in a WRONG orientation now miss and come back from the watchdog in
        # xgandalf's -- at the price of sending every lattice-free frame to the watchdog's blind search.
        # The constants are specific to that peak finder, detector, cell family, HKL_TOL and search depth
        # (the floor transferred to the relock cell 87.5/87.6/109.5 at 1.2% and to Proteinase K at 0.7% on
        # the same frames): re-fit with experiments/live_gate_null.py before relying on them elsewhere.
        # It applies wherever the live gate does (_gate_count), so a cascade, watchdog or warm-up rescue
        # is held to the same floor. Default None: the gate is exactly the two bars above.
        if null_floor is None:
            self.null_floor = None
        else:
            # ORDERED containers only: the coefficients are positional, and a set would hand them over in
            # hash order -- a different gate, not an error (Copilot review of #214)
            vals = (list(null_floor) if isinstance(null_floor, (list, tuple))
                    or (isinstance(null_floor, np.ndarray) and null_floor.ndim == 1) else None)
            if (vals is None or len(vals) not in (2, 3)
                    or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
                           for v in vals)
                    or not all(np.isfinite(float(v)) for v in vals)):
                raise ValueError("null_floor must be (a, b) or (a, b, c), finite numbers: a registration then "
                                 f"needs n_inl >= a*n + b + c*sqrt(n) of its n peaks; got {null_floor!r}")
            self.null_floor = tuple(float(v) for v in vals) + (0.0,) * (3 - len(vals))
        self.n_null_floor_refused = 0                   # registrations the floor alone turned down
        self.extra = []; self._watch = None; self.n_relock = 0
        if (self.adaptive_relock or retry_cascade) and not hasattr(self, "_blind_index"):
            try:
                from glint.glint_fast import index_blind_nbest
                self._blind_index = index_blind_nbest
            except Exception:                                # pragma: no cover - CPU-only unit env (no torch)
                self._blind_index = None                     # tests inject a fake _blind_index via the seam

        # Retry CASCADE on gate-failing frames (opt-in, glint#75). Once locked, the driver runs
        # known-cell ONLY: a frame the batched pass registers badly enough to fail the accept gate is
        # never retried, it just goes to the miss path. Offline does better only because it runs a
        # whole ARSENAL per frame. With retry_cascade=True, a frame that fits no ACTIVE cell gets the
        # measured arm union (glint.retry_cascade) before the miss path claims it: blind N-best k=10,
        # then the PER-FRAME known-cell indexer.
        #
        # Both arms, not one. On the 42 frames the batched pass rejects of the cxidb-120 set the arms
        # recover 11 and 9 -- and 16 TOGETHER, because they fail on different frames
        # (experiments/offline_retry_arsenal.py). That takes the streaming arrangement to 94/120
        # against offline's shipped 91, at 47 blind solves instead of 120. The per-frame known-cell
        # arm is not redundant with the batched pass that already rejected these frames: the two
        # disagree on 18 of 120 at the same cell and gate, 9 each way (batched_vs_perframe.py).
        #
        # THE CASCADE'S YIELD IS SET BY THE LIVE GATE AS MUCH AS BY THE ARMS, and this is the part
        # that is easy to get wrong. The cascade fires on _fits failures, and the shipped _fits
        # (min_inlier_frac=0.15) is deliberately far looser than the strict research bar -- so a
        # badly-registered frame is ACCEPTED live, never fails, and is never retried, even though the
        # strict rescore will not count it. Measured on the real cxidb-120 set, cold-started, with
        # the driver unmocked (experiments/test_streamdriver_vs_offline.py, A100):
        #     gate                       cascade OFF   cascade ON   retried/rescued
        #     shipped (frac 0.15)          78/120       79/120         4 / 2
        #     research (10, frac 0.25)     78/120     * 94/120 *      42 / 16   (11 blind + 5 known)
        # The 94 -- which beats offline's shipped 91 on the same frames -- needs BOTH flags:
        # retry_cascade=True AND min_inliers=10, min_inlier_frac=0.25. The 16/42 split reproduces the
        # offline arsenal frame for frame. Turning the cascade on at the default gate is nearly free
        # and nearly pointless; the pairing is the result.
        #
        # DEFAULT OFF. It trades latency for yield -- a retried frame costs a blind index (~26 ms)
        # plus a per-frame registration, against ~0.21 ms (default B=64) for its share of the
        # batched pass -- and
        # the cost model at DAQ rates is unmeasured. Off, nothing here is constructed or called and
        # the emitted stream is byte-identical.
        self.retry_cascade = bool(retry_cascade)
        self.retry_nbest = int(retry_nbest) if retry_nbest else DEFAULT_NBEST
        self.n_cascade_retried = 0                           # frames the cascade was actually run on
        self.n_cascade_rescued = 0                           # ...of which it recovered
        self.n_cascade_by_arm = {}                           # which arm did it, so a dead arm is visible
        # The PER-FRAME known-cell indexer, deliberately NOT self._known_index (which is the BATCHED
        # rgb.index_fused -- the very pass that just rejected these frames). Resolved only when the
        # cascade is on, so the default path takes no new import. Test seam, like _blind_index.
        self._known_perframe = None
        if self.retry_cascade:
            try:
                from glint.replica_gpu import index_known_gpu_cell
                self._known_perframe = index_known_gpu_cell
            except Exception:                                # pragma: no cover - CPU-only unit env (no torch)
                self._known_perframe = None                  # tests inject a fake via the seam

        # Miss-buffer retroactive rescue (opt-in, INDEX-ONLY). Frames that fit NO active cell are the
        # "rose/unindexed" bars during a sample change; today they are dropped once the batch flushes.
        # With rescue_buffer>0 (needs adaptive_relock) their q-vectors are buffered, and when the
        # watchdog LOCKS a new cell they are re-indexed against it -- recovering the INDEXING rate of
        # the pre-lock misses. q-only => tiny (no raw-pixel ring). With rescue_pixels (below) their
        # pixels are kept as well and a rescued frame is integrated into the new cell's merge.
        # `fanout` batches the watchdog's blind indexing across workers (default = serial
        # single-GPU loop, BIT-IDENTICAL to today); opt-in glint.warmup_batch.mpi_fanout cuts the
        # detection latency roughly in proportion to the workers, down to a floor of about five
        # blind-frame-times set by the host-side consensus vote -- NOT the "~one blind-frame-time"
        # this comment used to claim; measured in glint/warmup_batch.py's header (job 37198405).
        # Both seams (`_fanout`, `_known_index`) let CPU tests inject
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
        # The PRIMARY blind lock is gated too, not just the watchdog's relock (see _gate_lock). That
        # needs each voting frame's q AND the leader's lattice in that frame's orientation, which
        # RunningConsensus does not keep -- it is a histogram of cells with no frame identity. So the
        # blind path retains (q, cells) per indexed frame while it is still hunting for a cell. Bounded:
        # the gate only needs enough frames to vote, and a stream that never locks must not grow a
        # buffer forever. None (gate off) => nothing is retained => bit-identical.
        self._gate_buf = deque(maxlen=256) if alias_gate is not None else None
        self._gate_last_support = -1                     # re-gate only when new votes arrived (see _gate_lock)
        self.n_gate_refused = 0                          # proposed locks NOT committed (total)
        self.n_gate_deferred = 0                         # ...the no-voter subset of those (watchdog)
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
        # push), so this is index-only like _missbuf's rescue, not a full integration -- unless
        # rescue_pixels keeps them (below). Default off keeps warm-up bit-identical to before.
        self.warmup_rescue = bool(warmup_rescue)
        self._warmup_buf = [] if self.warmup_rescue else None
        self._warmup_ev = [] if self.warmup_rescue else None    # arrival index per _warmup_buf entry (rescued_warmup events)
        self.n_warmup_rescued = 0
        # Pixels for the retroactive rescues (opt-in). Both rescues above re-index a frame LONG after
        # its batch left the ring, so they can only count it: the pixels are gone and the frame never
        # reaches the merge, which is what index-before-compress needs it for. rescue_pixels=N keeps up
        # to N such frames on the device (preallocated; oldest evicted first): each voting warm-up
        # frame until the lock, each buffered miss until a relock. A frame the rescue then accepts is
        # integrated like any other (counted in n_warmup_integrated / n_rescued_integrated; its
        # `integrated` record follows the rescued_* one). Only frames pushed with pixels can be kept;
        # push_q/push_peaks frames stay index-only. Memory: N frames of `shape` x `dtype`.
        if (isinstance(rescue_pixels, (bool, np.bool_)) or not isinstance(rescue_pixels, (int, np.integer))
                or rescue_pixels < 0):
            raise ValueError(f"rescue_pixels must be an integer >= 0, got {rescue_pixels!r}")
        if rescue_pixels and not (self.warmup_rescue or self._missbuf is not None):
            raise ValueError("rescue_pixels keeps frames for the warm-up rescue (warmup_rescue=True) or the "
                             "relock rescue (adaptive_relock=True with rescue_buffer > 0); with neither "
                             "on, nothing would ever read them")
        self._pix = _PixelStore(rescue_pixels, self.shape, self.dtype, xp) if rescue_pixels else None
        self.n_warmup_integrated = self.n_rescued_integrated = 0
        self.hits_only = bool(hits_only)
        # Adaptive effort (opt-in): the known-cell search depth follows the hit rate and the spare budget buys
        # deep, chance-controlled searches on the misses -- see _EffortPolicy. Off (None) nothing here is
        # constructed, the known-cell calls carry no depth arguments, and the emitted stream is byte-identical.
        self._eff = _EffortPolicy(effort) if effort is not None else None
        # Per-frame confidence flag (opt-in, DIAGNOSTIC-only -- never affects what gets integrated).
        # The live accept gate above (_fits) is deliberately looser than the
        # matched_frac>=25% bar used for the paper's offline comparison numbers: on a DRP time/
        # bandwidth budget it's better to send a lower-completeness registration downstream than to
        # drop it, AS LONG AS it's flagged so a later pass (refinement, QC, re-merge) can sort it out
        # rather than silently trusting it at face value. With qc_frac_threshold set, every
        # integrated frame's matched_frac is compared against it and, if below, recorded in
        # self.low_conf_frames -- purely an annotation, computed from data already available at
        # integrate time, at the cost of one division and one comparison per frame.
        self.qc_frac_threshold = qc_frac_threshold
        self.low_conf_frames = [] if qc_frac_threshold is not None else None
        self.n_low_confidence = 0
        # Watchdog individual rescue (bundled into adaptive_relock, no separate flag): _watchdog
        # already blind-indexes every missed frame to pool votes toward NEW-cell detection -- that
        # compute is spent regardless. Checking each frame's own N-best candidates against the
        # ALREADY-active cell(s) first is nearly free on top of it, and rescues a same-cell miss
        # immediately instead of only ever detecting a genuinely different cell (see _watchdog).
        self.n_watchdog_rescued = 0
        # A dead fan-out (dead GPU worker, poisoned queue) must degrade to the miss path, not
        # take flush() down with it -- glint#147. _fanout_guarded counts every invocation that
        # raised; the consecutive-failure streak is what tells a persistent failure from a
        # transient one in the warning text. Reported in stats() as n_fanout_errors.
        self.n_fanout_errors = 0
        self._fanout_fail_streak = 0
        # ...and the FRAMES those failures cost: with rescue_buffer=0 a degraded frame is in no
        # other ledger -- the miss buffer is unarmed, and the adaptive branch never counts
        # n_gate_rejected -- so without this it would be visible only as pushed minus integrated
        # (Copilot review of #156, suppressed comment). n_fanout_errors counts invocations;
        # this counts frames.
        self.n_fanout_missed = 0

        # Cell registry (glint#199). A metadata layer over the primary cell + the adaptive-relock
        # extras -- NOT a new container: tests and callers keep reading self.Mc/self.grid/self.acc and
        # self.extra[k] exactly as before. registry[k] is aligned with _all_cells()[k] and records
        # who the cell is (a roster name, or cell<k>), where it came from (given / warmup / relock),
        # when it locked, and how many frames it has taken, so an operator can tell "one sample
        # changed" from "two crystal forms, mixed" without parsing a .stream. Maintained always
        # (O(1) per accepted frame); only EVENT emission is opt-in, because it costs a fit count.
        self.cell_window = max(1, int(cell_window))
        self._roster = None
        if roster is not None:
            self._roster = []
            for name, spec in dict(roster).items():
                a = np.asarray(spec, float)
                if a.shape == (6,):
                    Mr = cell_to_Ar(*a)                         # real-space basis, same convention as Mc
                elif a.shape == (3, 3):
                    Mr = a
                else:
                    raise ValueError(f"roster[{name!r}] must be 6 cell parameters or a 3x3 basis, got shape {a.shape}")
                self._roster.append((str(name), Mr))
        self.registry = []
        self._recent = deque(maxlen=self.cell_window)           # cell id of each recent attribution (recent_share)
        self._on_event = on_event
        self._events_on = bool(events) or on_event is not None
        self.events = [] if events else None
        # Cell ASSIGNMENT when more than one cell is active. "first" is the cascade every published
        # number was recorded with: the primary cell is tried first and keeps whatever passes the
        # live gate, the extras only see its misses. That is cheap (one known-cell call per batch
        # plus one per extra cell on the leftovers) and blind to a specific failure: a dense frame of
        # crystal B gives cell A enough chance near-integer peaks to pass the live gate, so A claims
        # it and B never sees it (the two-species replay of glint#202 measured 85 of 446 Proteinase K
        # frames claimed by the lysozyme cell this way, median 30 live inliers on 170-peak frames,
        # none clearing the strict bar under that cell). "best" indexes every frame against EVERY
        # active cell; the first-fit cell stays the incumbent, and a challenger takes the frame only
        # when it explains at least max(assign_margin, assign_margin_frac * n_peaks) MORE peaks.
        # The margin is what makes this safe on weak frames: a wrong cell's best orientation fits
        # 12-15% of the peaks by chance, more for a larger cell, so on a frame that is marginal
        # under every cell a raw count comparison is a coin flip biased toward the bigger cell (on
        # the same stream, plain argmax moved 40 lysozyme frames into the Proteinase K cell, for a
        # +1..+6 count difference, while the frames it exists for win by ~70). With margin 0 this
        # IS the plain argmax. The live gate is applied to the winner; the gate is monotone in the
        # count for a given frame, so a winner that fails it means no cell fits. Cost: one known-cell
        # call per active cell per batch. With a single active cell the branch is never taken, so
        # the single-cell path and the 480-frame gate are untouched.
        if assign not in ("first", "best"):
            raise ValueError(f"assign must be 'first' or 'best', got {assign!r}")
        self.assign = assign
        self.assign_margin, self.assign_margin_frac = int(assign_margin), float(assign_margin_frac)
        if self.assign_margin < 0 or self.assign_margin_frac < 0:
            raise ValueError("assign_margin and assign_margin_frac must be >= 0")
        # PER-LATTICE SCORING (opt-in). In a double hit -- two crystals of the same protein in one shot,
        # which the gated rule finds in about a third of the frames that index on cxidb-17
        # (experiments/joint_ceiling/RESULTS_second_lattice.md) -- the live gate's denominator counts the
        # SECOND crystal's peaks against the first. A lattice that explains 13 % of all the peaks can
        # explain 20 % of the peaks the other crystal does not claim. per_lattice=True scores the
        # STRONGER of the two lattices against the peaks the other does not claim (see _pl_record for
        # why never the weaker):
        #   * a frame that fails the live gate under every active cell is searched for a second lattice
        #     in the peaks its best rejected registration leaves -- the shipped double-hit rule,
        #     multilattice.second_lattice_verdict: same cell as lattice 1, >= 15 deg away. If one is
        #     found and the stronger lattice passes the live gate on the peaks the other does not claim,
        #     the frame is accepted under THAT lattice (outcome "rescued_per_lattice"). On the cxidb-17
        #     480-frame replay every such rescue kept the residual's lattice, and the rejected
        #     registration had explained 11-14 % of the peaks -- about what a wrong orientation of the
        #     right cell explains by chance -- so this path mostly recovers frames the known-cell search
        #     mis-registered, and the kept lattice usually passes the whole-frame gate as well;
        #   * an accepted frame whose lattice-1 share is below per_lattice_below is searched the same
        #     way; when a second lattice is found the per-lattice fraction replaces the whole-frame one in
        #     the QC low-confidence flag, and both are reported (frame event, .stream chunk).
        # The search is a blind solve on the residual (~26 ms on an A100), so it runs only where the
        # score is marginal. A 1-in-16 azimuth-scramble null on the rescue path reports the rule's own
        # false-accept floor live, as double_hit's does. Only the lattice the frame is kept under is
        # integrated; the other is reported, never merged as a second crystal.
        self.per_lattice = bool(per_lattice)
        self.per_lattice_below = float(per_lattice_below)
        if not 0.0 <= self.per_lattice_below <= 1.0:
            raise ValueError(f"per_lattice_below must be in [0, 1], got {per_lattice_below!r}")
        self.n_pl_searched = self.n_pl_found = self.n_pl_rescued = self.n_pl_swapped = 0
        self.n_pl_null_tested = self.n_pl_null_found = self.n_pl_null_rescued = 0
        self._pl_n = 0
        self._pl_rng = np.random.default_rng(0x9E4C3)
        if not self.double_hit:
            self._dh_index = None                               # resolved on first use (keeps construction torch-free)

        # Blind warm-up: with Mc=None the driver has no cell yet, so it indexes the first frames
        # blind (~26 ms/frame) one at a time, accumulating cross-frame consensus; when the running
        # histogram LOCKS the cell it builds the hkl grid and drops into the batched known-cell path
        # below (~0.21 ms/frame; this flushes at self.B, default 64). Warm-up frames are spent on
        # discovery (not integrated) -- ~6-10
        # frames, negligible for completeness. See glint.running_consensus.
        self._blind = Mc is None
        self.n_warmup = 0; self.locked_after = None; self.consensus_support = None
        # Hypotheses FOLDED into the locked group, >= consensus_support: the two differ only when a
        # multi-match merge absorbed cells the locked representative does not cover (glint#182).
        self.consensus_members = None
        if self._blind:
            self.Mc = None; self.grid = None; self.n_theoretical = None
            # The pool-keyed acceptance gate (glint.running_consensus.consensus_accept). ON by
            # default because it is measured free: over 400 random arrival orders of the cxidb-120
            # benchmark NOT ONE changes its lock frame, cell, support or runner-up against the
            # gap-only rule, so the driver's published lock behaviour is untouched -- while on
            # mfxx49820 r0016 (unrefined geometry) the gap-only rule locks a WRONG lattice on 194 of
            # 400 orders and this leaves 2. Set lock_frac=0 / lock_lead=1 / lock_pool_switch=0 to
            # restore the bare gap rule. The watchdog's relock consensus below is deliberately NOT
            # gated: its pool grows over a whole run so it is arguably more exposed, but a false
            # REFUSAL there means missing a genuine sample change, and that trade has not been
            # measured.
            self._rc = RunningConsensus(min_support=lock_support, gap=lock_gap, adaptive=adaptive_gap,
                                        min_frac=lock_frac, min_lead=lock_lead,
                                        pool_switch=lock_pool_switch)
            from glint.glint_fast import index_blind_nbest      # torch; imported only in blind mode
            self._blind_index = index_blind_nbest
        else:
            self._lock(np.asarray(Mc, float), standardize=True)

    def _standardize(self, M, ref=None):
        """Cell setting for the merge operators (see _lock), decided by the Laue CLASS rather than by
        lengths alone: `standardize_axes(M, laue=self.laue)` puts the unique axis in c for the
        tetragonal/trigonal/hexagonal classes (where laue_ops rotates about c), orders a primitive,
        body- or face-centered orthorhombic cell a <= b <= c, and leaves base-centered cells of any
        class, plus triclinic, monoclinic, rhombohedral and cubic, exactly as handed in --
        permuting a base-centered cell changes its centering letter, no length rule can locate a
        monoclinic unique axis or a rhombohedral 3-fold, and for cubic every permutation is standard.

        The class is the input a length rule cannot supply: a cell whose three axes all sit inside the
        equal-length tolerance has no pair a tolerance can identify, and guessing one made the setting
        flip between frames on a refine-sized change (glint#185 review). The same function and the
        same class canonicalize every frame in _integrate_one, so the grid, the prediction and the
        merge cannot land in different settings.

        For the classes it leaves alone there IS no canonical setting to land in, so a reference is
        used instead when one is given: the known-cell indexer returns its axes shortest-first
        whatever order the reference was written in, and `_relabel_like` undoes exactly that
        permutation (glint#186 review). That is exact only because the engine returns the reference's
        setting in the first place (replica_gpu._both_hands / _ref_setting, review r2 s4-01); a basis in
        another setting of the lattice is not a permutation away and no relabel repairs it. The two halves compose -- the class fixes the setting where
        one exists, the reference supplies it where none does.

        With explicit `ops` this is the identity: `laue` is then a reporting label only, and using it
        to permute axes would impose a setting the supplied operators never asked for."""
        M = np.asarray(M, float)
        # Whether a canonical setting was imposed is a property of the CLASS, not of object identity:
        # standardize_axes returns a copy for the classes it leaves alone, so an `out is M` test would
        # silently stop relabelling the moment that copy was introduced.
        # Base-centering blocks the permutation in EVERY class, not just the orthorhombic one: the
        # centering letter names the face by the axes, so a permuted oC cell is an oA cell wearing
        # the wrong label (glint#185). standardize_axes refuses it; this mirrors the refusal so the
        # reference relabel still runs, which is the only setting such a cell can get.
        base_centered = str(self.stream_symmetry.get("centering", "")).upper() in ("A", "B", "C")
        imposed = (not self._ops_explicit) and (not base_centered) and (
            self.laue in UNIQUE_C_LAUE or self.laue in LENGTH_ORDER_LAUE)
        out = M if self._ops_explicit else standardize_axes(
            M, laue=self.laue, centering=self.stream_symmetry.get("centering"))
        if ref is not None and not imposed:
            out = _relabel_like(out, ref)
        return out

    def _lock(self, Mc, support=None, standardize=False):
        """Fix the cell: build the hkl grid + theoretical-unique count, and leave blind mode.

        standardize: put the cell in the setting self.ops assume (_standardize: for the tetragonal
        classes the 4-fold axis in c, so laue_ops / theoretical_unique count in the conventional
        setting; other classes are left as received). Needed for both consensus-locked and supplied
        known cells so the grid, prediction and merge share one setting."""
        Mc = np.asarray(Mc, float)
        if standardize:
            Mc = self._standardize(Mc)
        source = "warmup" if self._blind else "given"
        if self._blind:
            self.locked_after = self.n_pushed; self.consensus_support = support
        self.Mc = Mc
        self.grid = HKLGrid(self.Mc, self.dmin, gpu=self.gpu,   # built once, reused every frame
                            panels=self.panels, clen_m=self.clen_m, wavelength_A=self.wavelength_A,
                            )
        self.n_theoretical = theoretical_unique(self.Mc, self.dmin, self.ops)
        self._blind = False
        self._register_cell(0, self.Mc, source)
        if self._warmup_buf:                                    # retroactive index-only rescue (see __init__)
            qs, self._warmup_buf = self._warmup_buf, []
            evs = list(self._warmup_ev or [])
            if self._warmup_ev is not None:
                self._warmup_ev = []
            evs += [None] * (len(qs) - len(evs))                # harnesses that fill _warmup_buf directly carry no ev
            Ms = self._known_index(qs, self.Mc, B=max(len(qs), 1))
            pix = getattr(self, "_pix", None)
            for q, M, ev in zip(qs, Ms, evs):                   # same count as the old sum(...)
                if M is not None and abs(np.linalg.det(np.asarray(M, float))) >= 1.0 \
                        and self._fits(q, np.asarray(M, float)):
                    self.n_warmup_rescued += 1
                    self._attribute(0, ev)
                    if self._events_on:
                        self._emit_retro(ev, q, M, "rescued_warmup", 0)
                    if pix is not None and self._integrate_stored(ev, q, np.asarray(M, float), self.grid,
                                                                  self.acc, 0, "rescued_warmup"):
                        self.n_warmup_integrated += 1
            if pix is not None:
                pix.release(evs)                                # a warm-up frame has no later rescue

    def _gate_lock(self, Mc, support):
        """Alias-gate a BLIND consensus lock. Returns the cell to lock (possibly a tighter derivative
        lattice) or None to refuse it.

        The vote share cannot tell a cell from its own index<=N super-cell: a doubled axis collects
        exactly the same peaks, they just sit on every OTHER node, so both cells win the same votes.
        The gate is the deterministic complement -- it scores the systematically absent nodes.

        Fed PER FRAME, each frame's own agreeing N-best hypothesis being the leader's lattice in that
        frame's orientation. Never `confirm(Mc, pooled_q)`: coverage and occupancy are computed in the
        leader's frame, so pooling many orientations puts every candidate at chance coverage and the
        score collapses to a 1/V preference for smaller cells (measured: that refused the TRUE
        cxidb-62 cell, every half-volume derivative scoring 1.5-1.8x the leader).

        A refusal does NOT reset the vote histogram: the frames keep accumulating so a later, cleaner
        lock can still fire. Re-gating is skipped until support actually grows, so a standing refusal
        costs one gate call, not one per frame."""
        if self._alias_gate is None or not self._gate_buf:
            return Mc
        if support is not None and support == self._gate_last_support:
            return None                                  # already refused on exactly these votes
        voters = [(q, next(c for c in cells if same_lattice(c, Mc)))
                  for q, cells in self._gate_buf
                  if any(same_lattice(c, Mc) for c in cells)]
        if not voters:
            return Mc                                    # nothing to testify -> leave the vote alone
        Mg = self._alias_gate.confirm_frames(Mc, voters)
        if Mg is None:
            self.n_gate_refused += 1
            self._gate_last_support = support
        return Mg

    # ------------------------------------------------------------------ registry ----------------
    def _cell_name(self, M, k):
        """Roster name of lattice M (first same_lattice hit), else the positional fallback cell<k>."""
        if self._roster:
            for name, Mr in self._roster:
                if same_lattice(M, Mr):
                    return name
        return f"cell{k}"

    def _register_cell(self, k, M, source, lock_z=None):
        """Enter (or re-enter: tests re-_lock the primary) active cell k in the registry."""
        e = dict(id=k, name=self._cell_name(M, k), source=source, locked_at=self.n_pushed,
                 lock_generation=self.n_relock, lock_z=lock_z, n_frames=0, last_seen=None,
                 recent=deque(maxlen=self.cell_window))
        if k < len(self.registry):
            self.registry[k] = e
        else:
            self.registry.append(e)
        return e

    def _attribute(self, k, ev):
        """Count one accepted frame (arrival index ev) for active cell k. Guarded: a hand-built
        `extra` list (tests) may have no registry entry, and that must not crash a flush."""
        if k < len(self.registry):
            e = self.registry[k]
            e["n_frames"] += 1; e["last_seen"] = ev; e["recent"].append(ev)
            self._recent.append(k)

    def recent_share(self):
        """Fraction of the last `cell_window` attributed frames that went to each active cell."""
        if not self._recent:
            return {}
        c = Counter(self._recent); n = len(self._recent)
        return {k: c.get(k, 0) / n for k in range(len(self.registry))}

    def cell_table(self):
        """JSON-friendly snapshot of the registry, one row per active cell incl. the primary."""
        share = self.recent_share(); cells = self._all_cells()
        rows = []
        for e in self.registry:
            k = e["id"]
            M = cells[k] if k < len(cells) and cells[k] is not None else None
            rows.append(dict(id=k, name=e["name"], source=e["source"], locked_at=e["locked_at"],
                             lock_generation=e["lock_generation"], lock_z=e["lock_z"],
                             n_frames=e["n_frames"], last_seen=e["last_seen"],
                             recent_share=share.get(k, 0.0),
                             axes=list(np.linalg.norm(M, axis=0).round(2)) if M is not None else None))
        return rows

    @property
    def locked(self):
        return not self._blind

    def _buffer_len(self):
        if self._blind:
            return len(self._warmup_buf) if self._warmup_buf is not None else 0
        return len(self._missbuf) if self._missbuf is not None else 0

    # ------------------------------------------------------------------ events ------------------
    # One record per pushed frame, emitted where the decision is made (batching is invisible to the
    # consumer): terminal outcomes blank / warmup_vote / warmup_lock / indexed / escalated / rescued_watchdog /
    # rescued_cascade / miss / gate_rejected -- exactly one per frame, guaranteed by the control flow
    # (each stage returns only what it did not accept). Retroactive outcomes rescued_warmup /
    # rescued_relock refer to a PAST ev (at > ev) and never replace a terminal one; "relock" marks a
    # new active cell. `ev` = arrival index, `at` = n_pushed when emitted.
    def _emit(self, rec):
        if self.events is not None:
            self.events.append(rec)
        if self._on_event is not None:
            self._on_event(rec)

    def _event_base(self, ev, outcome, cell):
        return dict(ev=ev, at=self.n_pushed, outcome=outcome, cell=cell,
                    cell_name=(self.registry[cell]["name"] if cell is not None and cell < len(self.registry) else None),
                    n_active=(0 if self._blind else 1 + len(self.extra)), buffer=self._buffer_len())

    def _emit_frame(self, i, outcome, cell=None, M=None):
        q = self._q[i]
        rec = self._event_base(self._idx[i], outcome, cell)
        rec["slot"] = int(i)                                # the ring slot, valid until this flush ends
        rec["n_peaks"] = 0 if q is None else len(q)
        rec["frac"] = rec["n_inl"] = rec["M"] = None
        if M is not None:
            rec["M"] = np.asarray(M, float).tolist()
            if q is not None and len(q):
                n = self._inliers(q, np.asarray(M, float)); rec["n_inl"] = n; rec["frac"] = n / len(q)
        alt = getattr(self, "_inl_alt", None)
        if alt is not None and alt[i] is not None:
            rec["inl_by_cell"] = [int(v) for v in alt[i]]   # assign="best": what every active cell saw
        pl = getattr(self, "_pl", None)
        if pl is not None and pl[i] is not None:
            rec["second_lattice"] = dict(pl[i])             # per_lattice: the frame's second lattice, both fractions
        if getattr(self, "events_keep_q", False) and q is not None:
            rec["q"] = np.asarray(q, float).tolist()        # opt-in (plain attribute): a recorder that never
        self._emit(rec)                                     # saw the pixels can still score the frame

    def _emit_retro(self, ev, q, M, outcome, cell):
        rec = self._event_base(ev, outcome, cell)
        rec["n_peaks"] = 0 if q is None else len(q)
        M = np.asarray(M, float); rec["M"] = M.tolist()
        n = self._inliers(q, M) if q is not None and len(q) else 0
        rec["n_inl"] = n; rec["frac"] = n / len(q) if q is not None and len(q) else None
        if getattr(self, "events_keep_q", False) and q is not None:
            rec["q"] = np.asarray(q, float).tolist()
        self._emit(rec)

    def _emit_blind(self, ev, n_peaks, outcome, sup, lead, locked, q=None):
        rec = self._event_base(ev, outcome, 0 if locked else None)
        rec.update(n_peaks=int(n_peaks), frac=None, n_inl=None, M=None, support=sup, lead=lead)
        if getattr(self, "events_keep_q", False) and q is not None:
            rec["q"] = np.asarray(q, float).tolist()
        self._emit(rec)

    def _emit_blank(self, ev, n_peaks):
        rec = self._event_base(ev, "blank", None)
        rec.update(n_peaks=int(n_peaks), frac=None, n_inl=None, M=None)
        self._emit(rec)

    def _emit_relock(self, k, source, lock_z):
        rec = self._event_base(self.n_pushed, "relock", k)
        rec.update(source=source, lock_z=lock_z)
        self._emit(rec)

    # ------------------------------------------------------------------ ingest ------------------
    def _push_blind(self, frame, src=None):
        """Warm-up ingest: peak-find + blind-index this frame, feed the running consensus, lock on fire.

        Runs per frame (not batched) so the cell can lock at the first few frames; slot 0 is scratch
        (the frame is not kept -- there is no cell to integrate against yet)."""
        xp = cp if self.gpu else np
        self.n_pushed += 1
        self._ring[0][...] = xp.asarray(frame)
        pk = self.finder.find(self._ring[0]); fs = pk["x"]; ss = pk["y"]
        pi = pk["intensity"] if self.stream_peaks else None
        if self.gpu:
            fs = cp.asnumpy(fs); ss = cp.asnumpy(ss)
            if pi is not None:
                pi = cp.asnumpy(pi)
        qq = None
        pkq = None
        if fs.size >= self.min_peaks:
            qq = peaks_to_q(fs, ss, self.panels, self.clen_m, self.wavelength_A)
            ok = np.isfinite(qq).all(1)
            qq = qq[ok]
            if self.stream_peaks and len(qq) >= self.min_peaks:
                pkq = np.stack([fs[ok], ss[ok], pi[ok]], 1)
        if (getattr(self, "_pix", None) is not None and self._warmup_buf is not None
                and qq is not None and len(qq) >= self.min_peaks):
            self._pix.put(self.n_pushed - 1, self._ring[0], src=src, pkq=pkq)   # a voting frame: kept for the warm-up rescue
        self._ingest_blind_q(qq, n_peaks=int(fs.size))

    def _ingest_blind_q(self, qq, n_peaks=None):
        """The q-only half of the warm-up ingest, shared by _push_blind (pixels) and push_q/push_peaks
        (no pixels): blind-index, feed the running consensus, lock on fire. The caller has already
        counted the frame in n_pushed, so this frame's arrival index is n_pushed - 1."""
        ev = self.n_pushed - 1
        voted = qq is not None and len(qq) >= self.min_peaks
        if voted:
            nb = self._blind_index(qq, self.warmup_nbest)       # N-best candidate cells for this frame
            cells = [c for c, _ in nb if c is not None]
            self._rc.add_frame(cells)
            self.n_warmup += 1
            if self._gate_buf is not None:
                self._gate_buf.append((qq, cells))              # per-frame evidence for the lock-time gate
            if self._warmup_buf is not None:
                self._warmup_buf.append(qq)                     # retained for post-lock rescue (see __init__)
                if self._warmup_ev is not None:
                    self._warmup_ev.append(ev)
        Mc, sup, lead = self._rc.verdict()
        if Mc is not None:
            Mc = self._gate_lock(Mc, sup)                       # may tighten the cell, or refuse the lock
        locked = False
        if Mc is not None:
            self.consensus_members = self._rc.leader_counts()[1]   # folded >= sup (glint#182)
            self._lock(Mc, sup, standardize=True)               # -> known-cell batched path from here
            locked = True
        if self._events_on:
            npk = len(qq) if qq is not None else (n_peaks or 0)
            outcome = "warmup_lock" if (locked and voted) else ("warmup_vote" if voted else "blank")
            self._emit_blind(ev, npk, outcome, sup, lead, locked, q=qq)

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

        `warm_topk` DEFAULT IS 32, RAISED FROM 16 (2026-08-04). Measured on real MFX xtc1
        (mfxx49820 r0016, 2228 frames, Epix10ka2M): at top-16 the triaged vote reached only 3 of 48
        hypotheses (6.2%) and REFUSED to lock even on the correct refined geometry; top-32 locked
        correctly (11/96, 11.5%), as did 64/128/512. 16 was tuned on denser synthetic stacks and is
        too tight for real sparse stills.

        Note what triage buys, because it is the opposite of a threshold: on the WRONG (unrefined)
        geometry this path REFUSED at every k from 16 to 512, where both the pooled offline consensus
        and the sequential streaming lock returned a confident wrong cell. Capping the pool at
        k*nbest hypotheses starves a chance cluster of the votes it needs, rather than trying to
        out-run it with a bigger bar -- and it costs nothing on good data.
        """
        if not self._blind:
            return True                                          # already locked -- nothing to warm up
        xp = cp if self.gpu else np
        counts, qmap = [], []
        pkqmap = [] if self.stream_peaks else None
        for fr in frames:                                        # cheap peak-find every frame
            pk = self.finder.find(xp.asarray(fr)); fs, ss = pk["x"], pk["y"]
            pi = pk["intensity"] if self.stream_peaks else None
            if self.gpu:
                fs, ss = cp.asnumpy(fs), cp.asnumpy(ss)
                if pi is not None:
                    pi = cp.asnumpy(pi)
            q = None
            pkq = None
            if fs.size >= self.min_peaks:
                q = peaks_to_q(fs, ss, self.panels, self.clen_m, self.wavelength_A)
                ok = np.isfinite(q).all(1)
                q = q[ok]
                if self.stream_peaks and len(q) >= self.min_peaks:
                    pkq = np.stack([fs[ok], ss[ok], pi[ok]], 1)
                q = q if len(q) >= self.min_peaks else None
            counts.append(len(q) if q is not None else 0)
            qmap.append(q)
            if pkqmap is not None:
                pkqmap.append(pkq)
        return self._warmup_from_qmap(qmap, fanout, frames=frames, pkqmap=pkqmap)

    def warmup_batch_q(self, qs, fanout=None):
        """warmup_batch for a startup stack that arrives as q-vectors (n_i, 3) per frame instead of
        pixels -- the peaks-in twin. Same triage, consensus, gate and lock; blind mode only."""
        if not self._blind:
            return True
        qmap = []
        for q in qs:
            q = None if q is None else np.asarray(q, float).reshape(-1, 3)
            if q is not None:
                q = q[np.isfinite(q).all(1)]
            qmap.append(q if q is not None and len(q) >= self.min_peaks else None)
        return self._warmup_from_qmap(qmap, fanout)

    def _warmup_from_qmap(self, qmap, fanout, frames=None, pkqmap=None):
        """Shared tail of warmup_batch / warmup_batch_q: triage by peak count, pooled consensus, lock.
        `frames` (warmup_batch only): the pixels, kept for the picks when rescue_pixels is on."""
        from glint.warmup_batch import triage_order, warmup_consensus
        counts = [len(q) if q is not None else 0 for q in qmap]
        picks = triage_order(counts, self.warm_topk, self.warm_floor)   # rank by peak count; skip low-signal
        sel = [k for k in picks if qmap[k] is not None]
        qs = [qmap[k] for k in sel]
        ev0 = self.n_pushed
        self.n_pushed += len(qmap); self.n_warmup += len(qs)
        if self._warmup_buf is not None:
            self._warmup_buf.extend(qs)                          # retained for post-lock rescue (see __init__)
            if self._warmup_ev is not None:
                self._warmup_ev.extend(ev0 + k for k in sel)
            if getattr(self, "_pix", None) is not None and frames is not None:
                for k in sel:
                    self._pix.put(ev0 + k, frames[k], pkq=None if pkqmap is None else pkqmap[k])  # kept for the warm-up rescue
        Mc, sup = warmup_consensus(qs, self._blind_index, self._rc, self.warmup_nbest, fanout,
                                   sink=self._gate_buf)         # per-frame evidence for the lock-time gate
        if Mc is not None:
            Mc = self._gate_lock(Mc, sup)                        # may tighten the cell, or refuse the lock
        locked = False
        if Mc is not None:
            self.locked_after = self.n_pushed
            self.consensus_members = self._rc.leader_counts()[1]   # folded >= sup (glint#182)
            self._lock(Mc, sup, standardize=True)
            locked = True
        if self._events_on:                                      # events for the triaged picks only:
            _, sup_now, lead_now = self._rc.verdict()            # unpicked frames get theirs when pushed later
            last = max(sel) if sel else None
            for k in sel:
                outcome = "warmup_lock" if (locked and k == last) else "warmup_vote"
                self._emit_blind(ev0 + k, counts[k], outcome, sup_now, lead_now, locked,
                                 q=qmap[k] if k < len(qmap) else None)
        return locked

    def push(self, frame, src=None):
        """Ingest one detector frame (host numpy or already-device array). `src` = (image, event) or
        an image filename: where the frame came from, stamped on its .stream chunk instead of the
        `stream_image` placeholder and the arrival index (event None = one image per file, no Event line)."""
        if self._blind:
            self._push_blind(frame, src=src); return
        slot = self._n
        self._src[slot] = src
        xp = cp if self.gpu else np
        self._ring[slot][...] = xp.asarray(frame)            # single H2D into the resident slot
        pk = self.finder.find(self._ring[slot])              # peak-find ON DEVICE, no readback of pixels
        fs = pk["x"]; ss = pk["y"]
        pi = pk["intensity"] if self.stream_peaks else None
        if self.gpu:
            fs = cp.asnumpy(fs); ss = cp.asnumpy(ss)         # peaks are tiny; pixels stay put
            if pi is not None:
                pi = cp.asnumpy(pi)
        q = None
        if fs.size >= self.min_peaks:
            qq = peaks_to_q(fs, ss, self.panels, self.clen_m, self.wavelength_A)
            ok = np.isfinite(qq).all(1)                      # peaks_to_q returns NaN rows off-panel
            qq = qq[ok]
            if len(qq) >= self.min_peaks:
                q = qq
                if self.stream_peaks:
                    # keep the OBSERVED peaks, masked to exactly the rows q has, so a stream chunk can
                    # carry fs/ss/(1/d)/I that line up. This is what makes a flagged frame RE-INDEXABLE
                    # offline: predicted reflections are computed under the orientation the driver
                    # chose, so they cannot rescue a frame whose orientation was the problem -- only
                    # the observed peaks can.
                    self._pkq[slot] = np.stack([fs[ok], ss[ok], pi[ok]], 1)
        self._q[slot] = q
        if self._eff is not None:
            self._eff.seen(q is not None)                    # the hit rate the effort policy follows
        self._idx[slot] = self.n_pushed                      # arrival index, so a stream chunk names its frame
        self._haspix[slot] = True
        if q is None and self._events_on:
            self._emit_blank(self.n_pushed, int(fs.size))
        if q is None and getattr(self, "hits_only", False):
            self.n_pushed += 1                               # the next push reuses this slot: B counts hits
            return
        if self.geom_refine:
            self._pk[slot] = np.stack([fs, ss], 1) if fs.size else None
        self._n += 1
        self.n_pushed += 1
        if self._n == self.B:
            self.flush()

    def push_peaks(self, fs, ss, intensity=None, src=None):
        """Ingest one frame as its PEAK LIST (data-array fs/ss in px, as PeakFinderV4 emits them) --
        the DRP reducer->indexer path, where the pixels never reach this process. The slot is
        index-only: registered, counted, written to the .stream as a crystal with zero reflections
        (plus the observed peaks when stream_peaks is on), never integrated."""
        fs = np.asarray(fs, float).ravel(); ss = np.asarray(ss, float).ravel()
        if fs.size != ss.size:
            raise ValueError(f"fs and ss must have equal lengths, got {fs.size} and {ss.size}")
        pi = np.asarray(intensity, float).ravel() if intensity is not None else np.zeros(fs.size)
        if pi.size != fs.size:
            raise ValueError(f"intensity must match fs/ss length {fs.size}, got {pi.size}")
        q = pkq = None
        if fs.size >= self.min_peaks:
            qq = peaks_to_q(fs, ss, self.panels, self.clen_m, self.wavelength_A)
            ok = np.isfinite(qq).all(1); qq = qq[ok]
            if len(qq) >= self.min_peaks:
                q = qq
                if self.stream_peaks or self.geom_refine:
                    pkq = np.stack([fs[ok], ss[ok], pi[ok]], 1)
        self._queue_q(q, n_peaks=int(fs.size), pkq=pkq, src=src)

    def push_q(self, q, src=None):
        """Ingest one frame as reciprocal vectors (n, 3) in 1/A (q @ M = hkl). Index-only, see push_peaks."""
        q = np.asarray(q, float).reshape(-1, 3)
        q = q[np.isfinite(q).all(1)]
        self._queue_q(q if len(q) >= self.min_peaks else None, n_peaks=len(q), src=src)

    def _queue_q(self, q, n_peaks, pkq=None, src=None):
        """Slot bookkeeping shared by push_peaks/push_q: blind -> warm-up ingest, locked -> ring slot.
        The arrival index is the push position in both modes, as for push()."""
        if self._blind:
            self.n_pushed += 1
            self._ingest_blind_q(q, n_peaks=n_peaks)
            return
        slot = self._n
        self._q[slot] = q
        if self._eff is not None:
            self._eff.seen(q is not None)
        self._idx[slot] = self.n_pushed
        self._pkq[slot] = pkq if self.stream_peaks else None
        self._haspix[slot] = False
        self._src[slot] = src
        self._pk[slot] = None                                # nothing is predicted for an index-only slot
        if self.geom_refine and pkq is not None and len(pkq):
            self._pk[slot] = np.asarray(pkq[:, :2], float)   # ...but observed peaks still refine the geometry
        if q is None and self._events_on:
            self._emit_blank(self.n_pushed, n_peaks)
        if q is None and getattr(self, "hits_only", False):
            self.n_pushed += 1                               # the next push reuses this slot: B counts hits
            return
        self._n += 1
        self.n_pushed += 1
        if self._n == self.B:
            self.flush()

    # ------------------------------------------------------------------ batch -------------------
    def _inliers(self, q, M):
        """# of q peaks near-integer in cell M (hkl = q @ M) -- the 'does this frame fit this cell' test."""
        hf = np.asarray(q, float) @ M
        return int((np.abs(hf - np.round(hf)).max(1) < HKL_TOL).sum())

    def _fits(self, q, M):
        """THE live gate: does frame `q` fit cell `M` well enough to accept? Count AND fraction.

        Single definition so the five places that ask this question -- the batch miss-gate, the
        watchdog's candidate check, the lock-probe's frame pick, and the warm-up / miss-buffer
        rescues -- cannot drift apart. See min_inlier_frac in __init__ for why a count alone is not a
        sufficient test. With min_inlier_frac=0 this reduces exactly to the historical count gate."""
        return self._gate_count(self._inliers(q, M), len(q))

    def _gate_count(self, n, n_peaks):
        """_fits on an inlier count already in hand (best-fit assignment counts every cell once and
        must not count again). The one place the gate's arithmetic lives."""
        if n < self.min_inliers:
            return False
        if self.min_inlier_frac and not (n >= self.min_inlier_frac * n_peaks):
            return False
        nf = getattr(self, "null_floor", None)
        if nf is None or n >= nf[0] * n_peaks + nf[1] + nf[2] * np.sqrt(n_peaks):
            return True
        self.n_null_floor_refused += 1                  # the floor alone turned this registration down
        return False


    def _sl_laue(self):
        """The Laue class for the double-hit gate: this driver's class if it was GIVEN (caller or stream
        header), None if it is only the 4/mmm merge fallback -- then misorientation_deg infers it."""
        return self.laue if getattr(self, "_laue_known", False) else None

    def _sl_index(self):
        """The blind indexer the second-lattice search uses (index_blind_nbest), resolved lazily."""
        if getattr(self, "_dh_index", None) is None:
            from glint.glint_fast import index_blind_nbest
            self._dh_index = index_blind_nbest
        return self._dh_index

    def _second_lattice(self, q, M1):
        """The shipped gated double-hit search on the peaks M1 leaves: (verdict, residual), or
        (None, residual) when the residual is too small to search (min_peaks)."""
        from glint.multilattice import second_lattice_verdict
        resid = deflate_peaks(q, M1)
        if len(resid) < self.min_peaks:
            return None, resid
        return second_lattice_verdict(resid, M1, self._sl_index(), min_peaks=self.min_peaks,
                                      laue=self._sl_laue()), resid

    def _pl_record(self, i, q, v, resid, kept, M1):
        """Per-lattice numbers for slot i from an ACCEPTED second-lattice verdict (frame event, chunk).

        n1 is the deflation's own count of the peaks the FIRST registration claims; m2 the residual peaks
        the second lattice indexes. Only the DOMINANT lattice -- the one claiming more peaks -- is scored
        per lattice, against the peaks the other does not claim. The rule matters: when the first
        registration is the weaker crystal (or a wrong orientation of the right cell, which explains
        12-15 % of a dense frame by chance), the search finds the frame's real crystal in the residual,
        and removing ITS peaks from the first registration's denominator would reward the weaker one.
        On cxidb-17 the second lattice is the stronger in 5 of the 15 double hits among the frames no
        arm indexes, and in none of the 134 among the frames that index.
        `kept` says which lattice the frame was integrated under: "first", "second" or None. M1 is the
        lattice-1 matrix the residual was deflated with; both matrices are recorded so a scorer can
        rebuild each lattice's claim (lattice 1 first, as the deflation assigns them)."""
        n1 = len(q) - len(resid)
        m2 = self._inliers(resid, np.asarray(v["M2"], float))
        dominant = "first" if n1 >= m2 else "second"
        nd, no = (n1, m2) if dominant == "first" else (m2, n1)
        rec = dict(n1=int(n1), m2=int(m2), n_peaks=int(len(q)), misorientation=round(float(v["misorientation"]), 2),
                   dominant=dominant, kept=kept, frac_first=n1 / len(q),
                   frac_per_lattice=nd / max(len(q) - no, 1),       # the dominant lattice, per lattice
                   M1=np.asarray(M1, float).tolist(), M2=np.asarray(v["M2"], float).tolist())
        self._pl[i] = rec
        return rec

    def _qc_frac(self, i, frac):
        """The fraction the QC low-confidence flag reads: the per-lattice fraction when a second lattice
        was found under per_lattice AND the integrated lattice is the dominant one; the whole-frame
        fraction otherwise (a weaker lattice is never upgraded by removing a stronger one's peaks)."""
        pl = getattr(self, "_pl", None)
        rec = pl[i] if pl is not None else None
        if rec is not None and rec["kept"] is not None and rec["kept"] == rec["dominant"]:
            return rec["frac_per_lattice"]
        return frac

    def _per_lattice_rescue(self, slots):
        """per_lattice (see __init__): frames that failed the live gate under every active cell. Returns
        the slots still missed, for the retry cascade / watchdog / gate_rejected exactly as before."""
        from glint.multilattice import scramble_azimuth, second_lattice_verdict
        still = []
        for i in slots:
            rej = self._rej.get(i)
            if rej is None:                                 # nothing registered: no lattice 1 to score
                still.append(i)
                continue
            M, k, _ = rej
            q = self._q[i]
            v, resid = self._second_lattice(q, M)
            if v is None:
                still.append(i)
                continue
            self.n_pl_searched += 1
            self._pl_n += 1
            n1 = len(q) - len(resid)
            if self._pl_n % 16 == 1:                        # the rescue rule's own false-accept floor, live
                rs = scramble_azimuth(resid, self._pl_rng)
                vn = second_lattice_verdict(rs, M, self._sl_index(), min_peaks=self.min_peaks,
                                            laue=self._sl_laue())
                self.n_pl_null_tested += 1
                if vn["accepted"]:
                    self.n_pl_null_found += 1
                    m2n = self._inliers(rs, np.asarray(vn["M2"], float))
                    nd, no = (n1, m2n) if n1 >= m2n else (m2n, n1)
                    self.n_pl_null_rescued += bool(self._gate_count(nd, len(q) - no))
            if not v["accepted"]:
                still.append(i)
                continue
            self.n_pl_found += 1
            rec = self._pl_record(i, q, v, resid, kept=None, M1=M)
            nd, no = (rec["n1"], rec["m2"]) if rec["dominant"] == "first" else (rec["m2"], rec["n1"])
            if not self._gate_count(nd, len(q) - no):
                still.append(i)                             # a double hit, but the stronger lattice is too weak even alone
                continue
            rec["kept"] = rec["dominant"]
            self._pl_v[i] = (v, resid, M)                   # _integrate_one must not search this frame again
            self.n_pl_rescued += 1
            grid, acc = self._cell_sink(k)
            if rec["dominant"] == "first":                  # the registration in hand is the frame's crystal
                self._integrate_one(i, M, grid, acc, cell_id=k, known_cell=True, outcome="rescued_per_lattice")
            else:                                           # the residual held the stronger crystal: keep THAT one
                self.n_pl_swapped += 1
                self._integrate_one(i, np.asarray(v["M2"], float), grid, acc, cell_id=k, known_cell=False,
                                    outcome="rescued_per_lattice")
        return still
    def _integrate_one(self, i, M, grid, acc, cell_id=0, known_cell=False, outcome="indexed", retro=False):
        """Canonicalize + predict + integrate slot i under an ALREADY-ACCEPTED matrix M into acc.
        Split out of _index_integrate so _watchdog's individual rescue can integrate a validated
        blind candidate directly, without re-registering it through known-cell (which could just
        miss again for the same reason the frame was flagged in the first place).

        cell_id identifies WHICH active cell accepted this frame (0 = the primary self.Mc, 1..n = the
        adaptive-relock extras) -- recorded per chunk so an offline merger can separate the sub-runs
        instead of silently co-merging two different crystals.

        retro=True: a retroactive rescue integrated from the pixel store (_integrate_stored). The caller
        has already counted it (n_warmup_rescued / n_rescued), attributed it to its cell and emitted its
        rescued_* record, so it is only integrated here: no n_indexed, no registry count, no second
        record, and no second-lattice search (that runs on the live batch)."""
        if not retro:
            self.n_indexed += 1
        # ONE setting for the reference and every frame (glint#186 review). This used to be
        # _canonical_axes(M) -- an unconditional (long, long, short) sort -- while HKLGrid was built
        # from the reference cell as _standardize left it. The two disagree for any class whose
        # standardizer is not that sort: on a 30/40/50 orthorhombic cell the grid's h bound is sized
        # for 30 A while prediction would read h along 40 A, clipping valid reflections, and for the
        # monoclinic and trigonal settings the reorder can move the unique axis off the one the
        # selected operators assume. Standardizing frames the same way the reference was standardized
        # makes grid, prediction and merge share a setting by construction; where the standardizer is
        # the identity the frame inherits its setting from registration against Mc, which is the same
        # rule the docstring states for Mc itself.
        # `known_cell` says the matrix came from the known-cell indexer, whose output order is a
        # length-sort permutation of the reference's columns and can therefore be relabelled back
        # exactly. That holds because the engines guarantee it: until review r2 s4-01 they did not on
        # triclinic and most monoclinic cells (a mirrored, non-lattice basis on ~half the frames, or
        # another setting of the right lattice), and no relabel could repair that; replica_gpu's
        # _both_hands / _ref_setting now return the reference setting. A BLIND candidate cannot: it arrives through primitivize(buerger_reduce(...)) and
        # may differ from the reference by a general integer change of basis, not a permutation, so
        # undoing a sort would be a guess (glint#186 review). Those paths keep the canonical-setting
        # behaviour only -- see glint#188.
        Mcan = self._standardize(M, ref=getattr(grid, "Mc_ref", None) if known_cell else None)
        if not retro:
            self._attribute(cell_id, self._idx[i])          # registry: this cell took this frame
        pl_on = getattr(self, "per_lattice", False)
        if (self.double_hit or pl_on) and not retro:        # deflate-and-reindex: a 2nd crystal in this shot?
            q = self._q[i]
            cached = self._pl_v[i] if pl_on else None       # the per-lattice rescue already searched this frame
            if pl_on and cached is None:
                self._pl[i] = None                          # a record from another lattice 1 (a failed rescue) must not leak
            if cached is not None:                          # M1: the lattice the residual was deflated with --
                v, resid, M1 = cached                       # NOT the kept one when the rescue swapped lattices
            elif self.double_hit or (len(q) and self._inliers(q, Mcan) / len(q) < self.per_lattice_below):
                M1 = Mcan
                v, resid = self._second_lattice(q, M1)
                if pl_on and v is not None:
                    self.n_pl_searched += 1
            else:
                v = resid = M1 = None
            if v is not None and self.double_hit:
                # A 2nd crystal = the residual re-indexes to the SAME CELL as lattice 1 (SFX double
                # hits are the same protein at a new orientation) at a genuinely different
                # orientation. The earlier comment here claimed the residual-inlier test "already
                # rejects merely re-finding lattice 1" -- measured false on both counts: with a
                # median 62-peak residual the bare test accepted azimuth-SCRAMBLED residuals at 93%
                # (vs 95% real, job 34409542), and a mosaic-tail clone of lattice 1 passes it via
                # peaks just outside the deflation tolerance. second_lattice_verdict carries both
                # gates -- same cell, and >= 15 deg away, which is what tells a second crystal from
                # this crystal's own mosaic tail (the angle is bimodal on real data with a near-empty
                # 5-15 deg band, so the cut is not delicate). The periodic scramble null reports the
                # gated rule's own false-accept floor on the run actually in front of it.
                from glint.multilattice import scramble_azimuth, second_lattice_verdict
                self.n_double_raw += bool(v["raw"])
                self.n_double += bool(v["accepted"])
                self._dh_n += 1
                if self._dh_n % 16 == 1:
                    vn = second_lattice_verdict(scramble_azimuth(resid, self._dh_rng), M1,
                                                self._sl_index(), min_peaks=self.min_peaks,
                                                laue=self._sl_laue())
                    self.n_dh_null_tested += 1
                    self.n_dh_null_acc += bool(vn["accepted"])
            if v is not None and pl_on and cached is None and v["accepted"]:
                self.n_pl_found += 1
                self._pl_record(i, q, v, resid, kept="first", M1=M1)   # accepted as registered; QC credits it only if dominant
        if self._events_on and not retro:
            self._emit_frame(i, outcome, cell=cell_id, M=M)
        if not self._haspix[i]:                             # peaks-in slot: registered + counted, nothing to integrate
            if self._grefiner is not None and self._pk[i] is not None:
                self._grefiner.add_frame(recip_from_M(Mcan), self._pk[i], None)   # it predicts for itself
            if self._writer is not None:
                q = self._q[i]
                frac = self._inliers(q, M) / max(len(q), 1)
                qc = self._qc_frac(i, frac)                 # per-lattice when a second lattice was found
                low_conf = (qc < self.qc_frac_threshold) if self.qc_frac_threshold is not None else None
                self._writer.write(self._stream_record(i, Mcan, None, None, None, None, None, None,
                                                       cell_id, frac, low_conf))
            return
        pred = grid.predict(Mcan, self.panels, self.clen_m, self.wavelength_A, tol=self.tol)
        if len(pred) == 0:
            return
        if self._grefiner is not None and self._pk[i] is not None:
            self._grefiner.add_frame(recip_from_M(Mcan), self._pk[i], pred)
        if self.gpu:
            from glint.fused_integrate import integrate_fused
            I, sig, pkI, bg = integrate_fused(self._ring[i], pred, half=self.half, gap=self.gap,
                                              ring=self.ring_w, bg_mode=self.bg_mode)
        else:
            I, sig, pkI, bg = integrate_spots(self._ring[i], pred, half=self.half, gap=self.gap,
                                              ring=self.ring_w, bg_mode=self.bg_mode)
        hkl = np.stack([pred["h"], pred["k"], pred["l"]], 1)
        keep = I != 0.0                                     # off-frame boxes integrate to exactly 0
        if keep.any():
            frac = None
            if self.qc_frac_threshold is not None or self._writer is not None or self._events_on:
                frac = self._inliers(self._q[i], M) / len(self._q[i])   # the marker carries it too
            low_conf = None
            if self.qc_frac_threshold is not None:
                qc = self._qc_frac(i, frac)                 # per-lattice when a second lattice was found
                low_conf = qc < self.qc_frac_threshold
                if low_conf:
                    self.n_low_confidence += 1
                    self.low_conf_frames.append((self._frame_no, qc))
            acc.add_frame(hkl[keep], I[keep], sig[keep], self._frame_no)
            if self._writer is not None:
                self._writer.write(self._stream_record(i, Mcan, pred, I, sig, pkI, bg, keep,
                                                       cell_id, frac, low_conf))
            self.n_integrated += 1; self._frame_no += 1
            if self._events_on:                             # marker, not a terminal outcome: the frame's
                rec = self._event_base(self._idx[i], "integrated", cell_id)      # `indexed` record precedes it
                rec.update(slot=(None if retro else int(i)), n_pred=int(len(pred)), n_refl=int(keep.sum()),
                           frame_no=self._frame_no - 1, frac=frac, low_conf=low_conf)
                pl = getattr(self, "_pl", None)
                if pl is not None and pl[i] is not None and pl[i]["kept"] == pl[i]["dominant"]:
                    rec["frac_per_lattice"] = pl[i]["frac_per_lattice"]
                self._emit(rec)

    def _integrate_stored(self, ev, q, M, grid, acc, cell_id, outcome):
        """Integrate a frame the pixel store kept (rescue_pixels) under the rescue's matrix M, through the
        ordinary _integrate_one on a scratch slot one past the ring that exists only for this call.
        Releases the frame. Returns True if it was integrated -- False when no pixels were kept for it
        (evicted, or pushed without pixels) or no reflection landed on the detector."""
        pix = getattr(self, "_pix", None)
        kept = pix.get(ev) if pix is not None and ev is not None else None
        if kept is None:
            return False
        frame, src, pkq = kept
        slot_lists = (self._ring, self._q, self._pk, self._idx, self._pkq, self._haspix, self._src,
                      self._inl_alt, self._pl, self._pl_v)
        S = self.B
        for L, v in zip(slot_lists, (frame, q, None, ev, pkq, True, src, None, None, None)):
            L.append(v)
        n0 = self.n_integrated
        try:
            self._integrate_one(S, M, grid, acc, cell_id=cell_id, known_cell=True, outcome=outcome, retro=True)
        finally:
            for L in slot_lists:
                del L[S:]
            pix.release([ev])
        return self.n_integrated > n0

    def _buffer_misses(self, slots):
        """Hand frames that fit no active cell to the miss buffer as (arrival index, q), for the relock
        rescue -- and, with rescue_pixels, their pixels to the pixel store, so a frame that rescue
        accepts is integrated. Only a pixel slot has pixels to keep. The deque drops its oldest
        entries when full, and their pixels are released with them."""
        mb = self._missbuf
        pix = getattr(self, "_pix", None)
        if pix is not None:
            drop = len(mb) + min(len(slots), mb.maxlen) - mb.maxlen
            if drop > 0:
                pix.release([ev for ev, _ in list(mb)[:drop]])
            for i in slots[-mb.maxlen:]:
                if self._haspix[i]:
                    pix.put(self._idx[i], self._ring[i], src=self._src[i], pkq=self._pkq[i])
        mb.extend((self._idx[i], self._q[i].copy()) for i in slots)   # (arrival index, q)

    def _stream_record(self, i, Mcan, pred, I, sig, pkI, bg, keep, cell_id, frac, low_conf):
        """One .stream chunk's worth of this frame, stamped with the state IN EFFECT FOR IT.

        Geometry semantics, deliberately: `clen_m` reports the distance actually USED to predict these
        fs/ss (self.clen_m -- GeomRefiner is diagnostic-only and is never fed back, see its docstring),
        so the chunk stays internally consistent. The refiner's running estimate is reported SEPARATELY
        as the correction: det_shift (CrystFEL's own per-crystal field, converted px -> mm) and
        glint/dclen_m. An offline merger therefore gets both what was assumed and what was measured,
        and can apply the correction itself rather than having it silently baked in."""
        dx = dy = 0.0
        dclen = n_solves = None
        if self._grefiner is not None:
            c = self._grefiner.correction()                 # snapshot AFTER this frame was pooled
            dx, dy = self._shift_basis @ np.array([c["dfs"], c["dss"]], float)
            dclen, n_solves = c["dclen_m"], c["n_solves"]
        if pred is None:                                    # index-only chunk (push_peaks/push_q): crystal, no reflections
            rows = dict(pred=None, I=None, sigma=None, peak=None, bg=None); fno = None
        else:
            rows = dict(pred=pred[keep], I=I[keep], sigma=sig[keep], peak=pkI[keep], bg=bg[keep]); fno = self._frame_no
        src = self._src[i]
        if src is None:
            image, event = self.stream_image, self._idx[i]
        elif isinstance(src, (tuple, list)):
            image, event = src[0], (src[1] if len(src) > 1 else None)
        else:
            image, event = src, None
        rec = dict(image=image, event=event, M=Mcan, **rows,
                   clen_m=self.clen_m, det_shift_mm=(dx, dy), dclen_m=dclen,
                   geom_n_solves=n_solves, cell_id=cell_id, lock_generation=self.n_relock,
                   matched_frac=frac, low_confidence=low_conf, frame_no=fno,
                   **self.stream_symmetry)
        pl = getattr(self, "_pl", None)
        if pl is not None and pl[i] is not None:            # per_lattice: the second lattice and its fraction
            if pl[i]["kept"] is not None and pl[i]["kept"] == pl[i]["dominant"]:
                rec["matched_frac_per_lattice"] = pl[i]["frac_per_lattice"]
            rec["second_lattice_deg"] = pl[i]["misorientation"]
        # Observed peaks: always for "all", only for the flagged frames for "flagged". Paired with
        # |q| so the chunk carries a real (1/d); these are the rows a downstream re-index would use.
        if self.stream_peaks and self._pkq[i] is not None:
            if self.stream_peaks == "all" or low_conf:
                rec["peaks"] = self._pkq[i]
                rec["peaks_invd"] = np.linalg.norm(self._q[i], axis=1)
        return rec

    def close(self):
        """End the run: flush the resident batch, then finish the .stream (if any). Returns how many
        chunks carried an indexed crystal, or None when no stream is being written.

        flush() first because the last partial batch is still resident when a run ends -- closing
        without it would silently drop those frames from BOTH the merge and the stream. Idempotent.
        Nothing calls this automatically (there is no __del__), so a long-lived DAQ process must call
        it, or use the driver as a context manager."""
        self.flush()
        return self._writer.close() if self._writer is not None else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _index_integrate(self, slots, Mc, grid, acc, gate, cell_id=0):
        """Index `slots` against Mc, integrate the fits into `acc`. Returns the missed slots.

        `gate` says whether a rejected frame has SOMEWHERE ELSE TO GO -- with gate=True it is returned
        so the caller can try it against another active cell or the watchdog; with gate=False there is
        no second consumer, so it is dropped here and counted in n_gate_rejected.

        The FIT TEST ITSELF now runs either way. It used to be tied to `gate`, which meant the
        single-cell path (adaptive_relock=False, the default) applied NO ingest test at all: any
        non-None result with |det| >= 1 was integrated, however badly it fitted. That let a frame whose
        registration is statistically indistinguishable from noise contribute reflections to the merge.
        Note this is a chance-rejection floor (see min_inlier_frac), NOT a completeness bar -- a real
        but low-completeness registration still passes here and is flagged, not dropped, by
        qc_frac_threshold, which is the DRP-side call: send it downstream WITH a warning rather than
        silently discard it.

        cell_id just labels which active cell this is, for per-chunk stream provenance."""
        qs = [self._q[i] for i in slots]
        Ms = rgb.index_fused(qs, Mc, B=max(len(qs), 1), **self._depth_kw())
        missed = []
        for i, M in zip(slots, Ms):
            M = np.asarray(M, float) if M is not None else None
            if M is None or abs(np.linalg.det(M)) < 1.0 or not self._fits(self._q[i], M):
                if getattr(self, "per_lattice", False) and M is not None and abs(np.linalg.det(M)) >= 1.0:
                    n = self._inliers(self._q[i], M)        # the best rejected registration, for the rescue
                    if i not in self._rej or n > self._rej[i][2]:
                        self._rej[i] = (M, cell_id, n)
                if gate:
                    missed.append(i)                    # another cell / the watchdog may still take it
                else:
                    self.n_gate_rejected += 1           # nowhere left to send it -- dropped, but counted
                    if self._events_on:
                        self._emit_frame(i, "gate_rejected")
                continue
            self._integrate_one(i, M, grid, acc, cell_id=cell_id, known_cell=True)
        return missed

    def _depth_kw(self):
        """The fast path's search-depth arguments: none (the shipped call, verbatim) unless the adaptive effort
        policy is on, then its current tier's topa/nc/full_grid on every call."""
        eff = getattr(self, "_eff", None)
        return eff.depth_kw() if eff is not None else {}

    def _escalate_misses(self, slots):
        """Adaptive effort's deep search on this flush's fast-path misses: glint.retry_cascade.escalate_batch at
        the top tier against each active cell in turn. A fit is accepted only if it passes the live gate for that
        cell (_fits and same_lattice) AND matches more peaks than every one of its k_null azimuth-scrambled copies
        -- the chance control of glint#211, scored with the driver's own inlier count. Accepted frames are
        integrated under the accepting cell with outcome "escalated"; the rest continue down the miss path (retry
        cascade, miss buffer, watchdog) exactly as before. Copy k of the frame that arrived at index ev is
        scrambled with rng([seed, ev, k]), so a replay reproduces the decisions."""
        from glint.retry_cascade import escalate_batch
        eff = self._eff
        topa, nc, full_grid, _ = eff.tiers[-1]

        def search(qs, Mcell):
            return rgb.index_known_deep_batch(qs, Mcell, topa, nc, full_grid=full_grid, budget=eff.deep_budget,
                                              return_errors=True)

        def count(M, q):
            return self._inliers(q, M)

        still = list(slots)
        eff.n_tried += len(still)
        for k, Mk in enumerate(self._all_cells()):
            if not still:
                break

            def gate(M, q, Mk=Mk):
                return abs(np.linalg.det(M)) >= 1.0 and self._fits(q, M) and same_lattice(M, Mk)

            Ms, recs = escalate_batch([self._q[i] for i in still], Mk, search, count, gate,
                                      seeds=[[eff.seed, int(self._idx[i])] for i in still],
                                      k_null=eff.k_null, round_copies=eff.round_copies)
            eff.n_searches += sum(r["searches"] for r in recs)
            nxt = []
            for i, M in zip(still, Ms):
                if M is None:
                    nxt.append(i)
                    continue
                grid, acc = self._cell_sink(k)
                self._integrate_one(i, np.asarray(M, float), grid, acc, cell_id=k, known_cell=True, outcome="escalated")
                eff.n_accepted += 1
            still = nxt
        return still

    def _emit_effort(self):
        """The effort policy changed its mind (tier or deep search): one marker record, like "relock", so a
        recorded stream shows which depth every frame from `ev` on was searched at."""
        e = self._eff.log[-1]
        rec = self._event_base(e["at"], "effort", None)      # ev = the first frame the decision applies to
        rec.update(tier=e["tier"], deep=e["deep"], budget_ms=e["budget_ms"], hit_est=e["hit_est"], miss_frac=e["miss_frac"],
                   n_cells=e["n_cells"])
        self._emit(rec)

    def _index_best_fit(self, slots):
        """assign="best": index `slots` against EVERY active cell. The incumbent is the first cell (in
        cell order) whose registration passes the live gate -- first-fit's answer; the challenger is
        the cell with the most near-integer peaks, and it takes the frame only when it beats the
        incumbent by max(assign_margin, assign_margin_frac * n_peaks) peaks (see __init__). Integrates
        the frame under the winner. Returns the missed slots (no cell passes the live gate), for the
        retry cascade / watchdog exactly as the first-fit cascade returns them.

        Same registration call (rgb.index_fused), same non-degeneracy test and same live gate
        (_gate_count == _fits) as _index_integrate, so a frame that passes under exactly one cell
        lands where first-fit would have put it; only frames that pass under more than one can move."""
        qs = [self._q[i] for i in slots]
        cells = self._all_cells()
        fits = []                                              # fits[k][j] = (M or None, n_inliers) for cell k, slot j
        for Mk in cells:
            Ms = rgb.index_fused(qs, Mk, B=max(len(qs), 1), **self._depth_kw())
            row = []
            for i, M in zip(slots, Ms):
                M = np.asarray(M, float) if M is not None else None
                if M is None or abs(np.linalg.det(M)) < 1.0:
                    row.append((None, 0))
                else:
                    row.append((M, self._inliers(self._q[i], M)))
            fits.append(row)
        missed = []
        for j, i in enumerate(slots):
            npk = len(self._q[i])
            counts = [fits[k][j][1] for k in range(len(cells))]
            k0 = next((k for k, n in enumerate(counts) if self._gate_count(n, npk)), None)
            if k0 is None:                                      # the gate is monotone in the count: no cell fits
                if getattr(self, "per_lattice", False):
                    kb = int(np.argmax(counts))
                    if fits[kb][j][0] is not None:
                        self._rej[i] = (fits[kb][j][0], kb, counts[kb])
                missed.append(i)
                continue
            kb = int(np.argmax(counts))                          # argmax takes the FIRST maximum: cell order on ties
            k = k0
            if kb != k0 and counts[kb] - counts[k0] >= max(self.assign_margin, self.assign_margin_frac * npk):
                k = kb
            self._inl_alt[i] = counts                           # the event shows what every cell saw
            grid, acc = self._cell_sink(k)
            self._integrate_one(i, fits[k][j][0], grid, acc, cell_id=k, known_cell=True)
        return missed

    def _all_cells(self):
        return [self.Mc] + [e["Mc"] for e in self.extra]

    def _cell_sink(self, k):
        """(grid, acc) for active cell k -- 0 is the primary, 1..n the adaptive-relock extras."""
        return ((self.grid, self.acc) if k == 0
                else (self.extra[k - 1]["grid"], self.extra[k - 1]["acc"]))

    def _cascade_accept(self, q, M, Mk):
        """Is candidate `M` an acceptable registration of frame `q` under ACTIVE cell `Mk`?

        Three bars, and all three are needed:
          * a non-degenerate matrix, as everywhere else in the batch path;
          * `same_lattice(M, Mk)` -- the blind arm returns whatever cell fits BEST, which need not be
            the cell whose accumulator we are about to add to. Without this a blind candidate for
            some other lattice would be integrated into cell k's merge as if it were cell k, which is
            corruption rather than a miss. (This is the same rule _watchdog's individual rescue
            applies to its blind candidates, and it mirrors the strict research gate's own
            same_lattice term.)
          * `_fits` -- THE live accept gate, identical to the one the frame just failed. A retry that
            accepted on a looser bar than the batch pass would not be recovering frames, it would be
            lowering the gate.
        """
        if M is None:
            return False
        M = np.asarray(M, float)
        if abs(np.linalg.det(M)) < 1.0:
            return False
        return same_lattice(M, Mk) and self._fits(q, M)

    def _cascade_retry(self, slots):
        """glint#75: run the measured retry arms on frames that fit NO active cell, integrating any
        that a retry recovers. Returns `(still_missed_slots, nbest_by_slot)`, where the first
        continues down the existing miss path (miss buffer / watchdog / gate_rejected) untouched and
        the second is this pass's blind N-best keyed by slot, for `_watchdog` to reuse.

        Called once per frame per flush, AFTER every active cell has had its ordinary batched pass --
        so a rescued frame is integrated exactly once and never also counted as a miss, and a frame
        the cascade cannot save behaves exactly as it does today.

        The blind index is fanned out over the whole retry set through `self._fanout` (default: the
        serial loop, which is what the offline measurement runs), because that call is by far the
        expensive part and the watchdog already batches it the same way. Its N-best candidates are
        then handed to the arm rather than recomputed -- and handed onward to the watchdog, which
        would otherwise blind-solve every unrescued frame a SECOND time (see _watchdog_nbest).
        """
        if not slots:
            return slots, {}
        cells = self._all_cells()
        qs = [self._q[i] for i in slots]
        self.n_cascade_retried += len(slots)
        # ONE blind index per frame, shared by every active cell below. Guarded (glint#147): an
        # exception degrades every slot to None and a short return is padded -- either way each
        # affected frame goes through the per-frame fallback below, and is cached like any other
        # result. The guard also counts/warns, so a dead fan-out is visible, not just survived.
        nbs = self._fanout_guarded(qs, self.retry_nbest)
        arm1 = f"blind_nbest_k{self.retry_nbest}"
        still = []
        for j, (i, q) in enumerate(zip(slots, qs)):
            if q is None:                                    # pragma: no cover - flush() filters these
                still.append(i)
                continue
            nb = nbs[j]
            if nb is None:
                # The fan-out raised, or declined this one frame. Solve it HERE and write the result
                # back, rather than letting arm_blind_nbest solve it privately with candidates=None:
                # that result would be discarded, the slot would be missing from the cache below, and
                # _watchdog_nbest would blind-solve the frame a SECOND time -- the exact ~26 ms double
                # solve this reuse exists to remove, surviving in the failure path.
                nb = nbs[j] = self._blind_candidates(q)
            hit = {}

            def blind_gate(c, q=q, hit=hit):
                """Accept candidate `c` for WHICHEVER active cell it is a registration of, and
                remember which -- so the frame lands in that cell's accumulator, not cell 0's."""
                for k, Mk in enumerate(cells):
                    if self._cascade_accept(q, c, Mk):
                        hit["k"] = k
                        return True
                return False

            # arm 1: blind N-best at k=retry_nbest. `candidates` is ALWAYS a concrete list by now
            # (fanned out, or solved just above), so the arm never runs a blind index of its own --
            # which is what keeps every solve visible to the cache. A frame whose blind indexer is
            # genuinely broken arrives here as [], and the arm simply finds nothing.
            M = arm_blind_nbest(q, blind_gate, self._blind_index, self.retry_nbest,
                                candidates=nb if nb is not None else [])
            arm = arm1
            if M is None and self._known_perframe is not None:
                arm = "known_perframe"                       # arm 2: PER-FRAME known-cell registration
                for k, Mk in enumerate(cells):
                    M = arm_known_perframe(q, Mk, lambda c, Mk=Mk: self._cascade_accept(q, c, Mk),
                                           self._known_perframe)
                    if M is not None:
                        hit["k"] = k
                        break
            if M is None:
                still.append(i)
                continue
            k = hit.get("k", 0)
            grid, acc = self._cell_sink(k)
            self._integrate_one(i, M, grid, acc, cell_id=k, known_cell=(arm == "known_perframe"),
                                outcome="rescued_cascade")
            self.n_cascade_rescued += 1
            self.n_cascade_by_arm[arm] = self.n_cascade_by_arm.get(arm, 0) + 1
        # Only the UNRESCUED slots can reach the watchdog, so only their candidates are worth
        # carrying. A None entry survives just one way now -- the blind indexer itself ERRORED, so we
        # do not know what its candidates would have been. Those stay out of the cache and the
        # watchdog re-attempts them, exactly as it would have before. An empty LIST is a real answer
        # ("this frame yielded no candidates") and IS cached, so it is not re-solved.
        keep = set(still)
        cache = {i: nb for i, nb in zip(slots, nbs) if i in keep and nb is not None}
        return still, cache

    def _blind_candidates(self, q):
        """One frame's blind N-best at `retry_nbest`, or None if the indexer errored.

        The None/[] distinction is load-bearing for the cache above: [] means "solved, found
        nothing" (cacheable, do not re-solve), None means "we never got an answer" (not cacheable).
        """
        if self._blind_index is None:
            return None
        try:
            return list(self._blind_index(q, self.retry_nbest))
        except Exception:                                    # pragma: no cover - a dead blind indexer
            return None

    def _fanout_guarded(self, qs, k):
        """`self._fanout`, degraded instead of fatal (glint#147): ALWAYS returns a list of
        len(qs), where a None entry means "no answer for this frame".

        The watchdog's fan-out invocation was unguarded, so a fanout that raises persistently --
        a dead GPU worker, a poisoned queue -- propagated out of flush() and took the streaming
        driver down with it, even though every one of those frames already has a defined miss
        path. On an exception every slot degrades to None (counted in n_fanout_errors, surfaced
        in stats(), and warned with the consecutive-failure streak so a dead fan-out reads
        differently from a transient one). A SHORT return is padded with None rather than zipped
        away: a bare zip silently dropped the tail slots, so they were neither retried nor
        returned to the miss path -- they just vanished (the defect _cascade_retry's fan-out
        already fixed for itself; this centralizes it). On the no-exception, full-length path the
        fan-out's own results pass through untouched, so flag-off behaviour stays bit-identical:
        the guard only changes what happens on the paths that used to crash or drop frames."""
        try:
            out = list(self._fanout(qs, k))
        except Exception as exc:
            self.n_fanout_errors += 1
            self._fanout_fail_streak += 1
            warnings.warn(
                f"blind fan-out raised ({self._fanout_fail_streak} consecutive failure(s)): "
                f"{exc!r} -- degrading {len(qs)} frame(s) to the miss path; a persistent count "
                f"here means the fan-out workers are dead (see stats()['n_fanout_errors'])",
                RuntimeWarning, stacklevel=3)
            return [None] * len(qs)
        self._fanout_fail_streak = 0
        if len(out) < len(qs):
            out += [None] * (len(qs) - len(out))
        return out

    def _watchdog_nbest(self, missed, cached):
        """N-best candidates for the watchdog's missed frames, reusing the retry cascade's blind
        solves instead of paying for a second one (glint#145 review).

        Without this, `adaptive_relock` + `retry_cascade` blind-index every unrescued frame TWICE --
        once in _cascade_retry at `retry_nbest`, once here at `warmup_nbest` -- and the blind solve is
        the ~26 ms that dominates the retry's whole cost. So the frames that gain nothing from the
        cascade were the ones paying double for it.

        THE REUSE IS EXACT, NOT AN APPROXIMATION, and only because of how index_blind_nbest is built:
        `N` is consulted ONLY in the final dedup loop (`if len(out) >= N: break`) -- the M3 refine, the
        triplet anneal and the GPU metric-dedup that produce the ranked list all run before it. So
        top-N is a verbatim PREFIX of top-M for any N <= M, and slicing the cached list to
        `warmup_nbest` yields exactly the list a fresh `_fanout(q, warmup_nbest)` would have returned.
        Verified on the real cxidb frames as well as read off the source. The watchdog's consensus
        therefore votes on identical candidates and can discover identical cells.

        When `warmup_nbest > retry_nbest` the cache is a prefix but too SHORT, and there is no honest
        cheap fix: the truncation is the only thing N controls, so "topping up the delta" means
        re-running the entire solve anyway and then merging two ranked lists. This falls back to the
        ordinary single fan-out for those frames instead -- no saving in that configuration, but the
        watchdog's candidate set is preserved exactly, which is the property worth keeping. The
        shipped defaults (warmup_nbest=3, retry_nbest=10) are in the reusable regime.
        """
        qs = [self._q[i] for i in missed]
        if not cached or self.warmup_nbest > self.retry_nbest:
            return self._fanout_guarded(qs, self.warmup_nbest)   # historical path (guarded, glint#147)
        out = [None] * len(missed)
        todo, todo_at = [], []
        for j, i in enumerate(missed):
            nb = cached.get(i)
            if nb is None:
                todo.append(qs[j]); todo_at.append(j)         # not cached -> still needs a solve
            else:
                out[j] = list(nb)[:self.warmup_nbest]
        if todo:                                              # one batched call for the remainder,
            for j, nb in zip(todo_at, self._fanout_guarded(todo, self.warmup_nbest)):
                out[j] = nb                                   # so the fan-out is still fanned out
        return out

    def _watchdog(self, missed, cached_nbest=None):
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
        non-crystals.

        `cached_nbest` (slot -> N-best) lets the retry cascade hand over the blind solves it already
        paid for, so an unrescued frame is not indexed twice; None (the default, and always the case
        with retry_cascade off) takes the historical fan-out verbatim. See _watchdog_nbest."""
        if self._watch is None:
            self._watch = RunningConsensus(min_support=3, gap=2, adaptive=False)
        cells = self._all_cells()
        still_missed = []; watch_ev = []
        # Fan the independent blind indexes across workers (default `_fanout` is the serial
        # single-GPU loop, so the votes -- and the lock -- are BIT-IDENTICAL to the old per-frame
        # loop; RunningConsensus is a histogram, add_frame order does not change the verdict.
        # Opt-in mpi_fanout distributes the frames across GPUs).
        for i, nb in zip(missed, self._watchdog_nbest(missed, cached_nbest)):
            q = self._q[i]
            if nb is None:
                # The fan-out died for this frame (glint#147). No candidates means no rescue and
                # NO VOTE -- absence of evidence must not feed the consensus histogram as a frame
                # that voted for nothing. The frame stays a miss: when a miss buffer is armed it
                # is already in it (flush() buffers before calling here). Counted regardless --
                # with rescue_buffer=0 this counter is the ONLY ledger these frames appear in
                # (Copilot review of #156, suppressed comment).
                self.n_fanout_missed += 1
                still_missed.append(i)
                continue
            rescued = False
            for c, _ in nb:
                c = np.asarray(c, float)
                if abs(np.linalg.det(c)) < 1.0 or not self._fits(q, c):
                    continue
                for k, Mk in enumerate(cells):
                    if same_lattice(c, Mk):
                        grid, acc = self._cell_sink(k)
                        self._integrate_one(i, c, grid, acc, cell_id=k, outcome="rescued_watchdog")
                        self.n_watchdog_rescued += 1
                        if getattr(self, "_pix", None) is not None:
                            self._pix.release([self._idx[i]])   # integrated now: a relock must not do it again
                        rescued = True
                        break
                if rescued:
                    break
            if rescued:
                continue
            # NOT `cells`: that name holds the ACTIVE cell list this loop rescues against, and
            # rebinding it here made every later frame check itself against the previous frame's blind
            # candidates instead -- a self-match that "rescued" the frame into cell 0's accumulator and
            # kept it out of watch_ev, starving the gate below of the frames its min_frames needs.
            vote_cells = [c for c, _ in nb if c is not None]
            self._watch.add_frame(vote_cells)
            if q is not None:
                watch_ev.append((q, vote_cells))        # per-frame evidence for the gate below
            still_missed.append(i)
        missed = still_missed
        if self._events_on:
            for i in missed:
                self._emit_frame(i, "miss")
        Mn = self._watch.verdict()[0]
        if Mn is None:
            return
        Mn = self._standardize(Mn)
        if self._alias_gate is not None:
            # Deterministic relock confirmation, scored PER FRAME and voted -- the missed frames are at
            # different orientations, and coverage/occupancy are only meaningful within one orientation.
            # Each voting frame supplies its own N-best hypothesis that agrees with the voted cell, i.e.
            # that lattice as oriented on that frame. (This used to score the single best-fitting frame
            # via `confirm`, which is one frame's opinion -- exactly what the gate's min_frames exists to
            # refuse.) Returns the leader (confirmed/abstained), a tighter derivative lattice (adopt
            # mode), or None (refuse -> do NOT reset self._watch, so the histogram keeps accumulating
            # for a later, cleaner lock).
            voters = [(q, next(c for c in cs if same_lattice(c, Mn)))
                      for q, cs in watch_ev
                      if any(same_lattice(c, Mn) for c in cs)]
            if not voters:
                # No frame in THIS batch supports the verdict: the histogram retained it from an
                # earlier batch (a refusal deliberately keeps accumulating), and this batch added
                # nothing that agrees -- every entry was None from a dead fan-out (glint#147), every
                # frame was rescued, or the votes went elsewhere. confirm_frames cannot run without
                # voters, and skipping it here would commit a cell the gate may already have
                # REFUSED, precisely on the batch with the least evidence (Copilot review of #156).
                # Refuse instead: same bookkeeping, histogram kept, a later batch with real
                # supporting frames confirms or refuses on its own evidence.
                self.n_gate_refused += 1                     # total keeps its #83 semantics...
                self.n_gate_deferred += 1                    # ...and the subset is now separable
                return
            Mg = self._alias_gate.confirm_frames(Mn, voters)
            if Mg is None:
                self.n_gate_refused += 1
                return
            Mn = self._standardize(Mg)
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
            best_q, best_M, best_frac = None, None, -1.0
            for q, Mi in zip(qs, Ms):
                if Mi is None:
                    continue
                frac = self._inliers(q, Mi) / max(len(q), 1)     # rank by FRACTION, not count, so the
                if frac > best_frac:                             # probe frame is the best-FITTING one,
                    best_q, best_M, best_frac = q, Mi, frac      # not merely the most peak-rich one
            lock_z = (null_margin(best_q, best_M, n_null=self._probe_null,
                                  rng=np.random.default_rng(self.n_relock))["z"]
                      if best_M is not None and self._fits(best_q, best_M) else 0.0)
            self.lock_z = lock_z
            if self._lock_min_z is not None and lock_z < self._lock_min_z:
                return
        k_new = len(self.extra) + 1
        self.extra.append(dict(Mc=Mn, grid=HKLGrid(Mn, self.dmin, gpu=self.gpu, panels=self.panels,
                                                   clen_m=self.clen_m, wavelength_A=self.wavelength_A,
                                                   ),
                               nth=theoretical_unique(Mn, self.dmin, self.ops),
                               acc=MergeAccumulator(self.snr_bins, self.ops), lock_z=lock_z,
                               name=self._cell_name(Mn, k_new)))
        self.n_relock += 1
        self._register_cell(k_new, Mn, "relock", lock_z=lock_z)
        if self._events_on:
            self._emit_relock(k_new, "relock", lock_z)
        if self._missbuf:                                       # retroactive INDEX-ONLY rescue of buffered misses
            # Re-index the buffered pre-lock misses against the newly locked cell Mn with the same
            # q-only fast-path indexer the batch uses (rgb.index_fused: q in, [M or None] out, no
            # pixels). Selection-misses (the recurring new cell) pass the same _inliers gate; blank/
            # spurious "generation" misses are correctly refused. Index-only, unless rescue_pixels kept
            # the frame's pixels: then it is integrated into the new cell's merge as well.
            evs = [e for e, _ in self._missbuf]; qs = [q for _, q in self._missbuf]
            Ms = self._known_index(qs, Mn, B=max(len(qs), 1))
            pix = getattr(self, "_pix", None)
            for ev, q, M in zip(evs, qs, Ms):                   # same count as the old sum(...)
                if M is not None and abs(np.linalg.det(np.asarray(M, float))) >= 1.0 and self._fits(q, M):
                    self.n_rescued += 1
                    self._attribute(k_new, ev)
                    if self._events_on:
                        self._emit_retro(ev, q, M, "rescued_relock", k_new)
                    if pix is not None:
                        grid, acc = self._cell_sink(k_new)
                        if self._integrate_stored(ev, q, np.asarray(M, float), grid, acc, k_new, "rescued_relock"):
                            self.n_rescued_integrated += 1
            if pix is not None:
                pix.release(evs)
            self._missbuf.clear()
        self._watch = RunningConsensus(min_support=3, gap=2, adaptive=False)    # reset for the next change

    def flush(self):
        """Index the resident batch, integrate each frame against its still-resident pixels."""
        if self._blind or self._n == 0:                         # nothing to integrate without a cell
            return
        slots = [i for i in range(self._n) if self._q[i] is not None]
        eff = self._eff
        if eff is not None:                                     # adaptive effort: (tier, deep) for THIS flush
            eff.n_flush += 1
            if (eff.n_flush - 1) % eff.every == 0:
                at = int(self._idx[0]) if self._n else self.n_pushed          # the first frame this decision applies to
                if eff.decide(at, n_cells=1 + len(self.extra)) and self._events_on:
                    self._emit_effort()
        if slots and not self.adaptive_relock:
            # gate=self.retry_cascade, not gate=False: with the cascade OFF this is the historical
            # single-cell call verbatim (rejects counted and dropped inside _index_integrate). With
            # it ON the rejects are handed back so the cascade can retry them, and whatever it cannot
            # save is counted in n_gate_rejected here -- the same frames, the same counter.
            pl = getattr(self, "per_lattice", False)
            missed = self._index_integrate(slots, self.Mc, self.grid, self.acc,
                                           gate=self.retry_cascade or pl or eff is not None)
            if eff is not None:
                eff.flushed(len(slots), len(missed))
            if missed and pl:                               # per-lattice first: it scores the registration in hand
                missed = self._per_lattice_rescue(missed)
            if missed and eff is not None and eff.deep:     # then the deep known-cell search with its null
                missed = self._escalate_misses(missed)
            if missed:
                # no watchdog here -> no cache to carry; without the cascade the misses are just counted
                still, _ = self._cascade_retry(missed) if self.retry_cascade else (missed, None)
                self.n_gate_rejected += len(still)
                if self._events_on:
                    for i in still:
                        self._emit_frame(i, "gate_rejected")
        elif slots and self.assign == "best" and self.extra:
            remaining = self._index_best_fit(slots)            # every cell sees every frame; most inliers wins
            if eff is not None:
                eff.flushed(len(slots), len(remaining))
            if remaining and getattr(self, "per_lattice", False):
                remaining = self._per_lattice_rescue(remaining)
            if remaining and eff is not None and eff.deep:      # deep known-cell search before the blind retry
                remaining = self._escalate_misses(remaining)
            cached_nbest = None
            if remaining and self.retry_cascade:
                remaining, cached_nbest = self._cascade_retry(remaining)
            if remaining:
                if self._missbuf is not None:
                    self._buffer_misses(remaining)
                self._watchdog(remaining, cached_nbest)
        elif slots:
            remaining = self._index_integrate(slots, self.Mc, self.grid, self.acc, gate=True)
            for k, e in enumerate(self.extra, 1):               # try each additional active cell in turn
                if not remaining:
                    break
                remaining = self._index_integrate(remaining, e["Mc"], e["grid"], e["acc"], gate=True,
                                                  cell_id=k)
            if eff is not None:
                eff.flushed(len(slots), len(remaining))
            if remaining and getattr(self, "per_lattice", False):
                remaining = self._per_lattice_rescue(remaining)
            if remaining and eff is not None and eff.deep:      # deep known-cell search before the blind retry
                remaining = self._escalate_misses(remaining)
            cached_nbest = None
            if remaining and self.retry_cascade:                # retry BEFORE the miss buffer claims them
                remaining, cached_nbest = self._cascade_retry(remaining)
            if remaining:                                       # fit no active cell -> blind watchdog
                if self._missbuf is not None:                   # buffer for retroactive rescue on lock
                    self._buffer_misses(remaining)
                # hand over the cascade's blind solves so these frames are not indexed a second time
                self._watchdog(remaining, cached_nbest)
        self._n = 0
        self._q = [None] * self.B
        self._pk = [None] * self.B
        self._idx = [0] * self.B
        self._pkq = [None] * self.B
        self._haspix = [True] * self.B
        self._src = [None] * self.B
        self._inl_alt = [None] * self.B
        self._pl = [None] * self.B
        self._pl_v = [None] * self.B
        self._rej = {}

    def stats(self, thr=0.0):
        if self._blind:                                         # not yet locked -- warm-up in progress
            _, sup, lead = self._rc.verdict()
            # gate_refused belongs HERE most of all. `_gate_lock` runs only on the blind path
            # (_push_blind, warmup_batch), and a refusal is precisely what leaves the driver blind
            # -- so the moment this counter is the whole explanation of what the operator is
            # looking at is the moment the locked branch below never runs. Reporting it only after
            # a successful lock would have hidden it at the one time it answers the question
            # "there is support but no cell -- why?" (Copilot review of glint#164).
            s = dict(locked=False, pushed=self.n_pushed, warmup_indexed=self.n_warmup,
                     consensus_support=sup, consensus_lead=lead,
                     consensus_members=self._rc.leader_counts()[1],   # folded >= support (glint#182)
                     gate_refused=self.n_gate_refused)
            if getattr(self, "_pix", None) is not None:
                s["pixels_held"] = len(self._pix)
                s["pixels_evicted"] = self._pix.n_evicted
            return s
        s = self.acc.stats(thr=thr, n_theoretical=self.n_theoretical)
        s.update(locked=True, locked_after=self.locked_after, consensus_support=self.consensus_support,
                 consensus_members=self.consensus_members,      # folded >= support (glint#182)
                 pushed=self.n_pushed, indexed=self.n_indexed, integrated=self.n_integrated,
                 theoretical_unique=self.n_theoretical,
                 # the Laue class the numbers above were merged under (None = explicit `ops`), so a
                 # disagreement with the stream header is visible where the numbers are (glint#180)
                 laue=self.laue,
                 # frames the ingest gate refused outright. Reported unconditionally: dropping data
                 # must never be silent, and a rising count is the signal that the cell has drifted
                 # away from the sample (or that min_inlier_frac is set too high for this run).
                 gate_rejected=self.n_gate_rejected,
                 # alias-gate REFUSALS, in the broad sense: proposed locks that were NOT
                 # committed. Counted since glint#83 but never reported, so the one diagnostic
                 # that says "a cell was proposed and turned down" was invisible to anyone
                 # watching a live run. The total has two sources -- the alias gate actually
                 # scoring a cell and refusing it, and the watchdog deferring a verdict whose
                 # batch carried no supporting voters (every entry rescued, votes elsewhere, or a
                 # dead fan-out; the #156 fix) -- and the neighbours CANNOT disambiguate them: a
                 # healthy fan-out defers too, so a flat n_fanout_errors proves nothing (Copilot
                 # review of glint#164, round 3). The subset is therefore reported under its own
                 # key: gate_refused - gate_deferred_no_voters = verdicts the gate itself scored
                 # and rejected, which is the "alias hypotheses are being held back" diagnostic;
                 # gate_deferred_no_voters rising alone means verdicts are outliving their
                 # evidence, not that aliases are being generated. The total keeps its #83
                 # semantics so nothing downstream moves.
                 gate_refused=self.n_gate_refused,
                 gate_deferred_no_voters=self.n_gate_deferred,
                 # fan-out invocations (retry cascade or watchdog) that raised and were degraded to
                 # the miss path (glint#147). Reported unconditionally for the same reason: a dead
                 # fan-out silently turns every retry/relock mechanism off, and this counter is the
                 # only place that shows. n_fanout_missed is the FRAMES the watchdog could not
                 # attempt because the fan-out gave no answer -- with rescue_buffer=0 they appear
                 # in no other ledger.
                 n_fanout_errors=self.n_fanout_errors,
                 n_fanout_missed=self.n_fanout_missed)
        if self.warmup_rescue:
            s["n_warmup_rescued"] = self.n_warmup_rescued         # warm-up frames recovered the instant the cell locked
        if getattr(self, "_pix", None) is not None:                # rescue_pixels: the rescues that reached the merge
            s["n_warmup_integrated"] = self.n_warmup_integrated
            s["n_rescued_integrated"] = self.n_rescued_integrated
            s["pixels_held"] = len(self._pix)                      # frames the store holds right now
            s["pixels_evicted"] = self._pix.n_evicted              # kept frames pushed out before any rescue
        if getattr(self, "null_floor", None) is not None:        # opt-in chance floor on the live gate
            s["null_floor"] = list(self.null_floor)
            # registrations that cleared the count and fraction bars and failed only the floor -- per
            # registration tested, so a frame tried under two cells or by a rescue can count twice
            s["n_null_floor_refused"] = self.n_null_floor_refused
        if self.retry_cascade:                                   # glint#75, opt-in
            s["n_cascade_retried"] = self.n_cascade_retried      # gate failures the cascade was run on
            s["n_cascade_rescued"] = self.n_cascade_rescued      # ...INTEGRATED after a retry (not misses)
            # per arm, because the whole finding is that the arms are complementary: an arm that
            # stops contributing on live data is the signal that the cascade has become one arm.
            s["n_cascade_by_arm"] = dict(self.n_cascade_by_arm)
        if self.qc_frac_threshold is not None:
            s["n_low_confidence"] = self.n_low_confidence          # integrated but below qc_frac_threshold -- sent, flagged
            s["low_conf_frac"] = self.n_low_confidence / max(self.n_integrated, 1)
        if self._writer is not None:                               # offline-merge handoff
            s["stream_out"] = self.stream_out
            s["stream_chunks"] = self._writer.n_chunks
            s["stream_indexed"] = self._writer.n_indexed
        if self.adaptive_relock:                                # adaptive: report the active cell set
            s["n_cells"] = 1 + len(self.extra); s["n_relock"] = self.n_relock
            s["n_rescued"] = self.n_rescued                      # buffered pre-lock misses recovered on relock
            s["n_watchdog_rescued"] = self.n_watchdog_rescued    # same-cell misses the watchdog rescued individually
            if self._lock_probe:
                s["lock_z"] = self.lock_z                        # signal-strength z of the most recent relock
            s["extra_cells"] = [dict(axes=list(np.linalg.norm(e["Mc"], axis=0).round(1)),
                                     **e["acc"].stats(thr=thr, n_theoretical=e["nth"])) for e in self.extra]
        if self.adaptive_relock or getattr(self, "_events_on", False) or getattr(self, "_roster", None) is not None:
            s["cells"] = self.cell_table()                       # registry snapshot, ALL active cells incl. the primary
        if self.double_hit:
            s["n_double"] = self.n_double                    # GATED: same cell AND >= 15 deg away
            s["double_hit_rate"] = self.n_double / max(self.n_indexed, 1)
            s["n_double_raw"] = self.n_double_raw            # the old bare rule, kept for comparison
            s["double_hit_rate_raw"] = self.n_double_raw / max(self.n_indexed, 1)
            # the gated rule's own false-accept floor, measured live on azimuth-scrambled residuals;
            # a double_hit_rate is only meaningful read AGAINST this
            s["n_dh_null"] = self.n_dh_null_tested
            s["dh_null_rate"] = self.n_dh_null_acc / max(self.n_dh_null_tested, 1)
        if getattr(self, "per_lattice", False):
            s["n_per_lattice_searched"] = self.n_pl_searched
            s["n_per_lattice_found"] = self.n_pl_found
            s["n_per_lattice_rescued"] = self.n_pl_rescued
            s["n_per_lattice_swapped"] = self.n_pl_swapped       # rescued under the stronger, second lattice
            # the rescue rule's own false-accept floor on azimuth-scrambled residuals (1 search in 16):
            # a second lattice found at all, and one that would also have rescued the frame
            s["n_pl_null"] = self.n_pl_null_tested
            s["pl_null_found_rate"] = self.n_pl_null_found / max(self.n_pl_null_tested, 1)
            s["pl_null_rescued_rate"] = self.n_pl_null_rescued / max(self.n_pl_null_tested, 1)
        eff = getattr(self, "_eff", None)
        if eff is not None:                                     # adaptive effort: what the policy is doing now
            s["effort"] = eff.summary()
        if self._grefiner is not None:
            s["geom_correction"] = self._grefiner.correction()
        return s
