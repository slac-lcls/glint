"""Fused GPU box-integration (cupy RawKernel) matching glint.predict.integrate_spots exactly.

One WARP per reflection: the 32 lanes cooperatively gather the (2R+1)^2 patch straight from the
detector image in its NATIVE dtype (no whole-image upcast), warp-reduce the box sum and box max, and
stage the annulus values in shared memory.

The background needs a MEDIAN -- a SELECTION, not a reduction, so it cannot be warp-reduced. Rather
than sort, each lane rank-counts its own values against the whole annulus:
    v is the k-th order statistic  <=>  count(<v) <= k < count(<v) + count(==v)
which is EXACT and handles ties correctly. numpy's even-n median is the mean of sorted[(n-1)/2] and
sorted[n/2]; both are found in the same pass (they coincide for odd n). All lanes that satisfy a
condition hold the same value, so the shared-slot write races are benign.

BG MODES, mirroring glint.predict.integrate_spots exactly (glint#131 changed the default from the
median to a MAD-clipped mean; leaving this file behind would have put the GPU streaming path on a
different estimator from the offline one, silently):
  0 clipmean  the default. A SECOND rank-count pass -- the same routine over |v - med|, recomputed
              on the fly so no extra shared buffer is needed -- gives the MAD; the surviving
              |v - med| <= NSIG*max(1.4826*MAD, sqrt(max(med,1))) values are warp-reduced to a sum
              and a count. Costs one more O(NA^2)/32 pass per reflection.
  1 median    bit-for-bit what this kernel did before.
  2 mean      the same masked reduction with an infinite window.

Accumulation is float64 throughout, as in numpy.

Measured agreement with integrate_spots, ALL THREE bg modes (A100-SXM4-40GB, cupy 13.6.0, driver
12090; experiments/bench_integrate_fused.py, which exits nonzero if any of this stops holding):
  uint16 / int32 / float32 input -- BIT-EXACT on all four outputs, in all three modes, including
    NON-INTEGER gain-corrected float32, edge-straddling boxes, and frames seeded with saturated
    pixels in the annulus. (float32 widened to double sums exactly for a 49-pixel box regardless
    of order, so summation order cannot bite.)
  float64 non-integer input -- peak is exact (a max is order-independent) and bg is exact for the
    median (a selection), but I, sigma, and the two mean-like backgrounds differ, because the warp
    reduction sums in a different order than numpy's pairwise summation. Measured max|dI| 5.1e-13
    counts on a lambda=6 frame and 3.6e-12 on a lambda=50 one -- it scales with the box sum, so it
    is a rounding of the signal, not a constant. Quote it ABSOLUTELY: ~1e-12 counts is twelve
    orders below one photon, while the same difference on the near-zero reflections reads ~1e-11
    as a RATIO and means nothing.

Getting there needed __dmul_rn/__dadd_rn/__dsub_rn on the final I and sigma expressions; see the
comment at that line. Bit-exactness with a numpy reference is not something a CPU emulation of a
kernel can establish -- the emulation of THIS kernel said "bit-exact" while the device did not,
because the divergence was introduced by the compiler, not by the algorithm (glint#131).

CAVEAT ON SPEED: the kernel is ~0.33 ms and nearly flat in frame size and spot count, but a
host->device copy of a 16 Mpix frame costs ~3-5 ms -- an order of magnitude more than the kernel.
This only pays when the frame is ALREADY GPU-resident, which is the live-DRP case (calibration and
peakfinding already run on device). Called with a host array it is copy-bound, not compute-bound."""
import numpy as np, cupy as cp

from glint.predict import BG_MODES, BG_NSIG      # ONE definition of the estimator's constants

_SRC = r"""
extern "C" __global__ void integrate_fused(
    const DT_IN* __restrict__ img,
    const int* __restrict__ cs, const int* __restrict__ cf,
    double* __restrict__ Iout, double* __restrict__ Sout,
    double* __restrict__ Pout, double* __restrict__ Bout,
    const int n, const int H, const int W,
    const int half, const int gap, const int ring, const int bgmode)
{
    const int i = blockIdx.x;
    if (i >= n) return;
    const int lane = threadIdx.x;
    const int R = half + gap + ring;
    const int P = 2 * R + 1;
    const int npatch = P * P;

    extern __shared__ double ann[];              // annulus staging, >= nann doubles
    __shared__ int nann_s;
    __shared__ double med1_s, med2_s, mad1_s, mad2_s;

    const int c_s = cs[i], c_f = cf[i];
    if (c_s - R < 0 || c_s + R >= H || c_f - R < 0 || c_f + R >= W) {   // numpy `valid` gate
        if (lane == 0) { Iout[i] = 0.0; Sout[i] = 0.0; Pout[i] = 0.0; Bout[i] = 0.0; }
        return;
    }
    if (lane == 0) { nann_s = 0; med1_s = 0.0; med2_s = 0.0; mad1_s = 0.0; mad2_s = 0.0; }
    __syncwarp();

    double bsum = 0.0, bmax = -1.0e300;
    for (int p = lane; p < npatch; p += 32) {
        const int dy = p / P - R, dx = p % P - R;
        const int ady = dy < 0 ? -dy : dy, adx = dx < 0 ? -dx : dx;
        const int ad = ady > adx ? ady : adx;                 // Chebyshev radius
        const double v = (double)img[(long)(c_s + dy) * (long)W + (long)(c_f + dx)];
        if (ad <= half) {
            bsum += v; if (v > bmax) bmax = v;
        } else if (ad > half + gap && ad <= half + gap + ring) {
            ann[atomicAdd(&nann_s, 1)] = v;                   // order irrelevant for a median
        }
    }
    for (int off = 16; off; off >>= 1) {                      // warp-reduce sum + max
        bsum += __shfl_down_sync(0xffffffff, bsum, off);
        const double o = __shfl_down_sync(0xffffffff, bmax, off);
        if (o > bmax) bmax = o;
    }
    __syncwarp();

    const int NA = nann_s;
    const int k1 = (NA - 1) / 2, k2 = NA / 2;                 // numpy even-n median indices
    for (int j = lane; j < NA; j += 32) {
        const double v = ann[j];
        int cl = 0, ce = 0;
        for (int q = 0; q < NA; ++q) { const double w = ann[q]; cl += (w < v); ce += (w == v); }
        if (cl <= k1 && k1 < cl + ce) med1_s = v;             // benign race: same value
        if (cl <= k2 && k2 < cl + ce) med2_s = v;
    }
    __syncwarp();
    const double med = 0.5 * (med1_s + med2_s);

    // --- background estimator (mirrors glint.predict.integrate_spots bg_mode) ---------------
    double bg = med;
    if (bgmode != 1) {
        double thr = __longlong_as_double(0x7ff0000000000000LL);   // "mean": an infinite window
        if (bgmode == 0) {                                    // "clipmean": MAD-clip, same rank-count
            for (int j = lane; j < NA; j += 32) {
                const double v = fabs(ann[j] - med);
                int cl = 0, ce = 0;
                for (int q = 0; q < NA; ++q) {
                    const double w = fabs(ann[q] - med); cl += (w < v); ce += (w == v);
                }
                if (cl <= k1 && k1 < cl + ce) mad1_s = v;
                if (cl <= k2 && k2 < cl + ce) mad2_s = v;
            }
            __syncwarp();
            const double mad = 0.5 * (mad1_s + mad2_s);        // numpy's operand ORDER, so the
            const double sc = 1.4826 * mad;                    // rounding matches bit for bit
            const double flo = sqrt(med > 1.0 ? med : 1.0);   // Poisson floor: MAD is 0 on sparse data
            thr = NSIG * (sc > flo ? sc : flo);
        }
        double asum = 0.0; int acnt = 0;
        for (int j = lane; j < NA; j += 32) {
            const double v = ann[j];
            if (fabs(v - med) <= thr) { asum += v; acnt += 1; }
        }
        for (int off = 16; off; off >>= 1) {
            asum += __shfl_down_sync(0xffffffff, asum, off);
            acnt += __shfl_down_sync(0xffffffff, acnt, off);
        }
        if (acnt > 0) bg = asum / (double)acnt;               // lanes>0 hold partials; only lane 0 writes
    }

    if (lane == 0) {
        const int NBOX = (2 * half + 1) * (2 * half + 1);
        const double bgc = bg > 0.0 ? bg : 0.0;
        // ROUND THE PRODUCT, THEN ADD -- numpy's two roundings, not one. Written `bsum - NBOX*bg`
        // the compiler contracts this into an FMA (nvrtc defaults to -fmad=true), which rounds
        // ONCE and lands up to half an ulp away from `sig_sum - nbox * bg`. That was invisible
        // while the background was a median: a median of integer counts is an integer or a
        // half-integer, so NBOX*bg is EXACT and the two agree bit for bit. A clipmean (or mean)
        // background is asum/acnt, generally non-terminating, so the product is inexact and the
        // contraction shows up -- measured on an A100 as max|dI| 2.8e-14 on uint16/int32/float32
        // and 2.3e-10 with saturated pixels in the annulus, on otherwise IDENTICAL bg. Physically
        // nothing; but it broke this file's bit-exactness contract, and a CPU emulation cannot
        // see it, which is how it got past review.
        double s = __dadd_rn(bsum, __dmul_rn((double)NBOX, bgc)); if (s < 1.0) s = 1.0;
        Iout[i] = __dsub_rn(bsum, __dmul_rn((double)NBOX, bg));
        Sout[i] = sqrt(s);
        Pout[i] = bmax;
        Bout[i] = bg;
    }
}
"""

_DT = {np.dtype(np.float32): "float", np.dtype(np.float64): "double",
       np.dtype(np.uint16): "unsigned short", np.dtype(np.int32): "int",
       np.dtype(np.uint32): "unsigned int", np.dtype(np.int16): "short"}
_CACHE = {}


def _kern(dt):
    k = _CACHE.get(dt)
    if k is None:
        if dt not in _DT:
            raise TypeError(f"unsupported detector dtype {dt}")
        k = _CACHE[dt] = cp.RawKernel(f"#define DT_IN {_DT[dt]}\n#define NSIG {BG_NSIG!r}\n" + _SRC,
                                      "integrate_fused")
    return k


def integrate_fused(data_gpu, pred, half=3, gap=2, ring=3, bg_mode="clipmean"):
    """data_gpu: cupy (H,W) in the detector's native dtype. pred: numpy struct array with fs/ss.
    Returns (I, sigma, peak, bg) as numpy float64, matching integrate_spots.

    ``bg_mode`` is integrate_spots' and means the same thing; "median" reproduces what this kernel
    computed before glint#131."""
    if bg_mode not in BG_MODES:
        raise ValueError(f"bg_mode must be one of {BG_MODES}, got {bg_mode!r}")
    H, W = data_gpu.shape
    n = len(pred)
    if n == 0:
        z = np.zeros(0)
        return z, z.copy(), z.copy(), z.copy()
    cs = cp.asarray(np.rint(pred["ss"]).astype(np.int32))
    cf = cp.asarray(np.rint(pred["fs"]).astype(np.int32))
    I = cp.zeros(n, cp.float64); S = cp.zeros(n, cp.float64)
    Pk = cp.zeros(n, cp.float64); B = cp.zeros(n, cp.float64)
    R = half + gap + ring
    P = 2 * R + 1
    nann = int(((np.maximum(np.abs(np.mgrid[-R:R + 1, -R:R + 1][0]),
                            np.abs(np.mgrid[-R:R + 1, -R:R + 1][1])) > half + gap) &
                (np.maximum(np.abs(np.mgrid[-R:R + 1, -R:R + 1][0]),
                            np.abs(np.mgrid[-R:R + 1, -R:R + 1][1])) <= half + gap + ring)).sum())
    _kern(data_gpu.dtype)((n,), (32,),
                          (data_gpu, cs, cf, I, S, Pk, B,
                           np.int32(n), np.int32(H), np.int32(W),
                           np.int32(half), np.int32(gap), np.int32(ring),
                           np.int32(BG_MODES.index(bg_mode))),
                          shared_mem=nann * 8)
    return cp.asnumpy(I), cp.asnumpy(S), cp.asnumpy(Pk), cp.asnumpy(B)
