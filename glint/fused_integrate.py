"""Fused GPU box-integration (cupy RawKernel) matching glint.predict.integrate_spots exactly.

One WARP per reflection: the 32 lanes cooperatively gather the (2R+1)^2 patch straight from the
detector image in its NATIVE dtype (no whole-image upcast), warp-reduce the box sum and box max, and
stage the annulus values in shared memory.

The background is np.median over the annulus -- a SELECTION, not a reduction, so it cannot be warp-
reduced. Rather than sort, each lane rank-counts its own values against the whole annulus:
    v is the k-th order statistic  <=>  count(<v) <= k < count(<v) + count(==v)
which is EXACT and handles ties correctly. numpy's even-n median is the mean of sorted[(n-1)/2] and
sorted[n/2]; both are found in the same pass (they coincide for odd n). All lanes that satisfy a
condition hold the same value, so the shared-slot write races are benign.

Accumulation is float64 throughout, as in numpy.

Measured agreement with integrate_spots (A100):
  uint16 / int32 / float32 input -- BIT-EXACT on all four outputs, including non-integer
    gain-corrected float32 and edge-straddling boxes. (float32 widened to double sums exactly for a
    49-pixel box regardless of order, so summation order cannot bite.)
  float64 input -- bg and peak are exact (a median is a selection and a max is order-independent),
    but I and sigma can differ by ~5e-12 relative, because the warp reduction sums the box in a
    different order than numpy's pairwise summation. Far below Poisson noise, but not zero.

CAVEAT ON SPEED: the kernel is ~0.33 ms and nearly flat in frame size and spot count, but a
host->device copy of a 16 Mpix frame costs ~3-5 ms -- an order of magnitude more than the kernel.
This only pays when the frame is ALREADY GPU-resident, which is the live-DRP case (calibration and
peakfinding already run on device). Called with a host array it is copy-bound, not compute-bound."""
import numpy as np, cupy as cp

_SRC = r"""
extern "C" __global__ void integrate_fused(
    const DT_IN* __restrict__ img,
    const int* __restrict__ cs, const int* __restrict__ cf,
    double* __restrict__ Iout, double* __restrict__ Sout,
    double* __restrict__ Pout, double* __restrict__ Bout,
    const int n, const int H, const int W,
    const int half, const int gap, const int ring)
{
    const int i = blockIdx.x;
    if (i >= n) return;
    const int lane = threadIdx.x;
    const int R = half + gap + ring;
    const int P = 2 * R + 1;
    const int npatch = P * P;

    extern __shared__ double ann[];              // annulus staging, >= nann doubles
    __shared__ int nann_s;
    __shared__ double med1_s, med2_s;

    const int c_s = cs[i], c_f = cf[i];
    if (c_s - R < 0 || c_s + R >= H || c_f - R < 0 || c_f + R >= W) {   // numpy `valid` gate
        if (lane == 0) { Iout[i] = 0.0; Sout[i] = 0.0; Pout[i] = 0.0; Bout[i] = 0.0; }
        return;
    }
    if (lane == 0) { nann_s = 0; med1_s = 0.0; med2_s = 0.0; }
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

    if (lane == 0) {
        const int NBOX = (2 * half + 1) * (2 * half + 1);
        const double bg = 0.5 * (med1_s + med2_s);
        const double bgc = bg > 0.0 ? bg : 0.0;
        double s = bsum + (double)NBOX * bgc; if (s < 1.0) s = 1.0;
        Iout[i] = bsum - (double)NBOX * bg;
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
        k = _CACHE[dt] = cp.RawKernel(f"#define DT_IN {_DT[dt]}\n" + _SRC, "integrate_fused")
    return k


def integrate_fused(data_gpu, pred, half=3, gap=2, ring=3):
    """data_gpu: cupy (H,W) in the detector's native dtype. pred: numpy struct array with fs/ss.
    Returns (I, sigma, peak, bg) as numpy float64, matching integrate_spots."""
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
                           np.int32(half), np.int32(gap), np.int32(ring)),
                          shared_mem=nann * 8)
    return cp.asnumpy(I), cp.asnumpy(S), cp.asnumpy(Pk), cp.asnumpy(B)
