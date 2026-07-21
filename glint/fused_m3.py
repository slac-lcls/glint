"""Fused cupy RawKernel for M3, the blind front end's dominant stage (#28).

WHY THIS AND NOT BATCHING. Profiling (`experiments/profile_blind_frontend.py`) showed M3 is 66% of
the blind frame and COMPUTE-bound, not host-dispatch-bound: it scales linearly with peak count above
a ~2 ms launch floor. So the rescue's batch -> CUDA-graph -> fuse playbook does not transfer; what
dominates is memory traffic through the (S, P) intermediates, S = 70,400 starts.

The stock path, per `refine_vec` step, materialises roughly ten S x P arrays -- proj, round, the
difference, abs, the mask, cos, sin, the weighted mask and two products -- each ~28 MB at P = 100,
and does it 8 times. This kernel keeps T and vel in REGISTERS across all steps and never
materialises any of them: one thread per start vector, the frame's peaks staged once in shared
memory, a 4-register accumulator (f is not even computed -- `refine_vec` discards it inside the
loop and uses only the gradient).

Two things it deliberately reproduces exactly, because they are easy to get subtly wrong:
  * sharp: refine_vec's default sharp_last=25 with STEPS=8 means `s >= steps - sharp_last` is
    ALWAYS true, so every step uses the sharpened form. The kernel hardcodes nothing -- it takes
    sharp_from and compares, so a caller changing STEPS/sharp_last still matches.
  * the gradient is the SHARP derivative dc = -PI*sin(2*PI*proj) with the cos^2 form, which is not
    the derivative of the non-sharp c. That asymmetry is in the original and is preserved.

NOT bit-exact vs torch: the per-start reduction over peaks runs sequentially here while torch does
it as a matmul, so summation order differs. Validate on RATE and CELL STABILITY, not equality --
see `experiments/bench_fused_m3.py`.
"""
import numpy as np, torch

try:
    import cupy as cp
    _HAVE_CP = True
except Exception:                                   # CPU box / no cupy -> caller falls back
    _HAVE_CP = False

PI = float(np.pi)

# nvrtc has no math.h, so M_PI is undefined -- spell the constants out, as fused_kernels.py does.
_SRC = ("#define PI_F %.9ff\n#define TWO_PI_F %.9ff\n" % (PI, 2.0 * PI)) + r"""
extern "C" __global__ void refine_m3_fused(
    const float* __restrict__ T0,      // (S,3) start vectors
    const float* __restrict__ Q,       // (P,3) rlps
    const float* __restrict__ W,       // (P,)  per-peak weight
    float* __restrict__ Tout,          // (S,3)
    const int S, const int P, const int steps, const int sharp_from,
    const float step0, const float tol, const float mom)
{
    extern __shared__ float sh[];                       // Q (3P) then W (P)
    float* Qs = sh; float* Ws = sh + 3*P;
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        Qs[3*p] = Q[3*p]; Qs[3*p+1] = Q[3*p+1]; Qs[3*p+2] = Q[3*p+2]; Ws[p] = W[p];
    }
    __syncthreads();

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    float tx = T0[3*s], ty = T0[3*s+1], tz = T0[3*s+2];
    float vx = 0.f, vy = 0.f, vz = 0.f;                 // momentum, in registers across steps

    for (int it = 0; it < steps; ++it) {
        const float lr = step0 * (1.f - 0.7f * (float)it / (float)steps);
        const bool sharp = (it >= sharp_from);
        float gx = 0.f, gy = 0.f, gz = 0.f;
        for (int p = 0; p < P; ++p) {
            const float qx = Qs[3*p], qy = Qs[3*p+1], qz = Qs[3*p+2];
            const float proj = tx*qx + ty*qy + tz*qz;
            const float d = proj - rintf(proj);
            if (fabsf(d) < tol) {                       // the hard inlier mask
                // sharp: c = cos(PI*proj)^2, dc = -PI*sin(2*PI*proj)
                // else : c = cos(2*PI*proj), dc = -2*PI*sin(2*PI*proj)
                // sinf, NOT __sinf: the fast intrinsic loses accuracy for large arguments, and
                // 2*PI*proj reaches ~377 here (|q||v| ~ 60), which is exactly where it degrades.
                const float sp = sinf(TWO_PI_F*proj);
                const float dc = sharp ? (-PI_F * sp) : (-TWO_PI_F * sp);
                const float a = dc * Ws[p];
                gx += a*qx; gy += a*qy; gz += a*qz;
            }
        }
        const float gn = sqrtf(gx*gx + gy*gy + gz*gz) + 1e-12f;
        vx = mom*vx + gx/gn; vy = mom*vy + gy/gn; vz = mom*vz + gz/gn;
        tx += lr*vx; ty += lr*vy; tz += lr*vz;
    }
    Tout[3*s] = tx; Tout[3*s+1] = ty; Tout[3*s+2] = tz;
}
"""

_K = None


def _kernel():
    global _K
    if _K is None:
        _K = cp.RawKernel(_SRC, "refine_m3_fused")
    return _K


# Shared memory is 4*(3P + P) bytes; keep well inside the 48 KB default limit.
MAX_P = int(48 * 1024 / (4 * 4)) - 64


def available(Q):
    return _HAVE_CP and Q.is_cuda and Q.dtype == torch.float32 and int(Q.shape[0]) <= MAX_P


def refine_vec_fused(T, Q, w, qmax, steps=8, tol=0.18, sharp_last=25, mom=0.5, block=256):
    """Drop-in for glint_index.refine_vec, fused into one kernel. Same arguments, same semantics.

    Falls back to nothing -- callers must check `available(Q)` first; this raises otherwise, rather
    than silently running a different algorithm than the one being benchmarked."""
    if not available(Q):
        raise RuntimeError("fused M3 unavailable for this input (needs cupy, cuda, fp32, P<=%d)" % MAX_P)
    S = int(T.shape[0]); P = int(Q.shape[0])
    T0 = T.contiguous(); Qc = Q.contiguous(); wc = w.contiguous().to(torch.float32)
    out = torch.empty_like(T0)
    grid = ((S + block - 1) // block,)
    with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        _kernel()(grid, (block,),
                  (cp.asarray(T0), cp.asarray(Qc), cp.asarray(wc), cp.asarray(out),
                   np.int32(S), np.int32(P), np.int32(steps), np.int32(steps - sharp_last),
                   np.float32(0.25 / qmax), np.float32(tol), np.float32(mom)),
                  shared_mem=4 * (3 * P + P))
    return out
