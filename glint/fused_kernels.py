"""Fused cupy RawKernels (nvrtc-JIT) for the batched known-cell engine's per-candidate hot loops:
  anneal_fused  -- the 20/10-iter anneal (bit-exact fp64, ports anneal_b+solve3x3)
  obj_fused     -- obj_b scoring (inlier count + log2 sub-score) for K axis-vector candidates
  refine_fused  -- refine_b (S-step sin-gradient) for K axis-vector candidates
All share the pattern: one thread-BLOCK per frame, one thread per candidate (k-strided), the frame's
peaks staged in shared memory once, per-candidate state in registers.  Launch on torch's CURRENT
stream (cupy ExternalStream) -> correct ordering, no host sync.  Precision follows KC_FP (rgb.FP)."""
import numpy as np, torch, cupy as cp
import glint.replica_gpu_batch as rgb
from glint.replica_gpu import TRIML, TRIMH, DELTA

IS32 = rgb.FP == torch.float32
PI = float(np.pi); TWO_PI = 2.0 * PI

# ---- shared type defs ----
_DEFS = ("#define DT float\n#define RINT rintf\n#define FMAX fmaxf\n#define FMIN fminf\n"
         "#define FABS fabsf\n#define SIN sinf\n#define LOG2 log2f\n" if IS32 else
         "#define DT double\n#define RINT rint\n#define FMAX fmax\n#define FMIN fmin\n"
         "#define FABS fabs\n#define SIN sin\n#define LOG2 log2\n")
_CONST = f"#define TRIML (DT){TRIML}\n#define TRIMH (DT){TRIMH}\n#define DELTA (DT){DELTA}\n" \
         f"#define TWO_PI (DT){TWO_PI}\n"

# ---------------------------------------------------------------------------------- anneal --------
_ANNEAL = r"""
extern "C" __global__ void anneal_fused(
    const DT* __restrict__ M0, const DT* __restrict__ Q, const int* __restrict__ npk,
    DT* __restrict__ Mout, const int F, const int K, const int Pmax, const int max_iter,
    const DT thr0, const DT contract, const DT min_thr)
{
    const int f = blockIdx.x; if (f >= F) return; const int P = npk[f];
    extern __shared__ unsigned char sh_raw[]; DT* Qs = reinterpret_cast<DT*>(sh_raw);
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const DT* qp = Q + ((long)f*Pmax + p)*3; Qs[3*p]=qp[0]; Qs[3*p+1]=qp[1]; Qs[3*p+2]=qp[2]; }
    __syncthreads();
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const DT* mp = M0 + ((long)f*K + k)*9;
        DT m0=mp[0],m1=mp[1],m2=mp[2],m3=mp[3],m4=mp[4],m5=mp[5],m6=mp[6],m7=mp[7],m8=mp[8];
        DT thr = thr0;
        for (int it = 0; it < max_iter; ++it) {
            DT A0=0,A1=0,A2=0,A4=0,A5=0,A8=0, r0=0,r1=0,r2=0,r3=0,r4=0,r5=0,r6=0,r7=0,r8=0; int cnt=0;
            for (int p = 0; p < P; ++p) {
                DT qx=Qs[3*p], qy=Qs[3*p+1], qz=Qs[3*p+2];
                DT Hx=qx*m0+qy*m3+qz*m6, Hy=qx*m1+qy*m4+qz*m7, Hz=qx*m2+qy*m5+qz*m8;
                DT hx=RINT(Hx), hy=RINT(Hy), hz=RINT(Hz);
                if (FMAX(FABS(Hx-hx),FMAX(FABS(Hy-hy),FABS(Hz-hz))) < thr) {
                    ++cnt; A0+=qx*qx;A1+=qx*qy;A2+=qx*qz;A4+=qy*qy;A5+=qy*qz;A8+=qz*qz;
                    r0+=qx*hx;r1+=qx*hy;r2+=qx*hz;r3+=qy*hx;r4+=qy*hy;r5+=qy*hz;r6+=qz*hx;r7+=qz*hy;r8+=qz*hz; } }
            if (cnt >= 6) {
                DT a00=A0+(DT)1e-9,a11=A4+(DT)1e-9,a22=A8+(DT)1e-9, a01=A1,a02=A2,a12=A5,a10=A1,a20=A2,a21=A5;
                DT c00=a11*a22-a12*a21,c01=a12*a20-a10*a22,c02=a10*a21-a11*a20;
                DT c10=a02*a21-a01*a22,c11=a00*a22-a02*a20,c12=a01*a20-a00*a21;
                DT c20=a01*a12-a02*a11,c21=a02*a10-a00*a12,c22=a00*a11-a01*a10;
                DT det=a00*c00+a01*c01+a02*c02;
                DT i00=c00/det,i01=c10/det,i02=c20/det,i10=c01/det,i11=c11/det,i12=c21/det,i20=c02/det,i21=c12/det,i22=c22/det;
                m0=i00*r0+i01*r3+i02*r6; m1=i00*r1+i01*r4+i02*r7; m2=i00*r2+i01*r5+i02*r8;
                m3=i10*r0+i11*r3+i12*r6; m4=i10*r1+i11*r4+i12*r7; m5=i10*r2+i11*r5+i12*r8;
                m6=i20*r0+i21*r3+i22*r6; m7=i20*r1+i21*r4+i22*r7; m8=i20*r2+i21*r5+i22*r8; }
            thr = FMAX(thr*contract, min_thr); }
        DT* op = Mout + ((long)f*K + k)*9;
        op[0]=m0;op[1]=m1;op[2]=m2;op[3]=m3;op[4]=m4;op[5]=m5;op[6]=m6;op[7]=m7;op[8]=m8; }
}
"""

# ---------------------------------------------------------------------------------- obj -----------
# proj = V.Q_p ; d=|proj-round|; inl += (d<TRIMH); sub += log2(clamp(d,TRIML,TRIMH)+DELTA); sub/=npk
_OBJ = r"""
extern "C" __global__ void obj_fused(
    const DT* __restrict__ V, const DT* __restrict__ Q, const int* __restrict__ npk,
    int* __restrict__ inl_out, DT* __restrict__ sub_out, const int F, const int K, const int Pmax)
{
    const int f = blockIdx.x; if (f >= F) return; const int P = npk[f];
    extern __shared__ unsigned char sh_raw[]; DT* Qs = reinterpret_cast<DT*>(sh_raw);
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const DT* qp = Q + ((long)f*Pmax + p)*3; Qs[3*p]=qp[0]; Qs[3*p+1]=qp[1]; Qs[3*p+2]=qp[2]; }
    __syncthreads();
    DT invn = (DT)1.0 / (DT)(P < 1 ? 1 : P);
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const DT* vp = V + ((long)f*K + k)*3; DT v0=vp[0],v1=vp[1],v2=vp[2];
        int inl = 0; DT sub = 0;
        for (int p = 0; p < P; ++p) {
            DT proj = v0*Qs[3*p] + v1*Qs[3*p+1] + v2*Qs[3*p+2];
            DT d = FABS(proj - RINT(proj));
            if (d < TRIMH) ++inl;
            sub += LOG2(FMIN(FMAX(d, TRIML), TRIMH) + DELTA); }
        inl_out[(long)f*K + k] = inl; sub_out[(long)f*K + k] = sub * invn; }
}
"""

# ---------------------------------------------------------------------------------- refine --------
# S steps of: g = sum_p sin(2pi V.Q_p) Q_p ; V -= lr[f]*2pi*g
_REFINE = r"""
extern "C" __global__ void refine_fused(
    DT* __restrict__ V, const DT* __restrict__ Q, const int* __restrict__ npk,
    const DT* __restrict__ lr, const int F, const int K, const int Pmax, const int steps)
{
    const int f = blockIdx.x; if (f >= F) return; const int P = npk[f];
    extern __shared__ unsigned char sh_raw[]; DT* Qs = reinterpret_cast<DT*>(sh_raw);
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const DT* qp = Q + ((long)f*Pmax + p)*3; Qs[3*p]=qp[0]; Qs[3*p+1]=qp[1]; Qs[3*p+2]=qp[2]; }
    __syncthreads();
    DT lrf = lr[f] * TWO_PI;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        DT* vp = V + ((long)f*K + k)*3; DT v0=vp[0],v1=vp[1],v2=vp[2];
        for (int s = 0; s < steps; ++s) {
            DT g0=0,g1=0,g2=0;
            for (int p = 0; p < P; ++p) {
                DT qx=Qs[3*p], qy=Qs[3*p+1], qz=Qs[3*p+2];
                DT sn = SIN(TWO_PI*(v0*qx+v1*qy+v2*qz));
                g0+=sn*qx; g1+=sn*qy; g2+=sn*qz; }
            v0-=lrf*g0; v1-=lrf*g1; v2-=lrf*g2; }
        vp[0]=v0; vp[1]=v1; vp[2]=v2; }
}
"""

_KA = cp.RawKernel(_DEFS + _CONST + _ANNEAL, "anneal_fused")
_KO = cp.RawKernel(_DEFS + _CONST + _OBJ, "obj_fused")
_KR = cp.RawKernel(_DEFS + _CONST + _REFINE, "refine_fused")
_SC = (lambda x: np.float32(x)) if IS32 else (lambda x: np.float64(x))
_IB = 4 if IS32 else 8
_cp = cp.asarray

# The stock torch ops, captured at import BEFORE patch() can swap them out: they are the fallback
# for frames whose peaks do not fit in shared memory (see _smem_cap below).  Bound here rather than
# read off rgb at call time so a fallback inside a patched region cannot recurse into itself.
_ORIG = {"anneal_b": rgb.anneal_b, "obj_b": rgb.obj_b, "refine_b": rgb.refine_b}

# ------------------------------------------------------------- dynamic shared memory (issue #69) --
# All three kernels stage the frame's peaks in dynamic shared memory as Pmax*3*_IB bytes.  CUDA caps
# a launch's dynamic shared memory at 48 KB per block unless the FUNCTION opts in via
# cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, N) -- so without the opt-in
# Pmax > 48*1024/(3*_IB) = 2048 (fp64) / 4096 (fp32) failed the launch with CUDA_ERROR_INVALID_VALUE,
# which propagated out of index_fused and killed StreamDriver.flush().  Opt in to whatever the device
# itself reports as its per-block maximum (A100: 166,912 B, 3.4x the default), and treat that number
# as a real ceiling: above it the peaks genuinely do not fit, so fall back to the stock torch ops
# rather than raise.  (Lifting the ceiling entirely would mean tiling the peak loop, which is a
# different change: even 163 KB runs out near 7k/14k peaks.)
_DEFAULT_SMEM = 48 * 1024
_SMEM_CAP = None


def _smem_cap():
    """Bytes of dynamic shared memory the three kernels may request on this device, opting them in
    once on first use.  Returns the 48 KB default if there is no device or the driver refuses."""
    global _SMEM_CAP
    if _SMEM_CAP is None:
        cap = _DEFAULT_SMEM
        try:
            want = int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlockOptin", 0) or 0)
            if want > cap:
                for k in (_KA, _KO, _KR):
                    k.max_dynamic_shared_size_bytes = want   # -> cudaFuncSetAttribute, per function
                cap = want
        except Exception:
            pass                                             # no GPU / opt-in unsupported: 48 KB
        _SMEM_CAP = cap
    return _SMEM_CAP


def max_peaks():
    """Largest Pmax the fused kernels can stage on this device (3 coords x _IB bytes per peak)."""
    return _smem_cap() // (3 * _IB)


def _stream():
    return cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)


def anneal_fused(M0, Q, m, thr0=0.25, contract=0.85, max_iter=15, min_thr=0.02, block=128):
    F, K = M0.shape[0], M0.shape[1]; Pmax = Q.shape[1]
    if Pmax > max_peaks():                                   # will not fit in shared memory
        return _ORIG["anneal_b"](M0, Q, m, thr0, contract, max_iter, min_thr)
    npk = m.sum(1).to(torch.int32).contiguous()
    M0f = M0.reshape(F, K, 9).contiguous(); Mout = torch.empty_like(M0f)
    with _stream():
        _KA((F,), (block,), (_cp(M0f), _cp(Q.contiguous()), _cp(npk), _cp(Mout),
             np.int32(F), np.int32(K), np.int32(Pmax), np.int32(max_iter),
             _SC(thr0), _SC(contract), _SC(min_thr)), shared_mem=Pmax*3*_IB)
    return Mout.reshape(F, K, 3, 3)


def obj_fused(V, Q, m, block=128):
    F, K = V.shape[0], V.shape[1]; Pmax = Q.shape[1]
    if Pmax > max_peaks():
        return _ORIG["obj_b"](V, Q, m)
    npk = m.sum(1).to(torch.int32).contiguous()
    Vc = V.contiguous()
    inl = torch.empty(F, K, dtype=torch.int32, device=V.device)
    sub = torch.empty(F, K, dtype=V.dtype, device=V.device)
    with _stream():
        _KO((F,), (block,), (_cp(Vc), _cp(Q.contiguous()), _cp(npk), _cp(inl), _cp(sub),
             np.int32(F), np.int32(K), np.int32(Pmax)), shared_mem=Pmax*3*_IB)
    return inl.long(), sub


def refine_fused(V, Q, m, steps=30, block=128):
    F, K = V.shape[0], V.shape[1]; Pmax = Q.shape[1]
    if Pmax > max_peaks():
        return _ORIG["refine_b"](V, Q, m, steps)
    npk = m.sum(1).to(torch.int32).contiguous()
    qmax = (Q.norm(dim=2) * m).amax(1).clamp(min=1e-9); npkf = m.sum(1).clamp(min=1)
    lr = (1.0 / (4 * PI**2 * npkf * qmax**2)).to(V.dtype).contiguous()
    Vout = V.contiguous().clone()
    with _stream():
        _KR((F,), (block,), (_cp(Vout), _cp(Q.contiguous()), _cp(npk), _cp(lr),
             np.int32(F), np.int32(K), np.int32(Pmax), np.int32(steps)), shared_mem=Pmax*3*_IB)
    return Vout


# ---------------------------------------------------------------------------- GPU cpu_stage -------
# The host tail (_cpu_stage) is 99% the per-frame numpy buerger_reduce inside same_lattice (~0.46 ms/
# fr, precision-independent).  buerger_reduce is fully batchable: M@_BN.T over 342 integer combos +
# two "first vector satisfying a condition" picks (argmax-first-true).  Do it ON GPU over the batch
# (pol/best already on device) and D2H only the final F*3*3 -> the CPU tail ~vanishes.  Bit-faithful
# to lattice.buerger_reduce / reduced_params / multishot.same_lattice.
_BR = torch.arange(-3, 4, device=rgb.DEV)
_BN = torch.stack(torch.meshgrid(_BR, _BR, _BR, indexing="ij"), -1).reshape(-1, 3)
_BN = _BN[(_BN != 0).any(1)].to(rgb.FP)                         # (342,3)


def _buerger_batch(M):
    """Batched buerger_reduce: M (F,3,3) -> reduced (sorted lens (F,3), sorted |cos| (F,3), ok (F,))."""
    F = M.shape[0]; ar = torch.arange(F, device=M.device)
    V = torch.einsum('fcd,nd->fnc', M, _BN.to(M.dtype))        # integer combos of the columns
    L = V.norm(dim=2)
    Ls, o = L.sort(dim=1); Vs = torch.gather(V, 1, o[..., None].expand(-1, -1, 3))
    v1 = Vs[:, 0]; L0 = Ls[:, :1]; rest = Vs[:, 1:]; Lr = Ls[:, 1:]
    crn = torch.cross(rest, v1[:, None].expand_as(rest), dim=2).norm(dim=2)
    cond2 = crn > 1e-3 * Lr * L0
    idx2 = cond2.float().argmax(1); v2 = rest[ar, idx2]
    nrm = torch.cross(v1, v2, dim=1); nn = nrm.norm(dim=1, keepdim=True)
    cond3 = (rest * nrm[:, None]).sum(2).abs() > 1e-3 * Lr * nn
    idx3 = cond3.float().argmax(1); v3 = rest[ar, idx3]
    ok = cond2.any(1) & cond3.any(1)
    lens = torch.stack([v1.norm(dim=1), v2.norm(dim=1), v3.norm(dim=1)], 1)
    l = lens
    cs = torch.stack([(v1 * v2).sum(1).abs() / (l[:, 0] * l[:, 1]),
                      (v1 * v3).sum(1).abs() / (l[:, 0] * l[:, 2]),
                      (v2 * v3).sum(1).abs() / (l[:, 1] * l[:, 2])], 1)
    return lens.sort(1).values, cs.sort(1).values, ok


def _cpu_stage_gpu(best, pol, mp, mainb, Mc, rtol=0.05, ctol=0.06, vtol=0.10):
    """GPU replacement for rgb._cpu_stage: choose polished vs pre-polish per frame with a batched,
    on-device same_lattice, D2H once.  Returns list of (3,3) np arrays."""
    Mct = torch.as_tensor(np.asarray(Mc, float), dtype=pol.dtype, device=pol.device)
    l2, c2, _ = _buerger_batch(Mct[None]); l2 = l2[0]; c2 = c2[0]
    dett = rgb.det3(Mct[None]).abs()
    l1, c1, ok = _buerger_batch(pol)
    detp = rgb.det3(pol).abs()
    vol_ok = (detp - dett).abs() <= vtol * dett
    len_ok = ((l1 - l2).abs() <= rtol * l2).all(1)
    ang_ok = ((c1 - c2).abs() <= ctol).all(1)
    lattice_ok = ok & vol_ok & len_ok & ang_ok
    use_pol = (mp >= mainb) & lattice_ok
    out = torch.where(use_pol[:, None, None], pol, best)
    return [m for m in out.cpu().numpy()]


_STOCK = {}


def patch(anneal=True, obj=True, refine=True, cpu=False):
    if anneal: _STOCK["anneal_b"] = rgb.anneal_b; rgb.anneal_b = anneal_fused
    if obj:    _STOCK["obj_b"] = rgb.obj_b; rgb.obj_b = obj_fused
    if refine: _STOCK["refine_b"] = rgb.refine_b; rgb.refine_b = refine_fused
    if cpu:    _STOCK["_cpu_stage"] = rgb._cpu_stage; rgb._cpu_stage = _cpu_stage_gpu


def unpatch():
    for k, v in _STOCK.items():
        setattr(rgb, k, v)
    _STOCK.clear()


def run_fused(frames, Mc, B=32):
    """Fully-fused known-cell indexing: swap in the fused anneal/obj/refine kernels + the on-device
    cpu_stage for the duration of a batched pass, then restore.  Output is IDENTICAL (bit-exact fp64)
    to looping index_known_gpu_cell_batch with the stock ops."""
    patch(anneal=True, obj=True, refine=True, cpu=True)
    try:
        return [M for i in range(0, len(frames), B)
                for M in rgb.index_known_gpu_cell_batch(frames[i:i + B], Mc)]
    finally:
        unpatch()
