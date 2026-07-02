"""Cleaned-up radial-integration benchmark (fixes drp-benchmarks/radial_integration/testing2.py):
no external jpg, synthetic detector image, a COMMON binning, and every method compared for correctness
and speed. Isolates the two questions:

  REDUCTION speed (all NEAREST-bin, same result):  their CSR SpMV  vs  cupy bincount  vs  a custom
    privatized-shared-memory RawKernel  (the "kernel beats bincount" hypothesis -- both do the same
    histogram, but the kernel privatizes per-block to dodge global-atomic contention on hot outer bins).
  ACCURACY (pixel-splitting):  our radial.RadialLUT (split bincount)  vs  pyFAI.

Every backend is guarded -> runs whatever is installed (numpy always; scipy/cupy/pyFAI if present).
Run:  python radial_bench.py [N]         (N -> image is 2N x 2N).  On S3DF GPU: srun ... in a cupy env.
Credit: pyFAI (Kieffer et al.; Ashiotis et al., J. Appl. Cryst. 48, 510, 2015)."""
import sys, time
import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024          # image is (2N,2N)


def make_image(N):
    yy, xx = np.mgrid[-N:N, -N:N].astype(np.float32)
    r = np.sqrt(xx * xx + yy * yy)                           # pixel radius
    img = np.zeros_like(r)
    for rr in (0.18, 0.33, 0.51, 0.72) :                     # a few rings (fractions of rmax)
        img += np.exp(-((r / N - rr) ** 2) / (2 * 0.004 ** 2))
    img += 0.05 * np.random.default_rng(0).random(img.shape).astype(np.float32)
    return np.ascontiguousarray(img), r


img, r = make_image(N)
nbin = int(r.max()) + 1
binidx = np.clip(r.ravel().round().astype(np.int32), 0, nbin - 1)   # nearest-bin index (their scheme)
GB = img.nbytes / (1024 ** 3)
results = {}                                                 # name -> (I, ms)


def timeit(fn, warm=3, rep=50):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e3


# ---- their CSR (nearest), scipy CPU + cupy GPU ----------------------------------------------------
def build_csr(sp):
    nnz = r.size
    ru, Krow, rc = np.unique(binidx, return_inverse=True, return_counts=True)
    Kval = (np.float32(1.0 / rc))[Krow]                     # 1/count baked in -> K@img = average (their scheme)
    return sp.coo_matrix((Kval, (Krow, np.arange(nnz))), shape=(int(Krow.max()) + 1, nnz)).tocsr()


try:
    import scipy.sparse as sps
    K = build_csr(sps); flat = img.ravel()
    results["CSR SpMV  scipy(CPU)"] = ((K @ flat), timeit(lambda: K @ flat))
except Exception as e:
    print("skip CSR-scipy:", repr(e)[:80])

try:
    import cupy, cupyx.scipy.sparse as csp
    import scipy.sparse as sps
    gK = csp.csr_matrix(build_csr(sps)); gflat = cupy.asarray(img.ravel())
    cupy.cuda.Stream.null.synchronize()
    results["CSR SpMV  cupy(GPU)"] = (cupy.asnumpy(gK @ gflat),
                                      timeit(lambda: (gK @ gflat, cupy.cuda.Stream.null.synchronize())))
except Exception as e:
    print("skip CSR-cupy:", repr(e)[:80])


# ---- bincount (nearest) : numpy CPU + cupy GPU -- the load-balanced reduction ----------------------
def bincount_nearest(xp, bi, im):
    num = xp.bincount(bi, im, minlength=nbin)[:nbin]
    cnt = xp.bincount(bi, minlength=nbin)[:nbin].astype(num.dtype)
    return num / xp.where(cnt == 0, xp.nan, cnt)


results["bincount  numpy(CPU)"] = (bincount_nearest(np, binidx, img.ravel()),
                                   timeit(lambda: bincount_nearest(np, binidx, img.ravel())))
try:
    import cupy
    gbi = cupy.asarray(binidx); gim = cupy.asarray(img.ravel())
    cupy.cuda.Stream.null.synchronize()
    results["bincount  cupy(GPU)"] = (cupy.asnumpy(bincount_nearest(cupy, gbi, gim)),
                                      timeit(lambda: (bincount_nearest(cupy, gbi, gim),
                                                      cupy.cuda.Stream.null.synchronize())))
except Exception as e:
    print("skip bincount-cupy:", repr(e)[:80])


# ---- custom RawKernel (nearest): per-block privatized histogram in shared memory --------------------
_SRC = r"""
extern "C" __global__ void radhist(const float* img, const int* b, int n, int nbin,
                                    float* num, float* cnt){
    extern __shared__ float sh[];                 // [0:nbin]=num, [nbin:2*nbin]=cnt
    for(int i=threadIdx.x;i<2*nbin;i+=blockDim.x) sh[i]=0.f;
    __syncthreads();
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
        int bb=b[i]; atomicAdd(&sh[bb], img[i]); atomicAdd(&sh[nbin+bb], 1.0f);
    }
    __syncthreads();
    for(int i=threadIdx.x;i<nbin;i+=blockDim.x){
        atomicAdd(&num[i], sh[i]); atomicAdd(&cnt[i], sh[nbin+i]);
    }
}"""
try:
    import cupy
    ker = cupy.RawKernel(_SRC, "radhist")
    gim = cupy.asarray(img.ravel()); gbi = cupy.asarray(binidx)
    thr, blk = 256, 512; shmem = 2 * nbin * 4

    n_i32 = np.int32(gim.size); nb_i32 = np.int32(nbin)

    def run_kernel():
        num = cupy.zeros(nbin, cupy.float32); cnt = cupy.zeros(nbin, cupy.float32)
        ker((blk,), (thr,), (gim, gbi, n_i32, nb_i32, num, cnt), shared_mem=shmem)
        cupy.cuda.Stream.null.synchronize()
        return num, cnt

    num, cnt = run_kernel()
    I = cupy.asnumpy(num / cupy.where(cnt == 0, cupy.nan, cnt))
    results["RawKernel  cupy(GPU)"] = (I, timeit(run_kernel))
except Exception as e:
    print("skip RawKernel:", repr(e)[:120])


# ---- our split-bincount (sub-bin accurate) ---------------------------------------------------------
try:
    sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
    from radial import RadialLUT
    for sp, tag in (("linear", "split-LUT"), ("area", "area-LUT ")):
        lut = RadialLUT(r, nbin=nbin, qmin=0.0, qmax=float(nbin), split=sp)
        results[f"{tag} numpy(CPU)"] = (lut.integrate(img)[1], timeit(lambda l=lut: l.integrate(img)))
        try:
            import cupy
            glut = RadialLUT(cupy.asarray(r), nbin=nbin, qmin=0.0, qmax=float(nbin), split=sp)
            gimg = cupy.asarray(img); cupy.cuda.Stream.null.synchronize()
            results[f"{tag} cupy(GPU)"] = (cupy.asnumpy(glut.integrate(gimg)[1]),
                                           timeit(lambda g=glut: (g.integrate(gimg), cupy.cuda.Stream.null.synchronize())))
        except Exception as e:
            print(f"skip {tag}-cupy:", repr(e)[:80])
except Exception as e:
    print("skip split-LUT:", repr(e)[:80])


# ---- pyFAI ----------------------------------------------------------------------------------------
try:
    import logging; logging.getLogger("pyFAI").setLevel(logging.ERROR)   # silence method-fallback spam
    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except Exception:
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    px = 1e-4
    ai = AzimuthalIntegrator(dist=1.0, pixel1=px, pixel2=px, wavelength=1e-10)
    ai.poni1 = px * N; ai.poni2 = px * N
    for meth in [("full", "csr", "opencl"), ("full", "csr", "cython"), "csr"]:
        try:
            f = lambda: ai.integrate1d(img, nbin, method=meth, unit="r_mm")
            res = f()
            results[f"pyFAI {meth if isinstance(meth,str) else meth[2]}"] = (res[1], timeit(f))
            break
        except Exception:
            continue
except Exception as e:
    print("skip pyFAI:", repr(e)[:100])


# ---- report ---------------------------------------------------------------------------------------
print(f"\nimage {2*N}x{2*N} ({GB*1024:.1f} MB), {nbin} bins, {img.size} pixels\n")
print(f"{'method':22s} {'ms':>8}  {'GB/s':>7}   corr-vs-bincount")
base = results.get("bincount  numpy(CPU)", (None,))[0]
for name, (I, ms) in sorted(results.items(), key=lambda kv: kv[1][1]):
    m = np.isfinite(I) & np.isfinite(base) if base is not None else None
    corr = np.corrcoef(I[m], base[m])[0, 1] if (base is not None and m.sum() > 5) else float("nan")
    print(f"{name:22s} {ms:8.3f}  {GB/(ms/1e3):7.1f}   {corr:.4f}")
