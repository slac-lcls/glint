"""Throughput of MANY radial averages: single-frame loop (per-call cupy/cuSPARSE overhead) vs batched SpMM
(M @ stacked-frames, amortises overhead) vs a CUDA graph (replay, kills launch overhead) -- and the H2D
transfer cost, which is the real ceiling. Run in a cupy env on a GPU node. `python radial_tp.py [N] [B]`."""
import sys, time
import numpy as np
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
from radial import RadialLUT
import cupy

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024              # image (2N,2N)
B = int(sys.argv[2]) if len(sys.argv) > 2 else 100              # number of frames
yy, xx = cupy.mgrid[-N:N, -N:N].astype(cupy.float32)
r = cupy.sqrt(xx * xx + yy * yy)
nbin = int(r.max()) + 1
lut = RadialLUT(r, nbin=nbin, qmin=0.0, qmax=float(nbin))
frames = cupy.asarray(np.random.default_rng(0).random((B, 2 * N, 2 * N)).astype(np.float32))   # on GPU
MB = frames[0].nbytes / 1024 ** 2
sync = cupy.cuda.Stream.null.synchronize


def per_frame(fn, warm=2):
    for _ in range(warm):
        fn(); sync()
    sync(); t0 = time.perf_counter(); fn(); sync()
    return (time.perf_counter() - t0) * 1e3 / B


# 1. single-frame loop (each M@x pays the per-call wrapper/descriptor overhead)
t_loop = per_frame(lambda: [lut.integrate(frames[i]) for i in range(B)])

# 2. batched SpMM: one M @ X over all B frames
t_batch = per_frame(lambda: lut.integrate_batch(frames))

# 3. CUDA graph: capture one M@x, replay B times (isolates op cost, no input change)
t_graph = None
try:
    xbuf = frames[0].reshape(-1).astype(lut.M.dtype)
    _ = lut.M @ xbuf; sync()
    s = cupy.cuda.Stream(non_blocking=True)
    s.begin_capture()
    out = lut.M @ xbuf
    g = s.end_capture(); s.synchronize()
    t0 = time.perf_counter()
    for _ in range(B):
        g.launch()
    s.synchronize(); t_graph = (time.perf_counter() - t0) * 1e3 / B
except Exception as e:
    print("graph capture failed:", repr(e)[:120])

# 4. H2D transfer of one frame (the real ceiling if frames start on the host)
hf = np.ascontiguousarray(cupy.asnumpy(frames[0]))
sync(); t0 = time.perf_counter()
for _ in range(50):
    d = cupy.asarray(hf)
sync(); t_h2d = (time.perf_counter() - t0) * 1e3 / 50

print(f"\n{B} frames of {2*N}x{2*N} ({MB:.0f} MB each), {nbin} bins, M.dtype={lut.M.dtype}\n")
print(f"{'per-frame':32s}{'ms':>8}{'GB/s':>9}")
def row(name, ms):
    print(f"{name:32s}{ms:8.4f}{MB/1024/(ms/1e3):9.1f}")
row("single-frame loop  (M@x)", t_loop)
row("batched SpMM       (M@X)", t_batch)
if t_graph is not None:
    row("CUDA graph replay  (M@x)", t_graph)
row("H2D transfer only  (1 frame)", t_h2d)
print(f"\nspeedup batch vs loop: {t_loop/t_batch:.1f}x   (compute-only; add {t_h2d:.3f} ms/frame if data starts on host)")
