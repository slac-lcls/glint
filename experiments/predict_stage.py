"""Per-stage microbenchmark in the SAME protocol as the check_numbers.py FACTS streaming block:
driver built OUTSIDE the timer, steady state, 5 warmups + 40 reps per stage, min + median reported.

Measures predict (the number under review) plus three CONTROL stages that PR #68 cannot touch --
peakfind, integrate, h2d. If the controls reproduce the FACTS entries (1.16 / 0.31 / 0.19), the
harness is calibrated to the same protocol and the predict number drops straight into the table.
"""
import os, sys, time, statistics as st
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import cupy as cp
from glint_fast import cell_to_Ar
import glint.stream_driver as sd
import glint.fused_integrate as fi
import glint.replica_gpu_batch as rgb

SIM = os.environ.get("GLINT_SIM", "/sdf/home/s/smarches/glint_sim")
imgs = np.load(f"{SIM}/images.npy"); t = np.load(f"{SIM}/truth.npz")
N = int(t["det_n"]); pix_mm = float(t["pix_mm"]); dist_mm = float(t["dist_mm"]); wave = float(t["wave_A"])
cell = t["cell"]
panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
               res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
               coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
Mc = cell_to_Ar(float(cell[0]), float(cell[1]), float(cell[2]), 90, 90, 90); clen = dist_mm / 1000.0
TOL = float(os.environ.get("TOL", "0.002"))          # the DRIVER's tol, not predict()'s signature default
B = int(os.environ.get("B", "40"))
WARM = int(os.environ.get("WARM", "5"))
REPS = int(os.environ.get("REPS", "40"))

print(f"device : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print(f"frame  : {N}x{N} {imgs.dtype}   tol={TOL} B={B} dmin=2.0   warmups={WARM} reps={REPS}")

# ---- build the driver ONCE, outside every timer (this is the point of the protocol) ----
drv = sd.StreamDriver(Mc, panels, clen, wave, (N, N), dtype=imgs.dtype, B=B, dmin=2.0, tol=TOL)
for f in imgs:                       # drive it to the locked, post-lock batched steady state
    drv.push(f)
drv.flush(); cp.cuda.Stream.null.synchronize()

# ---- a real device-resident frame and a real locked orientation, both prepared outside timing ----
g = cp.asarray(imgs[0])
pk = drv.finder.find(g)
fsn = cp.asnumpy(pk["x"]); ssn = cp.asnumpy(pk["y"])
q = sd.peaks_to_q(fsn, ssn, panels, clen, wave); q = q[np.isfinite(q).all(1)]
M = rgb.index_fused([q], Mc, B=1)[0]
Mcan = sd._canonical_axes(np.asarray(M, float))
pred = drv.grid.predict(Mcan, panels, clen, wave, tol=TOL)
print(f"steady state: {len(pk['x'])} peaks, {len(q)} q-vectors, {len(pred)} predicted reflections\n")


def bench(label, fn):
    """5 warmups + 40 reps, wall clock with a full device sync on both sides."""
    for _ in range(WARM):
        fn()
    cp.cuda.Stream.null.synchronize()
    out = []
    for _ in range(REPS):
        cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize(); out.append((time.perf_counter() - t0) * 1e3)
    print(f"  {label:<34} min {min(out):7.4f}  med {st.median(out):7.4f}  mean {st.mean(out):7.4f} ms")
    return min(out), st.median(out)


print("=== stage microbenchmarks (ms/frame) ===")
res = {}
res["predict"] = bench("predict   (grid.predict)  <-- UNDER REVIEW",
                       lambda: drv.grid.predict(Mcan, panels, clen, wave, tol=TOL))
res["peakfind"] = bench("peakfind  (finder.find)   [CONTROL 1.16]",
                        lambda: drv.finder.find(g))
res["integrate"] = bench("integrate (integrate_fused) [CONTROL 0.31]",
                         lambda: fi.integrate_fused(g, pred, half=drv.half, gap=drv.gap, ring=drv.ring_w))
ring0 = drv._ring[0]
host = np.ascontiguousarray(imgs[0])
res["h2d"] = bench("h2d       (ring[...]=asarray) [CTRL 0.19]",
                   lambda: ring0.__setitem__(Ellipsis, cp.asarray(host)))
res["peaks_to_q"] = bench("peaks_to_q (host)         [CONTROL 0.08]",
                          lambda: sd.peaks_to_q(fsn, ssn, panels, clen, wave))

print(f"\nRATIO peakfind/predict = {res['peakfind'][1]/res['predict'][1]:.2f}x (median), "
      f"{res['peakfind'][0]/res['predict'][0]:.2f}x (min)")
print("DONE")
