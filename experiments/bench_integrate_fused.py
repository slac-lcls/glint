"""Validate + time the fused GPU box-integration (glint.fused_integrate) against the numpy
integrate_spots, across detector dtypes and edge cases. GPU node + cupy.

  python experiments/bench_integrate_fused.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cupy as cp
from glint.predict import BG_MODES, integrate_spots
from glint.fused_integrate import integrate_fused

rng = np.random.default_rng(0)


def mk(H, W, n, dtype, edge=False, noninteger=False):
    a = rng.poisson(50, size=(H, W)).astype(float)
    if noninteger:
        a = a * rng.uniform(0.83, 1.19, size=(H, W)) - rng.uniform(0.3, 0.7, size=(H, W))
    img = a.astype(dtype)
    pred = np.zeros(n, dtype=[("fs", float), ("ss", float)])
    lo, hi = (-5, 5) if edge else (20, 20)
    pred["fs"] = rng.uniform(lo, W - hi, n); pred["ss"] = rng.uniform(lo, H - hi, n)
    return img, pred


print("=== fused GPU integration vs numpy integrate_spots ===")
cases = [("uint16 (raw)", 800, 800, 300, np.uint16, False, False),
         ("int32", 800, 800, 300, np.int32, False, False),
         ("float32", 800, 800, 300, np.float32, False, False),
         ("float32 non-integer", 1500, 1500, 500, np.float32, False, True),
         ("float64 non-integer", 1500, 1500, 500, np.float64, False, True),
         ("edge-straddling", 400, 400, 400, np.float32, True, False),
         ("dense 2000 spots", 4000, 4000, 2000, np.uint16, False, False)]
for name, H, W, n, dt, edge, ni in cases:
    img, pred = mk(H, W, n, dt, edge, ni)
    # EVERY bg_mode, not just the default: glint#131 changed the default estimator and put a second
    # rank-count pass in the kernel, so a mode-specific divergence is exactly what can hide here.
    for mode in BG_MODES:
        a = integrate_spots(img, pred, bg_mode=mode)
        b = integrate_fused(cp.asarray(img), pred, bg_mode=mode)
        exact = all(np.array_equal(x, y) for x, y in zip(a, b))
        rel = (np.abs(a[0] - b[0]) / np.maximum(np.abs(a[0]), 1e-12)).max()
        print(f"  {name:22s} n={n:5d} {mode:9s} bit-exact={str(exact):5s}  max rel dI={rel:.2g}  "
              f"bg-eq={np.array_equal(a[3], b[3])}  peak-eq={np.array_equal(a[2], b[2])}")

print("\n=== timing (min-of-10) ===")


def tmin(fn, reps=10):
    fn(); cp.cuda.Stream.null.synchronize(); best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t0)
    return 1e3 * best


for H, W, n in ((2000, 2000, 400), (4000, 4000, 800), (4000, 4000, 2000)):
    img, pred = mk(H, W, n, np.uint16)
    g = cp.asarray(img)
    tc = tmin(lambda: integrate_spots(img, pred))
    tg = tmin(lambda: integrate_fused(g, pred))
    th = tmin(lambda: integrate_fused(cp.asarray(img), pred))
    print(f"  {H}x{W} ({H*W/1e6:2.0f} Mpix), {n:5d} spots:  numpy {tc:7.2f}   fused {tg:6.3f}"
          f"   ({tc/tg:5.1f}x)   [incl. H2D {th:6.3f} -- copy-bound]")
