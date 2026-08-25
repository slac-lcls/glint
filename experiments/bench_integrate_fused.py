"""Validate + time the fused GPU box-integration (glint.fused_integrate) against the numpy
integrate_spots, across detector dtypes and edge cases. GPU node + cupy.

  python experiments/bench_integrate_fused.py      # exit 0 = the agreement contract holds

THE CONTRACT, asserted here and not merely printed: on a COUNTING dtype (uint16/int32/float32
holding integers) the kernel is BIT-EXACT with integrate_spots in every bg_mode, saturated pixels
in the annulus included; on non-integer float64 it agrees to <=1e-11 relative, because the warp
reduction sums the box in a different order than numpy's pairwise summation.

A saturated-pixel case is here because it is the one that separates the failure modes: it drives
the background estimators apart (a clipmean rejects the pixel, a mean does not), so a per-mode
kernel bug shows up as a LARGE disagreement rather than a rounding one. Exit-code enforcement is
here because this file printed `bit-exact=False` for two of three modes on an A100 and still
exited 0 -- see the FMA note in fused_integrate.py's kernel.
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
FAILS = []


def mk(H, W, n, dtype, edge=False, noninteger=False, saturate=0):
    a = rng.poisson(50, size=(H, W)).astype(float)
    if noninteger:
        a = a * rng.uniform(0.83, 1.19, size=(H, W)) - rng.uniform(0.3, 0.7, size=(H, W))
    img = a.astype(dtype)
    if saturate:                       # enough that many of the 168-px annuli catch one
        hi = 65535 if np.issubdtype(np.dtype(dtype), np.integer) else 1.0e6
        img[rng.integers(0, H, saturate), rng.integers(0, W, saturate)] = hi
    pred = np.zeros(n, dtype=[("fs", float), ("ss", float)])
    lo, hi_m = (-5, 5) if edge else (20, 20)
    pred["fs"] = rng.uniform(lo, W - hi_m, n); pred["ss"] = rng.uniform(lo, H - hi_m, n)
    return img, pred


props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device(0).id)
print(f"=== fused GPU integration vs numpy integrate_spots ({props['name'].decode()}, "
      f"cupy {cp.__version__}) ===")
cases = [("uint16 (raw)", 800, 800, 300, np.uint16, False, False, 0),
         ("int32", 800, 800, 300, np.int32, False, False, 0),
         ("float32", 800, 800, 300, np.float32, False, False, 0),
         ("uint16 + saturated px", 800, 800, 300, np.uint16, False, False, 6000),
         ("float32 + saturated px", 800, 800, 300, np.float32, False, False, 6000),
         ("float32 non-integer", 1500, 1500, 500, np.float32, False, True, 0),
         ("float64 non-integer", 1500, 1500, 500, np.float64, False, True, 0),
         ("edge-straddling", 400, 400, 400, np.float32, True, False, 0),
         ("dense 2000 spots", 4000, 4000, 2000, np.uint16, False, False, 0)]
for name, H, W, n, dt, edge, ni, sat in cases:
    img, pred = mk(H, W, n, dt, edge, ni, sat)
    g = cp.asarray(img)
    # EVERY bg_mode, not just the default: glint#131 changed the default estimator and put a second
    # rank-count pass in the kernel, so a mode-specific divergence is exactly what can hide here.
    for mode in BG_MODES:
        a = integrate_spots(img, pred, bg_mode=mode)
        b = integrate_fused(g, pred, bg_mode=mode)
        exact = all(np.array_equal(x, y) for x, y in zip(a, b))
        rel = (np.abs(a[0] - b[0]) / np.maximum(np.abs(a[0]), 1e-12)).max()
        counting = not ni                       # integer-valued input, whatever the container type
        ok = exact if counting else rel <= 1e-11
        print(f"  {name:22s} n={n:5d} {mode:9s} bit-exact={str(exact):5s}  max rel dI={rel:.2g}  "
              f"bg-eq={np.array_equal(a[3], b[3])}  peak-eq={np.array_equal(a[2], b[2])}"
              f"{'' if ok else '   <- CONTRACT VIOLATED'}")
        if not ok:
            FAILS.append(f"{name}/{mode}: "
                         + (f"not bit-exact on counting data (max rel dI={rel:.2g})" if counting
                            else f"max rel dI={rel:.2g} > 1e-11"))
    del g
    cp.get_default_memory_pool().free_all_blocks()

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
    # what the clipmean default costs over the median it replaced: one more O(NA^2)/32 pass
    per = {m: tmin(lambda m=m: integrate_fused(g, pred, bg_mode=m)) for m in BG_MODES}
    print("      per bg_mode: " + "  ".join(f"{m} {per[m]:6.3f}" for m in BG_MODES)
          + f"   clipmean/median = {per['clipmean']/per['median']:.2f}x")
    del g
    cp.get_default_memory_pool().free_all_blocks()

print(f"\nFAILURES: {len(FAILS)}")
for f in FAILS:
    print("  !!", f)
sys.exit(1 if FAILS else 0)
