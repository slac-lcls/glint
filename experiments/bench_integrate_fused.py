"""Validate + time the fused GPU box-integration (glint.fused_integrate) against the numpy
integrate_spots, across detector dtypes and edge cases. GPU node + cupy.

  python experiments/bench_integrate_fused.py      # exit 0 = the agreement contract holds
  python experiments/bench_integrate_fused.py --self-test   # checker only, no GPU needed

WITHOUT CUPY it runs --self-test and exits 0 on the GPU half, the way test_gate_project.py skips:
that keeps `compare_outputs` -- which is where the pass/fail rules actually live -- covered by the
CPU CI job, instead of only ever executing on a GPU node.

THE CONTRACT, asserted here and not merely printed, over ALL FOUR outputs (I, sigma, peak, bg):
on a COUNTING dtype (uint16/int32/float32 holding integers) the kernel is BIT-EXACT with
integrate_spots in every bg_mode, saturated pixels in the annulus included; on non-integer float64
I/sigma/bg agree to ATOL_F64 counts ABSOLUTE, because the warp reduction sums in a different order
than numpy's pairwise summation. Two outputs are held EXACT unconditionally, whatever the dtype,
because no summation reaches them: `peak` is a max, and `bg` in median mode is a selection.

The float64 bound is absolute, not relative, because I is a photon count and the thing it has to
be small against is one photon -- a relative bound is meaningless on the reflections that matter
here, the ones whose I is near zero, and on those it reads ~1e-11 while the absolute difference is
~5e-13 counts. ATOL_F64 is set at a nanocount: nine orders below the Poisson floor, and the
measured difference sits three orders below IT. It is a ceiling on "physically nothing", not a
line fitted to what the device happened to produce.

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
try:                                   # so the checker below is importable/testable without a GPU
    import cupy as cp
except ImportError:
    cp = None
from glint.predict import BG_MODES, integrate_spots

rng = np.random.default_rng(0)
ATOL_F64 = 1e-9          # counts; see the docstring -- a ceiling on "negligible", not a fitted line
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


def compare_outputs(a, b, counting, mode, atol=ATOL_F64):
    """(I, sigma, peak, bg) from integrate_spots vs integrate_fused -> list of contract violations.

    Pure numpy and free of cupy on purpose: this is where the pass/fail rules live, so it has to be
    runnable -- and sabotage-testable -- on the CPU CI job rather than only on a GPU node."""
    why = []
    # NON-FINITE FIRST, and never as part of the tolerance comparison. `nan > atol` is False, so a
    # NaN anywhere in I/sigma/bg used to make the bound PASS: the check was blind to exactly the
    # regression it most needs to catch. np.max propagates a NaN, but relying on that would still
    # end in a False comparison, so the guard is explicit and separate.
    for nm, j in (("I", 0), ("sigma", 1), ("peak", 2), ("bg", 3)):
        for side, arr in (("integrate_spots", a[j]), ("integrate_fused", b[j])):
            bad = int(np.count_nonzero(~np.isfinite(np.asarray(arr, float))))
            if bad:
                why.append(f"{side} returned {bad} non-finite {nm} value(s) -- invalid, not merely "
                           f"out of tolerance")
    d = {nm: float(np.max(np.abs(np.asarray(a[j], float) - np.asarray(b[j], float))))
         for nm, j in (("I", 0), ("sigma", 1), ("bg", 3))}      # (I, sigma, peak, bg) -> the 3 summed
    peak_eq = np.array_equal(a[2], b[2])
    exact = all(np.array_equal(x, y) for x, y in zip(a, b))
    if why:                                   # a non-finite d is meaningless; report it and stop
        return why, d, peak_eq, exact
    worst = max(d.values())
    if counting and not exact:
        why.append(f"not bit-exact on exactly-summable data (worst |d|={worst:.3g} counts)")
    if not counting and not worst <= atol:    # NOT `worst > atol`: that reads False on a NaN
        why.append(f"worst |d|={worst:.3g} > ATOL_F64={atol:g} counts "
                   + "(" + ", ".join(f"{k} {v:.3g}" for k, v in d.items()) + ")")
    # peak is a MAX: order-independent, so it is exact for every dtype and mode, always.
    if not peak_eq:
        why.append("peak differs -- a max is order-independent, so this is a real bug")
    # ...and a median is a SELECTION, so median-mode bg is exact for every dtype too.
    if mode == "median" and d["bg"] != 0.0:
        why.append(f"median-mode bg differs by {d['bg']:.3g} -- a selection cannot round")
    return why, d, peak_eq, exact


def self_test():
    """Sabotage the checker: it must REJECT what it is supposed to reject. Runs without a GPU."""
    n = 8
    base = [np.arange(n, dtype=float) + 1.0 for _ in range(4)]
    ok = lambda: [x.copy() for x in base]
    bad = 0

    def expect(label, why, want):
        nonlocal bad
        got = bool(why)
        print(f"  {'ok  ' if got == want else 'FAIL'}  {label}"
              + ("" if got == want else f"  <- wanted flagged={want}, got {why}"))
        bad += got != want

    expect("identical outputs pass", compare_outputs(ok(), ok(), True, "clipmean")[0], False)
    for nm, j in (("I", 0), ("sigma", 1), ("peak", 2), ("bg", 3)):
        for side in (0, 1):
            for val, tag in ((np.nan, "NaN"), (np.inf, "inf")):
                x, y = ok(), ok()
                (x if side == 0 else y)[j][3] = val
                expect(f"{tag} in {nm} ({'spots' if side == 0 else 'fused'}) is REJECTED",
                       compare_outputs(x, y, False, "clipmean")[0], True)
    x = ok(); x[0][2] += 1e-3
    expect("a real over-tolerance difference is REJECTED",
           compare_outputs(x, ok(), False, "clipmean")[0], True)
    x = ok(); x[0][2] += 1e-12
    expect("a difference inside ATOL_F64 passes",
           compare_outputs(x, ok(), False, "clipmean")[0], False)
    x = ok(); x[2][1] += 1.0
    expect("a peak difference is REJECTED in any mode",
           compare_outputs(x, ok(), False, "clipmean")[0], True)
    x = ok(); x[3][1] += 1e-13
    expect("median-mode bg must be EXACT (a selection cannot round)",
           compare_outputs(x, ok(), False, "median")[0], True)
    print(f"  self-test failures: {bad}")
    return bad


if "--self-test" in sys.argv or cp is None:
    print("=== checker self-test (no GPU needed) ===")
    rc = self_test()
    if cp is None:
        print("SKIP: cupy unavailable -- the device comparison below needs a GPU node.")
    if "--self-test" in sys.argv or cp is None:
        raise SystemExit(1 if rc else 0)

props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device(0).id)
print(f"=== fused GPU integration vs numpy integrate_spots ({props['name'].decode()}, "
      f"cupy {cp.__version__}) ===")
if self_test():                          # the checker itself, before trusting it on the device
    FAILS.append("checker self-test failed -- the pass/fail rules are broken, not the kernel")
cases = [("uint16 (raw)", 800, 800, 300, np.uint16, False, False, 0),
         ("int32", 800, 800, 300, np.int32, False, False, 0),
         ("float32", 800, 800, 300, np.float32, False, False, 0),
         ("uint16 + saturated px", 800, 800, 300, np.uint16, False, False, 6000),
         ("float32 + saturated px", 800, 800, 300, np.float32, False, False, 6000),
         ("float32 non-integer", 1500, 1500, 500, np.float32, False, True, 0),
         ("float64 non-integer", 1500, 1500, 500, np.float64, False, True, 0),
         ("edge-straddling", 400, 400, 400, np.float32, True, False, 0),
         ("dense 2000 spots", 4000, 4000, 2000, np.uint16, False, False, 0)]
from glint.fused_integrate import integrate_fused
for name, H, W, n, dt, edge, ni, sat in cases:
    img, pred = mk(H, W, n, dt, edge, ni, sat)
    g = cp.asarray(img)
    # EVERY bg_mode, not just the default: glint#131 changed the default estimator and put a second
    # rank-count pass in the kernel, so a mode-specific divergence is exactly what can hide here.
    for mode in BG_MODES:
        a = integrate_spots(img, pred, bg_mode=mode)
        b = integrate_fused(g, pred, bg_mode=mode)
        counting = not (ni and dt == np.float64)   # exactly-summable input, whatever the container
        why, d, peak_eq, exact = compare_outputs(a, b, counting, mode)
        print(f"  {name:22s} n={n:5d} {mode:9s} bit-exact={str(exact):5s}  "
              + "  ".join(f"max|d{k}|={v:9.3g}" for k, v in d.items())
              + f"  peak-eq={peak_eq}{'' if not why else '   <- CONTRACT VIOLATED'}")
        FAILS.extend(f"{name}/{mode}: {w}" for w in why)
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
