"""Check GpuCalibrator._common_mode against psana's UtilsCommonMode, in numpy.

The GPU version replaces np.ma.median with a sort-and-pick-the-middle, reshapes psana's per-segment
python loop into batched axes, and turns `arr[bmask] -= m[bmask]` into a multiply by the mask. Each
of those is a place to be off by one, which is what this checks, to a tolerance of 2e-3.

Two reference paths, so the check is never silently a no-op. Without psana it compares against
golden outputs generated once from real psana and checked in beside this file — at a geometry
chosen so every common-mode variant actually fires (see common_mode_cases.py: at the old 64-column
width, rows-mode groups fell under npix_min and mode&1 was compared against all-zero goldens), and
it asserts per-mode-bit nonzero corrections so that gap cannot silently return. With psana it also
compares at full detector size against the live library, and re-derives the goldens so a stale
golden file is caught rather than trusted.

The case grid, input construction and psana reference driver are imported from
common_mode_cases.py — one copy, shared with the generator, so the two cannot drift.
"""
import os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_mode_cases import (GOLDEN_NCOL, GOLDEN_NSEG, assert_modes_fire, make_inputs,
                               psana_reference)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_mode_golden.npz")
try:
    import Detector.UtilsCommonMode                     # noqa: F401
    HAVE_PSANA = True
except ImportError:
    HAVE_PSANA = False

# --- import gpu_calib with numpy standing in for cupy ------------------------------------------
fake = types.ModuleType("cupy")
for name in ("sort", "take_along_axis", "maximum", "where", "abs", "inf", "int32", "float32",
             "uint16", "ascontiguousarray", "asarray", "ones", "zeros"):
    setattr(fake, name, getattr(np, name))
sys.modules["cupy"] = fake
import gpu_calib


def gpu_under_test(arrf, gmask, mode, cormax, npixmin):
    gc = gpu_calib.GpuCalibrator.__new__(gpu_calib.GpuCalibrator)
    gc.cp, gc.dt = fake, np.float32
    gc.mode, gc.cormax, gc.npixmin = mode, cormax, npixmin
    a = arrf.copy()
    gc._common_mode(a, gmask)
    return a


fails = 0

# ---- always: compare against the golden outputs, so CI covers this with or without psana --------
z = np.load(GOLDEN)
print(f"golden reference: {GOLDEN} (psana {str(z['psana_release'])})")
fails += len(assert_modes_fire({k: z[k] for k in z.files if k.startswith("corr")}))
for key, arrf, gmask, mode, cormax in make_inputs(GOLDEN_NSEG, GOLDEN_NCOL):
    want = arrf + z[key]
    got = gpu_under_test(arrf, gmask, mode, cormax, 10)
    d = np.abs(got - want).max()
    ok = d < 2e-3
    fails += not ok
    print(f"[golden] {key:15s} cormax={cormax:<7g} maxdiff={d:.3e} {'OK' if ok else 'MISMATCH'}")

# the boundary case earns its own assertions, not just a maxdiff
_, arrf, gmask, mode, cormax = make_inputs(GOLDEN_NSEG, GOLDEN_NCOL)[-1]
got = gpu_under_test(arrf, gmask, mode, cormax, 10)
same_col0 = np.array_equal(got[0, :10, 0], arrf[0, :10, 0])
changed_col1 = not np.array_equal(got[0, :11, 1], arrf[0, :11, 1])
fails += not (same_col0 and changed_col1)
print(f"[golden] npix_min boundary: col(10 good) untouched={same_col0} "
      f"col(11 good) corrected={changed_col1}")

# ---- with psana: the strong check at full detector size, and a staleness check on the golden ----
if HAVE_PSANA:
    for key, arrf, gmask, mode, cormax in make_inputs(3, 384):
        want = psana_reference(arrf, gmask, mode, cormax, 10)
        got = gpu_under_test(arrf, gmask, mode, cormax, 10)
        d = np.abs(got - want).max()
        ok = d < 2e-3
        fails += not ok
        print(f"[psana ] {key:15s} cormax={cormax:<7g} maxdiff={d:.3e} {'OK' if ok else 'MISMATCH'}")

    stale = 0
    for key, arrf, gmask, mode, cormax in make_inputs(GOLDEN_NSEG, GOLDEN_NCOL):
        d = np.abs(psana_reference(arrf, gmask, mode, cormax, 10) - (arrf + z[key])).max()
        stale += d >= 2e-3
    fails += stale > 0
    print(f"[psana ] golden file still reproduces: {'yes' if stale == 0 else f'NO -- {stale} cases stale'}")
else:
    print("psana not importable: the full-size comparison was skipped, the golden checks above ran.")

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
