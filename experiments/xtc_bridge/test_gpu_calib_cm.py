"""Check GpuCalibrator._common_mode against psana's UtilsCommonMode, in numpy.

The GPU version replaces np.ma.median with a sort-and-pick-the-middle, reshapes psana's per-segment
python loop into batched axes, and turns `arr[bmask] -= m[bmask]` into a multiply by the mask. Each
of those is a place to be off by one, which is what this checks, to a tolerance of 2e-3.

Two reference paths, so the check is never silently a no-op. Without psana it compares against
golden outputs generated once from real psana and checked in beside this file. With psana it also
compares at full detector size against the live library, and re-derives the goldens so a stale
golden file is caught rather than trusted.

Inputs are drawn with the legacy RandomState rather than the newer Generator: numpy guarantees
stream compatibility for the former across releases but explicitly does not for Generator
distributions, and the goldens are only valid for the exact arrays that produced them.
"""
import os, sys, types
import numpy as np

# --- reference implementation -------------------------------------------------------------------
# The five common-mode functions were once transcribed verbatim from psana
# Detector/UtilsCommonMode.py (ana-4.0.58-py3, M. Dubrovin). psana declares no licence, so that
# copy could not be covered by this repository's licence and has been removed.
#
# Two reference paths now, so the check is never silently a no-op:
#   * psana present  -> import the real functions and compare against them at full detector size.
#                       Also re-derives the golden file below, so a stale golden is caught.
#   * psana absent   -> compare against GOLDEN outputs generated once from real psana
#                       (common_mode_golden.npz, ana-4.0.59-py3-minipytorch). Smaller arrays, but
#                       every mode, the bank split and the npix_min boundary are still exercised.
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_mode_golden.npz")
try:
    from Detector.UtilsCommonMode import (          # noqa: F401
        common_mode_rows,
        common_mode_cols,
        common_mode_2d,
        common_mode_rows_hsplit_nbanks,
        common_mode_2d_hsplit_nbanks,
    )
    HAVE_PSANA = True
except ImportError:
    HAVE_PSANA = False

def psana_reference(arrf, gmask, mode, cormax, npixmin):
    """calib_epix10ka_any's common-mode block: per segment, banks then rows then cols."""
    a = arrf.copy()
    hrows = 176
    for s in range(a.shape[0]):
        if mode & 4:
            common_mode_2d_hsplit_nbanks(a[s, :hrows, :], mask=gmask[s, :hrows, :], nbanks=8,
                                         cormax=cormax, npix_min=npixmin)
            common_mode_2d_hsplit_nbanks(a[s, hrows:, :], mask=gmask[s, hrows:, :], nbanks=8,
                                         cormax=cormax, npix_min=npixmin)
        if mode & 1:
            common_mode_rows_hsplit_nbanks(a[s, ], mask=gmask[s, ], nbanks=8,
                                           cormax=cormax, npix_min=npixmin)
        if mode & 2:
            common_mode_cols(a[s, :hrows, :], mask=gmask[s, :hrows, :],
                             cormax=cormax, npix_min=npixmin)
            common_mode_cols(a[s, hrows:, :], mask=gmask[s, hrows:, :],
                             cormax=cormax, npix_min=npixmin)
    return a


# --- import gpu_calib with numpy standing in for cupy ------------------------------------------
fake = types.ModuleType("cupy")
for name in ("sort", "take_along_axis", "maximum", "where", "abs", "inf", "int32", "float32",
             "uint16", "ascontiguousarray", "asarray", "ones", "zeros"):
    setattr(fake, name, getattr(np, name))
sys.modules["cupy"] = fake
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_calib


def gpu_under_test(arrf, gmask, mode, cormax, npixmin):
    gc = gpu_calib.GpuCalibrator.__new__(gpu_calib.GpuCalibrator)
    gc.cp, gc.dt = fake, np.float32
    gc.mode, gc.cormax, gc.npixmin = mode, cormax, npixmin
    a = arrf.copy()
    gc._common_mode(a, gmask)
    return a


CASES = [(mode, t, cormax, frac)
         for mode in (2, 1, 4, 3, 7)
         for t, (cormax, frac) in enumerate([(10.0, 0.95), (10.0, 0.5), (100.0, 0.99),
                                             (10.0, 0.03), (1e9, 0.8)])]


def make_inputs(nseg, ncol):
    """Regenerate every case's arrays from the seed, in the order the golden file was built."""
    rng = np.random.RandomState(0)          # stream-stable across numpy releases
    out = []
    for mode, t, cormax, frac in CASES:
        arrf = (rng.normal(0, 6, (nseg, 352, ncol))).astype(np.float32)
        arrf[:, :, ::37] += 25.0                     # big offsets so the cormax veto is exercised
        gmask = (rng.random_sample((nseg, 352, ncol)) < frac).astype(np.uint8)
        out.append((f"corr_{mode}_{t}", arrf, gmask, mode, cormax))
    arrf = rng.normal(0, 3, (1, 352, ncol)).astype(np.float32)
    gmask = np.zeros((1, 352, ncol), np.uint8)
    gmask[0, :10, 0] = 1        # exactly 10 good -> npix > 10 is False -> no correction
    gmask[0, :11, 1] = 1        # exactly 11 good -> corrected
    out.append(("corr_boundary", arrf, gmask, 2, 10.0))
    return out


fails = 0

# ---- always: compare against the golden outputs, so CI covers this with or without psana --------
z = np.load(GOLDEN)
print(f"golden reference: {GOLDEN} (psana {str(z['psana_release'])})")
for key, arrf, gmask, mode, cormax in make_inputs(1, 64):
    want = arrf + z[key]
    got = gpu_under_test(arrf, gmask, mode, cormax, 10)
    d = np.abs(got - want).max()
    ok = d < 2e-3
    fails += not ok
    print(f"[golden] {key:15s} cormax={cormax:<7g} maxdiff={d:.3e} {'OK' if ok else 'MISMATCH'}")

# the boundary case earns its own assertions, not just a maxdiff
_, arrf, gmask, mode, cormax = make_inputs(1, 64)[-1]
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
    for key, arrf, gmask, mode, cormax in make_inputs(1, 64):
        d = np.abs(psana_reference(arrf, gmask, mode, cormax, 10) - (arrf + z[key])).max()
        stale += d >= 2e-3
    fails += stale > 0
    print(f"[psana ] golden file still reproduces: {'yes' if stale == 0 else f'NO -- {stale} cases stale'}")
else:
    print("psana not importable: the full-size comparison was skipped, the golden checks above ran.")

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
