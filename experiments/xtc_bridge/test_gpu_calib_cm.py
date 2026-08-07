"""Check GpuCalibrator._common_mode against a transcription of psana's UtilsCommonMode, in numpy.

The GPU version replaces np.ma.median with a sort-and-pick-the-middle, reshapes psana's per-segment
python loop into batched axes, and turns `arr[bmask] -= m[bmask]` into a multiply by the mask. Each
of those is a place to be off by one. This runs both on the same random frames and requires exact
agreement to float32 rounding.
"""
import os, sys, types
import numpy as np

# --- psana's Detector/UtilsCommonMode.py, transcribed verbatim (ana-4.0.58-py3) ----------------
def common_mode_rows(arr, mask=None, cormax=None, npix_min=10):
    rows, cols = arr.shape
    if mask is None:
        cmode = np.median(arr, axis=1)
    else:
        marr = np.ma.array(arr, mask=mask < 1)
        cmode = np.ma.median(marr, axis=1)
        npix = mask.sum(axis=1)
        cmode = np.select((npix > npix_min,), (cmode,), default=0)
    if cormax is not None:
        cmode = np.select((np.fabs(cmode) < cormax,), (cmode,), default=0)
    _, m2 = np.meshgrid(np.zeros(cols, dtype=np.int16), cmode)
    if mask is None:
        arr -= m2
    else:
        bmask = mask > 0
        arr[bmask] -= m2[bmask]


def common_mode_cols(arr, mask=None, cormax=None, npix_min=10):
    rows, cols = arr.shape
    if mask is None:
        cmode = np.median(arr, axis=0)
    else:
        marr = np.ma.array(arr, mask=mask < 1)
        cmode = np.ma.median(marr, axis=0)
        npix = mask.sum(axis=0)
        cmode = np.select((npix > npix_min,), (cmode,), default=0)
    if cormax is not None:
        cmode = np.select((np.fabs(cmode) < cormax,), (cmode,), default=0)
    m1, _ = np.meshgrid(cmode, np.zeros(rows, dtype=np.int16))
    if mask is None:
        arr -= m1
    else:
        bmask = mask > 0
        arr[bmask] -= m1[bmask]


def common_mode_2d(arr, mask=None, cormax=None, npix_min=10):
    if mask is None:
        cmode = np.median(arr)
        if cormax is None or abs(cmode) < cormax:
            arr -= cmode
    else:
        arr1 = np.ones_like(arr, dtype=np.int16)
        bmask = mask > 0
        npix = arr1[bmask].sum()
        if npix < npix_min:
            return
        cmode = np.median(arr[bmask])
        if cormax is None or abs(cmode) < cormax:
            arr[bmask] -= cmode


def common_mode_rows_hsplit_nbanks(data, mask=None, nbanks=4, cormax=None, npix_min=10):
    bdata = np.hsplit(data, nbanks)
    bmask = np.hsplit(mask, nbanks)
    for b, m in zip(bdata, bmask):
        common_mode_rows(b, m, cormax, npix_min)
    data[:] = np.hstack(bdata)[:]


def common_mode_2d_hsplit_nbanks(data, mask=None, nbanks=4, cormax=None, npix_min=10):
    bdata = np.hsplit(data, nbanks)
    bmask = np.hsplit(mask, nbanks)
    for b, m in zip(bdata, bmask):
        common_mode_2d(b, m, cormax, npix_min)
    data[:] = np.hstack(bdata)[:]


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


rng = np.random.default_rng(0)
NSEG = 3
fails = 0
for mode in (2, 1, 4, 3, 7):
    for trial, (cormax, frac) in enumerate([(10.0, 0.95), (10.0, 0.5), (100.0, 0.99),
                                            (10.0, 0.03), (1e9, 0.8)]):
        arrf = (rng.normal(0, 6, (NSEG, 352, 384))).astype(np.float32)
        # a few groups get a big offset so the cormax veto is exercised in both directions
        arrf[:, :, ::37] += 25.0
        gmask = (rng.random((NSEG, 352, 384)) < frac).astype(np.uint8)
        want = psana_reference(arrf, gmask, mode, cormax, 10)
        got = gpu_under_test(arrf, gmask, mode, cormax, 10)
        d = np.abs(got - want).max()
        ok = d < 2e-3
        fails += not ok
        print(f"mode={mode} cormax={cormax:<7g} good_frac={frac:<5} maxdiff={d:.3e} "
              f"{'OK' if ok else 'MISMATCH'}")

# npix_min boundary: a group with exactly 10 and exactly 11 good pixels
arrf = rng.normal(0, 3, (1, 352, 384)).astype(np.float32)
gmask = np.zeros((1, 352, 384), np.uint8)
gmask[0, :10, 0] = 1          # 10 good -> npix > 10 is False -> no correction
gmask[0, :11, 1] = 1          # 11 good -> corrected
want = psana_reference(arrf, gmask, 2, 10.0, 10)
got = gpu_under_test(arrf, gmask, 2, 10.0, 10)
d = np.abs(got - want).max()
same_col0 = np.array_equal(got[0, :10, 0], arrf[0, :10, 0])
changed_col1 = not np.array_equal(got[0, :11, 1], arrf[0, :11, 1])
fails += not (d < 2e-3 and same_col0 and changed_col1)
print(f"npix_min boundary: maxdiff={d:.3e} col(10 good) untouched={same_col0} "
      f"col(11 good) corrected={changed_col1}")

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
