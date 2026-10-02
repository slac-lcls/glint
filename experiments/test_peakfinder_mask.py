"""Peak finding must not depend on what a MASKED pixel holds, and a NaN/inf pixel is a bad pixel.

Two defects, one root: the finders let pixels they should ignore into their arithmetic.

  NaN    v4/pf9 (CPU) applied the mask by multiplying, `I * goodf`. A masked NaN/inf times 0.0 is NaN,
         and uniform_filter's running sum carried that NaN across the rest of the frame below and to
         the right of it: one masked NaN at (60,60) of a 256^2 frame cost 3 of 6 planted peaks, and
         frames_from_cxi (the `glint --images` default) dropped that frame. In peakfinder8's
         `M @ (I * keep)` a NaN on a good pixel emptied its resolution ring of peaks, and a masked NaN
         made a spurious peak at the radial origin.
  hot    the local-max test ignored the mask, so a masked HOT pixel next to a peak's brightest pixel
         stopped every pixel of the peak from being a local maximum -- no seed, peak dropped.

The checks, in order:
  A  v4 / pf9 (classes and one-shots): masked pixels set to NaN, +-inf, a hot 5000 or 1e30 give
     EXACTLY the peak list of the same mask with 0.0 there.
  B  a NaN/+-inf on a GOOD pixel gives exactly the peak list of that pixel masked (v4, pf9, pf8).
  C  pf8 (CPU): the same masked-value invariance; no spurious peak at the radial origin.
  D  on a finite frame the new v4/pf9 ring background equals the pre-fix `I * goodf` arithmetic.
  E  the CUDA stats kernels (v4 fp64, v4 fp32, pf9), compiled as plain C++ and run on the CPU: the
     same invariances, and the seed beside a masked hot pixel. Skips if there is no C++ compiler.
     Not a GPU run -- it checks the kernel SOURCE, which is what the fix changed.

Plain script: prints ok/FAIL lines, exits 1 on any failure. numpy + scipy; a C++ compiler for E.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import scipy.ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", RuntimeWarning)

import glint.peakfinder_v4 as V4
import glint.peakfinder9 as P9
import glint.peakfinder8 as P8

FAILS = []
KEYS = ("x", "y", "intensity", "snr", "npix")


def check(name, ok, detail=""):
    print(("ok    " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def section(name, fn):
    """Run one block; an exception (e.g. an API an older tree lacks) is one FAIL, not a crash."""
    try:
        fn()
    except Exception as e:                                       # noqa: BLE001 -- report and go on
        check(f"{name}: ran", False, f"{type(e).__name__}: {e}")


def same(a, b):
    """Two peak dicts equal bit for bit."""
    for k in KEYS:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return False
    return True


def recovered(pk, peaks, tol=1.5):
    x, y = np.asarray(pk["x"]), np.asarray(pk["y"])
    return sum(bool(np.any(np.hypot(x - px, y - py) < tol)) for (py, px) in peaks)


# ---------------------------------------------------------------- the frame used by A-E
H = W = 160
PEAKS = [(30, 30), (30, 120), (120, 30), (80, 80), (120, 120), (64, 64)]
rng = np.random.default_rng(601)
yy, xx = np.mgrid[0:H, 0:W].astype(float)
lam = np.full((H, W), 10.0)
for (py, px) in PEAKS:
    lam += 300.0 * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / 2.0)
CLEAN = rng.poisson(lam).astype(np.float32)
MASK = rng.random((H, W)) > 5e-3                                # ~0.5% bad pixels...
for (py, px) in PEAKS:
    MASK[py - 2:py + 3, px - 2:px + 3] = True                  # ...none on a planted peak,
MASK[8, 8] = False                                             # one early in raster order (a NaN
                                                               # there poisoned everything after it),
MASK[64, 65] = False                                           # one beside the (64,64) peak
ZEROED = np.where(MASK, CLEAN, 0.0).astype(np.float32)
MASKED_VALUES = {"NaN": np.nan, "+inf": np.inf, "-inf": -np.inf, "hot 5000": 5000.0, "1e30": 1e30}
GOOD_PX = (100, 40)                                            # a good pixel away from every peak

FINDERS = {
    "v4": lambda img, m: V4.PeakFinderV4(m).find(img),
    "v4 one-shot": lambda img, m: V4.peakfinder_v4(img, mask=m),
    "pf9": lambda img, m: P9.PeakFinder9(m).find(img),
    "pf9 one-shot": lambda img, m: P9.peakfinder9(img, mask=m),
}


def part_a():
    for name, run in FINDERS.items():
        ref = run(ZEROED, MASK)
        check(f"A {name}: reference (masked px = 0) recovers all {len(PEAKS)} planted peaks",
              recovered(ref, PEAKS) == len(PEAKS), f"{recovered(ref, PEAKS)}/{len(PEAKS)}")
        for tag, val in MASKED_VALUES.items():
            img = np.where(MASK, CLEAN, val).astype(np.float32)
            pk = run(img, MASK)
            check(f"A {name}: every masked px = {tag} -> identical peak list", same(pk, ref),
                  f"{recovered(pk, PEAKS)}/{len(PEAKS)} planted, {len(pk['x'])} vs {len(ref['x'])} peaks")


def part_b():
    m2 = np.ones((H, W), bool); m2[GOOD_PX] = False
    for name, cls in (("v4", V4.PeakFinderV4), ("pf9", P9.PeakFinder9)):
        ref = cls(m2).find(CLEAN)
        for tag in ("NaN", "+inf", "-inf"):
            img = CLEAN.copy(); img[GOOD_PX] = MASKED_VALUES[tag]
            pk = cls(np.ones((H, W), bool)).find(img)
            check(f"B {name}: {tag} on a GOOD pixel == that pixel masked", same(pk, ref),
                  f"{recovered(pk, PEAKS)}/{len(PEAKS)} planted")
    for tag, fn in (("v4 one-shot", V4.peakfinder_v4), ("pf9 one-shot", P9.peakfinder9)):
        img = CLEAN.copy(); img[8, 8] = np.nan                   # no mask passed at all
        pk = fn(img)
        check(f"B {tag}, no mask, NaN at (8,8): all planted peaks found", recovered(pk, PEAKS) == len(PEAKS),
              f"{recovered(pk, PEAKS)}/{len(PEAKS)}")


Q8 = np.hypot(yy - 80.3, xx - 79.6)                            # pf8 bins on |q|; a radius does
PF8_GOOD_PX = (151, 80)                                        # r = 70.7: the (30,30) peak's ring, 100 px away


def part_c():
    ref = P8.PeakFinder8(Q8, mask=MASK).find(ZEROED)
    off = [p for p in PEAKS if p != (80, 80)]                    # (80,80) sits on the radial origin
    check(f"C pf8: reference recovers all {len(off)} off-centre planted peaks", recovered(ref, off) == len(off),
          f"{recovered(ref, off)}/{len(off)}")
    for tag in ("NaN", "+inf", "-inf", "hot 5000"):
        img = np.where(MASK, CLEAN, MASKED_VALUES[tag]).astype(np.float32)
        pk = P8.PeakFinder8(Q8, mask=MASK).find(img)
        r = np.hypot(np.asarray(pk["y"]) - 80.3, np.asarray(pk["x"]) - 79.6)
        check(f"C pf8: every masked px = {tag} -> identical peak list", same(pk, ref),
              f"{len(pk['x'])} vs {len(ref['x'])} peaks, {int((r < 2).sum())} at the radial origin")
    m2 = MASK.copy(); m2[PF8_GOOD_PX] = False
    ref2 = P8.PeakFinder8(Q8, mask=m2).find(ZEROED)
    for tag in ("NaN", "+inf", "-inf"):
        img = ZEROED.copy(); img[PF8_GOOD_PX] = MASKED_VALUES[tag]
        pk = P8.PeakFinder8(Q8, mask=MASK).find(img)
        check(f"C pf8: {tag} on a GOOD pixel == that pixel masked", same(pk, ref2),
              f"{len(pk['x'])} vs {len(ref2['x'])} peaks")


def part_d():
    """Pre-fix arithmetic, frozen: bit-identity on finite frames is the promise."""
    img = (CLEAN - 12.0).astype(np.float64)                      # negatives too: the old I*0.0 gave -0.0
    for name, cls in (("v4", V4.PeakFinderV4), ("pf9", P9.PeakFinder9)):
        f = cls(MASK)
        r = f.r; goodf = MASK.astype(np.float64)
        w, wi = 2 * r + 1, 2 * r - 1
        bs = lambda a, s: ndi.uniform_filter(a, size=s, mode="constant") * float(s * s)
        Ig = img * goodf
        ring_sum = bs(Ig, w) - bs(Ig, wi); ring_sq = bs(Ig * img, w) - bs(Ig * img, wi)
        ring_n = bs(goodf, w) - bs(goodf, wi); nz = ring_n > 0.5
        den = np.where(nz, ring_n, 1.0); mu = ring_sum / den
        sig = np.sqrt(np.clip(ring_sq / den - mu * mu, 0.0, None))
        new = f._ring_bg(img)
        check(f"D {name}: finite frame, ring mu/sigma == the pre-fix I*goodf arithmetic",
              all(np.array_equal(a, b) for a, b in zip(new, (mu, sig, nz))))


# ---------------------------------------------------------------- E: CUDA source compiled as C++
_SHIM = r"""
#include <math.h>
#define __global__
struct glint_dim3 { int x, y, z; };
static glint_dim3 blockIdx = {0, 0, 0}, blockDim = {1, 1, 1}, threadIdx = {0, 0, 0};
"""
_DRIVER = r"""
extern "C" void run_kernel(const float* I, const float* good, int H, int W, int r, int lmr,
    float a0, float a1, float a2, float a3, float a4,
    float* snr, float* sub, float* var, unsigned char* f1, unsigned char* f2){
  for(int i = 0; i < H*W; ++i){ blockIdx.x = i; __CALL__; } }
"""
_CALLS = {"pfv4_stats": "pfv4_stats(I, good, H, W, r, lmr, a0, a1, a2, a3, a4, snr, sub, var, f1, f2)",
          "pf9_stats": "pf9_stats(I, good, H, W, r, lmr, a0, a1, a2, a3, snr, sub, var, f1, f2)"}


def _cxx():
    for c in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def part_e():
    cxx = _cxx()
    if cxx is None:
        print("SKIP  E: no C++ compiler on PATH (set CXX) -- the CUDA kernel sources are not checked here")
        return
    import ctypes
    kernels = (("v4 fp64 kernel", "pfv4_stats", lambda: V4._pfv4_source(False), (5.0, 8.0, 0.0, 0.0, 0.0)),
               ("v4 fp32 kernel", "pfv4_stats", lambda: V4._pfv4_source(True), (5.0, 8.0, 0.0, 0.0, 0.0)),
               ("pf9 kernel", "pf9_stats", lambda: P9._PF9_SRC, (7.0, 6.0, 0.0, 0.0, 0.0)))
    tmp = tempfile.mkdtemp(prefix="glint_kemu_")
    try:
        for label, kname, src_fn, params in kernels:
            def one(label=label, kname=kname, src_fn=src_fn, params=params):
                cpp = os.path.join(tmp, kname + label.split()[1] + ".cpp"); so = cpp[:-4] + ".so"
                open(cpp, "w").write(_SHIM + src_fn() + _DRIVER.replace("__CALL__", _CALLS[kname]))
                res = subprocess.run([cxx, "-O1", "-ffp-contract=off", "-shared", "-fPIC", "-o", so, cpp],
                                     capture_output=True, text=True)
                check(f"E {label}: source compiles as C++", res.returncode == 0, res.stderr.strip()[:300])
                if res.returncode != 0:
                    return
                fn = ctypes.CDLL(so).run_kernel
                P = ctypes.c_void_p
                fn.argtypes = [P, P] + [ctypes.c_int] * 4 + [ctypes.c_float] * 5 + [P] * 5

                def run(img, good):
                    I = np.ascontiguousarray(img, np.float32); g = np.ascontiguousarray(good, np.float32)
                    out = [np.empty((H, W), np.float32) for _ in range(3)] + [np.empty((H, W), np.uint8) for _ in range(2)]
                    fn(I.ctypes.data, g.ctypes.data, H, W, 4, 1, *params, *[o.ctypes.data for o in out])
                    return out                                   # snr, sub, var, grow|cand, seed|ismax

                def eq(a, b, usable):                          # sub is I-bg at every pixel, masked or not
                    return all(np.array_equal(a[i], b[i]) for i in (0, 2, 3, 4)) and \
                        np.array_equal(a[1][usable], b[1][usable])

                ref = run(ZEROED, MASK)
                check(f"E {label}: seed/max flag set at the (64,64) peak beside a masked pixel",
                      bool(ref[4][64, 64]))
                for tag in ("NaN", "+inf", "hot 5000"):
                    out = run(np.where(MASK, CLEAN, MASKED_VALUES[tag]), MASK)
                    check(f"E {label}: every masked px = {tag} -> identical stats", eq(out, ref, MASK))
                m2 = np.ones((H, W), bool); m2[GOOD_PX] = False
                ref2 = run(CLEAN, m2)
                img = CLEAN.copy(); img[GOOD_PX] = np.nan
                check(f"E {label}: NaN on a GOOD pixel == that pixel masked", eq(run(img, np.ones((H, W), bool)), ref2, m2))
            section(f"E {label}", one)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


for nm, fn in (("A", part_a), ("B", part_b), ("C", part_c), ("D", part_d), ("E", part_e)):
    section(nm, fn)

print()
if FAILS:
    print(f"{len(FAILS)} check(s) FAILED")
    sys.exit(1)
print("ALL PASS")
