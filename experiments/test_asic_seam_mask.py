"""ASIC seam masking for tiled modules, on synthetic frames.

WHY THIS EXISTS. A tiled module's ASICs do not butt together at one pixel pitch -- the pixels on an
ASIC edge collect over a wider area and read HIGH -- so every frame carries a bright, connected,
few-pixel line at each interior seam. peakfinder8 is built to report exactly that shape, and because
the seams do not move, the false peaks are reproducible and read as signal rather than noise.

The trap is that the facility mask does not cover it. psana's `_mask_edges()` masks the module
PERIMETER, and the seams are interior; the pixel-status mask does not flag them either, because they
are working pixels, just bigger ones. CrystFEL never meets the problem: its geometry declares every
ASIC its own panel (p0a0, p0a1, ...), so the seams are masked as panel edges for free. That asymmetry
is the reason to care -- a peak list built from a raw psana array and one built from a CrystFEL
geometry at nominally the SAME settings are not comparable until the seams are masked.

Measured on 400 raw Jungfrau 16M frames of mfx101555026 r0013, at the CrystFEL-matched settings
(--threshold=50 --min-snr=7 --min-pix-count=4 --max-pix-count=200 --min-res=50 --max-res=3000), with
the facility mask alone (psana `_mask_from_status()` & `_mask_edges(width=2)`). Of 8,009 returned
peaks, **57% sat within half a pixel of an interior seam and 79% within one pixel, against 0.5% and
1.5% of the unmasked area** -- an enrichment of 117x and 54x. The median peak-to-nearest-seam distance
was 0.5 px. Masking the seams costs 2.44% of the module and removes 78% of the peaks, taking the mean
from 20.0 to 4.4 per frame and the fraction of frames carrying any peak at all from 90% to 39%.

The definitions are part of the claim, because the previous ones were lost. `interior seam` means
ss = 256 (mod 512) and fs in {256, 512, 768}, EXCLUDING module perimeters -- those are
`_mask_edges`' job and counting them would flatter the result. The chance figures are the fraction of
UNMASKED pixels at the same distance, computed from the actual mask rather than asserted, which is
why they can be checked: they match the analytic interior-seam area at +-0 and +-1 px exactly.
`experiments/measure_asic_seams.py` is the script that produced all of it; run it on an LCLS analysis
node to re-check. That file exists because the figure this paragraph used to quote -- "~12% expected
by area" -- reconstructs from no natural definition of a seam (the interior-seam area fraction is
0.49% at +-0 px, 1.5% at +-1, 2.4% at +-2, and reaches 10% only at +-10 px), and nothing had been
kept that could be used to check it.

The properties worth pinning, in the order they would hurt if broken:
  * width=0 IS A TRUE NO-OP (all True), so the mask can be wired in unconditionally and disabled by a
    parameter rather than by an `if` at every call site.
  * It masks INTERIOR seams only and never the perimeter AS A PERIMETER -- that is `_mask_edges()`'s
    job, and a helper that silently did both would double-mask and change the perimeter policy. The
    corollary looks like a contradiction and is not: a seam runs the full width of the module, so the
    pixel where a seam MEETS the perimeter is dropped. It is an ASIC edge pixel twice over, and
    sparing it to keep the outer row pristine would leave a genuine seam pixel live.
  * A malformed tiling RAISES rather than returning an all-True mask. A mask that silently fails to
    mask is the precise failure this exists to prevent, and a negative ASIC size (empty range) or a
    negative width (looks like the width=0 opt-out) would both be silent without an explicit check.
  * It broadcasts against a raw (n_module, ss, fs) stack, because that is the shape psana hands you.
  * The point of the whole thing: a seam artifact IS found by the finder without the mask and is NOT
    found with it, while a real peak away from the seam survives both.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.peakfinder8 import PeakFinder8, asic_seam_mask

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


print("mask geometry")

m = asic_seam_mask((512, 1024), (256, 256))
check("returns the module shape", m.shape == (512, 1024), m.shape)
check("width=0 is a true no-op", asic_seam_mask((512, 1024), (256, 256), 0).all())

# Interior seams only. range() stops before the module edge, so a module whose size is an exact
# multiple of the ASIC size must NOT get a mask line at ss or fs.
masked_rows = np.where(~m[:, 5])[0]
masked_cols = np.where(~m[5, :])[0]
check("row seams at the interior ASIC boundary", list(masked_rows) == [254, 255, 256, 257], masked_rows)
check("col seams at every interior ASIC boundary",
      list(masked_cols) == [254, 255, 256, 257, 510, 511, 512, 513, 766, 767, 768, 769], masked_cols)
check("perimeter is not masked AS A PERIMETER (that is _mask_edges' job)",
      bool(m[0, 5]) and bool(m[511, 5]) and bool(m[5, 0]) and bool(m[5, 1023]))
# The corollary, pinned so it does not get "fixed" later: a seam runs the FULL width of the module, so
# the pixel where a seam meets the perimeter IS dropped. It is an ASIC edge pixel twice over, and
# sparing it to keep the outer row pristine would leave a genuine seam pixel live.
check("a seam still reaches the module edge",
      (not m[254, 0]) and (not m[254, 1023]) and (not m[0, 255]) and (not m[511, 255]),
      (m[254, 0], m[254, 1023], m[0, 255], m[511, 255]))
check("cost is a couple of percent", 0.01 < 1 - m.mean() < 0.04, 1 - m.mean())
check("broadcasts against a raw (n_module, ss, fs) stack",
      (np.ones((4, 512, 1024), bool) & m).shape == (4, 512, 1024))

# A malformed tiling must RAISE, not return an all-True mask: silently not masking is the exact
# failure this helper exists to prevent, and both slips below are quiet without an explicit check.
for bad_asic, bad_w, why in [((-256, 256), 2, "negative asic size"),
                             ((0, 256), 2, "zero asic size"),
                             ((256, 256), -3, "negative width")]:
    try:
        asic_seam_mask((512, 1024), bad_asic, bad_w)
        check(f"{why} raises rather than silently not masking", False, "returned a mask")
    except ValueError:
        check(f"{why} raises rather than silently not masking", True)

# A module that is NOT an exact multiple of the ASIC size still only gets interior seams.
m2 = asic_seam_mask((300, 300), (256, 256), width=1)
check("ragged module: one interior seam per axis",
      list(np.where(~m2[:, 0])[0]) == [255, 256] and list(np.where(~m2[0, :])[0]) == [255, 256],
      (np.where(~m2[:, 0])[0], np.where(~m2[0, :])[0]))


print("\nthe seam is found without the mask and not with it")

# One ASIC-tiled module: flat background + noise, a seam line reading high (the artifact), and one
# real peak parked well away from any seam. Radii are in pixels from a corner origin -- the finder
# only needs a monotone radial coordinate to build its rings.
H, W, A = 256, 512, 128
rng = np.random.default_rng(0)
img = 100.0 + rng.normal(0, 3.0, (H, W))

seam = asic_seam_mask((H, W), (A, A), width=0)          # all True -> locate seams for the scene
for b in range(A, H, A):
    img[b - 1:b + 1, :] += 900.0                        # the oversized edge pixels, reading high
for b in range(A, W, A):
    img[:, b - 1:b + 1] += 900.0

py, px = 60, 60                                          # a real peak, far from every seam
yy, xx = np.mgrid[0:H, 0:W]
img += 4000.0 * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * 1.4 ** 2))

q = np.hypot(yy - 0.0, xx - 0.0).astype(float)
base = dict(nbin=120, thr_snr=7.0, min_snr=7.0, min_pix=4, max_pix=200, thr_adu=150.0, n_iter=3)


def near_seam(res, tol=3):
    """Fraction of returned peaks within tol px of an interior ASIC seam."""
    if res["x"].size == 0:
        return 0.0
    ry = np.asarray(res["y"]); rx = np.asarray(res["x"])
    dr = np.min(np.abs(ry[:, None] - np.arange(A, H, A)[None, :]), axis=1)
    dc = np.min(np.abs(rx[:, None] - np.arange(A, W, A)[None, :]), axis=1)
    return float(np.mean(np.minimum(dr, dc) <= tol))


def found_real(res, tol=3.0):
    if res["x"].size == 0:
        return False
    return bool(np.any((np.abs(np.asarray(res["y"]) - py) <= tol)
                       & (np.abs(np.asarray(res["x"]) - px) <= tol)))


bare = PeakFinder8(q, mask=np.ones((H, W), bool), **base).find(img)
maskd = PeakFinder8(q, mask=asic_seam_mask((H, W), (A, A), width=2), **base).find(img)

print(f"    no seam mask : {bare['x'].size:4d} peaks, {100 * near_seam(bare):.0f}% of them on a seam")
print(f"    seam masked  : {maskd['x'].size:4d} peaks, {100 * near_seam(maskd):.0f}% of them on a seam")

check("without the mask the seam dominates the peak list", near_seam(bare) > 0.5, near_seam(bare))
check("with the mask the seam is gone", near_seam(maskd) == 0.0, near_seam(maskd))
check("masking removes peaks rather than adding them", maskd["x"].size < bare["x"].size,
      (bare["x"].size, maskd["x"].size))
check("the real peak survives the mask", found_real(maskd))
check("...and was found without it too (the mask did not create it)", found_real(bare))

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
