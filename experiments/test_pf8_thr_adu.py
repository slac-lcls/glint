"""PeakFinder8's absolute-ADU floor (CrystFEL's `--threshold`), on synthetic frames.

WHY THIS EXISTS. peakfinder8 gates a candidate pixel on TWO things and we only had one: a relative
`snr > thr_snr` and an absolute `value > threshold`. Every practitioner number quoted at us pairs
them -- mfx101555026 r0013 (Jungfrau 16M) runs `--threshold=110 --min-snr=5`, this experiment's
Cheetah used t100-s6, cxilu8823's indexamajig `--min-snr=3.5`. Comparing our bare min_snr against
those is comparing two different gates, and it shows: on 60 events of r0013 our snr 3-10 called 92%
of events hits against Cheetah's own 30%, with consensus REFUSING at every rung.

The properties worth pinning, in the order they would hurt if broken:
  * DEFAULT IS OFF and is a true no-op. `thr_adu=None` must give bit-identical peaks, because every
    number already measured with this finder was measured without it. `0.0` is NOT the no-op --
    it would additionally drop zero and negative pixels -- which is exactly why the default is None.
  * It REMOVES weak peaks and keeps strong ones (monotone in the threshold), and it is applied to
    the RAW pixel value, not the background-subtracted one.
  * It does NOT touch the background estimate. The background iteration excludes peak pixels on the
    relative test alone; adding an absolute floor there would bias the background itself, so a peak
    that survives the floor must have the SAME intensity/snr it had without it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.peakfinder8 import PeakFinder8

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def scene(H=192, W=192, seed=0):
    """Flat background + Gaussian noise + peaks of KNOWN, well-separated amplitudes.

    Peaks are kept OFF the beam centre on purpose: at q~0 the innermost radial bin holds almost no
    pixels, its mu/sigma are degenerate, and a peak placed there is not found at ANY setting -- an
    earlier version of this scene put one at the exact centre and the resulting "the floor dropped
    it" reading was an artifact of the scene, not of the floor.
    """
    rng = np.random.default_rng(seed)
    img = 100.0 + rng.normal(0, 3.0, (H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    #    raw peak value = 100 (background) + amplitude
    amps = [42.0, 120.0, 400.0]                    # raw ~142 (below a 150 floor), ~220, ~500
    pos = [(40, 40), (60, 140), (150, 150)]
    for a, (cy, cx) in zip(amps, pos):
        img += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.6 ** 2))
    q = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / H * 0.5
    return img, q, amps, pos


img, q, amps, pos = scene()
base = dict(nbin=48, thr_snr=4.0, min_snr=5.0, min_pix=2)

print("default is OFF, and is a true no-op")
a = PeakFinder8(q, **base).find(img)
b = PeakFinder8(q, **base, thr_adu=None).find(img)
check("thr_adu defaults to None", PeakFinder8(q, **base).p["thr_adu"] is None)
for k in ("x", "y", "intensity", "snr"):
    va, vb = np.asarray(a[k]), np.asarray(b[k])
    check(f"None is bit-identical to the default ({k})",
          va.shape == vb.shape and np.array_equal(va, vb))
check("the scene is found at all", len(a["x"]) >= 3, len(a["x"]))

print("\nthe floor removes weak peaks and keeps strong ones")
n_off = len(a["x"])
counts = {}
for t in (0.0, 110.0, 200.0, 450.0, 1e6):
    counts[t] = len(PeakFinder8(q, **base, thr_adu=t).find(img)["x"])
    print(f"    thr_adu {t:>9g} -> {counts[t]} peaks")
check("a floor above every pixel finds nothing", counts[1e6] == 0, counts[1e6])
check("peak count is monotone non-increasing in the floor",
      all(counts[x] >= counts[y] for x, y in ((0.0, 110.0), (110.0, 200.0), (200.0, 450.0))), counts)
check("110 ADU (this beamline's own) is selective, not a pass-through",
      counts[110.0] < n_off, (counts[110.0], n_off))

print("\napplied to the RAW value, and the background is untouched")
# THE DISCRIMINATOR: the mid peak is ~220 ADU raw and ~120 ADU above background. A floor of 150
# sits BETWEEN the two. Applied to the raw value (correct, CrystFEL) the peak SURVIVES; applied to
# the background-subtracted value it would be dropped. So finding it at thr_adu=150 is the test.
mid = PeakFinder8(q, **base, thr_adu=150.0).find(img)


def _has(pk, cy, cx, tol=3.0):
    return len(pk["x"]) and bool(
        (np.hypot(np.asarray(pk["y"]) - cy, np.asarray(pk["x"]) - cx) < tol).any())


check("a floor BETWEEN subtracted and raw keeps the peak (raw semantics)", _has(mid, *pos[1]),
      [(round(float(y)), round(float(x))) for y, x in zip(mid["y"], mid["x"])])
check("...while a floor above its raw value drops it",
      not _has(PeakFinder8(q, **base, thr_adu=260.0).find(img), *pos[1]))

strong_off = PeakFinder8(q, **base).find(img)
strong_on = PeakFinder8(q, **base, thr_adu=110.0).find(img)


def near(pk, cy, cx, tol=3.0):
    if not len(pk["x"]):
        return None
    d = np.hypot(np.asarray(pk["y"]) - cy, np.asarray(pk["x"]) - cx)
    i = int(np.argmin(d))
    return i if d[i] < tol else None


i_off, i_on = near(strong_off, *pos[2]), near(strong_on, *pos[2])
if i_off is not None and i_on is not None:
    di = abs(float(strong_off["intensity"][i_off]) - float(strong_on["intensity"][i_on]))
    ds = abs(float(strong_off["snr"][i_off]) - float(strong_on["snr"][i_on]))
    check("a surviving peak keeps its intensity (background not re-estimated)", di < 1e-6, di)
    check("...and its snr", ds < 1e-6, ds)
else:
    check("the strong peak is found with and without the floor", False, (i_off, i_on))

print("\nthe floor is what separates the finders' numbers")
# The point of the whole change: at a fixed min_snr, adding the beamline's floor changes how many
# frames would be called hits. A bare snr test admits pixels an absolute floor rejects.
noise, qn, _, _ = scene(seed=7)
noise = 100.0 + np.random.default_rng(11).normal(0, 3.0, noise.shape)   # background ONLY, no peaks
n_bare = len(PeakFinder8(qn, **base).find(noise)["x"])
n_floor = len(PeakFinder8(qn, **base, thr_adu=110.0).find(noise)["x"])
print(f"    pure-noise frame: bare snr -> {n_bare} peaks, with a 110 ADU floor -> {n_floor}")
check("the floor suppresses noise the bare snr test admits", n_floor <= n_bare, (n_bare, n_floor))

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
