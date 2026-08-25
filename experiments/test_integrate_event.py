"""`integrate_frames` must integrate each result against ITS OWN event, not event 0.

WHY THIS EXISTS. glint#136. `glint --peaks ... --integrate` dispatches to
`glint.predict.integrate_frames`, which loads its image through `_load_image`. That loader ended

    if a.ndim == 3:                     # (event|panel, ss, fs) -> single assembled 2D frame
        a = a[0] if a.shape[0] > 1 else a[0]

-- BOTH branches take index 0 -- and `integrate_frames` never looked at `r['event']` at all, even
though `glint.geom.read_crystfel_peaks` parses `Event://N` out of the peak stream and
`glint.hybrid_stream` attaches it to every result. On a stacked multi-event `.cxi` (what LUTE's
`PeakFinderSFX` emits, and the natural thing to point the peaks route at) every frame in the run was
therefore box-integrated against **event 0 of its file**: real I/sigma numbers, computed from the
wrong pixels, with no error and no warning. The orientations were right, so the stream looked
healthy; only the intensities were wrong.

What is pinned here, in the order it would hurt if broken:
  * Three events of the SAME spots at DIFFERENT flux integrate to THREE DIFFERENT intensities, in
    the right ratio. This is the whole defect: before the fix all three came back identical to
    event 0's, and this file's first check failed with ratios of exactly 1.
  * An event index past the end of the stack RAISES instead of silently re-reading frame 0 -- the
    failure mode that made the original bug invisible.
  * A genuine per-file (legacy) detector is untouched: a 2D dataset, and a (1, ss, fs) dataset whose
    stream still carries a run-global `Event://N`, both still load. That is the case
    `integrate_frames` documents itself for, and the fix must not cost it.

  PYTHONPATH=. python experiments/test_integrate_event.py     # exit 0 = all pass
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import h5py
except ImportError:                      # pragma: no cover - CI installs h5py; a bare tree may not
    print("SKIP: h5py not available (needed to build a stacked .cxi)")
    sys.exit(0)

from glint.geom import parse_geom
from glint.lattice import cell_to_Ar
from glint.predict import _load_image, integrate_frames, panels_from_geom, predict_spots

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


LAM = 1.322
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
NPX = 700
DATA = "/entry_1/data_1/data"
GEOM = (f"photon_energy = {12398.419843320026 / LAM:.2f}\nclen = 0.1\nres = 10000\ncoffset = 0.0\n"
        f"data = {DATA}\np0/min_fs = 0\np0/max_fs = {NPX-1}\n"
        f"p0/min_ss = 0\np0/max_ss = {NPX-1}\np0/corner_x = {-NPX//2}\np0/corner_y = {-NPX//2}\n"
        f"p0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n")

FLUX = (1000.0, 4000.0, 16000.0)         # event 0, 1, 2 -- distinct, and distinct in RATIO


def plant(shape_hw, spots, flux):
    """One frame: a unit-sum 7x7 Gaussian of total `flux` at each (fs, ss)."""
    img = np.zeros(shape_hw, np.float32)
    yy, xx = np.mgrid[-3:4, -3:4]
    g = np.exp(-(xx * xx + yy * yy) / 2.0)
    g /= g.sum()
    for f0, s0 in spots:
        cf, cs = int(round(f0)), int(round(s0))
        img[cs - 3:cs + 4, cf - 3:cf + 4] += flux * g
    return img


with tempfile.TemporaryDirectory() as d:
    gpath = os.path.join(d, "g.geom")
    open(gpath, "w").write(GEOM)
    geom = parse_geom(gpath)
    panels, clen = panels_from_geom(geom)

    pred = predict_spots(LYSO, panels, clen, LAM, dmin=5.0, tol=0.004)
    inb = pred[(pred["fs"] > 10) & (pred["fs"] < NPX - 11)
               & (pred["ss"] > 10) & (pred["ss"] < NPX - 11)]
    spots = [(p["fs"], p["ss"]) for p in inb]
    print(f"\n  {len(pred)} predicted reflections, {len(spots)} well inside the {NPX}px frame")
    check("the synthetic geometry actually predicts spots to integrate", len(spots) >= 20, len(spots))

    # --- a STACKED multi-event .cxi: same spots every event, flux x4 each event -------------
    stack = np.stack([plant((NPX, NPX), spots, f) for f in FLUX])
    cxi = os.path.join(d, "stack.cxi")
    with h5py.File(cxi, "w") as f:
        f.create_dataset(DATA, data=stack)

    results = [{"image": cxi, "event": e, "M": LYSO} for e in range(3)]
    nint, tot = integrate_frames(results, geom, image_dir=d, data_path=DATA,
                                 dmin=5.0, tol=0.004)
    check("all three frames integrated", nint == 3 and tot > 0, (nint, tot))

    sums = [float(np.asarray(r["I"]).sum()) for r in results]
    print(f"  integrated sum(I) per event: "
          + ", ".join(f"ev{e} {s:+.1f}" for e, s in enumerate(sums))
          + f"   (planted flux {FLUX[0]:.0f} : {FLUX[1]:.0f} : {FLUX[2]:.0f} per spot)")

    # THE DEFECT. Pre-fix these three numbers were bit-identical (every event read frame 0).
    check("per-event integrated intensities DIFFER (the #136 defect)",
          len(set(np.round(sums, 6))) == 3, sums)

    r10 = sums[1] / sums[0] if sums[0] else np.nan
    r20 = sums[2] / sums[0] if sums[0] else np.nan
    print(f"  intensity ratios: ev1/ev0 = {r10:.3f} (expect {FLUX[1]/FLUX[0]:.3f}), "
          f"ev2/ev0 = {r20:.3f} (expect {FLUX[2]/FLUX[0]:.3f})")
    check("...and they scale with the flux actually planted in each event",
          abs(r10 / (FLUX[1] / FLUX[0]) - 1) < 0.02 and abs(r20 / (FLUX[2] / FLUX[0]) - 1) < 0.02,
          (r10, r20))

    # --- and the loader underneath it, directly --------------------------------------------
    loaded = [_load_image(cxi, DATA, event=e) for e in range(3)]
    check("_load_image returns a DIFFERENT frame per event",
          all(not np.array_equal(loaded[i], loaded[j])
              for i in (0, 1, 2) for j in (0, 1, 2) if i < j),
          [float(x.sum()) for x in loaded])
    check("_load_image event e IS stack[e]",
          all(np.array_equal(loaded[e], stack[e]) for e in range(3)),
          [float(x.sum()) for x in loaded])

    # --- an event past the end must be LOUD, not silently event 0 --------------------------
    raised = None
    try:
        _load_image(cxi, DATA, event=7)
    except Exception as exc:                      # noqa: BLE001 - the message is the point
        raised = exc
    check("an out-of-range event raises instead of quietly reading frame 0",
          isinstance(raised, IndexError) and "integrate_cxi" in str(raised), repr(raised))

    # --- the legacy per-file route this function documents itself for ----------------------
    one = os.path.join(d, "legacy_2d.h5")
    with h5py.File(one, "w") as f:
        f.create_dataset(DATA, data=stack[2])                    # plain 2D frame
    check("a plain 2D dataset still loads (legacy per-file detector)",
          np.array_equal(_load_image(one, DATA, event=0), stack[2]))

    one3 = os.path.join(d, "legacy_1xhw.h5")
    with h5py.File(one3, "w") as f:
        f.create_dataset(DATA, data=stack[1][None])              # (1, ss, fs)
    # a peak stream over many single-frame files carries a RUN-GLOBAL event number; a length-1
    # stack must not be indexed by it
    check("a (1,ss,fs) dataset loads under a run-global event number",
          np.array_equal(_load_image(one3, DATA, event=41), stack[1]))

    res2 = [{"image": one, "event": 0, "M": LYSO}]
    n2, t2 = integrate_frames(res2, geom, image_dir=d, data_path=DATA,
                              dmin=5.0, tol=0.004)
    check("integrate_frames still works on the legacy per-file route", n2 == 1 and t2 > 0, (n2, t2))

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
