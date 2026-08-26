"""A .geom's `data = <path>` key must actually reach integration (glint#143).

WHY THIS EXISTS. `glint.geom.parse_geom` filtered global keys through `_GLOBAL`, which did not
include `data`, so a geometry's `data = /some/path` line was silently dropped. The consumer side
already existed -- `integrate_frames` looks up ``geom['global']['data']`` -- but the key was never
there, so integration always fell back to `/data/data`, and on any file whose images live
elsewhere the peaks + `--integrate` route died with a bare h5py KeyError naming neither the file
nor the key that would fix it. PR #142's regression test passed `data_path` explicitly, which
papered over exactly this (its .geom fixtures carry a `data =` line that did nothing).

What is pinned here:
  * `parse_geom` keeps `data` (a string) in the returned global dict, and numeric globals are
    still floats -- the key must ride the existing float-or-string fallback, not change it.
  * `integrate_frames` with NO `data_path` argument reads the frames at the geom's `data` path.
    This is the defect: pre-fix this exact call raised ``KeyError: '/data/data'``.
  * An explicit `data_path` still wins over the geom key (two datasets, different flux, and the
    integrated intensities say which one was read).
  * A wrong path fails with an error naming the file, the tried path, the .geom `data =` key and
    `--data-path`, and the image-like datasets the file actually has -- not a bare KeyError.
  * `glint_cli` exposes `--data-path` and forwards it to all three consumers
    (`frames_from_cxi`, `integrate_cxi`, `integrate_frames`) -- a flag that works on one route
    and is silently ignored on another is the same trap in a different coat.

  PYTHONPATH=. python experiments/test_geom_data_key.py     # exit 0 = all pass
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
    print("SKIP: h5py not available (needed to build the image files)")
    sys.exit(0)

from glint.geom import parse_geom
from glint.lattice import cell_to_Ar
from glint.predict import integrate_frames, panels_from_geom, predict_spots

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


LAM = 1.322
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
NPX = 700
DATA = "/entry_1/data_1/data"            # NOT /data/data: the fallback must not save the test
ALT = "/entry_1/data_2/data"
GEOM = (f"photon_energy = {12398.419843320026 / LAM:.2f}\nclen = 0.1\nres = 10000\ncoffset = 0.0\n"
        f"data = {DATA}\np0/min_fs = 0\np0/max_fs = {NPX-1}\n"
        f"p0/min_ss = 0\np0/max_ss = {NPX-1}\np0/corner_x = {-NPX//2}\np0/corner_y = {-NPX//2}\n"
        f"p0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n")


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

    # --- the parse itself -------------------------------------------------------------------
    gg = geom.get("global", {})
    check("parse_geom keeps the `data` key as a string", gg.get("data") == DATA, gg.get("data"))
    check("...and numeric globals are still floats",
          isinstance(gg.get("res"), float) and isinstance(gg.get("clen"), float),
          {k: type(v).__name__ for k, v in gg.items()})

    panels, clen = panels_from_geom(geom)
    pred = predict_spots(LYSO, panels, clen, LAM, dmin=5.0, tol=0.004)
    inb = pred[(pred["fs"] > 10) & (pred["fs"] < NPX - 11)
               & (pred["ss"] > 10) & (pred["ss"] < NPX - 11)]
    spots = [(p["fs"], p["ss"]) for p in inb]
    check("the synthetic geometry actually predicts spots to integrate", len(spots) >= 20, len(spots))

    # one file, the SAME spots at two dataset paths with DIFFERENT flux, so the integrated
    # intensity says which dataset was read -- no way to pass by reading the wrong one
    img = os.path.join(d, "frame.h5")
    with h5py.File(img, "w") as f:
        f.create_dataset(DATA, data=plant((NPX, NPX), spots, 1000.0))
        f.create_dataset(ALT, data=plant((NPX, NPX), spots, 4000.0))

    # --- THE DEFECT: no data_path argument -> the geom's `data =` key ------------------------
    res = [{"image": img, "event": 0, "M": LYSO}]
    try:
        n, tot = integrate_frames(res, geom, image_dir=d, dmin=5.0, tol=0.004)
    except KeyError as exc:              # pre-fix: KeyError('/data/data'), the dropped-key symptom
        n, tot = 0, 0
        check("integrate_frames reads the geom's `data` path when none is passed (glint#143)",
              False, repr(exc))
    else:
        check("integrate_frames reads the geom's `data` path when none is passed (glint#143)",
              n == 1 and tot > 0, (n, tot))
    sum_geom = float(np.asarray(res[0].get("I", [0.0])).sum())

    # --- an explicit data_path still wins over the geom key ----------------------------------
    res2 = [{"image": img, "event": 0, "M": LYSO}]
    n2, t2 = integrate_frames(res2, geom, image_dir=d, data_path=ALT, dmin=5.0, tol=0.004)
    sum_alt = float(np.asarray(res2[0].get("I", [0.0])).sum())
    print(f"  sum(I): geom key {sum_geom:+.1f}, explicit override {sum_alt:+.1f} "
          f"(planted flux 1000 vs 4000)")
    check("an explicit data_path overrides the geom key (4x the flux comes back)",
          n2 == 1 and sum_geom > 0 and abs(sum_alt / sum_geom - 4.0) < 0.05,
          (sum_geom, sum_alt))

    # --- the failure mode is a clear error, not a bare KeyError ------------------------------
    raised = None
    try:
        integrate_frames([{"image": img, "event": 0, "M": LYSO}], geom, image_dir=d,
                         data_path="/nowhere/at/all", dmin=5.0, tol=0.004)
    except Exception as exc:                      # noqa: BLE001 - the message is the point
        raised = exc
    msg = str(raised)
    check("a missing dataset raises naming the file and the tried path",
          isinstance(raised, KeyError) and "frame.h5" in msg and "/nowhere/at/all" in msg,
          repr(raised)[:120])
    check("...and the message names BOTH knobs that set the path (`data =` key, --data-path)",
          "data =" in msg and "--data-path" in msg, msg[:160])
    check("...and lists an image-like dataset the file actually has", DATA in msg, msg[:200])

# --- the flag has to be REACHABLE on every route that resolves a data path -------------------
import pathlib                                                       # noqa: E402
ROOT = pathlib.Path(__file__).resolve().parent.parent
cli = (ROOT / "glint/glint_cli.py").read_text()
check("glint_cli exposes --data-path", "--data-path" in cli)
check("...and forwards it to all three consumers (frames_from_cxi, integrate_cxi, integrate_frames)",
      cli.count("args.data_path") >= 3, cli.count("args.data_path"))

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
