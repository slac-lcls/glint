"""Un-assembled multi-panel input must integrate slab-locally, not to zero or a refusal (glint#148).

WHY THIS EXISTS. `_load_image` used to hand slab 0 of a (panel, ss, fs) dataset to
`integrate_spots`, so every reflection outside panel 0 integrated to EXACTLY ZERO, silently --
real-looking I=0/sigma=0 rows, not errors (measured in the issue: a planted 4500-count spot on
panel 1 came back 0.0). glint#136/#142 made that visible and #146 turned it into an honest
NotImplementedError. This lands the actual support: the .geom's integer ``dimN`` keys map each
panel to its slab (`_panel_slab` -> ``panels_from_geom``), `_load_image` returns the whole stack
when that mapping exists, and ``integrate_spots_stack`` integrates each reflection on ITS OWN
panel's slab at its own data-array coordinates -- which are within-slab by construction. No
assembled canvas is built, so no pixel is invented and the issue's "what is a gap" question
dissolves: an off-slab box is edge-gated (sigma=0, dropped by the caller) exactly like a
frame-edge box on an assembled frame.

The fixture is deliberately the hard layout: FOUR panels sharing TWO slabs (two asics tiling each
module, the CrystFEL 3-D shape), each panel planted with a DIFFERENT flux -- so a wrong slab, a
wrong window, or a wrong panel->reflection join all show up as the wrong intensity, and there is
no way to pass by reading the wrong pixels.

Pinned here:
  * `_panel_slab`: one integer dim -> the slab; '%'/'ss'/'fs' strings ignored; no dims -> None;
    two integers -> None (a layout the flat-panel model does not describe).
  * The end-to-end defect case: integrate_frames on a 2-slab/4-panel stack recovers each panel's
    own planted flux (pre-fix: NotImplementedError before a single reflection is integrated).
  * The 4-D (event, panel, ss, fs) path: per-event slab stacks integrate with EVENT addressing
    intact (this used to silently take a[0] -- the #148 defect surviving on the 4-D path).
  * A multi-panel stack with NO dim mapping is still refused, and the refusal now names the fix.
  * A slab-mapped .geom whose window exceeds the slab is a geometry/data mismatch -> ValueError,
    never wrong pixels.
  * An off-slab box is edge-gated to sigma=0 (dropped by integrate_frames' keep), not recorded as
    a measured zero.

  PYTHONPATH=. python experiments/test_panel_stack_integrate.py     # exit 0 = all pass
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
    print("SKIP: h5py not available (needed to build the panel-stack files)")
    sys.exit(0)

from glint.geom import parse_geom
from glint.lattice import cell_to_Ar
from glint.predict import (_load_image, _panel_slab, integrate_frames, integrate_spots_stack,
                           panels_from_geom, predict_spots)

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


LAM = 1.322
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
DATA = "/entry_1/data_1/data"
FS, SS = 700, 350                        # slab shape: 2 slabs of (SS, FS); asics split ss in half
E_KEV = f"photon_energy = {12398.419843320026 / LAM:.2f}"

# Four panels, TWO slabs: module p0 = slab 0 (asics a0/a1 tiling its ss range), module p1 =
# slab 1. Windows address WITHIN the slab; the lab position comes from corner_x/corner_y, tiling
# the four asics into a 700x700 lab plane around the beam.
PANEL_DEFS = [                           # (name, slab, win_ss0, corner_y)
    ("p0a0", 0, 0, -350), ("p0a1", 0, 175, -175), ("p1a0", 1, 0, 0), ("p1a1", 1, 175, 175)]
FLUX = {"p0a0": 1000.0, "p0a1": 2000.0, "p1a0": 4000.0, "p1a1": 8000.0}


def _geom_text(dims=True, ss_span=174):
    lines = [E_KEV, "clen = 0.1", "res = 10000", "coffset = 0.0", f"data = {DATA}"]
    for name, slab, s0, cy in PANEL_DEFS:
        lines += [f"{name}/min_fs = 0", f"{name}/max_fs = {FS-1}",
                  f"{name}/min_ss = {s0}", f"{name}/max_ss = {s0 + ss_span}",
                  f"{name}/corner_x = -350", f"{name}/corner_y = {cy}",
                  f"{name}/fs = +1.0x +0.0y", f"{name}/ss = +0.0x +1.0y"]
        if dims:
            lines += [f"{name}/dim0 = {slab}", f"{name}/dim1 = ss", f"{name}/dim2 = fs"]
    return "\n".join(lines) + "\n"


def plant(img, spots, flux):
    """A unit-sum 7x7 Gaussian of total `flux` at each (fs, ss) of one slab."""
    yy, xx = np.mgrid[-3:4, -3:4]
    g = np.exp(-(xx * xx + yy * yy) / 2.0)
    g /= g.sum()
    for f0, s0 in spots:
        cf, cs = int(round(f0)), int(round(s0))
        if 3 <= cs < img.shape[0] - 3 and 3 <= cf < img.shape[1] - 3:
            img[cs - 3:cs + 4, cf - 3:cf + 4] += flux * g


with tempfile.TemporaryDirectory() as d:
    gpath = os.path.join(d, "slabs.geom")
    open(gpath, "w").write(_geom_text())
    geom = parse_geom(gpath)

    # --- the parse and the slab mapping -----------------------------------------------------
    check("parse_geom keeps integer dims as floats and axis dims as strings",
          geom["panels"]["p0a1"]["dim0"] == 0.0 and geom["panels"]["p0a1"]["dim1"] == "ss",
          {k: geom["panels"]["p0a1"].get(k) for k in ("dim0", "dim1", "dim2")})
    check("_panel_slab: one integer dim is the slab, axis strings ignored",
          _panel_slab({"dim0": 1.0, "dim1": "ss", "dim2": "fs"}) == 1)
    check("_panel_slab: an event axis '%' does not hide the slab",
          _panel_slab({"dim0": "%", "dim1": 3.0, "dim2": "ss", "dim3": "fs"}) == 3)
    check("_panel_slab: no dims -> None (plain 2-D layout)", _panel_slab({}) is None)
    check("_panel_slab: two integer dims -> None (not this flat-panel model)",
          _panel_slab({"dim0": 0.0, "dim1": 1.0}) is None)

    panels, clen = panels_from_geom(geom)
    check("panels_from_geom carries slab per panel (shared slabs included)",
          [p["slab"] for p in panels] == [0, 0, 1, 1], [p["slab"] for p in panels])

    # --- predicted reflections land on all four panels, then get planted slab-locally --------
    pred = predict_spots(LYSO, panels, clen, LAM, dmin=4.0, tol=0.004)
    n_by_panel = [(pred["panel"] == i).sum() for i in range(4)]
    check("the fixture predicts reflections onto ALL FOUR panels", all(n >= 3 for n in n_by_panel),
          n_by_panel)

    stack = np.zeros((2, SS, FS), np.float32)
    for pi, p in enumerate(panels):
        on = pred[pred["panel"] == pi]
        plant(stack[p["slab"]], [(r["fs"], r["ss"]) for r in on], FLUX[p["name"]])

    cxi = os.path.join(d, "stack.h5")
    with h5py.File(cxi, "w") as f:
        f.create_dataset(DATA, data=stack)

    # --- THE DEFECT, end to end: pre-fix this raised NotImplementedError ----------------------
    res = [{"image": cxi, "event": 0, "M": LYSO}]
    n, tot = integrate_frames(res, geom, image_dir=d, data_path=DATA, dmin=4.0, tol=0.004)
    check("integrate_frames integrates an un-assembled 2-slab/4-panel stack (glint#148)",
          n == 1 and tot > 0, (n, tot))

    kept = res[0]["pred"]; I = np.asarray(res[0]["I"])
    med = {}
    for pi, p in enumerate(panels):
        m = kept["panel"] == pi
        med[p["name"]] = float(np.median(I[m])) if m.any() else np.nan
    print("  median I per panel: " + ", ".join(f"{k} {v:+.0f} (planted {FLUX[k]:.0f})"
                                               for k, v in med.items()))
    check("every panel recovers ITS OWN planted flux (wrong slab/window/join would mismatch)",
          all(np.isfinite(v) and abs(v / FLUX[k] - 1) < 0.25 for k, v in med.items()), med)
    check("...including off-slab-0 panels at nonzero I -- the issue's exact silent-zero case",
          all(v > 0.5 * FLUX[k] for k, v in med.items() if k.startswith("p1")), med)

    # --- the 4-D (event, panel, ss, fs) path: event addressing survives the stack ------------
    ev_scale = (1.0, 3.0, 9.0)
    cxi4 = os.path.join(d, "stack4d.h5")
    with h5py.File(cxi4, "w") as f:
        f.create_dataset(DATA, data=np.stack([stack * s for s in ev_scale]))
        f.create_dataset("/entry_1/result_1/nPeaks", data=np.array([5, 5, 5]))  # 3 events, says the file
    sums = []
    for ev in range(3):
        r4 = [{"image": cxi4, "event": ev, "M": LYSO}]
        integrate_frames(r4, geom, image_dir=d, data_path=DATA, dmin=4.0, tol=0.004)
        sums.append(float(np.asarray(r4[0]["I"]).sum()))
    check("a 4-D event+panel file integrates per event (used to silently take slab a[0])",
          sums[0] > 0 and abs(sums[1] / sums[0] - 3.0) < 0.05 and abs(sums[2] / sums[0] - 9.0) < 0.05,
          sums)

    # --- 4-D where the EVENT count equals the SLAB count (Copilot review of #157) -------------
    # The slab-count shortcut is reserved for 3-D data; on 4-D the leading axis is the event axis
    # (the geometry's '%'), even with no per-event metadata to say so. Pre-fix this classified the
    # file as a panel stack and handed all four axes to the 2-D integrator.
    cxi_co = os.path.join(d, "coincidence4d.h5")
    with h5py.File(cxi_co, "w") as f:
        f.create_dataset(DATA, data=np.stack([stack, stack * 5.0]))    # 2 events == 2 slabs, no metadata
    s_co = []
    for ev in range(2):
        rc = [{"image": cxi_co, "event": ev, "M": LYSO}]
        integrate_frames(rc, geom, image_dir=d, data_path=DATA, dmin=4.0, tol=0.004)
        s_co.append(float(np.asarray(rc[0]["I"]).sum()))
    check("4-D with n_events == n_slabs is EVENTS, not a panel stack",
          s_co[0] > 0 and abs(s_co[1] / s_co[0] - 5.0) < 0.05, s_co)

    # --- a ONE-event 4-D file: not 'stacked', but its single event is still a panel stack -----
    cxi_1e = os.path.join(d, "oneevent4d.h5")
    with h5py.File(cxi_1e, "w") as f:
        f.create_dataset(DATA, data=stack[None])                       # (1, panel, ss, fs)
    r1 = [{"image": cxi_1e, "event": 0, "M": LYSO}]
    n1e, t1e = integrate_frames(r1, geom, image_dir=d, data_path=DATA, dmin=4.0, tol=0.004)
    check("a one-event 4-D file integrates slab-locally (pre-fix: 4 axes hit the 2-D integrator)",
          n1e == 1 and t1e > 0 and abs(float(np.asarray(r1[0]["I"]).sum()) / sums[0] - 1.0) < 1e-6,
          (n1e, t1e))

    # --- no mapping -> still refused, and the refusal names the fix ---------------------------
    # A TWO-panel no-dims geometry over the 2-slab file, with per-event metadata saying ONE event
    # (so the ladder reads the axis as panels): the pre-#148 refusal, now telling the user that
    # integer dimN keys unlock slab-local integration. (The 4-panel no-dims case cannot even be
    # told from an event stack -- rule 4's "axis != n_panels -> events" -- which is exactly why
    # the mapping has to come from the .geom.)
    g2 = "\n".join([E_KEV, "clen = 0.1", "res = 10000", "coffset = 0.0", f"data = {DATA}",
                    f"m0/min_fs = 0", f"m0/max_fs = {FS-1}", "m0/min_ss = 0",
                    f"m0/max_ss = {SS-1}", "m0/corner_x = -350", "m0/corner_y = -350",
                    "m0/fs = +1.0x +0.0y", "m0/ss = +0.0x +1.0y",
                    f"m1/min_fs = 0", f"m1/max_fs = {FS-1}", "m1/min_ss = 0",
                    f"m1/max_ss = {SS-1}", "m1/corner_x = -350", "m1/corner_y = 0",
                    "m1/fs = +1.0x +0.0y", "m1/ss = +0.0x +1.0y"]) + "\n"
    gplain = os.path.join(d, "nodims.geom")
    open(gplain, "w").write(g2)
    cxi_meta = os.path.join(d, "stack_onemeta.h5")
    with h5py.File(cxi_meta, "w") as f:
        f.create_dataset(DATA, data=stack)
        f.create_dataset("/LCLS/eventNumber", data=np.array([7]))    # ONE event: axis is panels
    raised = None
    try:
        integrate_frames([{"image": cxi_meta, "event": 0, "M": LYSO}], parse_geom(gplain),
                         image_dir=d, data_path=DATA, dmin=4.0, tol=0.004)
    except Exception as exc:                      # noqa: BLE001 - the message is the point
        raised = exc
    check("a multi-panel stack with NO dim mapping is still refused",
          isinstance(raised, NotImplementedError) and "glint#148" in str(raised), repr(raised)[:120])
    check("...and the refusal says integer dimN keys are the fix", "dimN" in str(raised),
          str(raised)[:160])

    # --- a window that exceeds the slab is a mismatch, never wrong pixels ---------------------
    gbad = os.path.join(d, "badwin.geom")
    open(gbad, "w").write(_geom_text(ss_span=SS))               # max_ss = min_ss + 350 > slab
    raised_w = None
    try:
        integrate_frames([{"image": cxi, "event": 0, "M": LYSO}], parse_geom(gbad),
                         image_dir=d, data_path=DATA, dmin=4.0, tol=0.004)
    except Exception as exc:                      # noqa: BLE001
        raised_w = exc
    check("a window exceeding the slab raises a geometry/data mismatch",
          isinstance(raised_w, ValueError) and "WITHIN the panel's slab" in str(raised_w),
          repr(raised_w)[:120])

    # ...and so does a NEGATIVE bound -- the complete ordered window is validated, not just the
    # upper endpoints (Copilot review of #157): an un-addressable window must never surface as
    # silently dropped reflections.
    gneg = os.path.join(d, "negwin.geom")
    open(gneg, "w").write(_geom_text().replace("p0a1/min_ss = 175", "p0a1/min_ss = -5"))
    raised_n = None
    try:
        integrate_frames([{"image": cxi, "event": 0, "M": LYSO}], parse_geom(gneg),
                         image_dir=d, data_path=DATA, dmin=4.0, tol=0.004)
    except Exception as exc:                      # noqa: BLE001
        raised_n = exc
    check("a negative window bound raises the same mismatch, not silent drops",
          isinstance(raised_n, ValueError) and "does not address the slab" in str(raised_n),
          repr(raised_n)[:120])

    # --- an off-slab box is edge-gated, not a measured zero -----------------------------------
    edge = np.zeros(1, dtype=pred.dtype)
    edge["panel"] = 2; edge["fs"] = 350.0; edge["ss"] = 1.0      # box (R=8) leaves slab 1's edge
    Ie, se, pe, be = integrate_spots_stack(stack, edge, panels)
    check("an off-slab box comes back sigma=0 (edge-gated -> dropped by the keep filter)",
          Ie[0] == 0.0 and se[0] == 0.0, (float(Ie[0]), float(se[0])))

    # --- _load_image returns the whole stack only under a mapping ----------------------------
    got = _load_image(cxi, DATA, event=0, n_panels=4, panel_slabs=[0, 0, 1, 1])
    check("_load_image hands back the whole (panel, ss, fs) stack when slabs are mapped",
          got.ndim == 3 and got.shape == stack.shape, got.shape)

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
