"""`integrate_cxi` (the --images route) must read `data[event]` only when the leading axis IS events.

WHY THIS EXISTS. `integrate_cxi` took the frame as

    frame = np.asarray(dset[ev] if dset.ndim >= 3 else dset, np.float32)

-- any 3-D dataset was an event stack, by definition of the front end -- while the --peaks route's
`_load_image` had grown a real layout decision (glint#136, glint#148: per-event metadata, panel
count, slab mapping, refuse-when-ambiguous). Point the --images route at an un-assembled
``(panel, ss, fs)`` file under a multi-panel geometry and `data[ev]` is one PANEL's pixels integrated
as event `ev`'s assembled frame: real-looking I/sigma, wrong pixels, no error -- the #136 failure
shape on the other route. Both routes now share ONE decision (`predict._leading_axis_is_events`),
and this pins what it buys the --images route without costing it the stacked fast path.

What is pinned:
  * a stacked ``(event, ss, fs)`` .cxi with per-event metadata integrates every event against ITS
    OWN frame -- three planted fluxes come back as three intensities in the planted ratio;
  * the same stack with NO metadata under a one-panel geometry is still events (a one-panel geometry
    cannot describe a panel stack) and integrates bit-identically;
  * an un-assembled panel stack (leading axis == panel count, metadata says ONE event) is REFUSED
    with NotImplementedError naming glint#148 and both readings;
  * leading axis == panel count and NO metadata is ambiguous -> ValueError naming ``event_axis``,
    exactly as `_load_image`; ``event_axis=True`` then integrates it as events;
  * ``event_axis=False`` forces the panel reading even when metadata says events -> refused;
  * a 4-D ``(event, panel, ss, fs)`` file is events by rule but each frame is a panel stack, and a
    3-D frame is refused (this integrator works on ONE assembled 2-D image);
  * a plain 2-D dataset and a ``(1, ss, fs)`` singleton still integrate (legacy per-file layouts);
  * glint_cli forwards ``--event-axis`` to integrate_cxi as well as integrate_frames, so the
    refusal's named override is reachable from the shipped route.

  PYTHONPATH=. python experiments/test_integrate_cxi_layout.py     # exit 0 = all pass
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import h5py
except ImportError:                      # pragma: no cover - CI installs h5py; a bare tree may not
    print("SKIP: h5py not available (needed to build a synthetic .cxi)")
    sys.exit(0)

from glint.lattice import cell_to_Ar
from glint.lute_bridge import lambda_from_eV, parse_geom
from glint.predict import integrate_cxi, predict_spots

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as exc:                  # noqa: BLE001 - the type and message are the point
        return exc
    return None


LAM = 1.322
EV = 12398.419843320026 / LAM
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
NPX = 700
DATA = "/entry_1/data_1/data"
HEAD = f"photon_energy = {EV:.2f}\nclen = 0.1\nres = 10000\ncoffset = 0.0\ndata = {DATA}\n"
GEOM = (HEAD + f"p0/min_fs = 0\np0/max_fs = {NPX-1}\np0/min_ss = 0\np0/max_ss = {NPX-1}\n"
        f"p0/corner_x = {-NPX//2}\np0/corner_y = {-NPX//2}\np0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n")
NP = 4                                    # a 4-panel geometry of 60x60 panels stacked along ss
GEOM4 = HEAD + "".join(
    f"p{p}/min_fs = 0\np{p}/max_fs = 59\np{p}/min_ss = {60*p}\np{p}/max_ss = {60*p+59}\n"
    f"p{p}/corner_x = -30\np{p}/corner_y = {-30+60*p}\np{p}/fs = +1.0x +0.0y\np{p}/ss = +0.0x +1.0y\n"
    for p in range(NP))
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


def write(path, data, meta=None):
    with h5py.File(path, "w") as f:
        f.create_dataset(DATA, data=data)
        for k, v in (meta or {}).items():
            f.create_dataset(k, data=np.asarray(v))
    return path


def res(path, event):
    return [{"image": path, "event": event, "M": LYSO}]


with tempfile.TemporaryDirectory() as d:
    gpath = os.path.join(d, "one.geom"); open(gpath, "w").write(GEOM)
    g4path = os.path.join(d, "four.geom"); open(g4path, "w").write(GEOM4)
    panels, glob = parse_geom(gpath)
    clen_m = float(glob["clen"]) + float(glob.get("coffset", 0.0))
    wl = lambda_from_eV(float(glob["photon_energy"]))
    check("the 4-panel geometry parses as 4 panels", len(parse_geom(g4path)[0]) == NP)

    pred = predict_spots(LYSO, panels, clen_m, wl, dmin=5.0, tol=0.004)
    inb = pred[(pred["fs"] > 10) & (pred["fs"] < NPX - 11) & (pred["ss"] > 10) & (pred["ss"] < NPX - 11)]
    spots = [(p["fs"], p["ss"]) for p in inb]
    print(f"\n  {len(pred)} predicted reflections, {len(spots)} well inside the {NPX}px frame")
    check("the synthetic geometry actually predicts spots to integrate", len(spots) >= 20, len(spots))
    stack = np.stack([plant((NPX, NPX), spots, f) for f in FLUX])

    # --- the fast path: a STACKED (event, ss, fs) .cxi with per-event metadata ---------------
    cxi = write(os.path.join(d, "stack.cxi"), stack, {"/entry_1/result_1/nPeaks": [12, 9, 31]})
    results = [{"image": cxi, "event": e, "M": LYSO} for e in range(3)]
    nint, tot = integrate_cxi(results, gpath, dmin=5.0, tol=0.004)
    check("all three events integrated", nint == 3 and tot > 0, (nint, tot))
    sums = [float(np.asarray(r["I"]).sum()) for r in results]
    print("  integrated sum(I) per event: " + ", ".join(f"ev{e} {s:+.1f}" for e, s in enumerate(sums))
          + f"   (planted flux {FLUX[0]:.0f} : {FLUX[1]:.0f} : {FLUX[2]:.0f} per spot)")
    check("per-event intensities DIFFER (each event read its own frame)",
          len(set(np.round(sums, 6))) == 3, sums)
    r10, r20 = sums[1] / sums[0], sums[2] / sums[0]
    check("...and scale with the flux planted in each event",
          abs(r10 / (FLUX[1] / FLUX[0]) - 1) < 0.02 and abs(r20 / (FLUX[2] / FLUX[0]) - 1) < 0.02,
          (r10, r20))

    # --- no metadata, one-panel geometry: still events (rule 5), bit-identical -----------------
    cxi_nm = write(os.path.join(d, "stack_nometa.cxi"), stack)
    results_nm = [{"image": cxi_nm, "event": e, "M": LYSO} for e in range(3)]
    n_nm, _ = integrate_cxi(results_nm, gpath, dmin=5.0, tol=0.004)
    check("a metadata-free stack under a 1-panel geometry is still events, bit-identical",
          n_nm == 3 and all(np.array_equal(a["I"], b["I"]) for a, b in zip(results, results_nm)))

    # --- an un-assembled PANEL stack under the 4-panel geometry --------------------------------
    panel_stack = np.stack([np.full((60, 60), 100.0 * (p + 1), np.float32) for p in range(NP)])
    pfile_meta = write(os.path.join(d, "panels_one_event.h5"), panel_stack, {"/LCLS/eventNumber": [5]})
    exc = raises(integrate_cxi, res(pfile_meta, 2), g4path, dmin=5.0, tol=0.004)
    check("panel stack (axis == panel count, metadata says 1 event) is REFUSED, not read as data[ev]",
          isinstance(exc, NotImplementedError) and "glint#148" in str(exc), repr(exc))
    check("...and the refusal names both readings and the override",
          exc is not None and "PANELS" in str(exc) and "event_axis" in str(exc), str(exc)[:160])

    # --- THE COINCIDENCE: axis == panel count, no metadata -> ambiguous -> refuse, never guess ---
    pfile = write(os.path.join(d, "panels_or_events.h5"), panel_stack)
    exc2 = raises(integrate_cxi, res(pfile, 1), g4path, dmin=5.0, tol=0.004)
    check("axis == panel count and the file says nothing -> ValueError naming event_axis (as _load_image)",
          isinstance(exc2, ValueError) and "event_axis" in str(exc2)
          and "EVENTS" in str(exc2) and "PANELS" in str(exc2), repr(exc2))
    r_ov = res(pfile, 1)
    n_ov, _ = integrate_cxi(r_ov, g4path, dmin=5.0, tol=0.004, event_axis=True)
    check("...event_axis=True resolves it as events and integrates", n_ov == 1 and "I" in r_ov[0], n_ov)

    # --- the override in the other direction beats metadata that says events -------------------
    ev_file = write(os.path.join(d, "four_events.h5"), panel_stack,
                    {"/entry_1/result_1/nPeaks": [12, 9, 31, 4]})
    n_ev, _ = integrate_cxi(res(ev_file, 2), g4path, dmin=5.0, tol=0.004)
    check("axis == panel count + per-event metadata of that length -> EVENTS, integrates", n_ev == 1, n_ev)
    exc3 = raises(integrate_cxi, res(ev_file, 2), g4path, dmin=5.0, tol=0.004, event_axis=False)
    check("event_axis=False forces the panel reading -> refused", isinstance(exc3, NotImplementedError),
          repr(exc3))

    # --- 4-D (event, panel, ss, fs): events by rule, but each frame is a panel stack -----------
    four = write(os.path.join(d, "four_d.h5"), np.stack([panel_stack, 2 * panel_stack]),
                 {"/entry_1/result_1/nPeaks": [3, 4]})
    exc4 = raises(integrate_cxi, res(four, 1), g4path, dmin=5.0, tol=0.004)
    check("a 4-D per-event panel stack is refused (one assembled 2-D frame per event only)",
          isinstance(exc4, NotImplementedError) and "glint#148" in str(exc4) and "2-D" in str(exc4),
          repr(exc4))

    # --- legacy layouts keep working ----------------------------------------------------------
    one = write(os.path.join(d, "plain2d.h5"), stack[2])
    r1 = res(one, 0)
    n1, _ = integrate_cxi(r1, gpath, dmin=5.0, tol=0.004)
    check("a plain 2-D dataset integrates, identically to the same frame out of the stack",
          n1 == 1 and np.array_equal(r1[0]["I"], results[2]["I"]))
    one3 = write(os.path.join(d, "singleton.h5"), stack[1][None])
    r13 = res(one3, 0)
    n13, _ = integrate_cxi(r13, gpath, dmin=5.0, tol=0.004)
    check("a (1, ss, fs) singleton integrates, identically to event 1 of the stack",
          n13 == 1 and np.array_equal(r13[0]["I"], results[1]["I"]))

# --- the override must be REACHABLE from the shipped --images route --------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "glint", "glint_cli.py")) as f:
    cli_src = f.read()
tree = ast.parse(cli_src)
calls = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in ("integrate_cxi", "integrate_frames"):
            calls[name] = {k.arg for k in node.keywords}
check("glint_cli calls integrate_cxi with event_axis=", "event_axis" in calls.get("integrate_cxi", ()),
      calls.get("integrate_cxi"))
check("...and still integrate_frames with event_axis=", "event_axis" in calls.get("integrate_frames", ()),
      calls.get("integrate_frames"))
check("--event-axis help no longer scopes itself to --peaks only",
      "--integrate --peaks:" not in cli_src and "--images" in cli_src.split("--event-axis", 1)[1][:600])

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
