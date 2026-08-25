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
from glint.predict import (_load_image, integrate_frames, integrate_spots, panels_from_geom,
                           predict_spots)

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

    # --- (panel, ss, fs) MUST NOT be read as (event, ss, fs) --------------------------------
    # The old loader documented its 3-D axis as "(event|panel, ss, fs)" and took index 0 for both.
    # Indexing a PANEL stack by the event would silently integrate panel N of a legacy per-file
    # detector whose peak stream carries a run-global Event://N -- a different wrong image, not a
    # fixed one. But matching the axis against the panel count does not DISAMBIGUATE it either
    # (see THE COINCIDENCE below), so the file's own per-event metadata decides, and a case that
    # nothing settles is refused instead of guessed.
    NP = 4
    panel_stack = np.stack([np.full((60, 60), 100.0 * (p + 1), np.float32) for p in range(NP)])
    pfile = os.path.join(d, "panelstack.h5")
    with h5py.File(pfile, "w") as f:
        f.create_dataset(DATA, data=panel_stack)
    # a 4-panel geometry: the leading axis matches the panel count, so it is panels
    gp = "\n".join([f"photon_energy = {12398.419843320026 / LAM:.2f}", "clen = 0.1", "res = 10000",
                    "coffset = 0.0", f"data = {DATA}"]
                   + [f"p{p}/min_fs = 0\np{p}/max_fs = 59\np{p}/min_ss = {60*p}\n"
                      f"p{p}/max_ss = {60*p+59}\np{p}/corner_x = -30\np{p}/corner_y = {-30+60*p}\n"
                      f"p{p}/fs = +1.0x +0.0y\np{p}/ss = +0.0x +1.0y" for p in range(NP)]) + "\n"
    gppath = os.path.join(d, "panels.geom")
    open(gppath, "w").write(gp)
    geom_p = parse_geom(gppath)
    panels_p, _ = panels_from_geom(geom_p)
    check("the 4-panel geometry parses as 4 panels", len(panels_p) == NP, len(panels_p))

    # A multi-panel PANEL stack is REFUSED, not silently reduced to slab 0. That reduction is what
    # this function did before glint#136 and it was wrong then: integration runs on ONE ASSEMBLED
    # frame, so a slab drops every other panel's reflections to exactly 0. Measured below.
    pfile_meta = os.path.join(d, "panelstack_with_meta.h5")
    with h5py.File(pfile_meta, "w") as f:
        f.create_dataset(DATA, data=panel_stack)
        f.create_dataset("/LCLS/eventNumber", data=np.array([5]))      # ONE event in this file
    raised_p = None
    try:
        _load_image(pfile_meta, DATA, event=2, n_panels=NP)
    except Exception as exc:                      # noqa: BLE001 - the message is the point
        raised_p = exc
    check("an un-assembled MULTI-panel stack is refused, not reduced to slab 0",
          isinstance(raised_p, NotImplementedError) and "glint#148" in str(raised_p), repr(raised_p))
    check("...and the refusal says what would have gone wrong (reflections integrating to 0)",
          raised_p is not None and "integrate to exactly 0" in str(raised_p), str(raised_p)[:100])

    # a ONE-panel geometry is unaffected: slab 0 really is the whole detector there
    one_panel_stack = os.path.join(d, "onepanel_slab.h5")
    with h5py.File(one_panel_stack, "w") as f:
        f.create_dataset(DATA, data=panel_stack)
        f.create_dataset("/LCLS/eventNumber", data=np.array([5]))
    check("a one-panel geometry still reads slab 0 (panel mode stays usable there)",
          np.array_equal(_load_image(one_panel_stack, DATA, event=2, n_panels=1, event_axis=False),
                         panel_stack[0]))

    # ...while the SAME file under a one-panel geometry is an event stack, because a one-panel
    # geometry cannot be describing a panel stack
    ev2 = _load_image(pfile, DATA, event=2, n_panels=1)
    check("the same file under a 1-panel geometry IS event-indexed",
          np.array_equal(ev2, panel_stack[2]), float(ev2.flat[0]))

    # ...and a leading axis that matches neither is EVENTS: it cannot be a panel stack
    check("a leading axis matching neither the panel count nor 1 is treated as events",
          np.array_equal(_load_image(pfile, DATA, event=1, n_panels=7), panel_stack[1]))

    # --- THE COINCIDENCE. n_events == n_panels ---------------------------------------------
    # Matching the leading axis against the panel count does not DISAMBIGUATE it: a 4-event .cxi
    # under a 4-panel geometry matches too, and reading it as panels puts every frame in the run
    # back on slab 0 -- glint#136 exactly, reintroduced by the fix for it. The file is asked
    # instead, and where it does not answer the ambiguity is refused.
    ev_file = os.path.join(d, "coincidence_events.cxi")
    with h5py.File(ev_file, "w") as f:
        f.create_dataset(DATA, data=panel_stack)                       # same 4 slabs...
        f.create_dataset("/entry_1/result_1/nPeaks", data=np.array([12, 9, 31, 4]))  # ...+ 4 events
    got_ev = _load_image(ev_file, DATA, event=2, n_panels=NP)
    check("n_events == n_panels + per-event metadata -> EVENTS (event=2 gives slab 2)",
          np.array_equal(got_ev, panel_stack[2]), float(got_ev.flat[0]))
    check("...which is exactly the frame the panel reading would have got WRONG",
          not np.array_equal(got_ev, panel_stack[0]))

    raised2 = None
    try:
        _load_image(pfile, DATA, event=1, n_panels=NP)     # same shape, NO per-event metadata
    except Exception as exc:                      # noqa: BLE001 - the message is the point
        raised2 = exc
    check("n_events == n_panels and the file says nothing -> RAISES rather than picking",
          isinstance(raised2, ValueError) and "event_axis" in str(raised2), repr(raised2))
    check("...and the message names BOTH readings so the user can choose",
          raised2 is not None and "EVENTS" in str(raised2) and "PANELS" in str(raised2),
          str(raised2)[:120])

    # metadata that DISAGREES with the leading axis says the axis is not events
    mism = os.path.join(d, "panels_with_one_event.h5")
    with h5py.File(mism, "w") as f:
        f.create_dataset(DATA, data=panel_stack)
        f.create_dataset("/LCLS/eventNumber", data=np.array([7]))      # one event, four slabs
    raised_m = None
    try:
        _load_image(mism, DATA, event=3, n_panels=NP)
    except Exception as exc:                      # noqa: BLE001
        raised_m = exc
    check("per-event metadata of a DIFFERENT length -> panels -> refused under a multi-panel geom",
          isinstance(raised_m, NotImplementedError), repr(raised_m))

    # the override beats the metadata, in both directions
    raised_o = None
    try:
        _load_image(ev_file, DATA, event=2, n_panels=NP, event_axis=False)
    except Exception as exc:                      # noqa: BLE001
        raised_o = exc
    check("event_axis=False overrides metadata that says events -- and then REFUSES multi-panel",
          isinstance(raised_o, NotImplementedError), repr(raised_o))
    check("event_axis=True overrides metadata that says panels",
          np.array_equal(_load_image(mism, DATA, event=3, n_panels=NP, event_axis=True),
                         panel_stack[3]))

    # the explicit override wins over the inference, in both directions
    check("event_axis=True forces event indexing under a multi-panel geometry",
          np.array_equal(_load_image(pfile, DATA, event=3, n_panels=NP, event_axis=True),
                         panel_stack[3]))
    check("event_axis=False forces slab 0 under a one-panel geometry",
          np.array_equal(_load_image(pfile, DATA, event=3, n_panels=1, event_axis=False),
                         panel_stack[0]))

    # --- WHY panel mode is refused: the measurement behind it (glint#148) -------------------
    # predict_spots emits ASSEMBLED fs/ss -- project_q adds each panel's min_fs/min_ss -- while
    # integrate_spots takes one assembled 2-D image. A single slab is not that canvas, so a real
    # spot on panel 1 integrates to exactly zero. This is what the pre-#136 loader did on every
    # multi-panel geometry, silently, and it is why "preserve the legacy path" was the wrong goal.
    g2 = "\n".join([f"photon_energy = {12398.419843320026 / LAM:.2f}", "clen = 0.1", "res = 10000",
                    "coffset = 0.0", f"data = {DATA}"]
                   + [f"q{p}/min_fs = 0\nq{p}/max_fs = 59\nq{p}/min_ss = {60*p}\n"
                      f"q{p}/max_ss = {60*p+59}\nq{p}/corner_x = -30\nq{p}/corner_y = {-30+60*p}\n"
                      f"q{p}/fs = +1.0x +0.0y\nq{p}/ss = +0.0x +1.0y" for p in range(2)]) + "\n"
    g2path = os.path.join(d, "two_panel.geom"); open(g2path, "w").write(g2)
    panels2, clen2 = panels_from_geom(parse_geom(g2path))
    pred2 = predict_spots(LYSO, panels2, clen2, LAM, dmin=3.0, tol=0.02)
    on1 = pred2[pred2["panel"] == 1]
    check("the 2-panel geometry predicts reflections onto panel 1", len(on1) > 0, len(on1))
    if len(on1):
        r1 = on1[len(on1) // 2]
        lf, ls = int(round(r1["fs"])), int(round(r1["ss"])) - 60          # panel-local
        slab = np.zeros((2, 60, 60), np.float32)
        slab[1, ls - 1:ls + 2, lf - 1:lf + 2] += 500.0                    # 4500 counts, on panel 1
        Iz, sz, pz, _ = integrate_spots(slab[0], np.asarray([r1]))
        print(f"  planted 4500 counts on panel 1; integrating slab 0 gives "
              f"I={Iz[0]:.1f} sigma={sz[0]:.1f} peak={pz[0]:.1f}")
        check("a real panel-1 spot integrates to EXACTLY ZERO off slab 0 (the legacy defect)",
              Iz[0] == 0.0 and pz[0] == 0.0, (float(Iz[0]), float(pz[0])))
        # ...and the same spot integrates correctly once the frame really is assembled
        canvas = np.zeros((120, 60), np.float32)
        canvas[60 + ls - 1:60 + ls + 2, lf - 1:lf + 2] += 500.0
        Ia, sa, pa, _ = integrate_spots(canvas, np.asarray([r1]))
        check("...while an ASSEMBLED frame recovers it (so the refusal is the honest answer)",
              Ia[0] > 4000, float(Ia[0]))

    two_slab = os.path.join(d, "two_panel_stack.h5")
    with h5py.File(two_slab, "w") as f:
        f.create_dataset(DATA, data=np.zeros((2, 60, 60), np.float32))
        f.create_dataset("/LCLS/eventNumber", data=np.array([1]))         # says: panels, not events
    raised_t = None
    try:
        _load_image(two_slab, DATA, event=0, n_panels=2)
    except Exception as exc:                      # noqa: BLE001
        raised_t = exc
    check("a 2-panel stack with a reflection on panel 1 is REFUSED, never zero-integrated",
          isinstance(raised_t, NotImplementedError), repr(raised_t))

# --- the override has to be REACHABLE, or the raise above is a dead end ---------------------
# _load_image now refuses an ambiguous stack. That is only an improvement if the user can then say
# which reading they meant WITHOUT editing Python, so the flag has to exist on the route that can
# hit it (--peaks + --integrate) and the task model that drives it.
import pathlib                                                       # noqa: E402
ROOT = pathlib.Path(__file__).resolve().parent.parent
cli = (ROOT / "glint/glint_cli.py").read_text()
check("glint_cli exposes --event-axis and forwards it to integrate_frames",
      "--event-axis" in cli and "event_axis=" in cli
      and all(f'"{v}"' in cli for v in ("auto", "event", "panel")))
check("the LUTE task model has an event_axis field rendering as --event-axis",
      "event_axis:" in (ROOT / "lute/glint_index.py").read_text()
      and 'rename_param="event-axis"' in (ROOT / "lute/glint_index.py").read_text())
# ...and it must NOT be in the launcher's xtc whitelist: glint_xtc.py takes frames from psana, has
# no such flag, and passing it there would be an argparse error rather than a no-op.
launch = (ROOT / "lute/glint_launch.sh").read_text()
whitelist = launch.split("XTC_FLAGS=", 1)[1].split('"')[1]
check("--event-axis is NOT whitelisted for the xtc route (glint_xtc has no such flag)",
      "--event-axis" not in whitelist and "--event-axis" in launch, whitelist[:60])

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
