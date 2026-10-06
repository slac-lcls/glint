"""The --images peak finder must read an un-assembled panel stack as ONE event (glint#148).

WHY THIS EXISTS. `lute_bridge.frames_from_cxi` (the self peak-find half of `glint --images`, default
`--peakfinder v4`) read ``data[i]`` of any 3-D dataset as event ``i``. On an un-assembled
``(panel, ss, fs)`` CXI that is one "event" per PANEL, and every one of them was mapped through
the whole panel list with first-match-wins -- i.e. through panel 0's corner and basis, since the
slab-local windows all overlap. Measured on a 2-panel file holding one event: 2 frames labelled
events 0 and 1, and panel 1's 8 peaks came back with the sign of q_x flipped. A 4-D
``(event, panel, ss, fs)`` file crashed inside the finder with ``too many values to unpack``.
The --integrate half of the same route (`integrate_cxi`) already refused such a file by name, so
an index-only run was the silently wrong one. Now frames_from_cxi makes the SAME layout decision
(`predict._leading_axis_is_events`, slab mapping from integer ``dimN`` keys) and peak-finds each
slab on its own, mapping its peaks through that slab's panels only.

What is pinned:
  * a (panel, ss, fs) file holding one event -> ONE frame, event 0, whose q are the per-panel truth
    (each slab peak-found alone, mapped through its own panel) -- also with no per-event metadata,
    and through glint_cli._load_frames as the CLI calls it;
  * a 4-D (event, panel, ss, fs) file under the CrystFEL 4-D geometry -> one frame per event, each
    its own event's per-panel truth, with v4 and pf9;
  * a .lst of two such files -> two frames (not four), and the .lst recursion forwards ring_focus;
  * the per-slab mask (``m[s]``, not panel 0's mask for every panel), the per-slab ring mask, and
    top_n taken over the whole event;
  * refusals by name instead of wrong q: a panel stack with no slab mapping; the ambiguous case
    (leading axis == panel count, no metadata), whose named override --event-axis now reaches this
    route; an assembled reading under a multi-slab geometry; stored (x, y) peaks under a multi-slab
    geometry; a 4-D file read as panels; a slab the data does not have;
  * UNCHANGED: an assembled (event, ss, fs) stack, a 2-D frame and a one-panel stack still give
    exactly what they gave before.

  PYTHONPATH=. python experiments/test_images_unassembled_cxi.py     # exit 0 = all pass
"""
from __future__ import annotations

import argparse
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

from glint.lute_bridge import frames_from_cxi, parse_geom, peaks_to_q
from glint.peakfinder_v4 import peakfinder_v4
from glint.peakfinder9 import peakfinder9
from glint.ring_mask import ring_qmask

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)[:300]}")
    if not ok:
        FAILS.append(name)


def attempt(fn):
    try:
        return fn(), None
    except Exception as exc:             # noqa: BLE001 - the exception IS the observation
        return None, exc


P, S, F = 2, 64, 64
WL, CLEN = 1.3, 0.1
CX = [-70.0, 6.0]                        # panel 0 left of the beam, panel 1 right of it
DATA = "/entry_1/data_1/data"
MASK = "/entry_1/data_1/mask"
CELL6 = [79.1, 79.1, 38.0, 90.0, 90.0, 90.0]
SPOTS = {0: [(10, 12), (30, 40), (50, 20), (20, 55), (45, 50), (35, 10), (55, 55)],
         1: [(8, 30), (25, 18), (40, 44), (52, 8), (15, 50), (58, 36), (33, 58), (47, 25)]}
rng = np.random.default_rng(12345)
yy, xx = np.mgrid[0:S, 0:F]


def one_event(shift=0):
    a = rng.poisson(5.0, (P, S, F)).astype(np.float32)
    for p in range(P):
        for (y, x) in SPOTS[p]:
            a[p] += 300 * np.exp(-((yy - y - shift) ** 2 + (xx - x) ** 2) / 2.0)
    return a


def geom_text(kind, mapped=True, mask=False):
    """kind '3d': (panel, ss, fs) with pN/dim0 = N; '4d': dim0 = % and pN/dim1 = N; 'canvas': the 2-D
    slab layout (panel p at ss rows p*S..p*S+S-1, no dimN keys) -- the standard Cheetah layout."""
    g = f"clen = {CLEN}\nphoton_energy = 9500\nres = 10000\ndata = {DATA}\n"
    if mask:
        g += f"mask = {MASK}\nmask_good = 0\n"
    if kind == "4d":
        g += "dim0 = %\n"
    for p, cx in enumerate(CX):
        if mapped and kind == "3d":
            g += f"p{p}/dim0 = {p}\np{p}/dim1 = ss\np{p}/dim2 = fs\n"
        elif mapped and kind == "4d":
            g += f"p{p}/dim1 = {p}\np{p}/dim2 = ss\np{p}/dim3 = fs\n"
        s0 = p * S if kind == "canvas" else 0
        g += (f"p{p}/min_fs = 0\np{p}/max_fs = {F-1}\np{p}/min_ss = {s0}\np{p}/max_ss = {s0+S-1}\n"
              f"p{p}/fs = +1.0x +0.0y\np{p}/ss = +0.0x +1.0y\n"
              f"p{p}/corner_x = {cx}\np{p}/corner_y = -32.0\n")
    return g


def truth(ev, panels, finder=peakfinder_v4, masks=None, top_n=0):
    """Each slab peak-found ALONE and mapped through ITS OWN panel, concatenated in slab order."""
    xs, ys, ws, ks = [], [], [], []
    for p in range(P):
        pk = finder(ev[p].astype(np.float32), mask=None if masks is None else masks[p])
        x = np.asarray(pk["x"], float)
        xs.append(x); ys.append(np.asarray(pk["y"], float))
        ws.append(np.asarray(pk["intensity"], float)); ks.append(np.full(len(x), p))
    x, y, w, k = (np.concatenate(a) for a in (xs, ys, ws, ks))
    if top_n and len(x) > top_n:
        keep = np.argsort(w)[::-1][:top_n]; x, y, k = x[keep], y[keep], k[keep]
    q = np.empty((len(x), 3))
    for p in range(P):
        r = k == p
        if r.any():
            q[r] = peaks_to_q(x[r], y[r], [panels[p]], CLEN, WL)
    return q


def same(a, b):
    return a is not None and b is not None and a.shape == b.shape and np.allclose(a, b, atol=1e-12)


def write(path, data, n_events=None, mask=None, stored=None):
    with h5py.File(path, "w") as f:
        f[DATA] = data
        if n_events is not None:
            f["/entry_1/result_1/nPeaks"] = np.full(n_events, 15, np.int32)
        if mask is not None:
            f[MASK] = mask
        if stored is not None:
            x, y = stored
            f["/entry_1/result_1/peakXPosRaw"] = np.asarray([x], np.float32)
            f["/entry_1/result_1/peakYPosRaw"] = np.asarray([y], np.float32)
            f["/entry_1/result_1/nPeaks"] = np.asarray([len(x)], np.int32)
    return path


with tempfile.TemporaryDirectory() as d:
    def gfile(name, *a, **k):
        p = os.path.join(d, name)
        open(p, "w").write(geom_text(*a, **k))
        return p

    g3, g4, gc = gfile("stack3.geom", "3d"), gfile("stack4.geom", "4d"), gfile("canvas.geom", "canvas")
    g3_nomap = gfile("nomap.geom", "3d", mapped=False)
    g3_mask = gfile("stack3_mask.geom", "3d", mask=True)
    pan3, _ = parse_geom(g3)
    pan4, _ = parse_geom(g4)

    evA = one_event()
    cA = write(os.path.join(d, "one_event_panels.cxi"), evA, n_events=1)
    qA = truth(evA, pan3)

    # ---- THE DEFECT: one (panel, ss, fs) event --------------------------------------------------
    print("one event, (panel, ss, fs), slab-mapped .geom (pN/dim0 = N)")
    got, exc = attempt(lambda: frames_from_cxi(cA, g3, wavelength_A=WL, min_peaks=6))
    fr, im = got if got else ([], [])
    check("one (2,64,64) panel-stack event comes back as ONE frame, labelled event 0",
          exc is None and len(fr) == 1 and [i["event"] for i in im] == [0],
          exc or (len(fr), [i["event"] for i in im]))
    check("...holding BOTH panels' peaks, each mapped through its own panel (= per-panel truth)",
          len(fr) == 1 and same(fr[0], qA), exc or [len(q) for q in fr])
    if len(fr) == 1 and same(fr[0], qA):
        n0 = len(peakfinder_v4(evA[0])["x"])
        check("...panel 1's peaks land RIGHT of the beam (q_x > 0) as its corner says",
              bool(np.all(fr[0][n0:, 0] > 0)) and bool(np.all(fr[0][:n0, 0] < 0)),
              np.sign(fr[0][:, 0]).tolist())

    args = argparse.Namespace(qframes=None, images=cA, geom=g3, wavelength=WL, N=0, min_peaks=6,
                              data_path=None, peakfinder="v4", top_peaks=0, ring_focus=False,
                              cell=None, ring_qlow=0.15)          # no event_axis attr: older callers
    from glint import glint_cli
    got, exc = attempt(lambda: glint_cli._load_frames(args))
    check("glint_cli._load_frames (--images, default v4) gives the same one frame",
          exc is None and len(got[0]) == 1 and same(got[0][0], qA), exc or [len(q) for q in got[0]])

    cA_nometa = write(os.path.join(d, "one_event_nometa.cxi"), evA)
    got, exc = attempt(lambda: frames_from_cxi(cA_nometa, g3, wavelength_A=WL, min_peaks=6))
    check("...also with NO per-event metadata (leading axis == panel count): the dimN keys decide",
          exc is None and len(got[0]) == 1 and same(got[0][0], qA), exc or [len(q) for q in got[0]])

    # A peak on no panel of its slab: panel 1's window covers only fs 0..31 of slab 1, so slab 1's
    # peaks at fs >= 32 map to NaN rows. They are dropped once per event, as on every other route
    # (glint.geom.clean_q, review s7-05), and reported; the rest keep their order and their panel.
    import contextlib
    import io
    from glint.geom import clean_q
    g3_half = gfile("stack3_half.geom", "3d")
    open(g3_half, "w").write(geom_text("3d").replace(f"p1/max_fs = {F-1}", "p1/max_fs = 31"))
    pan3h, _ = parse_geom(g3_half)
    raw = truth(evA, pan3h)
    n_off = int((~np.isfinite(raw).all(1)).sum())
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        got, exc = attempt(lambda: frames_from_cxi(cA, g3_half, wavelength_A=WL, min_peaks=6))
    check("a panel-stack peak on no panel of its slab is dropped, the rest stay aligned with their panels",
          n_off > 0 and exc is None and len(got[0]) == 1 and same(got[0][0], clean_q(raw))
          and len(got[0][0]) == len(raw) - n_off,
          exc or (n_off, [len(q) for q in got[0]]))
    check("...and the drop is reported once for the event",
          f"dropped {n_off} peak(s) in 1 frame(s)" in err.getvalue(), err.getvalue().strip()[:200])

    # ---- 4-D (event, panel, ss, fs) --------------------------------------------------------------
    print("4-D (event, panel, ss, fs), CrystFEL 4-D .geom (dim0 = %, pN/dim1 = N)")
    ev2 = [one_event(), one_event(shift=2)]
    c4 = write(os.path.join(d, "events_panels.cxi"), np.stack(ev2), n_events=2)
    for name, finder in (("v4", peakfinder_v4), ("pf9", peakfinder9)):
        got, exc = attempt(lambda: frames_from_cxi(c4, g4, wavelength_A=WL, min_peaks=6, peakfinder=name))
        fr, im = got if got else ([], [])
        check(f"[{name}] 2 events -> 2 frames, events [0, 1] (was: ValueError 'too many values to unpack')",
              exc is None and len(fr) == 2 and [i["event"] for i in im] == [0, 1],
              repr(exc) if exc else (len(fr), [i["event"] for i in im]))
        check(f"[{name}] ...each its OWN event's per-panel truth",
              len(fr) == 2 and all(same(fr[e], truth(ev2[e], pan4, finder)) for e in range(2)),
              exc or [len(q) for q in fr])

    # ---- .lst: two one-event files, and ring_focus forwarding ------------------------------------
    print(".lst of two one-event panel-stack files")
    cB = write(os.path.join(d, "second_event.cxi"), one_event(shift=1), n_events=1)
    lst = os.path.join(d, "two.lst")
    open(lst, "w").write(f"{cA}\n{cB}\n")
    got, exc = attempt(lambda: frames_from_cxi(lst, g3, wavelength_A=WL, min_peaks=6))
    check(".lst of 2 single-event files -> 2 frames (was 4)",
          exc is None and len(got[0]) == 2 and [i["image"] for i in got[1]] == [cA, cB],
          exc or len(got[0]))
    # ring_focus through a .lst: an ASSEMBLED (canvas) file, so this isolates the forwarding
    evC = one_event()
    cC1 = write(os.path.join(d, "canvas1.cxi"), evC.reshape(1, P * S, F), n_events=1)
    cC2 = write(os.path.join(d, "canvas2.cxi"), one_event(shift=1).reshape(1, P * S, F), n_events=1)
    lstC = os.path.join(d, "canvas.lst")
    open(lstC, "w").write(f"{cC1}\n{cC2}\n")
    rf = (CELL6, 0.15)
    per_file = [frames_from_cxi(c, gc, wavelength_A=WL, min_peaks=0, ring_focus=rf)[0][0] for c in (cC1, cC2)]
    no_rf = [frames_from_cxi(c, gc, wavelength_A=WL, min_peaks=0)[0][0] for c in (cC1, cC2)]
    check("(precondition) ring_focus changes this file's peak list, so the next check can fail",
          any(a.shape != b.shape for a, b in zip(per_file, no_rf)), [len(q) for q in per_file + no_rf])
    got, exc = attempt(lambda: frames_from_cxi(lstC, gc, wavelength_A=WL, min_peaks=0, ring_focus=rf))
    check("a .lst forwards ring_focus to each file (it was dropped in the recursion)",
          exc is None and len(got[0]) == 2 and all(same(a, b) for a, b in zip(got[0], per_file)),
          exc or ([len(q) for q in got[0]], [len(q) for q in per_file]))

    # ---- masks, ring masks, top_n: per slab -----------------------------------------------------
    print("per-slab mask / ring mask / top_n")
    m = np.zeros((P, S, F), np.uint16)
    m[1] = 1                                                   # slab 1 all bad (mask_good = 0)
    cM1 = write(os.path.join(d, "mask_slab1.cxi"), evA, n_events=1, mask=m)
    m0 = np.zeros((P, S, F), np.uint16)
    m0[0] = 1                                                  # slab 0 all bad
    cM0 = write(os.path.join(d, "mask_slab0.cxi"), evA, n_events=1, mask=m0)
    goodm = [np.ones((S, F), bool), np.zeros((S, F), bool)]
    got1, e1 = attempt(lambda: frames_from_cxi(cM1, g3_mask, wavelength_A=WL, min_peaks=0))
    got0, e0 = attempt(lambda: frames_from_cxi(cM0, g3_mask, wavelength_A=WL, min_peaks=0))
    check("a (slab, ss, fs) mask masks ITS slab: slab 1 bad -> only panel 0's truth survives",
          e1 is None and len(got1[0]) == 1 and same(got1[0][0], truth(evA, pan3, masks=goodm)),
          e1 or [len(q) for q in got1[0]])
    check("...slab 0 bad -> only panel 1's truth survives (was: slab 0's mask used for every panel)",
          e0 is None and len(got0[0]) == 1 and same(got0[0][0], truth(evA, pan3, masks=goodm[::-1])),
          e0 or [len(q) for q in got0[0]])

    rms = [ring_qmask([pan3[p]], CLEN, WL, CELL6, (S, F), qlow=0.15) for p in range(P)]
    rm_one = ring_qmask(pan3, CLEN, WL, CELL6, (S, F), qlow=0.15)    # one canvas for overlapping windows
    check("(precondition) per-slab ring masks give a different peak list than one shared canvas",
          not np.array_equal(rms[0], rms[1])
          and not same(truth(evA, pan3, masks=rms), truth(evA, pan3, masks=[rm_one, rm_one])),
          (len(truth(evA, pan3, masks=rms)), len(truth(evA, pan3, masks=[rm_one, rm_one]))))
    got, exc = attempt(lambda: frames_from_cxi(cA, g3, wavelength_A=WL, min_peaks=0, ring_focus=rf))
    check("ring_focus builds the ring mask PER SLAB from that slab's panels",
          exc is None and len(got[0]) == 1 and same(got[0][0], truth(evA, pan3, masks=rms)),
          exc or [len(q) for q in got[0]])
    got, exc = attempt(lambda: frames_from_cxi(cA, g3, wavelength_A=WL, min_peaks=6, top_n=10))
    check("top_n keeps the strongest 10 over the WHOLE event, each mapped through its own panel",
          exc is None and len(got[0]) == 1 and same(got[0][0], truth(evA, pan3, top_n=10)),
          exc or [len(q) for q in got[0]])

    # ---- refusals by name, not wrong q ----------------------------------------------------------
    print("refusals")
    got, exc = attempt(lambda: frames_from_cxi(cA, g3_nomap, wavelength_A=WL, min_peaks=6))
    check("panel stack (metadata: 1 event) under a 2-panel .geom with NO dimN mapping -> "
          "NotImplementedError naming glint#148 and the dimN fix",
          isinstance(exc, NotImplementedError) and "glint#148" in str(exc) and "dimN" in str(exc),
          repr(exc) if exc else f"returned {len(got[0])} frames")
    got, exc = attempt(lambda: frames_from_cxi(cA_nometa, g3_nomap, wavelength_A=WL, min_peaks=6))
    check("leading axis == panel count, no metadata, no mapping -> ValueError naming event_axis",
          isinstance(exc, ValueError) and "event_axis" in str(exc),
          repr(exc) if exc else f"returned {len(got[0])} frames")
    got, exc = attempt(lambda: frames_from_cxi(cA_nometa, g3_nomap, wavelength_A=WL, min_peaks=6,
                                               event_axis=True))
    check("...and event_axis=True is the way out: 2 frames read as events",
          exc is None and len(got[0]) == 2, exc or len(got[0]))
    cli = {ax: attempt(lambda ax=ax: glint_cli._load_frames(argparse.Namespace(
        **{**vars(args), "images": cA_nometa, "geom": g3_nomap, "event_axis": ax}))) for ax in ("auto", "event")}
    check("...reachable from the CLI: --event-axis auto refuses, --event-axis event reads 2 events",
          isinstance(cli["auto"][1], ValueError) and cli["event"][1] is None and len(cli["event"][0][0]) == 2,
          {ax: (repr(e) if e else len(g[0])) for ax, (g, e) in cli.items()})
    got, exc = attempt(lambda: frames_from_cxi(cA, g3, wavelength_A=WL, min_peaks=6, event_axis=True))
    check("an assembled (event, ss, fs) reading under a 2-slab dimN .geom -> refused (windows overlap)",
          isinstance(exc, NotImplementedError) and "slab" in str(exc),
          repr(exc) if exc else f"returned {len(got[0])} frames")
    got, exc = attempt(lambda: frames_from_cxi(c4, g4, wavelength_A=WL, min_peaks=6, event_axis=False))
    check("a 4-D file with event_axis=False -> NotImplementedError",
          isinstance(exc, NotImplementedError), repr(exc) if exc else f"returned {len(got[0])} frames")
    g3_bad = os.path.join(d, "slab2.geom")
    open(g3_bad, "w").write(geom_text("3d").replace("p1/dim0 = 1", "p1/dim0 = 2"))
    got, exc = attempt(lambda: frames_from_cxi(cA, g3_bad, wavelength_A=WL, min_peaks=6))
    check("a .geom slab the data does not have -> ValueError naming the slab",
          isinstance(exc, ValueError) and "slab 2" in str(exc),
          repr(exc) if exc else f"returned {len(got[0])} frames")
    # stored peaks: slab 1's spots as bare (x, y), which name no slab under this .geom
    sx = np.array([x for (_, x) in SPOTS[1]], float); sy = np.array([y for (y, _) in SPOTS[1]], float)
    cS = write(os.path.join(d, "stored.cxi"), evA, stored=(sx, sy))
    got, exc = attempt(lambda: frames_from_cxi(cS, g3, wavelength_A=WL, min_peaks=6, peakfinder="stored"))
    detail = repr(exc) if exc else ""
    if exc is None and got[0] and len(got[0][0]):
        q_as_p0 = peaks_to_q(sx, sy, [pan3[0]], CLEN, WL)
        detail = f"returned q equal to the PANEL-0 mapping: {same(got[0][0], q_as_p0)}"
    check("stored (x, y) peaks under a 2-slab dimN .geom -> NotImplementedError (they name no slab)",
          isinstance(exc, NotImplementedError) and "glint#148" in str(exc), detail)

    # ---- UNCHANGED: assembled layouts read exactly as before ------------------------------------
    print("unchanged layouts")
    panc, _ = parse_geom(gc)
    evs = [one_event().reshape(P * S, F), one_event(shift=1).reshape(P * S, F)]
    cE = write(os.path.join(d, "assembled_events.cxi"), np.stack(evs), n_events=2)
    want = [peaks_to_q(np.asarray(pk["x"], float), np.asarray(pk["y"], float), panc, CLEN, WL)
            for pk in (peakfinder_v4(e.astype(np.float32)) for e in evs)]
    got, exc = attempt(lambda: frames_from_cxi(cE, gc, wavelength_A=WL, min_peaks=6))
    check("assembled (event, ss, fs) 2-event stack under the canvas .geom: 2 frames, each data[i]",
          exc is None and len(got[0]) == 2 and all(same(a, b) for a, b in zip(got[0], want)),
          exc or [len(q) for q in got[0]])
    c2d = write(os.path.join(d, "frame2d.cxi"), evs[0])
    got, exc = attempt(lambda: frames_from_cxi(c2d, gc, wavelength_A=WL, min_peaks=6))
    check("a plain 2-D dataset: 1 frame, as before",
          exc is None and len(got[0]) == 1 and same(got[0][0], want[0]), exc or [len(q) for q in got[0]])
    g1 = os.path.join(d, "one_panel.geom")
    open(g1, "w").write(f"clen = {CLEN}\nphoton_energy = 9500\nres = 10000\ndata = {DATA}\n"
                        f"p0/min_fs = 0\np0/max_fs = {F-1}\np0/min_ss = 0\np0/max_ss = {P*S-1}\n"
                        f"p0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\np0/corner_x = -32\np0/corner_y = -64\n")
    pan1, _ = parse_geom(g1)
    c1 = write(os.path.join(d, "one_panel_events.cxi"), np.stack(evs))   # no metadata
    want1 = [peaks_to_q(np.asarray(pk["x"], float), np.asarray(pk["y"], float), pan1, CLEN, WL)
             for pk in (peakfinder_v4(e.astype(np.float32)) for e in evs)]
    got, exc = attempt(lambda: frames_from_cxi(c1, g1, wavelength_A=WL, min_peaks=6))
    check("a one-panel .geom with a 2-frame stack and no metadata: still 2 events",
          exc is None and len(got[0]) == 2 and all(same(a, b) for a, b in zip(got[0], want1)),
          exc or [len(q) for q in got[0]])

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + "; ".join(FAILS)))
sys.exit(1 if FAILS else 0)
