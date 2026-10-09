"""A GLOBAL `coffset` in the .geom must put the detector at z = clen + coffset on the --images route,
not z = clen + 2*coffset.

THE DEFECT. CrystFEL's coffset is "the offset of the panel along the z-direction from the position
given by clen" (geometry(5)); a top-level value is the default for the panels that follow, so it is
applied once. `glint.geom.parse_geom` (the --peaks route) does exactly that: panels inherit the
global coffset and z = clen + panel coffset. `glint.lute_bridge.parse_geom` (the --images route) also
lets every panel inherit it, but `frames_from_cxi` (indexing: the q of each peak, and the known-cell
ring mask) and `predict.integrate_cxi` (prediction for --integrate) then added the global coffset to
clen AGAIN before `peaks_to_q` / `predict_spots` added the panel's own: z = clen + 2*coffset. Both sides of the
round trip were off by the same amount, so indexing was self-consistent and nothing failed, but every
q and every predicted (fs, ss) belonged to a detector one coffset further away than the .geom says,
and the --peaks and --images routes disagreed about the same file. A per-panel coffset alone was
never affected (the global one is 0), which is why the measured data sets (mfx100848724 r51,
cxil1015922 r0033, mfx101343025 r194: no global coffset) were not.

WHAT IS PINNED, for three geometries -- a global coffset only, a per-panel coffset only (the control
the old code already got right), and both (the panel's own value wins) -- plus a global coffset with
the clen read per event from the .cxi in mm, the LCLS layout:
  * integrate_cxi's predicted (fs, ss) equal an independent forward model on the plane
    z = clen + coffset (written here, not glint's project_q);
  * frames_from_cxi's q for stored peaks equal an independent pixel -> q model on that plane, and its
    known-cell ring mask (ring_focus) is built on that plane;
  * the --images route agrees with the --peaks route on the same .geom (geom.peaks_to_q,
    predict.panels_from_geom + predict_spots);
  * integrate_cxi still reports clen WITHOUT coffset for the stream chunk header.
The geometry is chosen so the two planes are told apart: the check below insists the forward model
moves the predicted spots by more than half a pixel between z = clen + coffset and clen + 2*coffset.

The pre-fix tree fails 13 checks here: all five model, ring-mask and cross-route checks on the
global-only geometry and on the global + per-panel one, and the three model and ring-mask checks with
the per-event clen. The per-panel-only control passes there too, as it should.

  PYTHONPATH=. python experiments/test_global_coffset.py     # exit 0 = all pass
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
    print("SKIP: h5py not available (needed to build a synthetic .cxi)")
    sys.exit(0)

import glint.ring_mask as ring_mask
from glint.geom import parse_geom as geom_parse_geom, peaks_to_q as geom_peaks_to_q
from glint.lattice import cell_to_Ar
from glint.lute_bridge import frames_from_cxi
from glint.predict import integrate_cxi, panels_from_geom, predict_spots, recip_from_M

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


EV_TEXT = f"{12398.419843320026 / 1.322:.6f}"     # photon_energy as the .geom writes it
LAM = 12398.419843320026 / float(EV_TEXT)          # ...and the wavelength glint reads back from it
NPX = 700
RES = 10000.0                            # px per metre (100 um pixels)
CX = CY = -NPX / 2.0                     # beam on the panel centre
CLEN = 0.1
COFF = 0.0005                            # 0.5 mm: ~1.5 px at the panel edge
DMIN = 2.5
DATA = "/entry_1/data_1/data"
RL = "/entry_1/result_1"
CLEN_PATH = "/LCLS/detector_1/EncoderValue"


def _rot(ax, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    i, j = [(1, 2), (0, 2), (0, 1)][ax]
    R = np.eye(3); R[i, i] = R[j, j] = c; R[i, j] = -s; R[j, i] = s
    return R


# a lysozyme cell in a general orientation, so spots land all over the panel
M0 = _rot(2, 23.0) @ _rot(0, 37.0) @ _rot(1, 11.0) @ cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)

# stored peaks for the q check: a spread of (fs, ss) away from the beam centre
PEAKS = np.array([[40.5, 60.0], [650.0, 30.25], [20.0, 680.0], [690.0, 690.0], [350.0, 10.0],
                  [10.0, 350.0], [600.0, 400.0], [120.75, 500.5], [480.0, 140.0], [300.0, 620.0]])


def hand_project(q, z):
    """Independent forward model: q (1/A) -> (fs, ss) on the flat panel at lab z (m), fs = +x, ss = +y."""
    s = LAM * np.asarray(q, float) + np.array([0.0, 0.0, 1.0])
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    X = s * (z / s[:, 2])[:, None]
    return RES * X[:, 0] - CX, RES * X[:, 1] - CY


def hand_q(fs, ss, z):
    """Independent pixel -> q model: the pixel at (fs, ss) on the panel at lab z (m)."""
    r = np.stack([(CX + fs) / RES, (CY + ss) / RES, np.full(len(fs), z)], 1)
    s = r / np.linalg.norm(r, axis=1, keepdims=True)
    return (s - np.array([0.0, 0.0, 1.0])) / LAM


def write_geom(path, glob_coff=None, panel_coff=None, clen=CLEN):
    lines = [f"photon_energy = {EV_TEXT}", f"clen = {clen}", f"res = {RES:g}", f"data = {DATA}",
             f"peak_list = {RL}"]
    if glob_coff is not None:
        lines.append(f"coffset = {glob_coff}")
    lines += ["", f"p0/min_fs = 0", f"p0/max_fs = {NPX - 1}", f"p0/min_ss = 0", f"p0/max_ss = {NPX - 1}",
              f"p0/corner_x = {CX}", f"p0/corner_y = {CY}", "p0/fs = +1.0x +0.0y", "p0/ss = +0.0x +1.0y"]
    if panel_coff is not None:
        lines.append(f"p0/coffset = {panel_coff}")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def write_cxi(path, clen_mm=None):
    """One event: a flat Poisson background (so every box has a sigma) and the stored PEAKS."""
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as f:
        f.create_dataset(DATA, data=rng.poisson(20.0, (1, NPX, NPX)).astype(np.float32))
        x = np.zeros((1, 64)); y = np.zeros((1, 64))
        x[0, :len(PEAKS)] = PEAKS[:, 0]; y[0, :len(PEAKS)] = PEAKS[:, 1]
        f.create_dataset(RL + "/peakXPosRaw", data=x)
        f.create_dataset(RL + "/peakYPosRaw", data=y)
        f.create_dataset(RL + "/nPeaks", data=np.array([len(PEAKS)]))
        if clen_mm is not None:
            f.create_dataset(CLEN_PATH, data=np.array([clen_mm]))
    return path


def run_case(d, label, z_true, glob_coff=None, panel_coff=None, clen_from_h5=False):
    print(f"\n[{label}] z = clen + coffset = {z_true:.6f} m")
    gpath = write_geom(os.path.join(d, label + ".geom"), glob_coff, panel_coff,
                       clen=CLEN_PATH if clen_from_h5 else CLEN)
    cxi = write_cxi(os.path.join(d, label + ".cxi"), clen_mm=CLEN * 1000.0 if clen_from_h5 else None)

    # --- prediction: integrate_cxi (the --images --integrate route) --------------------------------
    r = {"image": cxi, "event": 0, "M": M0.copy()}
    n, tot = integrate_cxi([r], gpath, dmin=DMIN, tol=0.006)
    pred = r.get("pred")
    check("integrate_cxi integrated the frame and kept >= 50 reflections", n == 1 and tot >= 50, (n, tot))
    if pred is None or len(pred) == 0:
        return
    hkl = np.stack([pred["h"], pred["k"], pred["l"]], 1).astype(float)
    q = hkl @ recip_from_M(r["M"])                       # r["M"] is the canonical M it predicted with
    fs_t, ss_t = hand_project(q, z_true)
    dev = max(np.abs(pred["fs"] - fs_t).max(), np.abs(pred["ss"] - ss_t).max())
    check("predicted (fs, ss) = independent forward model at z = clen + coffset (< 1e-6 px)",
          dev < 1e-6, f"max |d| = {dev:.4f} px")
    if glob_coff:
        fs_2, ss_2 = hand_project(q, z_true + glob_coff)
        sep = max(np.abs(fs_2 - fs_t).max(), np.abs(ss_2 - ss_t).max())
        check("the geometry tells z = clen + coffset from z + global coffset apart (> 0.5 px)",
              sep > 0.5, f"{sep:.3f} px")
        dev2 = max(np.abs(pred["fs"] - fs_2).max(), np.abs(pred["ss"] - ss_2).max())
        print(f"        (prediction vs the double-counted plane z + {glob_coff:g}: max |d| = {dev2:.4f} px)")
    check("chunk-header clen_m is the .geom clen WITHOUT coffset",
          abs(r.get("clen_m", np.nan) - CLEN) < 1e-12, r.get("clen_m"))

    # --- indexing: frames_from_cxi stored peaks -> q (the --images front end) --------------------
    frames, _ = frames_from_cxi(cxi, gpath, peakfinder="stored", min_peaks=6)
    qi = frames[0]
    qt = hand_q(PEAKS[:, 0], PEAKS[:, 1], z_true)
    dq = np.abs(qi - qt).max() if qi.shape == qt.shape else np.inf
    check("frames_from_cxi q = independent pixel -> q model at z = clen + coffset (< 1e-9 1/A)",
          dq < 1e-9, f"max |dq| = {dq:.3e} 1/A, shape {qi.shape}")

    # --- the known-cell ring mask (frames_from_cxi ring_focus) is built on that plane too ----------
    # ring_qmask maps every pixel through peaks_to_q, so the panels add their coffset: clen_m must not.
    seen = []
    real = ring_mask.ring_qmask

    def spy(panels, clen_m, *a, **kw):
        seen.append(max(abs(clen_m + p["coffset"] - z_true) for p in panels))
        return real(panels, clen_m, *a, **kw)
    ring_mask.ring_qmask = spy
    try:
        frames_from_cxi(cxi, gpath, min_peaks=0, ring_focus=([79.02, 79.02, 37.98, 90, 90, 90], 0.3))
    finally:
        ring_mask.ring_qmask = real
    check("ring_focus mask is built at z = clen + coffset (< 1e-12 m)",
          bool(seen) and max(seen) < 1e-12, seen)

    if clen_from_h5:
        return                                       # the --peaks route has no file to read clen from
    # --- the two routes on the same .geom ----------------------------------------------------------
    gd = geom_parse_geom(gpath)
    qp = geom_peaks_to_q(PEAKS, gd)
    dqr = np.abs(qi - qp).max() if qi.shape == qp.shape else np.inf
    check("--images q = --peaks q (geom.peaks_to_q) for the same peaks (< 1e-9 1/A)", dqr < 1e-9,
          f"max |dq| = {dqr:.3e} 1/A")
    pan, clen = panels_from_geom(gd)
    pp = predict_spots(r["M"], pan, clen, LAM, dmin=DMIN, tol=0.006)
    key = {(int(a), int(b), int(c)): (fs, ss) for a, b, c, fs, ss in
           zip(pp["h"], pp["k"], pp["l"], pp["fs"], pp["ss"])}
    devr = max(max(abs(key[(int(a), int(b), int(c))][0] - fs), abs(key[(int(a), int(b), int(c))][1] - ss))
               if (int(a), int(b), int(c)) in key else np.inf
               for a, b, c, fs, ss in zip(pred["h"], pred["k"], pred["l"], pred["fs"], pred["ss"]))
    check("--images prediction = --peaks-route prediction (panels_from_geom + predict_spots, < 1e-6 px)",
          devr < 1e-6, f"max |d| = {devr:.4f} px")


with tempfile.TemporaryDirectory() as d:
    run_case(d, "global", CLEN + COFF, glob_coff=COFF)
    run_case(d, "panel", CLEN + COFF, panel_coff=COFF)
    run_case(d, "both", CLEN - 0.0003, glob_coff=COFF, panel_coff=-0.0003)
    run_case(d, "global_h5clen", CLEN + COFF, glob_coff=COFF, clen_from_h5=True)

print()
if FAILS:
    print(f"FAILED {len(FAILS)} check(s):")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL PASS")
