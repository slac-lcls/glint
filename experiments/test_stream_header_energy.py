"""`--images --integrate` must write each frame's own photon energy and camera length in its chunk header.

WHY THIS EXISTS. In an LCLS .geom, `photon_energy` and `clen` are usually HDF5 paths
(`/LCLS/photon_energy_eV`, `/LCLS/detector_1/EncoderValue`), because both change from shot to shot.
`integrate_cxi` already read those per event and predicted with them. But it stored only
M/pred/I/sigma/peak/bg, so `write_stream_integrated` fell back to its run-level kwargs. The CLI fills
those from the .geom, and a path is not a float, so every chunk said `photon_energy_eV = 9392.70` and
`average_camera_length = 0.150000 m`. Those are placeholders from the nanoBragg simulation, not
values from the run. The reflection rows (hkl, I, sigma, fs/ss) were right; the header that a merger
reads lambda from was not. `--wavelength` did not reach the header either.

WHAT IT CHECKS (a 3-event stacked .cxi, each event at its own energy and encoder value):
  1. integrate_cxi on the path-valued .geom: each integrated result carries its own event's photon_eV
     and clen_m; the stream written with the CLI's placeholder kwargs states them per chunk; and the
     written hkl, re-projected at the energy and camera length the chunk header states, land on the
     written fs/ss.
  2. wavelength_A (the CLI's --wavelength), on the path-valued and on a literal .geom: the header
     states that wavelength's energy.
  3. Literal .geom, no override: the header values are the .geom's, and the stream is byte-identical
     to the one written from the same results with the new per-result keys removed (the old output).
  4. A record without the new keys (what StreamDriver writes) still gets the run-level kwargs.
  5. The CLI itself (glint.glint_cli.main, indexer stubbed so no torch is needed): the path-valued
     .geom gives per-event headers, and with --wavelength every chunk states that energy, including
     a chunk with no crystal.

SKIPS (exit 0) without h5py, which the CPU CI job installs.

  PYTHONPATH=. python experiments/test_stream_header_energy.py     # exit 0 = all pass
"""
import os
import re
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import h5py
except ImportError:                      # pragma: no cover - CI installs h5py; a bare tree may not
    print("SKIP: h5py not available (needed to build a stacked .cxi)")
    sys.exit(0)

from glint.lattice import cell_to_Ar
from glint.lute_bridge import parse_geom as lb_parse_geom
from glint.predict import integrate_cxi, predict_spots, project_q, write_stream_integrated

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


HC = 12398.419843320026
EVS = [12000.0, 12010.0, 11990.0]        # per-event photon energy, eV
ENC = [120.0, 121.0, 119.0]              # per-event EncoderValue, mm -> clen 0.120 / 0.121 / 0.119 m
CLENS = [e * 1e-3 for e in ENC]
PLACEHOLDER = (9392.7, 0.15)             # what the CLI passes when the .geom value is not a float
CELL = (79.02, 79.02, 37.98, 90.0, 90.0, 90.0)
NPX = 600
DATA = "/entry_1/data_1/data"
RL = "/entry_1/result_1"
DMIN, TOL = 3.0, 0.006

rng = np.random.default_rng(7)
q4 = rng.normal(size=4); q4 /= np.linalg.norm(q4)
w, x, y, z = q4
ROT = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
M0 = ROT @ cell_to_Ar(*CELL)

PANEL = (f"p0/min_fs = 0\np0/max_fs = {NPX - 1}\np0/min_ss = 0\np0/max_ss = {NPX - 1}\n"
         f"p0/corner_x = {-NPX // 2}\np0/corner_y = {-NPX // 2}\np0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n")
TAIL = f"res = 10000\ncoffset = 0.0\ndata = {DATA}\npeak_list = {RL}\n" + PANEL
GEOM_PATH = "photon_energy = /LCLS/photon_energy_eV\nclen = /LCLS/detector_1/EncoderValue\n" + TAIL
GEOM_LIT = f"photon_energy = {EVS[0]:.2f}\nclen = {CLENS[0]}\n" + TAIL


def chunk_headers(path):
    """[(photon_energy_eV, average_camera_length_m, has_crystal)] per chunk, in file order."""
    out = []
    for ch in open(path).read().split("----- Begin chunk -----")[1:]:
        ev = re.search(r"^photon_energy_eV = ([\d.]+)$", ch, re.M)
        cl = re.search(r"^average_camera_length = ([\d.]+) m$", ch, re.M)
        out.append((float(ev.group(1)) if ev else None, float(cl.group(1)) if cl else None,
                    "--- Begin crystal" in ch))
    return out


def reprojection_px(path, panels):
    """Per crystal chunk: max |(fs, ss) written - (fs, ss) of the written hkl projected at the chunk's
    STATED energy and camera length|. The rows are written at 0.1 px, so a consistent chunk is < 0.1."""
    out = []
    for ch in open(path).read().split("----- Begin chunk -----")[1:]:
        if "--- Begin crystal" not in ch:
            continue
        ev = float(re.search(r"^photon_energy_eV = ([\d.]+)$", ch, re.M).group(1))
        cl = float(re.search(r"^average_camera_length = ([\d.]+) m$", ch, re.M).group(1))
        R = np.array([[float(v) for v in re.search(rf"^{n} = (\S+) (\S+) (\S+) nm\^-1$", ch, re.M).groups()]
                      for n in ("astar", "bstar", "cstar")]) / 10.0       # nm^-1 -> 1/A, rows a*, b*, c*
        body = ch.split("Reflections measured after indexing\n", 1)[1].split("End of reflections", 1)[0]
        rows = [ln.split() for ln in body.splitlines()[1:] if ln.strip()]
        hkl = np.array([[int(t) for t in r[:3]] for r in rows], float)
        fs_w = np.array([float(r[7]) for r in rows]); ss_w = np.array([float(r[8]) for r in rows])
        fs_p, ss_p, _ = project_q(hkl @ R, panels, cl, HC / ev)
        d = np.hypot(fs_p - fs_w, ss_p - ss_w)
        out.append(float(np.nanmax(d)) if np.isfinite(d).any() else float("inf"))
    return out


with tempfile.TemporaryDirectory() as d:
    gpath = os.path.join(d, "path_valued.geom"); open(gpath, "w").write(GEOM_PATH)
    glit = os.path.join(d, "literal.geom"); open(glit, "w").write(GEOM_LIT)
    panels, _ = lb_parse_geom(gpath)

    # --- the stacked .cxi: Poisson background plus a spot at every reflection of each event's geometry
    frames, peaks = [], []
    yy, xx = np.mgrid[-3:4, -3:4]
    gk = np.exp(-(xx * xx + yy * yy) / 2.0); gk /= gk.sum()
    for ev, clen in zip(EVS, CLENS):
        pr = predict_spots(M0, panels, clen, HC / ev, dmin=DMIN, tol=0.002)
        pr = pr[(pr["fs"] > 8) & (pr["fs"] < NPX - 9) & (pr["ss"] > 8) & (pr["ss"] < NPX - 9)]
        img = rng.poisson(5.0, size=(NPX, NPX)).astype(np.float32)
        for f0, s0 in zip(pr["fs"], pr["ss"]):
            cf, cs = int(round(f0)), int(round(s0))
            img[cs - 3:cs + 4, cf - 3:cf + 4] += 3000.0 * gk
        frames.append(img); peaks.append((pr["fs"].copy(), pr["ss"].copy()))
    npk = [len(p[0]) for p in peaks]; mp = max(npk)
    X = np.zeros((3, mp)); Y = np.zeros((3, mp)); PI = np.zeros((3, mp))
    for e, (fs, ss) in enumerate(peaks):
        X[e, :len(fs)], Y[e, :len(fs)], PI[e, :len(fs)] = fs, ss, 1000.0 + np.arange(len(fs))
    cxi = os.path.join(d, "stack.cxi")
    with h5py.File(cxi, "w") as f:
        f.create_dataset(DATA, data=np.stack(frames))
        f.create_dataset("/LCLS/photon_energy_eV", data=np.array(EVS))
        f.create_dataset("/LCLS/detector_1/EncoderValue", data=np.array(ENC))
        f.create_dataset(RL + "/peakXPosRaw", data=X)
        f.create_dataset(RL + "/peakYPosRaw", data=Y)
        f.create_dataset(RL + "/peakTotalIntensity", data=PI)
        f.create_dataset(RL + "/nPeaks", data=np.array(npk))
    print(f"3 events at {EVS} eV, EncoderValue {ENC} mm; planted spots per event {npk}")

    def fresh():
        return [{"image": cxi, "event": e, "M": M0.copy()} for e in range(3)]

    # ---- 1. path-valued .geom: per-event values reach the results, the header, and agree with the rows
    print("\n1. path-valued .geom (photon_energy, clen = HDF5 paths)")
    res = fresh()
    nint, tot = integrate_cxi(res, gpath, dmin=DMIN, tol=TOL)
    check("integrate_cxi integrates all 3 events", nint == 3 and tot > 0, (nint, tot))
    check("each result carries ITS event's photon_eV and clen_m",
          [r.get("photon_eV") for r in res] == EVS
          and all(r.get("clen_m") is not None and abs(r["clen_m"] - c) < 1e-12 for r, c in zip(res, CLENS)),
          [(r.get("photon_eV"), r.get("clen_m")) for r in res])
    out1 = os.path.join(d, "path.stream")
    write_stream_integrated(res, out1, geom_text=GEOM_PATH, photon_eV=PLACEHOLDER[0], clen_m=PLACEHOLDER[1])
    h1 = chunk_headers(out1)
    check("chunk photon_energy_eV = each event's own energy (not 9392.70)",
          [h[0] for h in h1] == EVS, [h[0] for h in h1])
    check("chunk average_camera_length = each event's own clen (not 0.150000 m)",
          [h[1] for h in h1] == [round(c, 6) for c in CLENS], [h[1] for h in h1])
    rp = reprojection_px(out1, panels)
    check("written hkl re-projected at the STATED energy/clen land on the written fs/ss (< 0.1 px)",
          len(rp) == 3 and max(rp) < 0.1, rp)

    # ---- 2. the wavelength override reaches the header ---------------------------------------------
    print("\n2. wavelength_A override (the CLI's --wavelength)")
    WL = HC / 12005.0
    for name, gp in (("path-valued", gpath), ("literal", glit)):
        r2 = fresh()
        integrate_cxi(r2, gp, wavelength_A=WL, dmin=DMIN, tol=TOL)
        o2 = os.path.join(d, f"wl_{name}.stream")
        write_stream_integrated(r2, o2, geom_text=GEOM_PATH, photon_eV=PLACEHOLDER[0], clen_m=PLACEHOLDER[1])
        e2 = [h[0] for h in chunk_headers(o2)]
        check(f"{name} .geom + wavelength_A: every chunk states 12005.00 eV", e2 == [12005.0] * 3, e2)

    # ---- 3. literal .geom without override: unchanged output ----------------------------------------
    print("\n3. literal .geom (photon_energy = 12000.00, clen = 0.12), no override")
    r3 = fresh()
    integrate_cxi(r3, glit, dmin=DMIN, tol=TOL)
    o3 = os.path.join(d, "lit.stream"); o3_old = os.path.join(d, "lit_oldkeys.stream")
    write_stream_integrated(r3, o3, geom_text=GEOM_LIT, photon_eV=EVS[0], clen_m=CLENS[0])
    stripped = [{k: v for k, v in r.items() if k not in ("photon_eV", "clen_m")} for r in r3]
    write_stream_integrated(stripped, o3_old, geom_text=GEOM_LIT, photon_eV=EVS[0], clen_m=CLENS[0])
    h3 = chunk_headers(o3)
    check("header = the .geom literals", all(h[:2] == (EVS[0], CLENS[0]) for h in h3), h3)
    check("byte-identical to the stream written without the per-result keys",
          open(o3, "rb").read() == open(o3_old, "rb").read())

    # ---- 4. records without the new keys keep the run-level kwargs ----------------------------------
    print("\n4. a record with no photon_eV / clen_m (StreamDriver's shape)")
    o4 = os.path.join(d, "kw.stream")
    write_stream_integrated([stripped[0]], o4, photon_eV=9000.0, clen_m=0.2)
    check("falls back to the photon_eV / clen_m kwargs", chunk_headers(o4)[0][:2] == (9000.0, 0.2),
          chunk_headers(o4))

    # ---- 5. the CLI: integrate_cxi -> write_stream_integrated as glint_cli wires them ---------------
    print("\n5. glint_cli --images ... --integrate (indexer stubbed: event 2 left unindexed in 5b)")
    INDEXED = {"set": {0, 1, 2}}

    def _hybrid_index(frames_, images_, **kw):
        results = [dict(im, M=(M0.copy() if i in INDEXED["set"] else None)) for i, im in enumerate(images_)]
        return results, {"n": len(frames_), "n_idx": sum(r["M"] is not None for r in results)}

    stubs = {"glint.hybrid_stream": types.ModuleType("glint.hybrid_stream"),
             "glint.glint_fast": types.ModuleType("glint.glint_fast")}
    stubs["glint.hybrid_stream"].hybrid_index = _hybrid_index
    stubs["glint.hybrid_stream"].dense_index = None
    stubs["glint.hybrid_stream"]._report = lambda stats, out: None
    stubs["glint.glint_fast"].CLUSTER_MIN = 10 ** 9
    saved = {k: sys.modules.get(k) for k in stubs}
    argv0 = sys.argv
    import glint.glint_cli as cli

    def run_cli(out, extra=()):
        sys.argv = ["glint", "--images", cxi, "--geom", gpath, "--peakfinder", "stored", "--mode", "sparse",
                    "--cell", " ".join(str(c) for c in CELL), "--integrate", "--int-dmin", str(DMIN),
                    "--device", "cpu", "-o", out, *extra]
        try:
            cli.main()
        except SystemExit as e:                 # an argparse/usage error must fail the check, not the file
            return f"SystemExit({e.code})"
        return None

    try:
        sys.modules.update(stubs)
        o5 = os.path.join(d, "cli.stream")
        err = run_cli(o5)
        h5 = chunk_headers(o5) if err is None and os.path.exists(o5) else []
        check("5a: path-valued .geom -> chunks state [12000, 12010, 11990] eV / [0.120, 0.121, 0.119] m",
              [h[:2] for h in h5] == [(e, round(c, 6)) for e, c in zip(EVS, CLENS)], err or h5)
        INDEXED["set"] = {0, 1}
        o5b = os.path.join(d, "cli_wl.stream")
        err = run_cli(o5b, ("--wavelength", repr(WL)))
        h5b = chunk_headers(o5b) if err is None and os.path.exists(o5b) else []
        check("5b: --wavelength -> every chunk states 12005.00 eV, the one with no crystal too",
              [h[0] for h in h5b] == [12005.0] * 3 and [h[2] for h in h5b] == [True, True, False], err or h5b)
    finally:
        sys.argv = argv0
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
