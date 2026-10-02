"""glint --integrate must predict at the wavelength the frames were INDEXED at, and say so in the stream.

WHY THIS EXISTS. ``--wavelength`` "overrides .geom", and the indexing side always honoured it
(``peaks_to_q(..., wavelength_A=args.wavelength)``). The --peaks integrator did not: glint_cli called
``integrate_frames`` with no wavelength, and ``integrate_frames`` read only ``geom['wavelength_A']``.
  * A .geom with an HDF5-path ``photon_energy`` (``/LCLS/photon_energy_eV``, the per-shot LCLS form)
    parses to wavelength None. The CLI tells that user to pass --wavelength; indexing then ran, and
    --integrate died in predict_spots with a TypeError on None, after all the indexing compute, with
    no stream written. Fixing only that call was not enough: the header line after it did
    ``float(gg["photon_energy"])`` on the same path string.
  * A .geom with a LITERAL energy that --wavelength overrides was integrated at the .geom energy
    while the frames had been indexed at the override. At a 4% mismatch (below) almost no predicted
    box sits on a real spot; the stream still exits 0 and stamps photon_energy_eV with the .geom value.
The --images route already predicted at --wavelength (integrate_cxi takes it) but stamped the header
from the .geom too.

WHAT IT CHECKS. Synthetic frames: one 1200x1200 panel, 100 um pixels, clen 0.1 m, a lysozyme cell at
two seeded orientations, a 7x7 Gaussian of flux 1000 planted at every spot predicted at LAM = 1.322 A
(9378.53 eV). The CLI runs in-process (glint_cli.main()) with an ORACLE indexer standing in for
glint.hybrid_stream: it returns each frame's true orientation, so everything this test sees comes from
the CLI's own integrate branch and the integrators, and no torch is needed. The oracle also records
whether the q it was handed index to the true cell, i.e. that the indexing side ran at LAM.
  --peaks route, every run with --int-tol 0.002 --int-dmin 2.5:
    C  literal .geom at the true energy + --wavelength LAM           (control)
    A  path-valued photon_energy, literal clen + --wavelength LAM     -> runs; rows == C; header hc/LAM
    B  literal 9000 eV .geom + --wavelength LAM                       -> rows == C; header hc/LAM
    D  literal 9000 eV .geom, NO --wavelength                         -> unchanged: predicted and stamped
                                                                         at the .geom's 9000 eV
    G  .geom keyed `wavelength = 1.322e-10` (no photon_energy)       -> header hc/LAM, not the 9392.70
                                                                         placeholder
  integrate_frames directly: no wavelength anywhere -> ValueError naming it (not a TypeError deep in
  predict_spots); wavelength_A=LAM on the path-valued .geom -> the same intensities as run C.
  --images route (--peakfinder stored, a 2-event .cxi): --wavelength over a literal 9000 eV .geom and
  over a path-valued one -> header hc/LAM; no --wavelength on a literal .geom -> that literal.

  PYTHONPATH=. python experiments/test_integrate_wavelength.py      # exit 0 = all pass
"""
import contextlib
import io
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

try:
    import h5py
except ImportError:
    print(f"SKIP {os.path.basename(__file__)} -- no h5py: it has to write the frame images")
    sys.exit(0)

from glint.geom import parse_geom                                          # noqa: E402
from glint.lattice import cell_to_Ar                                       # noqa: E402
from glint.predict import integrate_frames, panels_from_geom, predict_spots  # noqa: E402

HC = 12398.419843320026
LAM = 1.322
EV = HC / LAM                       # 9378.53 eV
NPX = 1200
FLUX = 1000.0
TOL, DMIN = 0.002, 2.5
CELL = (79.1, 79.1, 38.0, 90.0, 90.0, 90.0)
FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def geom_text(energy_line, data="/data/data"):
    return (f"{energy_line}\nclen = 0.1\nres = 10000\ncoffset = 0.0\nadu_per_eV = 0.001\n"
            f"data = {data}\n"
            f"p0/min_fs = 0\np0/max_fs = {NPX-1}\np0/min_ss = 0\np0/max_ss = {NPX-1}\n"
            f"p0/corner_x = {-NPX//2}\np0/corner_y = {-NPX//2}\n"
            f"p0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n")


def rot(a, b, c):
    def R(t, ax):
        s, k = np.sin(t), np.cos(t)
        m = np.eye(3); i, j = [x for x in range(3) if x != ax]
        m[i, i] = m[j, j] = k; m[i, j] = -s; m[j, i] = s
        return m
    return R(a, 0) @ R(b, 1) @ R(c, 2)


TRUE = [rot(0.3, 1.1, 2.0) @ cell_to_Ar(*CELL), rot(2.2, 0.4, 1.3) @ cell_to_Ar(*CELL)]


# --- the oracle indexer: glint.hybrid_stream / glint.glint_fast as the CLI imports them ------------
ORACLE = {"q_frac": []}


def _which(im):
    """Frame number of an `images` entry: f<k>.h5 on the --peaks route, the .cxi event on --images."""
    name = os.path.basename(str(im["image"]))
    return int(name[1]) if name.startswith("f") and name.endswith(".h5") else int(im["event"])


def _hybrid_index(frames, images, **_kw):
    res = []
    for q, im in zip(frames, images):
        M = TRUE[_which(im)]
        r = q @ M
        ORACLE["q_frac"].append(float((np.abs(r - np.rint(r)).max(1) < 0.15).mean()))
        res.append({"image": im["image"], "event": im["event"], "M": M, "q": q})
    return res, {"n_idx": len(res)}


_hs = types.ModuleType("glint.hybrid_stream")
_hs.hybrid_index = _hybrid_index
_hs.dense_index = _hybrid_index
_hs._report = lambda stats, out: None
_gf = types.ModuleType("glint.glint_fast")
_gf.CLUSTER_MIN = 10 ** 9                                   # --mode auto -> sparse -> the oracle
sys.modules["glint.hybrid_stream"] = _hs
sys.modules["glint.glint_fast"] = _gf
from glint import glint_cli                                                # noqa: E402


def run_cli(argv):
    """glint_cli.main() in-process. Returns (error or None, captured stdout)."""
    out = io.StringIO()
    old = sys.argv
    sys.argv = ["glint"] + argv
    try:
        with contextlib.redirect_stdout(out):
            glint_cli.main()
        return None, out.getvalue()
    except SystemExit as e:
        return (None if not e.code else f"SystemExit({e.code})"), out.getvalue()
    except Exception as e:                                                 # noqa: BLE001
        return f"{type(e).__name__}: {e}", out.getvalue()
    finally:
        sys.argv = old


def read_stream(path):
    """-> ([per chunk (n,4) array of fs, ss, I, sigma], [per chunk photon_energy_eV string])."""
    rows, evs, cur, inref = [], [], None, False
    for line in open(path):
        s = line.strip()
        if s.startswith("----- Begin chunk"):
            cur = []
        elif s.startswith("photon_energy_eV"):
            evs.append(s.split("=", 1)[1].strip())
        elif s.startswith("Reflections measured after indexing"):
            inref = True
        elif s.startswith("End of reflections"):
            inref = False
        elif s.startswith("----- End chunk"):
            rows.append(np.array(cur, float).reshape(-1, 4))
        elif inref:
            t = s.split()
            try:
                cur.append((float(t[7]), float(t[8]), float(t[3]), float(t[4])))
            except (ValueError, IndexError):
                pass
    return rows, evs


def on_spot(rows, planted):
    """Fraction of written reflections whose predicted (fs, ss) lies within 1.5 px of a planted spot."""
    n = hit = 0
    for R, P in zip(rows, planted):
        if len(R) and len(P):
            d = np.sqrt(((R[:, None, :2] - P[None, :, :]) ** 2).sum(-1)).min(1)
            n += len(R); hit += int((d <= 1.5).sum())
    return hit / max(n, 1)


with tempfile.TemporaryDirectory() as d:
    # --- synthesise the frames at LAM ---------------------------------------------------------------
    gtrue = os.path.join(d, "true.geom"); open(gtrue, "w").write(geom_text(f"photon_energy = {EV:.4f}"))
    panels, clen = panels_from_geom(parse_geom(gtrue))
    yy, xx = np.mgrid[-3:4, -3:4]
    kern = np.exp(-(xx * xx + yy * yy) / 2.0); kern /= kern.sum()
    rng = np.random.default_rng(20261001)
    planted, imgs = [], []
    for M in TRUE:
        pred = predict_spots(M, panels, clen, LAM, dmin=DMIN, tol=TOL)
        img = np.full((NPX, NPX), 5.0, np.float32) + rng.normal(0, 1.0, (NPX, NPX)).astype(np.float32)
        cen = []
        for p in pred:
            cf, cs = int(round(p["fs"])), int(round(p["ss"]))
            if 6 <= cs < NPX - 6 and 6 <= cf < NPX - 6:
                img[cs - 3:cs + 4, cf - 3:cf + 4] += FLUX * kern
                cen.append((cf, cs))
        planted.append(np.array(cen, float)); imgs.append(img)
    for k, img in enumerate(imgs):
        with h5py.File(os.path.join(d, f"f{k}.h5"), "w") as h:
            h["/data/data"] = img
    peaks = os.path.join(d, "peaks.stream")
    with open(peaks, "w") as f:
        f.write("CrystFEL stream format 2.3\n")
        for k, cen in enumerate(planted):
            f.write(f"----- Begin chunk -----\nImage filename: f{k}.h5\nEvent: //0\n"
                    "Peaks from peak search\n  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n")
            f.writelines(f"{a:.2f} {b:.2f} 1.0 {FLUX:.1f} p0\n" for a, b in cen)
            f.write("End of peak list\n----- End chunk -----\n")
    print(f"synthesised {len(TRUE)} frames at {LAM} A ({EV:.2f} eV), planted spots/frame "
          f"{[len(p) for p in planted]}")

    def peaks_run(tag, energy_line, wavelength):
        g = os.path.join(d, f"{tag}.geom"); open(g, "w").write(geom_text(energy_line))
        out = os.path.join(d, f"{tag}.stream")
        argv = ["--peaks", peaks, "--geom", g, "--integrate", "--image-dir", d,
                "--int-tol", str(TOL), "--int-dmin", str(DMIN), "--device", "cpu", "-o", out]
        if wavelength is not None:
            argv += ["--wavelength", str(wavelength)]
        ORACLE["q_frac"] = []
        err, _ = run_cli(argv)
        rows, evs = read_stream(out) if os.path.exists(out) else ([], [])
        return dict(err=err, written=os.path.exists(out), rows=rows, evs=sorted(set(evs)), geom=g,
                    q_frac=ORACLE["q_frac"])

    print("\n--peaks --integrate")
    C = peaks_run("C_control", f"photon_energy = {EV:.4f}", LAM)
    fC = on_spot(C["rows"], planted)
    check("C (control: .geom at the true energy, --wavelength) runs and lands on the planted spots",
          C["err"] is None and fC > 0.9, (C["err"], fC))
    check("...and stamps photon_energy_eV = hc/--wavelength", C["evs"] == [f"{EV:.2f}"], C["evs"])

    A = peaks_run("A_pathform", "photon_energy = /LCLS/photon_energy_eV", LAM)
    check("A: path-valued photon_energy + --wavelength integrates instead of crashing",
          A["err"] is None and A["written"], A["err"])
    check("...writing exactly the control's reflections",
          len(A["rows"]) == len(C["rows"]) and all(np.array_equal(a, c) for a, c in zip(A["rows"], C["rows"])),
          [len(r) for r in A["rows"]])
    check("...and photon_energy_eV = hc/--wavelength", A["evs"] == [f"{EV:.2f}"], A["evs"])

    B = peaks_run("B_literal_wrong", "photon_energy = 9000.0", LAM)
    fB = on_spot(B["rows"], planted)
    check("B: literal 9000 eV .geom + --wavelength predicts at --wavelength (on the planted spots)",
          B["err"] is None and fB > 0.9, (B["err"], round(fB, 3)))
    check("...writing exactly the control's reflections",
          len(B["rows"]) == len(C["rows"]) and all(np.array_equal(b, c) for b, c in zip(B["rows"], C["rows"])),
          [len(r) for r in B["rows"]])
    check("...and photon_energy_eV = hc/--wavelength, not the overridden .geom value",
          B["evs"] == [f"{EV:.2f}"], B["evs"])

    D = peaks_run("D_no_override", "photon_energy = 9000.0", None)
    lam_D = HC / 9000.0
    pD = [predict_spots(M, panels, clen, lam_D, dmin=DMIN, tol=TOL) for M in TRUE]
    at_geom = all(len(r) > 0 and {(round(a, 1), round(b, 1)) for a, b in r[:, :2]}
                  <= {(round(a, 1), round(b, 1)) for a, b in zip(p["fs"], p["ss"])}
                  for r, p in zip(D["rows"], pD))
    check("D: without --wavelength the .geom energy still rules prediction (unchanged default)",
          D["err"] is None and len(D["rows"]) == 2 and at_geom and on_spot(D["rows"], planted) < 0.2,
          (D["err"], at_geom, round(on_spot(D["rows"], planted), 3)))
    check("...and stamps the .geom's literal energy", D["evs"] == ["9000.00"], D["evs"])

    G = peaks_run("G_wavelength_key", f"wavelength = {LAM}e-10", None)
    check("G: a .geom keyed `wavelength =` stamps the energy it predicted at, not 9392.70",
          G["err"] is None and G["evs"] == [f"{EV:.2f}"] and on_spot(G["rows"], planted) > 0.9,
          (G["err"], G["evs"]))

    qf = {t: R["q_frac"] for t, R in (("C", C), ("A", A), ("B", B), ("G", G))}
    check("the oracle was handed q at the true wavelength in C, A, B and G (indexing side honours it)",
          all(len(v) == 2 and min(v) > 0.95 for v in qf.values()), qf)

    print("\nintegrate_frames directly")
    gpath = parse_geom(A["geom"])
    try:
        integrate_frames([{"image": "f0.h5", "event": 0, "M": TRUE[0]}], gpath, image_dir=d,
                         dmin=DMIN, tol=TOL)
        err = None
    except Exception as e:                                                 # noqa: BLE001
        err = e
    check("no wavelength anywhere -> a ValueError that names the wavelength",
          isinstance(err, ValueError) and "wavelength" in str(err), repr(err)[:160])
    try:
        rr = [{"image": f"f{k}.h5", "event": 0, "M": M} for k, M in enumerate(TRUE)]
        integrate_frames(rr, gpath, image_dir=d, dmin=DMIN, tol=TOL, wavelength_A=LAM)
        # the stream carries fs/ss to 0.1 px and I to 0.01: equal up to that rounding
        same = all(len(r["I"]) == len(c)
                   and np.abs(np.column_stack([r["pred"]["fs"], r["pred"]["ss"]]) - c[:, :2]).max() <= 0.0501
                   and np.abs(r["I"] - c[:, 2]).max() <= 0.00501
                   for r, c in zip(rr, C["rows"]))
        err = None
    except Exception as e:                                                 # noqa: BLE001
        err, same = e, False
    check("wavelength_A= on a path-valued .geom predicts and integrates what the CLI control wrote", err is None and same,
          repr(err)[:160])

    # --- the --images route -------------------------------------------------------------------------
    print("\n--images --integrate (--peakfinder stored)")
    cxi = os.path.join(d, "run.cxi")
    mp = max(len(p) for p in planted)
    with h5py.File(cxi, "w") as h:
        h["/entry_1/data_1/data"] = np.stack(imgs)
        h["/entry_1/result_1/nPeaks"] = np.array([len(p) for p in planted], np.int32)
        X = np.zeros((len(planted), mp), np.float32); Y = np.zeros_like(X)
        for k, p in enumerate(planted):
            X[k, :len(p)], Y[k, :len(p)] = p[:, 0], p[:, 1]
        h["/entry_1/result_1/peakXPosRaw"] = X
        h["/entry_1/result_1/peakYPosRaw"] = Y

    def images_run(tag, energy_line, wavelength):
        g = os.path.join(d, f"{tag}.geom")
        open(g, "w").write(geom_text(energy_line, data="/entry_1/data_1/data"))
        out = os.path.join(d, f"{tag}.stream")
        argv = ["--images", cxi, "--geom", g, "--peakfinder", "stored", "--integrate",
                "--int-tol", str(TOL), "--int-dmin", str(DMIN), "--device", "cpu", "-o", out]
        if wavelength is not None:
            argv += ["--wavelength", str(wavelength)]
        err, _ = run_cli(argv)
        rows, evs = read_stream(out) if os.path.exists(out) else ([], [])
        return dict(err=err, rows=rows, evs=sorted(set(evs)))

    IB = images_run("IB_literal_wrong", "photon_energy = 9000.0", LAM)
    check("--images: literal 9000 eV .geom + --wavelength lands on the planted spots",
          IB["err"] is None and on_spot(IB["rows"], planted) > 0.9, (IB["err"], on_spot(IB["rows"], planted)))
    check("...and photon_energy_eV = hc/--wavelength, not the overridden .geom value",
          IB["evs"] == [f"{EV:.2f}"], IB["evs"])
    IA = images_run("IA_pathform", "photon_energy = /LCLS/photon_energy_eV", LAM)
    check("--images: path-valued photon_energy + --wavelength stamps hc/--wavelength, not 9392.70",
          IA["err"] is None and IA["evs"] == [f"{EV:.2f}"], (IA["err"], IA["evs"]))
    ID = images_run("ID_no_override", f"photon_energy = {EV:.4f}", None)
    check("--images: without --wavelength a literal .geom energy is stamped as before",
          ID["err"] is None and ID["evs"] == [f"{EV:.2f}"] and on_spot(ID["rows"], planted) > 0.9,
          (ID["err"], ID["evs"]))

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
