"""B4 fast self-contained CPU smoke test for the productized fftindex/GLINT surface (no GPU, no real
data, ~1s): geom->panels, spot prediction + integration, the geom<->predict round trip, the CrystFEL
stream + --tofile writers, and the CLI entry point. Full blind-index ACCURACY is GPU-validated
separately (the 70k-seed search is impractical on CPU); this guards the plumbing so refactors can't
silently break the product surface.

  python test_cli_smoke.py     # exit 0 = all pass
"""
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from glint.lattice import cell_to_Ar
from glint.predict import (predict_spots, integrate_spots, write_solution_file,
                              write_stream_integrated, panels_from_geom, integrate_frames)
from glint.stream import write_stream
from glint.geom import parse_geom, peaks_to_q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90); LAM = 1.322
NPX = 1500
GEOM = (f"photon_energy = {12398.419843320026 / LAM:.2f}\nclen = 0.1\nres = 10000\ncoffset = 0.0\n"
        f"adu_per_eV = 0.001\ndata = /data/data\np0/min_fs = 0\np0/max_fs = {NPX-1}\n"
        f"p0/min_ss = 0\np0/max_ss = {NPX-1}\np0/corner_x = -750\np0/corner_y = -750\n"
        f"p0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n")
fails = []


def check(name, cond, msg=""):
    print(f"  {name:20s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


with tempfile.TemporaryDirectory() as d:
    gpath = os.path.join(d, "g.geom"); open(gpath, "w").write(GEOM)
    geom = parse_geom(gpath)

    panels, clen = panels_from_geom(geom)
    check("panels_from_geom", len(panels) == 1 and abs(clen - 0.1) < 1e-9 and panels[0]["min_fs"] == 0,
          f"{len(panels)} panel, clen {clen}")

    pred = predict_spots(LYSO, panels, clen, LAM, dmin=3.0, tol=0.004)
    use = pred[:60]
    img = np.zeros((NPX, NPX), np.float32)
    yy, xx = np.mgrid[-3:4, -3:4]; g = np.exp(-(xx * xx + yy * yy) / 2.0); g /= g.sum(); flux = 1000.0
    for p in use:
        cf, cs = int(round(p["fs"])), int(round(p["ss"]))
        if 5 <= cs < NPX - 5 and 5 <= cf < NPX - 5:
            img[cs - 3:cs + 4, cf - 3:cf + 4] += flux * g
    I, sig, pk, bg = integrate_spots(img, use)
    check("predict+integrate", len(pred) > 50 and (I > 0.5 * flux).sum() >= 20 and np.isfinite(sig).all(),
          f"{len(pred)} predicted, {(I > 0.5 * flux).sum()}/60 recover >50% flux")

    on = pred["panel"] >= 0
    q_back = peaks_to_q(np.column_stack([pred["fs"][on], pred["ss"][on]]), geom, wavelength_A=LAM)
    r = q_back @ LYSO; frac = float((np.abs(r - np.rint(r)).max(1) < 0.15).mean())   # 0.15 = the inlier window
    check("geom<->predict", frac > 0.95, f"{100*frac:.0f}% of back-projected spots index to lysozyme")

    results = [{"image": "a.h5", "event": i, "M": LYSO} for i in range(3)]
    sol = os.path.join(d, "s.sol"); n = write_solution_file(results, sol, "tPc")
    lines = [l for l in open(sol).read().splitlines() if l.strip()]

    def cellrec(v):
        Br = v.reshape(3, 3).T
        return np.sort(np.linalg.norm(np.linalg.inv(Br).T, axis=0)) * 10.0
    ok = (n == 3 and all(len(l.split()) == 14 and l.split()[-1] == "tPc" for l in lines)
          and all(np.allclose(cellrec(np.array(list(map(float, l.split()[2:11])))),
                              [37.98, 79.02, 79.02], rtol=0.06) for l in lines))
    check("write_solution_file", ok, f"{n} tPc solutions, cell ~lyso")

    r2 = [{"image": "a.h5", "event": i, "M": LYSO, "hkl": np.array([[1, 0, 0]]), "q": np.zeros((1, 3))}
          for i in range(2)]
    stp = os.path.join(d, "o.stream"); write_stream(r2, stp)
    s = open(stp).read()
    check("write_stream", "Begin chunk" in s and ("crystal" in s.lower() or "cell" in s.lower()),
          f"{s.count('Begin chunk')} chunks")

    rp = subprocess.run([sys.executable, "-m", "glint.glint_cli", "--help"],
                        cwd=ROOT, capture_output=True, text=True, timeout=120)
    check("glint --help", rp.returncode == 0 and "--tofile" in rp.stdout and "--mode" in rp.stdout
          and "--integrate" in rp.stdout, "entry point + new flags present")

print("\n%s (%d/6 passed)" % ("ALL PASS" if not fails else "FAILED: " + ",".join(fails), 6 - len(fails)))
sys.exit(1 if fails else 0)
