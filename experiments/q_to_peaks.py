"""Invert q-vector FRAME blocks into a CrystFEL .geom + peak-search stream (the inverse of the
geom bridge), so existing q datasets can exercise the productized --peaks/--geom CLI path and
serve as a realistic end-to-end test. Flat single panel; beam +z; lambda from --wavelength.

  python q_to_peaks.py frames_cxidb_clean.txt 1.322 out   # -> out.geom + out.stream
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from glint_fast import load

RES, CLEN, NPX, CORNER = 10000.0, 0.15, 3000, -1500.0     # px/m, m, panel size, corner (px)

GEOM = lambda lam: (f"photon_energy = {12398.419843320026 / lam:.4f}\nclen = {CLEN}\nres = {RES}\n"
                    f"coffset = 0.0\nadu_per_eV = 0.001\np0/min_fs = 0\np0/max_fs = {NPX-1}\n"
                    f"p0/min_ss = 0\np0/max_ss = {NPX-1}\np0/corner_x = {CORNER}\n"
                    f"p0/corner_y = {CORNER}\np0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n")


def q_to_fsss(q, lam):
    shat = np.array([0, 0, 1.0]) + lam * np.asarray(q, float)
    shat = shat / np.linalg.norm(shat, axis=1, keepdims=True)
    t = CLEN / shat[:, 2]
    fs = t * shat[:, 0] * RES - CORNER
    ss = t * shat[:, 1] * RES - CORNER
    on = (fs >= 0) & (fs < NPX) & (ss >= 0) & (ss < NPX) & (shat[:, 2] > 0)
    return fs[on], ss[on]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1.322
    out = sys.argv[3] if len(sys.argv) > 3 else "qpeaks"
    open(out + ".geom", "w").write(GEOM(lam))
    n = kept = 0
    with open(out + ".stream", "w") as f:
        f.write("CrystFEL stream format 2.3\n")
        for i, q in enumerate(load(path)):
            fs, ss = q_to_fsss(q, lam)
            f.write(f"----- Begin chunk -----\nImage filename: {os.path.basename(path)}\nEvent: //{i}\n")
            f.write(f"num_peaks = {len(fs)}\nPeaks from peak search\n  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n")
            for a, b in zip(fs, ss):
                f.write(f"  {a:.2f}  {b:.2f}  0.0  1000.0  p0\n")
            f.write("End of peak list\n----- End chunk -----\n")
            n += 1; kept += len(fs) >= 6
    print(f"wrote {out}.geom + {out}.stream: {n} chunks ({kept} with >=6 peaks), lambda={lam} A")
