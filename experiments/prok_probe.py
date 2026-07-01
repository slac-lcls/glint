"""NERSC probe: does GLINT blind-index real cxidb-45 Proteinase K (SACLA MPCCD) frames?

Load mnasser's MPCCD geometry, peakfind real frames from dermen's cxidb45 h5, map peaks -> q, GLINT
blind-index, and report the indexing rate + the cell GLINT converges to (expect ProK P43212 ~68/68/109).
This validates geometry + blind indexing on REAL data before the full predict/integrate/merge pipeline.

  module load pytorch/2.6.0
  python prok_probe.py [N]
"""
import os, sys, re
sys.path.insert(0, "/pscratch/sd/s/smarches/glint_real")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import h5py
from scipy.ndimage import maximum_filter
from glint.lute_bridge import peaks_to_q
from glint.glint_fast import index_blind_fast

GEOM = "/global/cfs/cdirs/lcls/mnasser/cxidb_62/mpccd-optimized.geom"
H5 = "/global/cfs/cdirs/lcls/dermen/cxidb45/data/run296940-0.h5"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def _vec(s):
    out = [0.0, 0.0, 0.0]
    for m in re.finditer(r"([-+0-9.eE]+)\s*([xyz])", s):
        out["xyz".index(m.group(2))] = float(m.group(1))
    return np.array(out)


def load_geom(path):
    glob, pan = {}, {}
    for line in open(path):
        line = line.split(";")[0].strip()
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        if "/" in k:
            p, prop = k.split("/", 1); pan.setdefault(p, {})[prop] = v
        else:
            glob[k] = v
    res_g = float(glob.get("res", 1.0)); coff_g = float(glob.get("coffset", 0.0))
    panels = []
    for name, d in pan.items():
        if "fs" not in d or "corner_x" not in d:
            continue
        panels.append(dict(name=name, fs=_vec(d["fs"]), ss=_vec(d["ss"]),
                           res=float(d.get("res", res_g)), cx=float(d["corner_x"]), cy=float(d["corner_y"]),
                           coffset=float(d.get("coffset", coff_g)),
                           min_fs=int(d["min_fs"]), max_fs=int(d["max_fs"]),
                           min_ss=int(d["min_ss"]), max_ss=int(d["max_ss"])))
    return panels, glob


def peakfind(img, nmax=160, snr=8.0):
    bg = np.median(img)
    sig = 1.4826 * np.median(np.abs(img - bg)) + 1e-3
    mx = maximum_filter(img, size=5)
    ys, xs = np.where((img == mx) & (img > bg + snr * sig))
    if len(xs) == 0:
        return np.empty((0, 2))
    order = np.argsort(img[ys, xs])[::-1][:nmax]
    return np.column_stack([xs[order], ys[order]]).astype(float)   # (fs, ss)


panels, glob = load_geom(GEOM)
clen = float(glob.get("clen", 0.055))
print(f"geom: {len(panels)} panels  clen={clen} m  res={panels[0]['res']}")

f = h5py.File(H5, "r")
tags = [k for k in f if k.startswith("tag")][:N]
index_blind_fast(np.random.rand(20, 3) * 0.1)   # numba warmup
nidx = 0
cells = []
for t in tags:
    img = f[t]["data"][()].astype(np.float32)
    lam = float(f[t]["photon_wavelength_A"][()])
    pk = peakfind(img)
    if len(pk) < 8:
        continue
    q = peaks_to_q(pk[:, 0], pk[:, 1], panels, clen, lam)
    q = q[~np.isnan(q).any(1)]
    M = index_blind_fast(q)
    if M is not None:
        nidx += 1
        a, b, c = (np.linalg.norm(M[:, i]) for i in range(3))
        cells.append(sorted([a, b, c]))
        if nidx <= 8:
            print(f"  {t}: npk={len(pk)} -> cell {a:.1f} {b:.1f} {c:.1f} A")
print(f"\nindexed {nidx}/{len(tags)} frames")
if cells:
    cells = np.array(cells)
    print(f"median sorted axes: {np.median(cells,axis=0).round(1)}  (ProK expect ~68 68 109)")
