"""Back-project the 120 cxidb q-frames onto a synthetic flat panel and write a CrystFEL CXI peak file
+ geom + event list, so indexamajig (asdf/taketwo/mosflm) indexes the SAME peaks GLINT/xgandalf/ffbidx
saw. q -> fs/ss via shat = zhat + lambda*q (exact: q=(shat-zhat)/lambda), so CrystFEL recovers the same
reciprocal lattice from the geometry. Reuses q_to_peaks geometry constants."""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
import numpy as np
import h5py
from glint_fast import load

RES, CLEN, NPX, CORNER = 10000.0, 0.15, 3000, -1500.0     # px/m, m, panel px, corner px
LAM = 1.322
PE = 12398.419843320026 / LAM
BASE = "/sdf/home/s/smarches/git/glint/experiments"
CXI = BASE + "/cf_peaks.cxi"


def q_to_fsss(q, lam):
    shat = np.array([0, 0, 1.0]) + lam * np.asarray(q, float)
    shat = shat / np.linalg.norm(shat, axis=1, keepdims=True)
    t = CLEN / shat[:, 2]
    fs = t * shat[:, 0] * RES - CORNER
    ss = t * shat[:, 1] * RES - CORNER
    on = (fs >= 0) & (fs < NPX) & (ss >= 0) & (ss < NPX) & (shat[:, 2] > 0)
    return fs[on], ss[on]


frames = list(load(BASE + "/frames_cxidb_clean.txt"))
pk = [q_to_fsss(q, LAM) for q in frames]
N = len(frames)
maxp = max(len(fs) for fs, _ in pk)
X = np.zeros((N, maxp), np.float32); Y = np.zeros((N, maxp), np.float32)
Iarr = np.zeros((N, maxp), np.float32); npk = np.zeros(N, np.int32)
for i, (fs, ss) in enumerate(pk):
    X[i, :len(fs)] = fs; Y[i, :len(ss)] = ss; Iarr[i, :len(fs)] = 1000.0; npk[i] = len(fs)

with h5py.File(CXI, "w") as f:
    f["/entry_1/data_1/data"] = np.zeros((N, 1, 1), np.float32)     # dummy; run with --no-image-data
    g = f.create_group("/entry_1/result_1")
    g["peakXPosRaw"] = X; g["peakYPosRaw"] = Y; g["peakTotalIntensity"] = Iarr; g["nPeaks"] = npk

with open(BASE + "/cf_events.lst", "w") as f:
    for i in range(N):
        f.write("%s //%d\n" % (CXI, i))

geom = (
    "photon_energy = %.4f\nclen = %.4f\nres = %.1f\nadu_per_eV = 0.001\n"
    "data = /entry_1/data_1/data\npeak_list = /entry_1/result_1\npeak_list_type = cxi\n"
    "dim0 = %%\ndim1 = ss\ndim2 = fs\n"
    "p0/min_fs = 0\np0/max_fs = %d\np0/min_ss = 0\np0/max_ss = %d\n"
    "p0/corner_x = %.1f\np0/corner_y = %.1f\np0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n"
) % (PE, CLEN, RES, NPX - 1, NPX - 1, CORNER, CORNER)
open(BASE + "/cf.geom", "w").write(geom)
print("wrote cf_peaks.cxi (%d events, maxp %d), cf.geom, cf_events.lst; lambda %.3f, PE %.1f eV"
      % (N, maxp, LAM, PE))
