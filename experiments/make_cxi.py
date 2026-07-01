"""Back-project q-frames onto a synthetic flat panel -> CrystFEL CXI + geom + event list + the EFFECTIVE
q's used (<outpref>_qeff.txt), so indexamajig indexes the SAME peaks. AUTO-infers the wavelength and
beam sign from the elastic condition (zhat.q = -lambda|q|^2/2), so any single-wavelength still dataset
round-trips exactly (negative inferred lambda => beam along -z => flip q_z). Override lambda with argv[3].
  python make_cxi.py <frames.txt> <outprefix> [lambda_A]"""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
import numpy as np, h5py
from glint_fast import load
RES, CLEN, NPX, CORNER = 10000.0, 0.15, 3000, -1500.0
frames_path = sys.argv[1] if len(sys.argv) > 1 else "/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt"
outpref = sys.argv[2] if len(sys.argv) > 2 else "/sdf/home/s/smarches/git/glint/experiments/cf_peaks"
frames = [np.asarray(q, float) for q in load(frames_path)]
allq = np.concatenate(frames)
lam_infer = float(np.median(-2 * allq[:, 2] / (np.sum(allq ** 2, 1) + 1e-12)))
flip = lam_infer < 0
LAM = float(sys.argv[3]) if len(sys.argv) > 3 else abs(lam_infer)
if flip:                                               # beam along -z -> flip q_z to the +z convention
    frames = [q * np.array([1, 1, -1.0]) for q in frames]
PE = 12398.419843320026 / LAM
CXI = outpref + ".cxi"
def q_to_fsss(q, lam):
    shat = np.array([0, 0, 1.0]) + lam * q
    shat = shat / np.linalg.norm(shat, axis=1, keepdims=True)
    t = CLEN / shat[:, 2]
    fs = t * shat[:, 0] * RES - CORNER; ss = t * shat[:, 1] * RES - CORNER
    on = (fs >= 0) & (fs < NPX) & (ss >= 0) & (ss < NPX) & (shat[:, 2] > 0)
    return fs[on], ss[on], on
pk = [q_to_fsss(q, LAM) for q in frames]
N = len(frames); maxp = max(len(fs) for fs, _, _ in pk)
X = np.zeros((N, maxp), np.float32); Y = np.zeros((N, maxp), np.float32)
Iarr = np.zeros((N, maxp), np.float32); npk = np.zeros(N, np.int32)
with open(outpref + "_qeff.txt", "w") as fq:                 # effective q (flipped, on-panel) for scoring/GLINT
    for i, (fs, ss, on) in enumerate(pk):
        X[i, :len(fs)] = fs; Y[i, :len(ss)] = ss; Iarr[i, :len(fs)] = 1000.0; npk[i] = len(fs)
        qe = frames[i][on]
        fq.write("FRAME %d %d\n" % (i, len(qe)))
        for a, b, c in qe:
            fq.write("%.6f %.6f %.6f\n" % (a, b, c))
with h5py.File(CXI, "w") as f:
    f["/entry_1/data_1/data"] = np.zeros((N, 1, 1), np.float32)
    g = f.create_group("/entry_1/result_1")
    g["peakXPosRaw"] = X; g["peakYPosRaw"] = Y; g["peakTotalIntensity"] = Iarr; g["nPeaks"] = npk
with open(outpref + "_events.lst", "w") as f:
    for i in range(N): f.write("%s //%d\n" % (CXI, i))
geom = ("photon_energy = %.4f\nclen = %.4f\nres = %.1f\nadu_per_eV = 0.001\n"
        "data = /entry_1/data_1/data\npeak_list = /entry_1/result_1\npeak_list_type = cxi\n"
        "dim0 = %%\ndim1 = ss\ndim2 = fs\np0/min_fs = 0\np0/max_fs = %d\np0/min_ss = 0\np0/max_ss = %d\n"
        "p0/corner_x = %.1f\np0/corner_y = %.1f\np0/fs = +1.0x +0.0y\np0/ss = +0.0x +1.0y\n"
        ) % (PE, CLEN, RES, NPX - 1, NPX - 1, CORNER, CORNER)
open(outpref + ".geom", "w").write(geom)
print("wrote %s.cxi (%d ev, maxp %d) + .geom + _events.lst + _qeff.txt; lambda %.4f (inferred %.4f, flip=%s)"
      % (outpref, N, maxp, LAM, lam_infer, flip))
