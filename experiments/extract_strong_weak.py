"""Stage A (psana2 env): dump REAL MFX run-51 frames ($GLINT_EXP) as STRONG + WEAK
reciprocal-q lists for the indexing-guided soft-completeness test on real data.

  strong = high-SNR peaks  (the normal peakfinder: son>=6, I>med+200)
  weak   = SUB-THRESHOLD peaks (son>=3, I>med+30) that are NOT already strong
           -> Cong's 'score peaks close to but below threshold', done CLASSICALLY
              (a detector-agnostic stand-in for PeakNet's p_peak; no learned
               weights needed -- PeakNet is a later drop-in upgrade of this set).

Geometry/photon-energy path copied verbatim from fftindex_on_real.py (validated).
Writes an .npz: n, s{i}=strong q (Ns,3), w{i}=weak q (Nw,3). q in A^-1.

  source /sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh
  python extract_strong_weak.py [N_FRAMES] [OUT.npz]
"""
import sys, os
import numpy as np
from scipy.ndimage import maximum_filter
from psana import DataSource
from psana.pscalib.geometry.GeometryAccess import GeometryAccess

HC = 12398.42
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 80
OUT = sys.argv[2] if len(sys.argv) > 2 else "/sdf/home/s/smarches/real_sw.npz"
ZDIST = float(os.environ.get("ZDIST", "0.246"))         # detector distance [m]
# strong / weak thresholds (son = sigma-over-noise; abs = floor over panel median)
S_SON, S_ABS = 6.0, 200.0
W_SON, W_ABS = float(os.environ.get("W_SON", "3.0")), float(os.environ.get("W_ABS", "30.0"))
CAP_S, CAP_W = 600, 1200

EXP = os.environ.get("GLINT_EXP")   # beamtime ID: proprietary, so not committed
if not EXP:
    sys.exit("set GLINT_EXP=<experiment id>; the ID is deliberately not in this file "
             "(proprietary beamtime data -- see experiments/README_beamtime.md)")
ds = DataSource(exp=EXP, run=int(os.environ.get("GLINT_RUN", "51")))
myrun = next(ds.runs())
det = myrun.Detector("jungfrau")
eb = myrun.Detector("ebeamh")
geo = GeometryAccess()
geo.load_pars_from_str(det.calibconst["geometry"][0])
X, Y, Z = (a.astype(float).reshape(32, 512, 1024) for a in geo.get_pixel_coords())
X, Y = X * 1e-6, Y * 1e-6
Zc = np.sign(np.nanmean(Z)) * ZDIST
kin = np.array([0.0, 0.0, np.sign(Zc)])
status = det.calibconst["pixel_status"][0].reshape(-1, 32, 512, 1024)
badmask = (status != 0).any(axis=0)
print(f"geometry+status loaded; bad pixels {100*badmask.mean():.1f}%; "
      f"weak thr son>={W_SON} abs>={W_ABS}", flush=True)


def peak_idx(img, son_min, abs_min):
    img = np.where(badmask, 0.0, img)
    mx = maximum_filter(img, size=5)
    keep = np.zeros(img.shape, bool)
    for p in range(32):
        panel = img[p]
        med = np.median(panel)
        sig = 1.4826 * np.median(np.abs(panel - med)) + 1e-6
        keep[p] = (panel == mx[p]) & (panel > med + son_min * sig) & (panel > med + abs_min)
    return np.argwhere(keep)


def to_q(idx, lam, img, cap):
    if len(idx) == 0:
        return np.zeros((0, 3))
    p, r, c = idx.T
    if cap and len(idx) > cap:
        sel = np.argsort(img[p, r, c])[::-1][:cap]
        p, r, c = p[sel], r[sel], c[sel]
    rr = np.stack([X[p, r, c], Y[p, r, c], np.full(len(p), Zc)], axis=1)
    s = rr / np.linalg.norm(rr, axis=1, keepdims=True)
    return (s - kin) / lam


frames = []
used = 0
for evt in myrun.events():
    if used >= N_FRAMES:
        break
    img = det.raw.calib(evt)
    if img is None:
        continue
    pe = eb.raw.ebeamPhotonEnergy(evt)
    lam = HC / (pe if pe and pe > 1000 else 11624.0)
    si = peak_idx(img, S_SON, S_ABS)
    if len(si) < 8:
        continue
    wi = peak_idx(img, W_SON, W_ABS)                    # low threshold = strong + weak
    qs = to_q(si, lam, img, CAP_S)
    qw_all = to_q(wi, lam, img, CAP_W)
    if len(qw_all) and len(qs):
        d = np.linalg.norm(qw_all[:, None, :] - qs[None, :, :], axis=2).min(1)
        qw = qw_all[d > 1e-4]                           # keep only SUB-threshold peaks
    else:
        qw = qw_all
    used += 1
    frames.append((qs.astype(np.float32), qw.astype(np.float32)))
    print(f"frame {used}: strong={len(qs)} weak_sub={len(qw)}", flush=True)

blob = {"n": len(frames)}
blob.update({f"s{i}": f[0] for i, f in enumerate(frames)})
blob.update({f"w{i}": f[1] for i, f in enumerate(frames)})
np.savez(OUT, **blob)
print(f"wrote {len(frames)} frames -> {OUT}")
