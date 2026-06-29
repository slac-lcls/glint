"""Stage A' (psana2): extract per-peak LAB X,Y + intensity + per-frame lambda for
real mfx run-51 frames, so the detector DISTANCE (the only free geometry param that
enters q via Z) can be swept OFFLINE without re-running psana. Strong + sub-threshold
weak peaks both saved.  q(ZDIST) = ( rr/|rr| - kin ) / lam,  rr = [X, Y, ZDIST].

  source .../psconda.sh ; python extract_pixels.py [N_FRAMES] [OUT.npz]
"""
import sys, os
import numpy as np
from scipy.ndimage import maximum_filter
from psana import DataSource
from psana.pscalib.geometry.GeometryAccess import GeometryAccess

HC = 12398.42
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 80
OUT = sys.argv[2] if len(sys.argv) > 2 else "/sdf/home/s/smarches/real_xy.npz"
S_SON, S_ABS = 6.0, 200.0
W_SON, W_ABS = float(os.environ.get("W_SON", "3.0")), float(os.environ.get("W_ABS", "30.0"))
CAP_S, CAP_W = 600, 1200

ds = DataSource(exp="mfx100848724", run=51)
myrun = next(ds.runs())
det = myrun.Detector("jungfrau")
eb = myrun.Detector("ebeamh")
geo = GeometryAccess()
geo.load_pars_from_str(det.calibconst["geometry"][0])
X, Y, Z = (a.astype(float).reshape(32, 512, 1024) for a in geo.get_pixel_coords())
X, Y = X * 1e-6, Y * 1e-6                                # um -> m
print(f"psana Z mean = {np.nanmean(Z)/1e6:.4f} m", flush=True)
status = det.calibconst["pixel_status"][0].reshape(-1, 32, 512, 1024)
badmask = (status != 0).any(axis=0)


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


def grab(idx, img, cap):
    if len(idx) == 0:
        return np.zeros((0, 3))                          # x, y, intensity
    p, r, c = idx.T
    inten = img[p, r, c]
    if cap and len(idx) > cap:
        sel = np.argsort(inten)[::-1][:cap]
        p, r, c, inten = p[sel], r[sel], c[sel], inten[sel]
    return np.stack([X[p, r, c], Y[p, r, c], inten], axis=1)


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
    wi = peak_idx(img, W_SON, W_ABS)
    S = grab(si, img, CAP_S)
    Wn = grab(wi, img, CAP_W)
    # weak = sub-threshold only (xy not among strong)
    if len(Wn) and len(S):
        d = np.linalg.norm(Wn[:, None, :2] - S[None, :, :2], axis=2).min(1)
        Wn = Wn[d > 1e-7]
    used += 1
    frames.append((lam, S.astype(np.float32), Wn.astype(np.float32)))
    print(f"frame {used}: strong={len(S)} weak_sub={len(Wn)} lam={lam:.4f}", flush=True)

blob = {"n": len(frames), "lam": np.array([f[0] for f in frames], np.float32)}
blob.update({f"s{i}": f[1] for i, f in enumerate(frames)})
blob.update({f"w{i}": f[2] for i, f in enumerate(frames)})
np.savez(OUT, **blob)
print(f"wrote {len(frames)} frames -> {OUT}")
