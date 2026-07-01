"""Stage 2: pin nanoBragg's detector convention by overlaying fftindex.predict_spots on the
simulated still. nanoBragg rendered sim_still0.npy at orientation B (columns a*,b*,c*). My predict
uses R with ROWS a*,b*,c* -> R = B.T. Build the matching single-panel geom (0.2mm px -> res=5000,
1024^2, beam at 512.5 -> corner=-512.5, dist 0.15m, fs=+x ss=+y) and check predicted spots land on
the simulated Bragg peaks. If a flip/transpose remains, the residual is structured -> adjust.

  source psconda.sh ; conda activate ana-4.0.58-py3-minipytorch ; python stage2_overlay.py
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from scipy.ndimage import maximum_filter
from glint.predict import predict_spots

HOME = "/sdf/home/s/smarches"
img = np.load(f"{HOME}/sim_still0.npy")
B = np.load(f"{HOME}/sim_still0_B.npy")          # columns a*,b*,c* (1/A)
R = B.T                                          # rows a*,b*,c* for predict (q = hkl @ R)
WAVE, CLEN, DMIN = 1.32, 0.15, 3.0

def geom(fs, ss, cx, cy, res=5000.0):
    return [dict(name="p0", fs=np.array(fs, float), ss=np.array(ss, float), res=res,
                 cx=cx, cy=cy, coffset=0.0, min_fs=0, max_fs=1023, min_ss=0, max_ss=1023)]

# simulated Bragg peaks (ss=slow, fs=fast)
mx = maximum_filter(img, size=7)
ys, xs = np.where((img == mx) & (img > 0.12 * img.max()))
sim = np.column_stack([xs, ys]).astype(float)     # (fs, ss)
print(f"{len(sim)} simulated Bragg maxima; img max={img.max():.0f}")

# try candidate detector conventions; the correct one makes predicted spots overlay the maxima
CANDS = {
    "fs=+x ss=+y  c=-512.5": (geom([1, 0, 0], [0, 1, 0], -512.5, -512.5)),
    "fs=+x ss=-y  c=-512.5/+512.5": (geom([1, 0, 0], [0, -1, 0], -512.5, 512.5)),
    "fs=-x ss=+y": (geom([-1, 0, 0], [0, 1, 0], 512.5, -512.5)),
    "fs=+y ss=+x (transpose)": (geom([0, 1, 0], [1, 0, 0], -512.5, -512.5)),
}
for label, panels in CANDS.items():
    pred = predict_spots(R, panels, CLEN, WAVE, dmin=DMIN, tol=0.004, is_recip=True)
    if len(pred) == 0:
        print(f"  {label:32}: 0 predicted"); continue
    pp = np.column_stack([pred["fs"], pred["ss"]])
    # nearest predicted spot to each simulated maximum
    d = np.sqrt(((sim[:, None, :] - pp[None, :, :]) ** 2).sum(-1)).min(1)
    matched = (d < 3.0).sum()
    print(f"  {label:32}: {len(pred):4d} pred,  {matched:3d}/{len(sim)} maxima matched <3px, "
          f"median nearest={np.median(d):.2f}px")
