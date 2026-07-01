import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
import numpy as np
import torch
sys.path.insert(0, "..")
from glint import simulate_shot
from glint.cnn import UNet3D
from glint.transform import fft_volume
from glint.peakfind import find_peaks_classical

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("cnn_peakfinder.pt", map_location=dev)
model = UNet3D(); model.load_state_dict(ck["model"]); model.to(dev).eval()

rng = np.random.default_rng(7000)
shot = simulate_shot(rng=rng, cell=(50, 55, 60, 90, 90, 90), n_target=40)
vol, x = fft_volume(shot.g, shot.meta["qmax"], n=96)
with torch.no_grad():
    v = torch.as_tensor(vol, dtype=torch.float32, device=dev)[None, None]
    heat = model(v)[0, 0].cpu().numpy()
print("vol max %.2f | heat: max %.3f mean %.4f" % (vol.max(), heat.max(), heat.mean()))

pk, amp = find_peaks_classical(heat, x, min_len=3.0, rel_thresh=0.3)
print("cnn heatmap peaks found:", len(pk))
if len(pk):
    fr = np.linalg.solve(shot.M, pk[:8].T).T
    print("  top-8 |M^-1 v - round| (0=lattice vec):",
          np.round(np.max(np.abs(fr - np.rint(fr)), axis=1), 2))
    print("  top-8 lengths:", np.round(np.linalg.norm(pk[:8], axis=1), 1))

cp, ca = find_peaks_classical(vol, x, min_len=3.0)
fc = np.linalg.solve(shot.M, cp[:8].T).T
print("classical top-8 |M^-1 v - round|:",
      np.round(np.max(np.abs(fc - np.rint(fc)), axis=1), 2))
print("true axis lengths:", np.round(np.sort(np.linalg.norm(shot.M, axis=0)), 1))
