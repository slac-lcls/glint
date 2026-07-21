"""Run PeakNet-673M on real MFX CXI frames ($GLINT_PEAKNET_CXI) and extract STRONG (above
threshold) + WEAK (sub-threshold p_peak) peaks -- the real-data weak-peak signal that
is Cong's whole idea and the input to soft-completeness scoring.

Validation hook: my strong-peak count should track the CXI's deployment nPeaks (the
pipeline's own PeakNet peaks) -> confirms the inference is faithful. The weak peaks
(0.1 < p_peak <= 0.5, local maxima) are what the strong threshold discards.

  python run_peaknet_cxi.py [N_FRAMES] [CXI_GLOB]
"""
import os, sys, glob
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_peaknet import load_model, seg_forward

CXI_DIR = os.environ.get("GLINT_PEAKNET_CXI")   # lives in a colleague's project area
if not CXI_DIR:
    sys.exit("set GLINT_PEAKNET_CXI=<dir of PeakNet CXI files>; the path names a beamtime\n"
             "run and sits in someone else's project space, so it is not committed")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
GLOB = sys.argv[2] if len(sys.argv) > 2 else CXI_DIR + "/*.cxi"
STRONG_CUT, WEAK_LO = 0.5, 0.10

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16 if dev == "cuda" else torch.float32
print(f"device={dev} dtype={dt}", flush=True)
m = load_model(dev, dt)


def instance_norm(img):
    """InstanceNorm over valid pixels (exclude assembled-gap fill value)."""
    fin = np.isfinite(img)
    fill = np.min(img[fin]) if fin.any() else 0.0
    valid = fin & (img > fill + 1e-6)
    if valid.sum() < 100:
        valid = fin
    mu = img[valid].mean()
    sd = img[valid].std() + 1e-6
    out = (img - mu) / sd
    out[~fin] = 0.0
    return out, valid


def pad32(t):
    H, W = t.shape[-2:]
    Hp, Wp = ((H + 31) // 32) * 32, ((W + 31) // 32) * 32
    return torch.nn.functional.pad(t, (0, Wp - W, 0, Hp - H)), (H, W)


def p_peak_map(img):
    xn, valid = instance_norm(img.astype(np.float32))
    t = torch.from_numpy(xn)[None, None]
    tp, (H, W) = pad32(t)
    tp = tp.to(dev, dt)
    with torch.no_grad():
        logit = seg_forward(m, tp).float()
    prob = torch.softmax(logit, dim=1)[0, 1, :H, :W]            # p_peak, on device
    prob = prob * torch.from_numpy(valid.astype(np.float32)).to(prob.device)
    return prob


def local_maxima(prob, lo, hi, k=5):
    """local maxima of p_peak in (lo, hi], via torch max_pool (GPU-native)."""
    p = prob[None, None]
    mx = F.max_pool2d(p, kernel_size=k, stride=1, padding=k // 2)
    loc = (p == mx) & (p > lo) & (p <= hi)
    return torch.argwhere(loc[0, 0]).cpu().numpy()


def maxima_with_p(prob, lo, hi, k=5):
    """local maxima coords (row,col) + their p_peak value, in (lo, hi]."""
    rc = local_maxima(prob, lo, hi, k)
    if len(rc) == 0:
        return np.zeros((0, 3), np.float32)
    pv = prob[rc[:, 0], rc[:, 1]].float().cpu().numpy()
    return np.concatenate([rc.astype(np.float32), pv[:, None]], axis=1)  # row,col,p


import h5py
OUT = os.environ.get("OUT", "/sdf/home/s/smarches/peaknet_sw.npz")
files = sorted(glob.glob(GLOB))
print(f"{len(files)} CXI files; reading first up to {N} frames", flush=True)
done = 0
frames = []                                                 # (energy_eV, strong[r,c,p], weak[r,c,p])
for f in files:
    if done >= N:
        break
    with h5py.File(f, "r") as h:
        data = h["/entry_1/data_1/data"]
        npk = h["/entry_1/result_1/nPeaks"][:]
        try:
            en = h["/LCLS/photon_energy_eV"][:]
        except Exception:
            en = np.full(data.shape[0], 9500.0, np.float32)
        ne = data.shape[0]
        for i in range(ne):
            if done >= N:
                break
            img = data[i][:]
            prob = p_peak_map(img)
            strong = maxima_with_p(prob, STRONG_CUT, 1.01)
            weak = maxima_with_p(prob, WEAK_LO, STRONG_CUT)
            done += 1
            frames.append((float(en[i]), strong, weak))
            print(f"frame {done}: CXI_nPeaks={int(npk[i])} eV={float(en[i]):.0f} | "
                  f"strong(p>{STRONG_CUT})={len(strong)} "
                  f"weak({WEAK_LO}-{STRONG_CUT})={len(weak)} "
                  f"pmax={float(prob.max()):.3f}", flush=True)

blob = {"n": len(frames), "energy": np.array([f[0] for f in frames], np.float32)}
blob.update({f"s{i}": f[1] for i, f in enumerate(frames)})
blob.update({f"w{i}": f[2] for i, f in enumerate(frames)})
np.savez(OUT, **blob)
print(f"wrote {len(frames)} frames -> {OUT}", flush=True)
print("done", flush=True)
