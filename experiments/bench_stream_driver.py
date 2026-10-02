"""Two gates on the device-resident streaming driver (glint.stream_driver).

(1) CORRECTNESS: the running accumulator must reproduce the BATCH merge_stats.py math on identical
    measurements. If they agree, a low CC1/2 is a property of the dataset size, not a bug.
(2) COST: where do the remaining ms/frame go, now that the hkl grid is hoisted?
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import cupy as cp
from glint_fast import cell_to_Ar
import glint.stream_driver as sd

SIM = os.environ.get("GLINT_SIM", "/sdf/home/s/smarches/glint_sim")
imgs = np.load(f"{SIM}/images.npy"); t = np.load(f"{SIM}/truth.npz")
N = int(t["det_n"]); pix_mm = float(t["pix_mm"]); dist_mm = float(t["dist_mm"]); wave = float(t["wave_A"])
cell = t["cell"]
panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
               res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
               coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
Mc = cell_to_Ar(float(cell[0]), float(cell[1]), float(cell[2]), 90, 90, 90); clen = dist_mm / 1000.0
TOL = float(os.environ.get("TOL", "0.002"))

# ---- record every measurement the driver folds in, so the batch path sees the SAME data ----
REC = {"fr": [], "hkl": [], "I": [], "S": []}
_orig = sd.MergeAccumulator.add_frame


def rec_add(self, hkl, I, sigma, frame_index):
    REC["fr"].append(np.full(len(I), frame_index)); REC["hkl"].append(np.asarray(hkl, int))
    REC["I"].append(np.asarray(I, float)); REC["S"].append(np.asarray(sigma, float))
    return _orig(self, hkl, I, sigma, frame_index)


sd.MergeAccumulator.add_frame = rec_add
drv = sd.StreamDriver(Mc, panels, clen, wave, (N, N), dtype=imgs.dtype, B=40, dmin=2.0, tol=TOL)
for f in imgs:
    drv.push(f)
drv.flush(); cp.cuda.Stream.null.synchronize()

frames = np.concatenate(REC["fr"]); H = np.concatenate(REC["hkl"])
I = np.concatenate(REC["I"]); S = np.maximum(np.concatenate(REC["S"]), 1e-3)
print(f"recorded {len(I)} measurements over {frames.max()+1} frames\n")

# ---------------- batch reference: exactly merge_stats.py's math ----------------
OPS = sd.laue_ops_4mmm()
ch = sd.canon(H, OPS)
key = ch[:, 0].astype(np.int64) * 10 ** 8 + ch[:, 1] * 10 ** 4 + ch[:, 2]
gmean = I[I > 0].mean()
nf = frames.max() + 1
# frames whose mean intensity is not measured are dropped, the live merge's rule (frame_scale,
# review r2 s1-01/s1-04); they used to stay in raw units with scale 1
scale = np.full(nf, np.nan)
for fr in range(nf):
    m = frames == fr
    fs = sd.frame_scale(I[m])
    if fs is not None:
        scale[fr] = gmean * fs
print(f"{int(np.isnan(scale).sum())}/{nf} frames not merged (mean(I) not measured)")
_ok = np.isfinite(scale[frames])
frames, H, I, S = frames[_ok], H[_ok], I[_ok], S[_ok]
Is = I * scale[frames]


def batch_merge(mask):
    k = key[mask]; v = Is[mask]; w = 1.0 / S[mask] ** 2
    o = np.argsort(k); k, v, w = k[o], v[o], w[o]
    uk, idx = np.unique(k, return_index=True)
    return uk, np.add.reduceat(w * v, idx) / np.add.reduceat(w, idx), np.add.reduceat(np.ones_like(w), idx)


snr = I / S
odd = (frames % 2) == 1
print(f"{'I/sig':>6} | {'BATCH  uniq':>12}{'common':>8}{'CC1/2':>8}{'CC*':>8}{'Rsplit%':>9} "
      f"| {'STREAM uniq':>12}{'common':>8}{'CC1/2':>8}{'CC*':>8}{'Rsplit%':>9} | match")
ok_all = True
for thr in (-np.inf, 0.0, 1.0, 2.0, 3.0, 5.0):          # -inf: no floor, I <= 0 kept (review r2 s1-05)
    sel = snr > thr
    uk_all, _, cnt_all = batch_merge(sel)
    k1, v1, _ = batch_merge(sel & odd); k2, v2, _ = batch_merge(sel & ~odd)
    common, i1, i2 = np.intersect1d(k1, k2, return_indices=True)
    a, b = v1[i1], v2[i2]
    cc = float(np.corrcoef(a, b)[0, 1]) if len(common) >= 10 else float("nan")
    ccs = float(np.sqrt(2 * cc / (1 + cc))) if cc > 0 else float("nan")
    rs = float((1 / np.sqrt(2)) * np.sum(np.abs(a - b)) / (0.5 * np.sum(a + b))) if len(common) else float("nan")
    z = drv.stats(thr=thr)
    m = (len(uk_all) == z["unique"] and len(common) == z["common"]
         and abs(cc - z["cc_half"]) < 1e-9 and abs(rs - z["rsplit"]) < 1e-9)
    ok_all &= m
    print(f"{thr:>6} | {len(uk_all):>12}{len(common):>8}{cc:>8.4f}{ccs:>8.4f}{100*rs:>9.2f} "
          f"| {z['unique']:>12}{z['common']:>8}{z['cc_half']:>8.4f}{z['cc_star']:>8.4f}{100*z['rsplit']:>9.2f} "
          f"| {'OK' if m else 'DIFF'}")
print(f"\nstreaming == batch : {'PASS' if ok_all else 'FAIL'}")

# ---------------- (2) where the per-frame time goes ----------------
print("\n=== per-frame stage cost (driver internals) ===")
from glint.lute_bridge import peaks_to_q
from glint.predict import _canonical_axes
from glint.fused_integrate import integrate_fused
import glint.replica_gpu_batch as rgb
g = cp.asarray(imgs[0])


def tmin(fn, reps=10):
    fn(); cp.cuda.Stream.null.synchronize(); best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t0)
    return 1e3 * best


pk = drv.finder.find(g); fs = cp.asnumpy(pk["x"]); ss = cp.asnumpy(pk["y"])
q = peaks_to_q(fs, ss, panels, clen, wave); q = q[np.isfinite(q).all(1)]
M = rgb.index_fused([q], Mc, B=1)[0]; Mcan = _canonical_axes(np.asarray(M, float))
pred = drv.grid.predict(Mcan, panels, clen, wave, tol=TOL)
Iv, Sv, _, _ = integrate_fused(g, pred)
hkl = np.stack([pred["h"], pred["k"], pred["l"]], 1)
acc2 = sd.MergeAccumulator(ops=OPS)
print(f"  peakfind (device)      {tmin(lambda: drv.finder.find(g)):8.3f} ms   ({len(fs)} peaks)")
print(f"  peaks_to_q             {tmin(lambda: peaks_to_q(fs, ss, panels, clen, wave)):8.3f} ms")
print(f"  predict (hoisted grid) {tmin(lambda: drv.grid.predict(Mcan, panels, clen, wave, tol=TOL)):8.3f} ms   ({len(pred)} refl)")
print(f"  integrate_fused        {tmin(lambda: integrate_fused(g, pred)):8.3f} ms")
print(f"  accumulate             {tmin(lambda: _orig(acc2, hkl, Iv, Sv, 0), reps=5):8.3f} ms")
print(f"  index (batched, /frame){tmin(lambda: rgb.index_fused([q]*40, Mc, B=40))/40:8.3f} ms")
