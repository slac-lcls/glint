"""Does GPU pedestal+gain match psana's det.calib, and what does dropping common mode cost?

STATUS.md item 4. det.calib is 145.8 ms of a 149.2 ms event; the DAQ does the same arithmetic on the
device in microseconds. Before any of that is used, two questions have to be answered with numbers:

  1. SPEED. What is the GPU path actually worth on this route?
  2. FIDELITY. det.calib applies pedestal, gain AND common mode -- a data-dependent per-ASIC/row
     median subtraction that Reader.cu has no equivalent of. The GPU path here does pedestal and gain
     only. That difference is the whole risk, and it is measured here rather than argued about.

Fidelity is reported three ways, because they answer different questions:
  * per-pixel residual        -- how far the images differ at all
  * residual vs the noise     -- is the difference below or above what the peak-finder calls signal
  * PEAK-SET agreement        -- the only one that matters downstream: run the SAME finder on both
                                 and compare the hits, because a bias that shifts every pixel
                                 equally changes nothing about which pixels are peaks
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

G = os.environ.get("GLINT_ROOT", "/sdf/home/s/smarches/glint_bench_pf8")
sys.path.insert(0, G); sys.path.insert(0, G + "/experiments/xtc_bridge")

EXP, RUN, DET = "mfxx49820", 16, "MfxEndstation.0:Epix10ka2M.0"
H = "/sdf/home/s/smarches/glint_gt_mfxx49820"
N = int(os.environ.get("NEV", "40"))

import psana
psana.setOption("psana.calib-dir", f"{H}/calib")
ds = psana.DataSource(f"exp={EXP}:run={RUN}")
det = psana.Detector(DET)

import cupy as cp
import gpu_calib, xtc_core

gc = gpu_calib.GpuCalibrator(det, RUN)
print(f"decode: psana gain modes via gain_maps_epix10ka_any, two precomputed plane pairs "
      f"(data bit 14 clear/set)  shape={gc.shape}")

pf = xtc_core.load_peakfinder_v4().PeakFinderV4
finders = None

t_cpu, t_gpu = [], []
res_rel, res_sig = [], []
same, only_cpu, only_gpu = 0, 0, 0
n = 0
for i, evt in enumerate(ds.events()):
    if n >= N:
        break
    raw = det.raw(evt)
    if raw is None:
        continue
    t0 = time.perf_counter(); ref = det.calib(evt); t1 = time.perf_counter()
    if ref is None:
        continue
    cp.cuda.Stream.null.synchronize()
    t2 = time.perf_counter(); got = gc(raw); cp.cuda.Stream.null.synchronize()
    t3 = time.perf_counter()
    t_cpu.append(t1 - t0); t_gpu.append(t3 - t2)

    g = cp.asnumpy(got).astype(np.float32)
    r = np.asarray(ref, np.float32)
    d = g - r
    rms_r = float(np.std(r))
    res_rel.append(float(np.sqrt(np.mean(d * d))))
    res_sig.append(res_rel[-1] / rms_r if rms_r else np.nan)

    if finders is None:
        try:
            m = det.mask(evt, status=True, calib=True, edges=False, central=False)
            good = m.astype(bool) if m is not None else np.ones(r.shape, bool)
        except Exception:
            good = np.ones(r.shape, bool)
        finders = [pf(cp.asarray(good[p]), dtype=cp.float32, min_pix=xtc_core.PF_MIN_PIX,
                      son_min=xtc_core.PF_SON_MIN, thr_high=xtc_core.PF_THR_HIGH,
                      thr_low=xtc_core.PF_THR_LOW) for p in range(r.shape[0])]

    def hits(frame):                       # (panel, rounded fs, rounded ss) set
        s = set()
        for p in range(frame.shape[0]):
            pk = finders[p].find(cp.asarray(frame[p], cp.float32))
            xs = cp.asnumpy(pk["x"]); ys = cp.asnumpy(pk["y"])
            for x, y in zip(np.rint(xs).astype(int), np.rint(ys).astype(int)):
                s.add((p, int(x), int(y)))
        return s

    a, b = hits(r), hits(g)
    same += len(a & b); only_cpu += len(a - b); only_gpu += len(b - a)
    n += 1

med = lambda v: 1e3 * float(np.median(v))
print(f"\n{n} events")
print(f"{'path':22s} {'median ms':>10s}")
print("-" * 34)
print(f"{'psana det.calib (CPU)':22s} {med(t_cpu):10.2f}")
print(f"{'GPU ped+gain':22s} {med(t_gpu):10.2f}")
print("-" * 34)
print(f"speedup: {med(t_cpu)/max(med(t_gpu),1e-9):.0f}x on the calibration stage alone")

print(f"\nFIDELITY (GPU ped+gain vs det.calib, which also does common mode)")
print(f"  per-pixel RMS residual   : {np.median(res_rel):.3f} ADU")
print(f"  residual / frame RMS     : {100*np.median(res_sig):.2f}%")
print(f"\nPEAK-SET agreement, same PeakFinderV4 and thresholds on both images:")
tot = same + only_cpu + only_gpu
print(f"  peaks found on BOTH      : {same}")
print(f"  only on det.calib        : {only_cpu}")
print(f"  only on GPU ped+gain     : {only_gpu}")
print(f"  Jaccard                  : {100.0*same/max(tot,1):.1f}%")
print("\nThe Jaccard is the number to judge on. A large per-pixel residual with a high Jaccard means")
print("common mode shifts the background but not which pixels are peaks; a low Jaccard means the")
print("fast path changes the hit set and must not be used without reinstating common mode.")
