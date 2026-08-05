"""Does the GPU path match psana's det.calib, and what is common mode worth? STATUS.md item 4.

det.calib is 145.8 ms of a 149.2 ms event; the same arithmetic runs on the device in about a
millisecond. Before any of that is used, two questions have to be answered with numbers:

  1. SPEED. What is the GPU path actually worth on this route?
  2. FIDELITY. Does it produce the same image, and -- the only part that matters downstream -- the
     same peaks?

Fidelity is reported three ways, because they answer different questions:
  * per-pixel residual        -- how far the images differ at all
  * residual vs the frame RMS -- is the difference below or above what the peak-finder calls signal
  * PEAK-SET agreement        -- run the SAME finder on both and compare the hits, because a bias
                                 that shifts every pixel equally changes nothing about which pixels
                                 are peaks, and a small bias in the wrong place changes everything

TWO GPU CONFIGURATIONS are measured against the one CPU reference, because their difference is the
answer to a question that was previously guessed at. `full` mirrors det.calib including common mode;
`no-cm` is pedestal and gain only, which is all that `Reader.cu` does and all that this module did
before 2026-08-05. Comparing their Jaccards says what reinstating common mode actually bought, and
comparing either against the earlier 58.6% says what the inverted-gain fix bought. Common mode is
capped at cormax ADU per group by construction, so if it moves the peak set much, that is worth
knowing before the DAQ's pedestal-and-gain-only path is trusted on this detector.
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

full = gpu_calib.GpuCalibrator(det, RUN)
nocm = gpu_calib.GpuCalibrator(det, RUN, cmpars=(7, 0, 0, 0))
print(f"shape={full.shape}  det.common_mode(run)={np.asarray(full.cmpars).tolist()}")
print(f"applied: mode={full.mode} (bitmask +4 banks +1 rows +2 cols)  "
      f"cormax={full.cormax} ADU  npixmin={full.npixmin}")
print("gain: psana gfac = 1/det.gain (ADU/keV -> keV/ADU), NOT a multiply by det.gain")

pf = xtc_core.load_peakfinder_v4().PeakFinderV4
finders = None

t_cpu, t_full, t_nocm = [], [], []
res = {"full": [], "no-cm": []}
rel = {"full": [], "no-cm": []}
agree = {"full": [0, 0, 0], "no-cm": [0, 0, 0]}     # shared, only-CPU, only-GPU
n = 0
for evt in ds.events():
    if n >= N:
        break
    raw = det.raw(evt)
    if raw is None:
        continue
    t0 = time.perf_counter(); ref = det.calib(evt); t1 = time.perf_counter()
    if ref is None:
        continue
    cp.cuda.Stream.null.synchronize()
    t2 = time.perf_counter(); g_full = full(raw); cp.cuda.Stream.null.synchronize()
    t3 = time.perf_counter(); g_nocm = nocm(raw); cp.cuda.Stream.null.synchronize()
    t4 = time.perf_counter()
    t_cpu.append(t1 - t0); t_full.append(t3 - t2); t_nocm.append(t4 - t3)

    r = np.asarray(ref, np.float32)
    rms_r = float(np.std(r))

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

    a = hits(r)
    for tag, dev in (("full", g_full), ("no-cm", g_nocm)):
        gg = cp.asnumpy(dev).astype(np.float32)
        d = gg - r
        res[tag].append(float(np.sqrt(np.mean(d * d))))
        rel[tag].append(res[tag][-1] / rms_r if rms_r else np.nan)
        b = hits(gg)
        agree[tag][0] += len(a & b); agree[tag][1] += len(a - b); agree[tag][2] += len(b - a)
    n += 1

med = lambda v: 1e3 * float(np.median(v))
print(f"\n{n} events")
print(f"{'path':28s} {'median ms':>10s} {'speedup':>9s}")
print("-" * 50)
print(f"{'psana det.calib (CPU)':28s} {med(t_cpu):10.2f} {'':>9s}")
print(f"{'GPU full (ped+cm+gain+mask)':28s} {med(t_full):10.2f} "
      f"{med(t_cpu)/max(med(t_full),1e-9):8.0f}x")
print(f"{'GPU no-cm (ped+gain+mask)':28s} {med(t_nocm):10.2f} "
      f"{med(t_cpu)/max(med(t_nocm),1e-9):8.0f}x")

print(f"\nFIDELITY vs det.calib")
print(f"{'config':10s} {'RMS residual':>14s} {'/frame RMS':>12s} {'shared':>8s} {'onlyCPU':>8s} "
      f"{'onlyGPU':>8s} {'Jaccard':>9s}")
print("-" * 74)
for tag in ("full", "no-cm"):
    s, oc, og = agree[tag]
    print(f"{tag:10s} {np.median(res[tag]):11.3f} ADU {100*np.median(rel[tag]):11.2f}% "
          f"{s:8d} {oc:8d} {og:8d} {100.0*s/max(s+oc+og,1):8.1f}%")

print("\nJaccard is the number to judge on. `full` is the one that decides whether this can be wired")
print("into the reader; `full` minus `no-cm` is what common mode is worth on this detector, and")
print("hence whether the DAQ's pedestal-and-gain-only Reader.cu would be safe here.")
