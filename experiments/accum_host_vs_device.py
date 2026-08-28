"""Is the DRP table's `accumulate 0.200 ms/frame` the HOST MergeAccumulator or the device one?

Static evidence says host: GLINT_DEVICE_MERGE is set NOWHERE in the repo (only read at
stream_driver.py:384), and check_numbers.py:97 annotates accumulate_ms as "on the host". But the
Confluence DRP comment credits part of the 5.58 -> 1.468 ms narrowing to "moving the running merge
accumulate onto the GPU", so either the measurement exported the flag by hand or the text credits a
change that was not active. Settle it by measuring both.

A fair comparison has to include the DRAIN. MergeAccumulatorDevice.add_frame does NOT do GPU work per
frame -- it appends to preallocated host staging arrays and sets _dirty; the sort+segment-reduce is
deferred to _drain() at stats() time. So its per-frame cost is a few slice assignments, and its real
cost shows up once at drain. Measuring only add_frame would flatter it enormously.

  python accum_host_vs_device.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")

MODE = os.environ.get("GLINT_DEVICE_MERGE", "0")
import cupy as cp
from glint.glint_fast import cell_to_Ar
import glint.stream_driver as sd

SIM = os.environ.get("GLINT_SIM", "/sdf/home/s/smarches/glint_sim")
imgs = np.load(f"{SIM}/images.npy")
t = np.load(f"{SIM}/truth.npz")
N = int(t["det_n"]); pix_mm = float(t["pix_mm"]); dist_mm = float(t["dist_mm"]); wave = float(t["wave_A"])
cell = t["cell"]
panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
               res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
               coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
Mc = cell_to_Ar(float(cell[0]), float(cell[1]), float(cell[2]), 90, 90, 90)
clen = dist_mm / 1000.0

# ---- instrument add_frame on whichever class the driver will actually construct ----------------
CLS = None
drv0 = sd.StreamDriver(Mc, panels, clen, wave, (N, N), dtype=imgs.dtype, B=40, dmin=2.0, tol=0.002)
CLS = type(drv0.acc)
print(f"GLINT_DEVICE_MERGE={MODE!r}  ->  driver constructed {CLS.__module__}.{CLS.__name__}")

_orig = CLS.add_frame
STATE = {"t": 0.0, "n": 0, "nref": 0}


def timed(self, hkl, I, sigma, frame_index, *a, **k):
    t0 = time.perf_counter()
    r = _orig(self, hkl, I, sigma, frame_index, *a, **k)
    STATE["t"] += time.perf_counter() - t0
    STATE["n"] += 1
    STATE["nref"] += len(np.asarray(I))
    return r


CLS.add_frame = timed

drv = sd.StreamDriver(Mc, panels, clen, wave, (N, N), dtype=imgs.dtype, B=40, dmin=2.0, tol=0.002)
t_all0 = time.perf_counter()
REP = int(os.environ.get("REP", "1"))          # cycle the frames so a one-time JIT/drain amortises
for _ in range(REP):
    for f in imgs:
        drv.push(f)
drv.flush()
cp.cuda.Stream.null.synchronize()
t_push = time.perf_counter() - t_all0

# stats() triggers the device path's deferred _drain(); time it separately
t_s0 = time.perf_counter()
s = drv.stats()
cp.cuda.Stream.null.synchronize()
t_stats = time.perf_counter() - t_s0

nfr = max(STATE["n"], 1)
print(f"\nframes accumulated : {STATE['n']}   reflections {STATE['nref']} "
      f"({STATE['nref']/nfr:.0f}/frame)")
print(f"add_frame TOTAL    : {1e3*STATE['t']:8.2f} ms   = {1e3*STATE['t']/nfr:6.3f} ms/frame")
print(f"stats()/drain      : {1e3*t_stats:8.2f} ms   = {1e3*t_stats/nfr:6.3f} ms/frame amortised")
print(f"ACCUMULATE TOTAL   : {1e3*(STATE['t']+t_stats):8.2f} ms   "
      f"= {1e3*(STATE['t']+t_stats)/nfr:6.3f} ms/frame   <-- comparable to the DRP table's 0.200")
print(f"whole push+flush   : {1e3*t_push:8.2f} ms   = {1e3*t_push/nfr:6.3f} ms/frame")
print(f"\nmerge stats: n_meas={s.get('n_meas')} unique={s.get('n_unique')} "
      f"cc12={s.get('cc12')} completeness={s.get('completeness')}")
