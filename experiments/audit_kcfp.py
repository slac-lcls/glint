"""AUDIT-ONLY part 3: KC_FP on the driver's actual per-frame indexer (rgb.index_fused).

Runs the 120 real cxidb frames through index_fused at the working precision selected by KC_FP
(set BEFORE import, since fused_kernels JITs against rgb.FP), times it, and writes the per-frame
orientation matrices so a second run at the other precision can be compared frame by frame.
"""
import os, sys, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import glint.replica_gpu_batch as rgb
from glint.replica_gpu import load
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
frames = [q for q in load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
n = len(frames)
B = int(os.environ.get("B", "64"))
FP = os.environ.get("KC_FP", "64")

def gpass(M, q):
    if M is None: return 0.0, 0
    h = np.asarray(q, float) @ np.asarray(M, float)
    m = int((np.abs(h - np.rint(h)).max(1) < 0.15).sum())
    return m / len(q), m

rgb.index_fused(frames[:B], LYSO, B=B); torch.cuda.synchronize()      # warm + JIT
REPS = 5
t0 = time.perf_counter()
for _ in range(REPS):
    Ms = rgb.index_fused(frames, LYSO, B=B)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / REPS

g = np.array([gpass(M, q) for M, q in zip(Ms, frames)])
lat = sum(1 for M in Ms if M is not None and same_lattice(np.asarray(M, float), LYSO))
print(f"KC_FP={FP}  dev={torch.cuda.get_device_name(0)}  n={n}  B={B}")
print(f"  index_fused: {1e3*dt/n:.3f} ms/frame  ({n/dt:.0f} frames/s)")
print(f"  frac>=0.25: {int((g[:,0]>=0.25).sum())}/{n}   >=10 refl: {int((g[:,1]>=10).sum())}/{n}   same_lattice: {lat}/{n}")
with open(f"/tmp/audit_kc_{FP}.pkl", "wb") as f:
    pickle.dump([None if M is None else np.asarray(M, float) for M in Ms], f)

other = "32" if FP == "64" else "64"
p = f"/tmp/audit_kc_{other}.pkl"
if os.path.exists(p):
    ref = pickle.load(open(p, "rb"))
    nd = sum(1 for a, b in zip(ref, Ms)
             if (a is None) != (b is None) or (a is not None and not same_lattice(a, np.asarray(b, float))))
    md = max((float(np.abs(a - np.asarray(b, float)).max()) for a, b in zip(ref, Ms)
              if a is not None and b is not None), default=float("nan"))
    fr = np.array([gpass(M, q) for M, q in zip(ref, frames)])
    print(f"  vs KC_FP={other}: lattice-differing frames {nd}/{n}   max |dM| {md:.3e}   "
          f"frac>=0.25 there {int((fr[:,0]>=0.25).sum())}/{n}")
