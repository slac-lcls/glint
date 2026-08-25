"""fused_kernels.py stages each frame's peaks in dynamic shared memory as Pmax*3*_IB bytes (_IB = 4
fp32 / 8 fp64) at three sites. CUDA caps a launch at 48 KB per block unless the FUNCTION opts in via
cudaFuncSetAttribute(MaxDynamicSharedMemorySize), so before issue #69 the ceiling was
48*1024/(3*_IB) = 2048 (fp64) / 4096 (fp32) and one peak more killed the run with an uncaught
CUDA_ERROR_INVALID_VALUE.

This checks the two halves of the fix together:
  1. the opt-in moved the ceiling to the device's own MaxSharedMemoryPerBlockOptin
     (A100: 166,912 B -> 6954 fp64 / 13909 fp32), so 2049 and 4097 now run on the KERNELS; and
  2. above that ceiling index_fused does not raise at all -- the frame is routed to the stock
     (non-fused) torch path, which has no shared-memory limit.

GPU node + cupy; exits 0 with a SKIP without them.

  KC_FP=64 PYTHONPATH=. python experiments/test_pmax_limit.py
  KC_FP=32 PYTHONPATH=. python experiments/test_pmax_limit.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # repo root (imports glint.*)
sys.path.insert(0, HERE)                                  # experiments/ (imports glint_fast)

try:
    import cupy as cp, torch
    import glint.replica_gpu_batch as rgb
    import glint.fused_kernels as fk
    from glint_fast import LYSO
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
except Exception as exc:                                  # CPU runner / no cupy -> nothing to test
    print(f"SKIP test_pmax_limit: {type(exc).__name__}: {exc}")
    sys.exit(0)

KC = os.environ.get("KC_FP", "64")
d = cp.cuda.Device()
default_cap = d.attributes["MaxSharedMemoryPerBlock"]
optin_cap = d.attributes.get("MaxSharedMemoryPerBlockOptin", 0)
old = 48 * 1024 // (3 * fk._IB)                           # the pre-#69 ceiling
new = fk.max_peaks()                                      # the ceiling after the opt-in

print(f"KC_FP={KC}  rgb.FP={rgb.FP}  _IB={fk._IB}")
print(f"  MaxSharedMemoryPerBlock      : {default_cap:>9,} B   -> Pmax {old}   (the old ceiling)")
print(f"  MaxSharedMemoryPerBlockOptin : {optin_cap:>9,} B")
print(f"  opted in to                  : {fk._smem_cap():>9,} B   -> Pmax {new}\n")

rng = np.random.default_rng(0)
fail = []
if new <= old:
    fail.append(f"opt-in did not raise the ceiling ({new} <= {old})")

# Points either side of the OLD ceiling must now run on the kernels, and points past the NEW ceiling
# must come back from the fallback -- in both cases without an exception escaping index_fused.
probes = sorted({512, old, old + 1, 2 * old, new, new + 1, new + 500, 2 * new})
for P in probes:
    q = rng.normal(0, 0.15, (P, 3))                       # one frame, P peaks
    lane = "fused" if P <= new else "fallback"
    try:
        M = rgb.index_fused([q], LYSO, B=1)
        got = "a matrix" if M[0] is not None else "None (miss)"
        print(f"  Pmax={P:6d}  {lane:8s}  OK    returned {got}")
    except Exception as e:
        print(f"  Pmax={P:6d}  {lane:8s}  RAISED {type(e).__name__}: {str(e)[:80]}")
        fail.append(f"Pmax={P} ({lane}) raised {type(e).__name__}")

print("\n" + ("FAIL: " + "; ".join(fail) if fail else
              f"PASS: fused ceiling {old} -> {new} peaks, and >{new} degrades to the stock path"))
sys.exit(1 if fail else 0)
