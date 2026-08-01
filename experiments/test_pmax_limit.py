"""fused_kernels.py sizes dynamic shared memory as Pmax*3*_IB bytes (_IB = 4 fp32 / 8 fp64) at three
sites (134/146/158) with no cudaFuncSetAttribute opt-in, so the 48 KB default cap predicts a hard
ceiling at Pmax = 48*1024/(3*_IB) = 2048 (fp64) / 4096 (fp32). Above it the launch should fail with
CUDA_ERROR_INVALID_VALUE rather than degrade. Find the ACTUAL threshold, check whether the failure is
caught anywhere, and report the A100's opt-in maximum so the size of the available fix is known.

  KC_FP=64 python test_pmax_limit.py ; KC_FP=32 python test_pmax_limit.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import cupy as cp, torch
import glint.glint_fast as gf
import glint.replica_gpu_batch as rgb
import glint.fused_kernels as fk

KC = os.environ.get("KC_FP", "64")
LYSO = gf.LYSO
d = cp.cuda.Device()
print(f"KC_FP={KC}  rgb.FP={rgb.FP}  _IB={fk._IB}")
print(f"GPU sharedMemPerBlock (default cap)  : {d.attributes['MaxSharedMemoryPerBlock']} B")
print(f"GPU sharedMemPerBlockOptin (opt-in)  : "
      f"{d.attributes.get('MaxSharedMemoryPerBlockOptin', 'n/a')} B")
pred = 48 * 1024 // (3 * fk._IB)
print(f"predicted Pmax ceiling at the 48 KB default: {pred}\n")

rng = np.random.default_rng(0)
lo, hi, last_ok, first_bad = 64, 16384, None, None
for P in (512, 1024, 2048, 2049, 3000, 4096, 4097, 6000, 8000):
    q = rng.normal(0, 0.15, (P, 3))                    # one frame, P peaks
    try:
        M = rgb.index_fused([q], LYSO, B=1)
        last_ok = P if (last_ok is None or P > last_ok) else last_ok
        print(f"  Pmax={P:6d}  OK      (returned {'a matrix' if M[0] is not None else 'None'})")
    except Exception as e:
        first_bad = P if first_bad is None else min(first_bad, P)
        print(f"  Pmax={P:6d}  FAIL    {type(e).__name__}: {str(e)[:90]}")

print(f"\nlast OK Pmax={last_ok}   first FAIL Pmax={first_bad}   predicted={pred}")
print(f"opt-in headroom: {d.attributes.get('MaxSharedMemoryPerBlockOptin',0)//(3*fk._IB)} peaks "
      f"if cudaFuncSetAttribute were used")
