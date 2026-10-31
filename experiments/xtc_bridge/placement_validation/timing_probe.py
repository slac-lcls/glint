"""Mona's timing point, measured: after `from mpi4py import MPI`, does a later CUDA_VISIBLE_DEVICES write
still take effect? Hide every GPU AFTER the MPI import; if CUDA was already initialized, CuPy still sees
the GPU (the write was ignored)."""
import os
from mpi4py import MPI  # noqa: F401
with open("/proc/self/maps") as f:
    mapped = any("libcuda.so" in line for line in f)
before = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import cupy
    n = cupy.cuda.runtime.getDeviceCount()
except Exception as e:
    n, why = 0, f" ({type(e).__name__})"
else:
    why = ""
print(f"TIMING mask before={before!r}; libcuda mapped after mpi4py import: {mapped}; "
      f"after hiding all GPUs post-import, CuPy sees {n} device(s){why} -> the write "
      f"{'was IGNORED (CUDA already initialized)' if n > 0 else 'took effect'}")
