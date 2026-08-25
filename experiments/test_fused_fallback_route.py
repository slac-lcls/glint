"""index_fused splits out frames with more peaks than fit in shared memory (issue #69) and runs them
through the stock torch ops instead, one at a time, then stitches the two lanes back together. Two
things there can go wrong silently: the fallback could give a different answer, and the stitch could
put a frame back in the wrong slot.

Force the split by lowering fused_kernels.max_peaks() to the median peak count -- so half of the real
120 cxidb frames take the fallback -- and compare per-frame against the all-fused run of the same
frames. fp64 must agree to round-off (the fused kernels are bit-exact ports of the stock ops); fp32
agrees on rate and lattice, differing only by equivalent-basis argmax ties, as index_fused documents.

GPU node + cupy; exits 0 with a SKIP without them.

  KC_FP=64 PYTHONPATH=. python experiments/test_fused_fallback_route.py
  KC_FP=32 PYTHONPATH=. python experiments/test_fused_fallback_route.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # repo root (imports glint.*)
sys.path.insert(0, HERE)                                  # experiments/ (imports glint_fast)

try:
    import torch
    import glint.replica_gpu_batch as rgb
    import glint.fused_kernels as fk
    from glint_fast import load, gpass, LYSO
    from glint.multishot import same_lattice
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
except Exception as exc:
    print(f"SKIP test_fused_fallback_route: {type(exc).__name__}: {exc}")
    sys.exit(0)

frames = [q for q in load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]
n = len(frames); npk = np.array([len(f) for f in frames])
tag = "fp32" if rgb.FP == torch.float32 else "fp64"


def rate(Ms):
    g = np.array([gpass(M, q) for M, q in zip(Ms, frames)]); return int(g[:, 0].sum()), int(g[:, 1].sum())


A = rgb.index_fused(frames, LYSO, B=32)                   # every frame on the fused kernels
cut = int(np.median(npk))
orig = fk.max_peaks
fk.max_peaks = lambda: cut                                # force the split
try:
    B = rgb.index_fused(frames, LYSO, B=32)
finally:
    fk.max_peaks = orig

routed = int((npk > cut).sum())
dmax = max(float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max()) for a, b in zip(A, B))
sl = sum(1 for a, b in zip(A, B) if same_lattice(np.asarray(a, float), np.asarray(b, float)))
nones = sum(1 for b in B if b is None)
rA, rB = rate(A), rate(B)

print(f"[{tag}] {n} cxidb frames, {npk.min()}-{npk.max()} peaks; ceiling forced to {cut} "
      f"-> {routed} frames on the non-fused fallback")
print(f"   rate      all-fused {rA}   split {rB}   None {nones}")
print(f"   per-frame max|dM| {dmax:.3e}   same_lattice {sl}/{n}")

fail = []
if routed == 0:               fail.append("nothing was routed to the fallback -- the test is vacuous")
if rA != rB:                  fail.append(f"rate changed {rA} -> {rB}")
if sl != n:                   fail.append(f"same_lattice {sl}/{n}")
if nones:                     fail.append(f"{nones} frames came back None")
if tag == "fp64" and dmax > 1e-9:
    fail.append(f"fp64 fallback is not bit-faithful (max|dM| {dmax:.3e})")
print("   " + ("FAIL: " + "; ".join(fail) if fail else "PASS"))
sys.exit(1 if fail else 0)
