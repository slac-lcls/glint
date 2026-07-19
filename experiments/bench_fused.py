"""Validate + time the fully-fused known-cell engine (replica_gpu_batch.index_fused) against the
CUDA-graph path (index_all_graph) on the real 120 cxidb LYSO frames: per-frame identity, indexing
rate, speedup, and the DRP GPU-count projection. Runs at both KC_FP=64 and KC_FP=32. GPU node + cupy.

index_fused swaps in custom fused CUDA kernels (cupy RawKernel, nvrtc-JIT) for the anneal/obj/refine
per-candidate hot loops PLUS an on-device cpu_stage (batched buerger same_lattice), replacing the many
small per-stage torch kernels and the host tail. Output is IDENTICAL to the stock engine (bit-exact at
fp64; rate + lattice identical at fp32 -- fp32 differs only by equivalent-basis argmax ties on losing
candidates, which selection washes out).

  KC_FP=64 python experiments/bench_fused.py     # bit-exact vs graph on A100
  KC_FP=32 python experiments/bench_fused.py     # fp32 path (RTX-Blackwell deployment precision)
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # repo root (imports glint.*)
sys.path.insert(0, HERE)                                  # experiments/ (imports glint_fast)
from glint_fast import load, gpass, LYSO
import glint.replica_gpu_batch as rgb
from glint.multishot import same_lattice

frames = [q for q in load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]; n = len(frames)
tag = "fp32" if rgb.FP == torch.float32 else "fp64"


def rate(Ms):
    g = np.array([gpass(M, q) for M, q in zip(Ms, frames)]); return int(g[:, 0].sum()), int(g[:, 1].sum())


def bench(fn, reps=10):
    fn(); torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        best = min(best, 1e3 * (time.perf_counter() - t0) / n)
    return best


Mg = rgb.index_all_graph(frames, LYSO, B=32); torch.cuda.synchronize()
Mf = rgb.index_fused(frames, LYSO, B=32); torch.cuda.synchronize()
maxd = max(float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max()) for a, b in zip(Mg, Mf))
sl = sum(1 for a, b in zip(Mg, Mf) if same_lattice(np.asarray(a, float), np.asarray(b, float)))
rg, rf = rate(Mg), rate(Mf)
tg = bench(lambda: rgb.index_all_graph(frames, LYSO, B=32))
tf = bench(lambda: rgb.index_fused(frames, LYSO, B=32))

print(f"[{tag}] index_fused vs index_all_graph on {n} cxidb frames")
print(f"   correctness   max|ΔM| {maxd:.2e}   same_lattice {sl}/{n}   rate graph {rg} / fused {rf}"
      f"   {'PASS' if rg == rf else 'RATE DIFF'}")
print(f"   speed         graph {tg:.3f}   fused {tf:.3f}  ms/frame   ({tg / tf:.2f}x)")
print(f"   DRP @10% hit / 35 kHz (N = 3.5 x ms/fr):  graph {3.5 * tg:.1f}  ->  fused {3.5 * tf:.1f}  GPUs")
