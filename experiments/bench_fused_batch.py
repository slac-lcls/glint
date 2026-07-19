"""index_fused throughput vs batch size B. Each frame is one thread-block, so B sets GPU occupancy;
B>=64 saturates an A100. The kernels loop each frame's real peak count (not Pmax), so a larger,
looser-padded batch adds blocks with no work penalty and output is batch-invariant. GPU node + cupy.

  KC_FP=32 python experiments/bench_fused_batch.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from glint_fast import load, gpass, LYSO
import glint.replica_gpu_batch as rgb
frames = [q for q in load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]; n = len(frames)
tag = "fp32" if rgb.FP == torch.float32 else "fp64"


def rate(Ms):
    g = np.array([gpass(M, q) for M, q in zip(Ms, frames)]); return int(g[:, 0].sum()), int(g[:, 1].sum())


def bench(fn, reps=12):
    fn(); torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        best = min(best, 1e3 * (time.perf_counter() - t0) / n)
    return best


tg = bench(lambda: rgb.index_all_graph(frames, LYSO, B=32))
print(f"[{tag}] N={n}   index_all_graph(B=32) {tg:.3f} ms/fr   index_fused rate {rate(rgb.index_fused(frames, LYSO, B=32))}")
for B in (16, 32, 48, 64, 96, 120):
    t = bench(lambda: rgb.index_fused(frames, LYSO, B=B))
    print(f"    index_fused B={B:3d}   {t:.3f} ms/fr   ({tg / t:.2f}x vs graph)   DRP@10%/35kHz {3.5 * t:.2f} GPU")
