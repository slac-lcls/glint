"""Validate + time the CUDA-graph known-cell engine (replica_gpu_batch.index_all_graph) against
the eager batched loop on the real 120 cxidb LYSO frames: bit-identity, rate, speedup. GPU node.

  python experiments/bench_kc_graph.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # repo root (imports glint.*)
sys.path.insert(0, HERE)                                  # experiments/ (imports glint_fast)
from glint_fast import load, gpass, LYSO
import glint.replica_gpu_batch as rgb

frames = [q for q in load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]; n = len(frames)
npk = np.array([len(f) for f in frames])
print(f"host {os.uname().nodename}  n={n}  peaks/frame med {int(np.median(npk))} max {npk.max()}  "
      f"adaptive_dirs {len(rgb._adaptive_dirs(LYSO, float(rgb._frame_qmax(frames).max())))}", flush=True)


def loop(B=32):
    return [M for i in range(0, n, B) for M in rgb.index_known_gpu_cell_batch(frames[i:i + B], LYSO)]


def graph():
    return rgb.index_all_graph(frames, LYSO, B=32)


def mism(A, Bx):
    return sum(1 for a, b in zip(A, Bx) if not (a is None and b is None)
               and ((a is None) != (b is None) or not np.allclose(np.asarray(a), np.asarray(b), 0, 0)))


def bench(fn, reps=6):
    fn(); torch.cuda.synchronize(); best = 1e9; r = None
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); torch.cuda.synchronize()
        best = min(best, 1e3 * (time.perf_counter() - t0) / n)
    return best, r


Ml, Mg = loop(), graph(); torch.cuda.synchronize()
print(f"BIT-IDENTICAL (graph vs eager loop): {mism(Ml, Mg)} mismatches / {n}", flush=True)
tl, _ = bench(loop); tg, Mg = bench(graph)
g = np.array([gpass(M, q) for M, q in zip(Mg, frames)])
print(f"eager loop  {tl:.3f} ms/frame ({1000/tl:.0f} f/s)")
print(f"CUDA graph  {tg:.3f} ms/frame ({1000/tg:.0f} f/s)   speedup {tl/tg:.2f}x")
print(f"rate (graph): frac {int(g[:, 0].sum())}/{n}  loose {int(g[:, 1].sum())}/{n}")
