"""Smoke-test the nbest port through the full production hybrid: final rate + cell must be unchanged
(116/120, LYSO), and the nbest front-end must show the on-device speedup."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import fftindex.glint_fast as gf
from fftindex.hybrid_stream import hybrid_index
from fftindex.glint_fast import index_blind_nbest
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)
index_blind_nbest(frames[0], 3)                                 # warmup
t0 = time.perf_counter()
for q in frames:
    index_blind_nbest(q, 3)
tnb = time.perf_counter() - t0
print(f"index_blind_nbest(3) front-end: {1e3*tnb/n:.1f} ms/frame")
t0 = time.perf_counter()
results, stats = hybrid_index(frames, nbest=3)
th = time.perf_counter() - t0
print(f"HYBRID n_idx={stats['n_idx']}/{stats['n']}  support={stats['support']}  edges={stats['edges']}  "
      f"n_resc={stats['n_resc']}  nbest_recovered={stats['n_nbest']}")
print(f"HYBRID end-to-end {1e3*th/n:.1f} ms/frame ({n/th:.1f} f/s)   (expected 116/120, LYSO ~[38,79,79])")
