"""Does consensus + rescue ABSORB a pruned M1 start grid? starts_sweep.py showed blind rate drops with
n_dir (2200->84, 1600->80, 1100->76 of 120). But blind misses are exactly what cross-frame consensus +
known-cell rescue recover -- so the HYBRID (end-to-end) rate may hold even when blind drops. If it does,
STARTS-pruning is ~free end-to-end (a real throughput win). Runs the full production hybrid at each n_dir.
Baseline (n_dir=2200) hybrid = 116-117/120, LYSO [~38,79,79]."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_index import sample
from glint.hybrid_stream import hybrid_index
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)

print("n_dir  starts   hybrid_rate  support  n_resc  nbest_rec   ms/frame   cell")
for n_dir in [2200, 1600, 1100, 700]:
    gf.STARTS = sample(n_dir=n_dir).to(gf.DEV)                      # rebind the module-global start grid
    hybrid_index(frames[:2], nbest=3)                              # warmup (this n_dir)
    t0 = time.perf_counter()
    results, st = hybrid_index(frames, nbest=3)
    ms = 1e3 * (time.perf_counter() - t0) / n
    cell = st.get("cell", None)
    cs = "[%.0f,%.0f,%.0f]" % (cell[0], cell[1], cell[2]) if cell is not None else "-"
    print(f"{n_dir:5d}  {gf.STARTS.shape[0]:6d}   {st['n_idx']}/{st['n']}      {st['support']:4d}     "
          f"{st['n_resc']:3d}     {st['n_nbest']:3d}      {ms:.1f}     {cs}", flush=True)
