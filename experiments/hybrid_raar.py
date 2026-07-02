"""Does the RAAR M3 refiner (blind 84->88, the first to beat GD) lift the END-TO-END hybrid rate / consensus
support? Compares the production hybrid with GD-8 vs RAAR-16 (beta 0.7). RAAR propagates through nbest +
rescue (both use _refine). Baseline hybrid = 117/120, support ~91."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf
import glint.glint_index as gi
from glint.hybrid_stream import hybrid_index
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)


def run(refiner, steps, beta=0.7):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = beta
    hybrid_index(frames[:2], nbest=3)                              # warmup (this config)
    t0 = time.perf_counter()
    results, st = hybrid_index(frames, nbest=3)
    ms = 1e3 * (time.perf_counter() - t0) / n
    cell = st.get("cell", None)
    cs = "[%.1f,%.1f,%.1f]" % tuple(cell[:3]) if cell is not None else "-"
    return st, ms, cs


print("refiner steps  hybrid    support  n_resc  nbest_rec   ms/frame   cell")
for r, s in [("grad", 8), ("raar", 16)]:
    st, ms, cs = run(r, s)
    print(f"  {r:4s}  {s:2d}    {st['n_idx']}/{st['n']}     {st['support']:4d}     {st['n_resc']:3d}     "
          f"{st['n_nbest']:3d}      {ms:.1f}     {cs}", flush=True)
