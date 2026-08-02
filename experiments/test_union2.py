"""Is there really a frame left on the table, or was that an artefact of comparing across cells?

where_are_the_3.py computed the arm union with the VOTED cell (92); test_union ran hybrid_index with
Mc_known=LYSO (91). Different cells, so the two were never comparable. This computes hybrid_index and
the explicit arm union with the SAME cell, both ways, so the question is actually answered.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import index_blind_nbest
from glint.multishot import same_lattice
from glint.replica_gpu import index_known_gpu_cell
from what_are_the_failures import strict_gate, LYSO
from glint.hybrid_stream import hybrid_index

frames = [np.asarray(f, float) for f in gf.load("frames_cxidb_clean.txt") if len(f) >= 6]
N = len(frames)

# the voted cell, as where_are_the_3 used
cells = []
for i in range(5):
    for c, _s in index_blind_nbest(frames[i], 3):
        if c is not None:
            cells.append(np.asarray(c, float))
        break
Mvote = max(cells, key=lambda a: sum(1 for b in cells if same_lattice(a, b)))

print(f"{'cell':12s} {'hybrid first':>13s} {'hybrid best':>12s} {'ARM UNION':>10s}")
print("-" * 52)
for name, Mc in (("LYSO", LYSO), ("voted", Mvote)):
    g = {}
    for mode in ("first", "best"):
        res, _ = hybrid_index(frames, Mc_known=Mc, warmup=True, select=mode)
        g[mode] = sum(1 for r, q in zip(res, frames) if strict_gate(r["M"], q, Mc))
    # explicit union of the two arms, same cell, same gate
    u = set()
    for i, q in enumerate(frames):
        for c, _s in index_blind_nbest(q, 3):
            if c is not None and strict_gate(np.asarray(c, float), q, Mc):
                u.add(i); break
        if i not in u:
            Mr = index_known_gpu_cell(q, Mc)
            if Mr is not None and strict_gate(np.asarray(Mr, float), q, Mc):
                u.add(i)
    print(f"{name:12s} {g['first']:10d}/{N:<3d} {g['best']:9d}/{N:<3d} {len(u):7d}/{N:<3d}")
