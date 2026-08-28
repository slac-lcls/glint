"""Where do the 3 frames between 91 and 94 actually come from -- the ORDER, or an extra arm?

The abstract says reversing discovery and registration "reaches 78% using a third as many discovery
solves", attributing the gain to the ORDER. But union is commutative: if both arrangements ended up
taking the union of the same arms, order could not change the yield at all -- only the cost. So the
gain must come from the arm sets differing.

  shipped hybrid_index  = blind N-best  UNION  per-frame known-cell           (91)
  arrangement B         = BATCHED known-cell  UNION  blind  UNION  per-frame  (94)

B has one arm the shipped pipeline never uses: the batched path, which was separately measured to
disagree with the per-frame one on 18 of 120 frames (9 each way). This computes every arm's pass set
over all frames with ONE cell and ONE gate, so the 3 frames can be attributed exactly.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import index_blind_nbest, index_blind_fast
from glint.multishot import same_lattice
from glint.replica_gpu import index_known_gpu_cell
from what_are_the_failures import strict_gate, LYSO
import glint.replica_gpu_batch as rgb

frames = [np.asarray(f, float) for f in gf.load("frames_cxidb_clean.txt") if len(f) >= 6]
N = len(frames)

# one cell for every arm: the vote over the first 5 frames (what arrangement B locks)
cells = []
for i in range(5):
    for c, _s in index_blind_nbest(frames[i], 3):
        if c is not None:
            cells.append(np.asarray(c, float))
        break
Mc = max(cells, key=lambda a: sum(1 for b in cells if same_lattice(a, b)))
print(f"{N} frames; voted cell same_lattice(Mc,LYSO)={same_lattice(Mc, LYSO)}\n")

def ok(M, q):
    return M is not None and strict_gate(np.asarray(M, float), q, Mc)

# --- every arm, over ALL frames, same cell and gate ---
A = {}
A["blind top-1"] = {i for i, q in enumerate(frames) if ok(index_blind_fast(q), q)}
s = set()
for i, q in enumerate(frames):
    for c, _sc in index_blind_nbest(q, 3):
        if ok(c, q):
            s.add(i); break
A["blind N-best(3)"] = s
Mb = rgb.index_fused(frames, Mc, B=N)
A["known-cell BATCHED"] = {i for i in range(N) if ok(Mb[i], frames[i])}
A["known-cell per-frame"] = {i for i, q in enumerate(frames) if ok(index_known_gpu_cell(q, Mc), q)}

print(f"{'arm':26s} {'passes':>7s}")
print("-" * 36)
for k, v in A.items():
    print(f"{k:26s} {len(v):7d}")

shipped = A["blind N-best(3)"] | A["known-cell per-frame"]
reorder = shipped | A["known-cell BATCHED"]
print("-" * 36)
print(f"{'UNION shipped arms':26s} {len(shipped):7d}   (blind N-best + per-frame KC)")
print(f"{'UNION + batched arm':26s} {len(reorder):7d}   (what arrangement B takes)")
print(f"\nframes ONLY the batched arm gets : {sorted(reorder - shipped)}")
print(f"frames ONLY the per-frame arm gets: {len(A['known-cell per-frame'] - A['known-cell BATCHED'])}")
print(f"\nSo the +3 is an EXTRA ARM, not the ordering." if len(reorder) > len(shipped)
      else "\nThe batched arm adds nothing; the gain must be elsewhere.")
