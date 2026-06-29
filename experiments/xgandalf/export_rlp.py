"""Export the SAME per-frame rlp q-vectors fftindex used (lyso_rlp.npz, frames[:60]
with >=10 refl) to a flat text file both the xgandalf C++ driver and fftindex read.
This guarantees a true same-spots comparison."""
import sys

import numpy as np

d = np.load("/sdf/home/s/smarches/lyso_rlp.npz")
rlp, ids = d["rlp"], d["id"]
out = open(sys.argv[1] if len(sys.argv) > 1 else "frames.txt", "w")
used = 0
for fid in np.unique(ids):
    q = rlp[ids == fid]
    if len(q) < 10:
        continue
    used += 1
    if used > 60:
        break
    out.write(f"FRAME {int(fid)} {len(q)}\n")
    for v in q:
        out.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
out.close()
print(f"wrote {used} frames")
