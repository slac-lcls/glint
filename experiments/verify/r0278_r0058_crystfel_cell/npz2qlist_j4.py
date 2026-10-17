#!/usr/bin/env python3
"""Write one of the per-frame check's q-list npz files (keys n, events, lams, q_0..q_{n-1}, q in 1/A) as a FRAME-block q list, the
format make_cxi.py and the shipped indexers read. Frames in numeric key order; the writer is the shipped
experiments/scramble_qlist.write_qlist, and the file is read back with glint.glint_fast.load and compared row for
row. Also writes <dst>.lams (one wavelength per frame) for the record.

    python npz2qlist_j4.py $J4OUT/r0278_qlists.npz r0278_real.txt
"""
import os
import re
import sys

import numpy as np

GLINT = os.environ["GLINT"]
sys.path.insert(0, GLINT)
sys.path.insert(0, os.path.join(GLINT, "experiments"))
from glint.glint_fast import load            # noqa: E402
from scramble_qlist import write_qlist       # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
z = np.load(src)
keys = sorted((k for k in z.files if re.fullmatch(r"q_\d+", k)), key=lambda k: int(k[2:]))
n = int(z["n"])
if [int(k[2:]) for k in keys] != list(range(n)):
    raise SystemExit(f"{src}: q keys are not q_0..q_{n - 1}")
frames = [np.asarray(z[k], float) for k in keys]
write_qlist(dst, frames)
back = load(dst)
assert len(back) == n and all(np.allclose(np.asarray(a, float), b, atol=1e-8) for a, b in zip(back, frames)), "read-back differs"
np.savetxt(dst + ".lams", np.asarray(z["lams"], float), fmt="%.6f")
print(f"{src}: {n} frames, {sum(len(f) for f in frames)} peaks -> {dst}; lambda median {np.median(z['lams']):.5f} A")
