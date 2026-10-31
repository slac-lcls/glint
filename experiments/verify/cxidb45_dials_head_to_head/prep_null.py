"""Null input: the shipped azimuth scramble (experiments/scramble_qlist.scramble_frames, 95d51ce) of the
907 cxidb-45 lists. Each peak is rotated by its own random azimuth about the beam (rng([seed, i]) for frame
i), which keeps |q| and q_z (so each frame stays on its own Ewald sphere and its wavelength is unchanged)
and destroys only the lattice. Run in the GLINT env with PYTHONPATH=<pinned tree>."""
import argparse
import hashlib
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--npz", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=20260928)      # scramble_qlist.py's default seed
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.tree, "experiments"))
from scramble_qlist import scramble_frames                  # noqa: E402

D = np.load(a.npz)
n = int(D["n"])
frames = [np.asarray(D["q_%d" % i], float) for i in range(n)]
scr = scramble_frames(frames, a.seed)
for q, s in zip(frames, scr):
    assert s.shape == q.shape
    assert np.allclose(np.linalg.norm(s, axis=1), np.linalg.norm(q, axis=1), atol=1e-9)
    assert np.allclose(s[:, 2], q[:, 2], atol=1e-12)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.savez(a.out, n=np.int64(n), seed=np.int64(a.seed), **{"q_%d" % i: s for i, s in enumerate(scr)})
h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
print("null input: %d frames, seed %d, |q| and q_z preserved, %s sha256 %s" % (n, a.seed, a.out, h))
