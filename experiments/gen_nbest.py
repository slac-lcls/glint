"""Reconstruct the producing call behind experiments/nbest_120.npz and apply it to a new frame set.

DATA_PROVENANCE.md says the exact producing run predates the file's commit and that no in-tree
generator reproduces it.  What the file records is 120 frames x top-3 blind hypotheses, frame-major,
with the M4 score and the run's wall time -- i.e. glint.glint_fast.index_blind_nbest(q, 3) looped
over glint_fast.load(frames.txt), which is what experiments/nbest_consensus.py does at NBEST=5.
This script is that loop at N=3, writing the same six keys.

  python gen_nbest.py <frames.txt> <out.npz> [N=3]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")           # nbest_consensus.py's default
ROOT = os.environ.get("GLINT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np
from glint.glint_fast import index_blind_nbest, load, LYSO
import torch

path, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3

frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
print(f"input {path}: {len(frames)} frames, npk min/med/max "
      f"{min(len(q) for q in frames)}/{int(np.median([len(q) for q in frames]))}/"
      f"{max(len(q) for q in frames)}", flush=True)
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)

index_blind_nbest(frames[0], N)                                   # warmup, not timed
cells, fid, score = [], [], []
t0 = time.time()
for i, q in enumerate(frames):
    for c, s in index_blind_nbest(q, N):
        cells.append(np.asarray(c, float)); fid.append(i); score.append(float(s))
dt = time.time() - t0

cells = np.asarray(cells, float); fid = np.asarray(fid, np.int64); score = np.asarray(score, float)
np.savez(out, cells=cells, fid=fid, score=score, lyso=np.asarray(LYSO, float),
         nframes=np.int64(len(frames)), index_s=float(dt))
u, c = np.unique(fid, return_counts=True)
print(f"wrote {out}: {cells.shape[0]} hypotheses over {len(u)} frames "
      f"(per-frame min/max {c.min()}/{c.max()}), index_s={dt:.3f} s "
      f"({1e3*dt/len(frames):.1f} ms/frame)", flush=True)
