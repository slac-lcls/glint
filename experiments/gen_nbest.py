"""Reconstruct the producing call behind experiments/nbest_120.npz and apply it to a new frame set.

DATA_PROVENANCE.md says the exact producing run predates the file's commit and that no in-tree
generator reproduces it.  What the file records is 120 frames x top-3 blind hypotheses, frame-major,
with the M4 score and the run's wall time -- i.e. glint.glint_fast.index_blind_nbest(q, 3) looped
over glint_fast.load(frames.txt), which is what experiments/nbest_consensus.py does at NBEST=5.
This script is that loop at N=3, writing the same six keys.

  python gen_nbest.py <frames.txt> <out.npz> [N=3]
"""
import os, sys, time

# glint_fast reads these at import and they change the output; an ambient override would
# silently break the bit-exactness this script promises, so refuse rather than inherit.
_OUTPUT_AFFECTING = (
    "STEPS", "M3_FUSED", "M3_CAP", "NTOP", "KEEP", "REFINER", "SCORER", "ANNEAL_ITERS",
    "ANNEAL_FP32", "QPOW", "TOL", "QHI", "QLO", "QAPSIG", "QDIST", "QDTOL", "DETREJ",
    "NEWTON_STEPS", "CLUSTER_MIN", "COHERENT_SEEDS", "FFTSEED_CAP", "BIGCELL_RLPS",
)
_set = [v for v in _OUTPUT_AFFECTING if v in os.environ]
if _set and os.environ.get("GEN_NBEST_ALLOW_ENV") != "1":
    sys.exit(f"gen_nbest: refusing to run with output-affecting overrides set: {_set} "
             "(unset them, or set GEN_NBEST_ALLOW_ENV=1 to accept a non-reference payload)")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["STEPS"] = "8"                     # the validated producing configuration
ROOT = os.environ.get("GLINT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np
from glint.glint_fast import index_blind_nbest, load, LYSO
import torch

path, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
if N < 1:
    sys.exit(f"gen_nbest: N must be >= 1 (got {N})")

frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
if not frames:
    sys.exit(f"gen_nbest: no frames with >=6 peaks in {path}; nothing to do")
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

if not cells:
    sys.exit("gen_nbest: no hypotheses produced for any frame; refusing to write a malformed pool")
cells = np.asarray(cells, float); fid = np.asarray(fid, np.int64); score = np.asarray(score, float)
np.savez(out, cells=cells, fid=fid, score=score, lyso=np.asarray(LYSO, float),
         nframes=np.int64(len(frames)), index_s=float(dt))
u, c = np.unique(fid, return_counts=True)
print(f"wrote {out}: {cells.shape[0]} hypotheses over {len(u)} frames "
      f"(per-frame min/max {c.min()}/{c.max()}), index_s={dt:.3f} s "
      f"({1e3*dt/len(frames):.1f} ms/frame)", flush=True)
