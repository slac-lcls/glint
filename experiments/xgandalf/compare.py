"""Same-spots head-to-head: fftindex vs xgandalf on identical rlp q-vectors.

Reads frames.txt (the shared input) and runs fftindex (blind + known-cell) on each
frame; reads the xgandalf driver outputs (xg_blind.txt / xg_known.txt). Scores BOTH
with fftindex's same_lattice vs the lysozyme cell, so the accuracy metric is identical.
Reports indexing rate and mean time/frame for each."""
import sys
import time

import numpy as np

sys.path.insert(0, "../..")
from glint import index_shot
from glint.lattice import cell_to_Ar
from glint.multishot import (index_known_pairangle, reference_lattice,
                                same_lattice)

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)


def read_frames(path):
    frames = []
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].startswith("FRAME"):
            i += 1
            continue
        _, fid, npk = lines[i].split()
        npk = int(npk)
        q = np.array([[float(x) for x in lines[i + 1 + j].split()] for j in range(npk)])
        frames.append((int(fid), q))
        i += 1 + npk
    return frames


def score_xg(path, frames):
    """xgandalf output -> (n_correct, mean_ms). Basis cols are vectors; if reciprocal
    (norms<1) convert real M=inv(B).T, then the same same_lattice check fftindex uses."""
    res = {}
    for line in open(path):
        p = line.split()
        if len(p) < 2:
            continue
        if p[1] == "NONE":
            res[int(p[0])] = (None, float(p[2]))
        else:
            B = np.array(list(map(float, p[1:10]))).reshape(3, 3)
            M = np.linalg.inv(B).T if np.median(np.linalg.norm(B, axis=0)) < 1 else B
            res[int(p[0])] = (M, float(p[10]))
    ok = ms = 0
    for fid, _ in frames:
        M, t = res.get(fid, (None, 0.0))
        ok += M is not None and same_lattice(M, LYSO)
        ms += t
    return ok, ms / len(frames)


frames = read_frames(sys.argv[1] if len(sys.argv) > 1 else "frames.txt")
N = len(frames)
print(f"frames: {N} (identical rlp q's fed to both indexers)\n")

# fftindex blind + known-cell, timed
fb = fk = tb = tk = 0
for fid, q in frames:
    qmax = float(np.linalg.norm(q, axis=1).max())
    t0 = time.perf_counter(); r = index_shot(q, qmax); tb += time.perf_counter() - t0
    fb += r.M is not None and same_lattice(r.M, LYSO)
    t0 = time.perf_counter()
    kn = index_known_pairangle(q, qmax, LYSO, Vref=reference_lattice(LYSO, qmax))
    tk += time.perf_counter() - t0
    fk += kn.M is not None and same_lattice(kn.M, LYSO)

print(f"  fftindex  BLIND     : {fb}/{N} ({100*fb/N:3.0f}%)   {1e3*tb/N:6.1f} ms/frame (CPU)")
print(f"  fftindex  KNOWN-CELL: {fk}/{N} ({100*fk/N:3.0f}%)   {1e3*tk/N:6.1f} ms/frame (CPU)")
for tag, path in (("BLIND", "xg_blind.txt"), ("KNOWN-CELL", "xg_known.txt")):
    try:
        ok, ms = score_xg(path, frames)
        print(f"  xgandalf  {tag:10s}: {ok}/{N} ({100*ok/N:3.0f}%)   {ms:6.1f} ms/frame (CPU)")
    except FileNotFoundError:
        print(f"  xgandalf  {tag:10s}: ({path} not found)")
