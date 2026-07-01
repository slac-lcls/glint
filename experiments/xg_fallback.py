"""Measure the xgandalf-paper indexer (paper_xg_gpu.index_blind, defect-greedy assembly) as a
FALLBACK for GLINT blind failures: does its different selection rescue frames GLINT's
triplet+coverage assembly misses, or do both hit the same spurious wall? Reports per-frame
correct-lysozyme rates, the rescue contribution, and the combined blind rate.

  python xg_fallback.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from glint_fast import index_blind_fast, load, LYSO
from paper_xg_gpu import index_blind as xg_blind
from glint.multishot import same_lattice


def correct(M):
    return M is not None and same_lattice(M, LYSO)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
    if len(sys.argv) > 2:
        frames = frames[:int(sys.argv[2])]
    n = len(frames)
    index_blind_fast(frames[0]); xg_blind(frames[0])                 # warmup both
    G = [index_blind_fast(q) for q in frames]
    X = [xg_blind(q) for q in frames]
    gc = [correct(m) for m in G]; xc = [correct(m) for m in X]
    gnone = [m is None for m in G]
    ng = sum(gc); nx = sum(xc)
    resc = sum(1 for i in range(n) if not gc[i] and xc[i])           # xg correct where GLINT not
    resc_none = sum(1 for i in range(n) if gnone[i] and xc[i])       # ...of which GLINT returned None
    lost = sum(1 for i in range(n) if gc[i] and not xc[i])           # GLINT correct where xg not
    combined = sum(1 for i in range(n) if gc[i] or xc[i])
    print(f"N={n}")
    print(f"GLINT blind correct-cell  : {ng}/{n} ({100*ng//n}%)   (hard None: {sum(gnone)})")
    print(f"xgandalf blind correct    : {nx}/{n} ({100*nx//n}%)")
    print(f"xg rescues GLINT failures : {resc}  (GLINT-None subset: {resc_none})")
    print(f"GLINT-only correct (xg miss): {lost}   [complementarity check]")
    print(f"COMBINED (either correct) : {combined}/{n} ({100*combined//n}%)   "
          f"-> fallback lift = +{combined-ng}")
