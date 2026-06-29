"""GLINT -> CrystFEL .stream CLI (productization seed for LUTE/LCLS). Reads a peak file
(FRAME blocks of reciprocal q-vectors), blind-indexes each frame, writes a .stream with the
per-frame cell + orientation (astar/bstar/cstar) + indexed reflections.

  python glint_stream.py [frames.txt] [N] [out.stream]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))             # experiments/
_ROOT = os.path.dirname(_HERE)                                 # repo root (has fftindex/)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)
from glint_fast import index_blind_fast, load
from fftindex.stream import write_stream

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    out = sys.argv[3] if len(sys.argv) > 3 else "glint.stream"
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    index_blind_fast(frames[0])
    results = []
    for i, q in enumerate(frames):
        M = index_blind_fast(q)
        hkl = qin = None
        if M is not None:
            H = q @ M; r = np.rint(H); inl = np.abs(H - r).max(1) < 0.15
            hkl = r[inl].astype(int); qin = q[inl]
        results.append({"image": os.path.basename(path), "event": i,
                        "M": M, "q": (qin if M is not None else q), "hkl": hkl})
    n_idx = write_stream(results, out)
    print(f"wrote {out}: {len(results)} chunks, {n_idx} indexed ({100*n_idx//len(results)}%)")
