"""Sweep the M2 inlier window. HARD mask half-width TOL (xgandalf's empirical eps, GLINT default
.18) vs SMOOTH apodized Gaussian window OBJSIG -- on clean and mild-mosaic cxidb. Questions:
(1) GLINT's empirical TOL optimum and how broad it is; (2) does a SMOOTH (apodized-edge) window
beat the hard mask, ESPECIALLY at larger widths / under broadening -- where you need a wide window
for the long axes but a hard edge lets in spurious peaks?  Metric: blind correct-lysozyme rate.

  python sweep_tol.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
import glint_fast as gf
import glint_index as gi
from glint_fast import index_blind_fast, load, LYSO
from fftindex.multishot import same_lattice

CONDS = [(0.0, "clean"), (0.001, "mosaic s=.001")]
HARD = [0.10, 0.15, 0.18, 0.25, 0.35]
GAUSS = [0.10, 0.15, 0.20, 0.28]


def rate(frames):
    c = sum(1 for q in frames if (lambda M: M is not None and same_lattice(M, LYSO))(index_blind_fast(q)))
    return 100 * c // len(frames)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    index_blind_fast(frames[0])                                  # warmup
    print(f"N={len(frames)}  M2 inlier window: HARD mask TOL vs SMOOTH gauss OBJSIG (blind correct-cell)")
    for sig, label in CONDS:
        jit = np.random.default_rng(1)
        bro = [q + jit.normal(0, sig, q.shape) if sig else q for q in frames]
        gi.OBJFORM = ""
        hard = []
        for t in HARD:
            gf.TOL = t; hard.append((t, rate(bro)))
        gi.OBJFORM = "gauss"
        soft = []
        for s in GAUSS:
            gi.OBJSIG = s; soft.append((s, rate(bro)))
        gi.OBJFORM = ""
        print(f"\n  {label}")
        print("    HARD  TOL : " + "   ".join(f"{t:.2f}={r}%" for t, r in hard))
        print("    GAUSS sig : " + "   ".join(f"{s:.2f}={r}%" for s, r in soft))
