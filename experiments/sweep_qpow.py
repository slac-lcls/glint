"""Tune the M2 objective weight exponent: w_i = |q_i|^(-p), used in both M3 ascent and M2
scoring. GLINT uses p=1.0; the xgandalf paper uses p=2.0 (smoother, fewer high-res local
maxima). Nobody has measured which is best -- sweep p on the 120 cxidb frames and report the
blind correct-lysozyme rate (and the wrong-cell count) for each.

  python sweep_qpow.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
import glint_fast as gf
from glint_fast import index_blind_fast, load, LYSO
from fftindex.multishot import same_lattice

POWERS = [float(x) for x in os.environ["POWS"].split(",")] if os.environ.get("POWS") \
    else [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
    if len(sys.argv) > 2:
        frames = frames[:int(sys.argv[2])]
    n = len(frames)
    index_blind_fast(frames[0])                                  # warmup
    print(f"N={n}   M2 weight  w_i = |q_i|^(-p)   [GLINT p=1.0, xgandalf-paper p=2.0]")
    print(f"{'p':>5} {'correct':>8} {'rate':>6} {'wrong':>6}")
    best = (None, -1)
    for p in POWERS:
        gf.QPOW = p                                              # mutate module global (read at call time)
        Ms = [index_blind_fast(q) for q in frames]
        c = sum(M is not None and same_lattice(M, LYSO) for M in Ms)
        w = sum(M is not None and not same_lattice(M, LYSO) for M in Ms)
        print(f"{p:>5.2f} {c:>8} {100*c//n:>5}% {w:>6}", flush=True)
        if c > best[1]:
            best = (p, c)
    print(f"\nBEST: p={best[0]}  {best[1]}/{n} ({100*best[1]//n}%)   "
          f"(GLINT default p=1.0)")
