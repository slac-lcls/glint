"""Sweep erf^2 q-resolution apodization on 120 cxidb blind correct-cell. GLINT has no resolution
band-pass (only the 1/|q| soft weight) -- does a smooth high-q taper (drop the noisy high-res
tail) or a low-q beamstop taper help candidate generation? QHI/QLO = taper edge as a fraction of
qmax (0=off); QAPSIG = taper width / qmax. The erf^2 window is C^1 at each edge.

  python sweep_qapod.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
import glint_fast as gf
from glint_fast import index_blind_fast, load, LYSO
from fftindex.multishot import same_lattice

CONFIGS = [  # (QHI, QLO, QAPSIG, label)
    (0.00, 0.00, 0.08, "off (baseline 1/|q|)"),
    (0.95, 0.00, 0.08, "hi-q taper edge .95 qmax"),
    (0.90, 0.00, 0.08, "hi-q taper edge .90 qmax"),
    (0.85, 0.00, 0.08, "hi-q taper edge .85 qmax"),
    (0.80, 0.00, 0.10, "hi-q taper edge .80 qmax"),
    (0.70, 0.00, 0.10, "hi-q taper edge .70 qmax"),
    (0.90, 0.10, 0.08, "hi-q .90 + lo-q beamstop .10"),
    (0.00, 0.15, 0.08, "lo-q beamstop taper .15 only"),
]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
    if len(sys.argv) > 2:
        frames = frames[:int(sys.argv[2])]
    n = len(frames)
    index_blind_fast(frames[0])                                  # warmup
    print(f"N={n}   erf^2 q-apodization sweep (blind correct-cell; baseline = no taper)")
    print(f"{'QHI':>6}{'QLO':>6}{'sig':>6}{'rate':>7}   note")
    best = (None, -1)
    for qhi, qlo, sig, label in CONFIGS:
        gf.QHI = qhi; gf.QLO = qlo; gf.QAPSIG = sig
        Ms = [index_blind_fast(q) for q in frames]
        c = sum(M is not None and same_lattice(M, LYSO) for M in Ms)
        print(f"{qhi:>6.2f}{qlo:>6.2f}{sig:>6.2f}{100*c//n:>6}%   {label}", flush=True)
        if c > best[1]:
            best = (label, c)
    print(f"\nBEST: {best[0]}  ({best[1]}/{n} = {100*best[1]//n}%)   [baseline no-taper = 70%]")
