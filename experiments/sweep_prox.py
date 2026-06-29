"""Sweep the M2 proximity-function FORM (and smooth-window width) inside GLINT's own blind
front-end on 120 cxidb. Does a SMOOTH window (wrapped Gaussian / von Mises / cos x Gaussian)
or the xgandalf linear TENT beat GLINT's default hard-mask cos/cos^2 for blind candidate
generation? Only M1-M3 (generation + ascent + ranking) changes; M4-M6 assembly is unchanged,
so this isolates the proximity's effect. Reports blind correct-lysozyme rate per form.

  python sweep_prox.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
import glint_index as gi
from glint_fast import index_blind_fast, load, LYSO
from fftindex.multishot import same_lattice

CONFIGS = [  # (OBJFORM, OBJSIG, OBJKAP, label)
    ("",         0.12,  8.0, "default cos/cos^2 + HARD mask (baseline)"),
    ("gauss",    0.08,  8.0, "wrapped Gaussian, sigma=.08"),
    ("gauss",    0.12,  8.0, "wrapped Gaussian, sigma=.12"),
    ("gauss",    0.18,  8.0, "wrapped Gaussian, sigma=.18"),
    ("vonmises", 0.12,  6.0, "von Mises, kappa=6"),
    ("vonmises", 0.12, 12.0, "von Mises, kappa=12"),
    ("vonmises", 0.12, 25.0, "von Mises, kappa=25"),
    ("softcos",  0.12,  8.0, "cos x Gaussian window, sigma=.12"),
    ("softcos",  0.18,  8.0, "cos x Gaussian window, sigma=.18"),
    ("tent",     0.12,  8.0, "xgandalf linear tent (1 .. -1)"),
]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
    if len(sys.argv) > 2:
        frames = frames[:int(sys.argv[2])]
    n = len(frames)
    index_blind_fast(frames[0])                                  # warmup
    print(f"N={n}   M2 proximity-function sweep (blind correct-cell; GLINT default = hard-mask cos)")
    print(f"{'form':>10}{'sig':>6}{'kap':>6}{'rate':>7}   note")
    best = (None, -1)
    for form, sig, kap, label in CONFIGS:
        gi.OBJFORM = form; gi.OBJSIG = sig; gi.OBJKAP = kap
        Ms = [index_blind_fast(q) for q in frames]
        c = sum(M is not None and same_lattice(M, LYSO) for M in Ms)
        print(f"{form or 'default':>10}{sig:>6.2f}{kap:>6.1f}{100*c//n:>6}%   {label}", flush=True)
        if c > best[1]:
            best = (label, c)
    print(f"\nBEST: {best[0]}  ({best[1]}/{n} = {100*best[1]//n}%)   [GLINT default cos/cos^2 = 70%]")
