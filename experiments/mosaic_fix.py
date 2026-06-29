"""D2 mosaic robustness fix. The failure map (failmodes.py) shows peak BROADENING is the steepest
cliff and it fails at candidate GENERATION (gen_miss) -- the M2 ascent can't land on the true axes
when peaks are jittered. Hypothesis: the default tight basins are the problem; a WIDER SMOOTH
proximity window (OBJFORM=gauss/softcos, larger OBJSIG) should tolerate the jitter and recover
generation. Test solved% vs mosaic sigma for several objective-window configs.

  python mosaic_fix.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
import glint_index as gi
import glint_fast as gf
from glint_fast import index_blind_fast, load, matched, LYSO
from fftindex.multishot import same_lattice

GATE, MININL = 0.25, 10
SIGS = [0.0, 0.0006, 0.001, 0.0015, 0.002]
OBJ_CONFIGS = [  # (OBJFORM, OBJSIG, QDIST, label)
    ("",        0.12, False, "default hard-mask cos"),
    ("gauss",   0.20, False, "gauss sigma=.20 (wide window)"),
    ("",        0.12, True,  "QDIST (recip-dist score)"),
    ("gauss",   0.20, True,  "gauss .20 + QDIST"),
    ("softcos", 0.25, True,  "softcos .25 + QDIST"),
]


def solved(q):
    M = index_blind_fast(q)
    if M is None:
        return False
    m = matched(M, q)
    return m / len(q) >= GATE and m >= MININL and same_lattice(M, LYSO)


def s_broaden(q, sig, rng):
    return q + rng.normal(0, sig, q.shape)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    n = len(frames)
    index_blind_fast(frames[0])                                  # warmup
    print(f"N={n}  mosaic-broadening recovery: solved% vs sigma (q-jitter, 1/A)")
    print("  " + "objective window".ljust(24) + "".join(f"s={s:<7}" for s in SIGS))
    for form, osig, qdist, label in OBJ_CONFIGS:
        gi.OBJFORM = form; gi.OBJSIG = osig; gf.QDIST = qdist
        row = []
        for sig in SIGS:
            rng = np.random.default_rng(0)
            c = sum(solved(s_broaden(q, sig, rng)) for q in frames)
            row.append(100 * c // n)
        print("  " + label.ljust(24) + "".join(f"{r:<9d}" for r in row), flush=True)
