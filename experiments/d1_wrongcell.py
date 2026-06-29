"""D1 diagnostic: WHAT are the baseline 'wrong_cell' leaks (indexed >=25% but not
same_lattice(LYSO))? Characterize each: reduced cell parameters (a,b,c,alpha,beta,gamma)
and volume ratio to LYSO. Tells us the fix:
  - sub/supercell of LYSO (lengths ~ simple multiples/divisors, vol ratio ~ simple fraction)
      -> better reduction / supercell detection (single-frame, cheap)
  - one very short axis / anomalous volume -> plausibility filter
  - random/idiosyncratic -> cross-frame consensus voting

  QDIST=0 python d1_wrongcell.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_fast import index_blind_fast, load, matched, LYSO, QDIST
from fftindex.multishot import same_lattice


def cellparams(M):
    G = M.T @ M
    L = np.sqrt(np.diag(G))
    ang = lambda i, j: np.degrees(np.arccos(np.clip(G[i, j] / (L[i] * L[j]), -1, 1)))
    return (*np.sort(L), ang(1, 2), ang(0, 2), ang(0, 1), abs(np.linalg.det(M)))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    index_blind_fast(frames[0])
    LV = abs(np.linalg.det(np.asarray(LYSO)))
    LL = np.sort(np.sqrt(np.diag(np.asarray(LYSO).T @ np.asarray(LYSO))))
    print(f"D1 wrong_cell characterization  QDIST={int(QDIST)}  N={len(frames)}")
    print(f"  LYSO reduced lengths ~ {np.round(LL,1)}  volume {LV:.0f}")
    print(f"  {'a':>6} {'b':>6} {'c':>6} {'al':>5} {'be':>5} {'ga':>5} {'vol':>9} {'vol/LYSO':>9}")
    nwrong = 0; vr = []
    for q in frames:
        M = index_blind_fast(q)
        if M is None:
            continue
        m = matched(M, q)
        if m / len(q) >= 0.25 and m >= 10 and not same_lattice(M, LYSO):
            a, b, c, al, be, ga, v = cellparams(M)
            nwrong += 1; vr.append(v / LV)
            print(f"  {a:6.1f} {b:6.1f} {c:6.1f} {al:5.0f} {be:5.0f} {ga:5.0f} {v:9.0f} {v/LV:9.3f}")
    vr = np.array(vr)
    print(f"\n  wrong cells: {nwrong}/{len(frames)}")
    if nwrong:
        print(f"  vol/LYSO: median {np.median(vr):.3f}  range [{vr.min():.3f}, {vr.max():.3f}]")
        for lo, hi, name in [(0, 0.6, "sub (<0.6x)"), (0.6, 1.4, "~LYSO (0.6-1.4x)"),
                             (1.4, 1e9, "super (>1.4x)")]:
            print(f"    {name:18}: {int(((vr >= lo) & (vr < hi)).sum())}")
