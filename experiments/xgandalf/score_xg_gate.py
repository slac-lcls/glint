"""Score xgandalf driver outputs (blind / known-cell) on the 120 cxidb frames under
Table 1s exact gate: correct lattice (same_lattice vs LYSO) AND >=25% spots indexed
(matched-frac, TOL=0.15); also report correct-lattice-only and >=10 refl, and mean ms.
Sanity: xg_blind120.txt should reproduce the papers ~71%."""
import os
import sys
import numpy as np
# THIS checkout first, resolved from __file__ the way score_glint_gate.py does: with the fixed
# /sdf checkout ahead of it, `glint` (and now the canonical GATE_* below) could resolve from a
# stale clone while score_glint_gate.py read the current one -- the exact drift the shared
# constants exist to prevent (Copilot review of glint#172).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, ROOT)
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar
# The gate's canonical home (glint#170): the same three constants gpass() applies, and the same
# names experiments/score_glint_gate.py imports -- one gate, so the two arms cannot drift apart.
from glint.glint_fast import GATE_TOL as TOL, GATE_FRAC, GATE_MIN
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)

def read_frames(path):
    frames = {}
    lines = open(path).read().split("\n"); i = 0
    while i < len(lines):
        if not lines[i].startswith("FRAME"):
            i += 1; continue
        _, fid, npk = lines[i].split(); npk = int(npk)
        q = np.array([[float(x) for x in lines[i+1+j].split()] for j in range(npk)])
        frames[int(fid)] = q; i += 1 + npk
    return frames

def parse_xg(path):
    res = {}
    for line in open(path):
        p = line.split()
        if len(p) < 2: continue
        if p[1] == "NONE":
            res[int(p[0])] = (None, float(p[2]))
        else:
            B = np.array(list(map(float, p[1:10]))).reshape(3,3)
            M = np.linalg.inv(B).T if np.median(np.linalg.norm(B, axis=0)) < 1 else B
            res[int(p[0])] = (M, float(p[10]))
    return res

def gate(M, q):
    if M is None or not same_lattice(M, LYSO):
        return (0,0,0)
    H = q @ M; inl = np.abs(H - np.rint(H)).max(1) < TOL; m = int(inl.sum())
    return (1, int(m/len(q) >= GATE_FRAC), int(m >= GATE_MIN))

frames = read_frames("frames_cxidb_clean.txt")
for tag, path in [("BLIND","xg_blind120.txt"), ("KNOWN-CELL","xg_known120.txt")]:
    try:
        res = parse_xg(path)
    except FileNotFoundError:
        print(f"{tag:10s}: {path} not found"); continue
    n = lat = g25 = g10 = 0; ms = 0.0
    for fid, q in frames.items():
        if len(q) < 6: continue
        n += 1
        M, t = res.get(fid, (None, 0.0))
        a,b,c = gate(M, q); lat += a; g25 += b; g10 += c; ms += t
    # ROUND, not floor -- the deliverables' convention since 0f456a0. Flooring here printed 86/120 as
    # "71%", which is the value that pair was RETIRED FROM (it is 72%), so the script that SOURCES
    # xgandalf's rate was reproducing the stale number and contradicting the paper it backs.
    print(f"xgandalf {tag:10s} N={n}: correct-lattice {lat}/{n}={round(100*lat/n)}%  "
          f">=25%(Table1) {g25}/{n}={round(100*g25/n)}%  >=10refl {g10}/{n}={round(100*g10/n)}%  {ms/n:.0f} ms/frame")
