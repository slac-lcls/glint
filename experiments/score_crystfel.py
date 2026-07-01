"""Score CrystFEL streams under the Table-1 gate on a given frames file.
  python score_crystfel.py <frames.txt> <stream1> [stream2 ...]"""
import os, sys, re
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
import numpy as np
import glint.glint_fast as gf
from glint.multishot import same_lattice
LYSO = gf.LYSO
frames = list(gf.load(sys.argv[1])); n = len(frames)
def parse_stream(path):
    out = {}; ev = a = b = c = None
    for ln in open(path):
        if ln.startswith("Event:"):
            mm = re.search(r"//(\d+)", ln); ev = int(mm.group(1)) if mm else None
        elif ln.startswith("astar ="): a = [float(x) for x in ln.split("=")[1].split()[:3]]
        elif ln.startswith("bstar ="): b = [float(x) for x in ln.split("=")[1].split()[:3]]
        elif ln.startswith("cstar ="):
            c = [float(x) for x in ln.split("=")[1].split()[:3]]
            if ev is not None and a and b:
                B = np.array([a, b, c]).T / 10.0; out[ev] = np.linalg.inv(B).T; a = b = c = None
    return out
def gate(M, q):
    if M is None or not same_lattice(M, LYSO): return (0, 0, 0)
    r = np.asarray(q) @ M; m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return (1, int(m / len(q) >= 0.25), int(m >= 10))
for tag in sys.argv[2:]:
    S = parse_stream(tag); lat = g25 = g10 = 0
    for i, q in enumerate(frames):
        a, b, c = gate(S.get(i), q); lat += a; g25 += b; g10 += c
    print("%-24s idx %3d | lattice %d/%d=%d%% >=25%% %d/%d=%d%% >=10refl %d/%d=%d%%"
          % (os.path.basename(tag), len(S), lat, n, 100*lat//n, g25, n, 100*g25//n, g10, n, 100*g10//n))
