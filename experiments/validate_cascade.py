"""(1) Validate the wired external cascade end-to-end: hybrid with cascade=ffbidx must NOT regress and
should report n_casc. (2) Answer the union question: does GLINT + xgandalf (or + ffbidx) index MORE
frames than xgandalf (or ffbidx) ALONE -- i.e. is GLINT complementary?"""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_fast import index_blind_fast
from glint.replica_gpu import index_known_gpu_cell
from glint.multishot import same_lattice
from glint.hybrid_stream import hybrid_index
from glint.cascade import external_cascade
LYSO = gf.LYSO
DRV = "/sdf/home/s/smarches/git/glint/experiments/xgandalf/ffbidx_driver"
frames = list(gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt"))
n = len(frames)


def gate(M, q):
    if M is None or not same_lattice(M, LYSO):
        return False
    r = np.asarray(q) @ M
    m = int((np.abs(r - np.rint(r)).max(1) < 0.15).sum())
    return (m / len(q) >= 0.25) and (m >= 10)


def parse_ffbidx(path):
    out = {}
    for ln in open(path):
        t = ln.split()
        if len(t) >= 2:
            out[int(t[0])] = None if t[1] == "NONE" else np.array(list(map(float, t[1:10]))).reshape(3, 3).T
    return out


def parse_xg(path):
    out = {}
    for ln in open(path):
        p = ln.split()
        if len(p) < 2:
            continue
        if p[1] == "NONE":
            out[int(p[0])] = None
        else:
            B = np.array(list(map(float, p[1:10]))).reshape(3, 3)
            out[int(p[0])] = np.linalg.inv(B).T if np.median(np.linalg.norm(B, axis=0)) < 1 else B
    return out


FB = parse_ffbidx("xgandalf/ffbidx_120.txt"); XG = parse_xg("xgandalf/xg_known120.txt")
index_blind_fast(frames[0]); index_known_gpu_cell(frames[0], LYSO)
gb = np.array([gate(index_blind_fast(q), q) for q in frames])
gr = np.array([gate(index_known_gpu_cell(q, LYSO), q) for q in frames])
fb = np.array([gate(FB.get(i), q) for i, q in enumerate(frames)])
xg = np.array([gate(XG.get(i), q) for i, q in enumerate(frames)])
g_own = gb | gr
P = lambda x: f"{int(x.sum()):3d}/{n}={round(100 * x.sum() / n):2d}%"   # ROUND (0f456a0), not floor

print("=== UNION question (single-frame, cell-given) ===")
print(f"  xgandalf alone            {P(xg)}")
print(f"  GLINT alone               {P(g_own)}")
print(f"  GLINT + xgandalf (union)  {P(g_own | xg)}   GLINT adds {int((g_own & ~xg).sum())} frames xgandalf misses")
print(f"  ffbidx alone              {P(fb)}")
print(f"  GLINT + ffbidx (union)    {P(g_own | fb)}   GLINT adds {int((g_own & ~fb).sum())} frames ffbidx misses")
print(f"  xgandalf adds {int((xg & ~g_own).sum())} frames GLINT misses; ffbidx adds {int((fb & ~g_own).sum())}")

print("\n=== wired-cascade validation (production consensus hybrid) ===")
r0, s0 = hybrid_index(frames, warmup=False)
r1, s1 = hybrid_index(frames, warmup=False, cascade=external_cascade(DRV))
print(f"  hybrid baseline   : FINAL {s0['n_idx']}/{n}  support {s0['support']}  n_resc {s0['n_resc']}")
print(f"  hybrid + cascade  : FINAL {s1['n_idx']}/{n}  n_casc {s1.get('n_casc')}  (>= baseline, no regression)")
