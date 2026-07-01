"""Cascade fallback analysis on the 120 cxidb frames (all cell-given LYSO, same gate as Table 1).
Per frame: GLINT blind, GLINT own GPU rescue, ffbidx, xgandalf-cell -> does a stronger fallback
boost the rate, and on what FRACTION of frames is each fallback decisive (GLINT fails, fallback solves)?"""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf
from glint.glint_fast import index_blind_fast
from glint.replica_gpu import index_known_gpu_cell
from glint.multishot import same_lattice
LYSO = gf.LYSO
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
        if len(t) < 2:
            continue
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


FB = parse_ffbidx("xgandalf/ffbidx_120.txt")
XG = parse_xg("xgandalf/xg_known120.txt")
index_blind_fast(frames[0])
index_known_gpu_cell(frames[0], LYSO)
gb = []; gr = []; fb = []; xg = []
for i, q in enumerate(frames):
    gb.append(gate(index_blind_fast(q), q))
    gr.append(gate(index_known_gpu_cell(q, LYSO), q))
    fb.append(gate(FB.get(i), q))
    xg.append(gate(XG.get(i), q))
gb, gr, fb, xg = map(np.array, (gb, gr, fb, xg))
g_own = gb | gr
P = lambda x: f"{int(x.sum()):3d}/{n} = {100 * x.sum() // n:2d}%"
print("individual (all given LYSO cell; GLINT-blind uses no cell):")
print(f"  GLINT blind            {P(gb)}")
print(f"  GLINT rescue (cell)    {P(gr)}")
print(f"  GLINT blind|rescue     {P(g_own)}")
print(f"  ffbidx (cell)          {P(fb)}")
print(f"  xgandalf (cell)        {P(xg)}")
c1 = g_own; c2 = g_own | fb; c3 = g_own | fb | xg
print("\ncascade (stack fallbacks on GLINT failures):")
print(f"  GLINT                        {P(c1)}")
print(f"  GLINT -> ffbidx              {P(c2)}   (+{int((c2 & ~c1).sum())} frames = ffbidx decisive on {100 * (c2 & ~c1).sum() / n:.1f}% of all frames)")
print(f"  GLINT -> ffbidx -> xgandalf  {P(c3)}   (+{int((c3 & ~c2).sum())} more from xgandalf-cell)")
gfail = ~g_own
nf = int(gfail.sum())
print(f"\nwhere the fallback helps -- GLINT's {nf} failures:")
print(f"  ffbidx solves    {int((gfail & fb).sum())}/{nf}")
print(f"  xgandalf solves  {int((gfail & xg).sum())}/{nf}")
print(f"  either solves    {int((gfail & (fb | xg)).sum())}/{nf}   (fallback decisive on {100 * (gfail & (fb | xg)).sum() / n:.1f}% of all frames)")
print(f"  neither          {int((gfail & ~fb & ~xg).sum())}/{nf}   (genuinely hard for every indexer)")
print(f"\nreverse -- GLINT catches {int((g_own & ~fb & ~xg).sum())} frames BOTH ffbidx and xgandalf miss")
