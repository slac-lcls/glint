"""Unified 3-way: GLINT-① vs xgandalf(blind) vs ffbidx(known) on the SAME 120 cxidb
frames, SAME gate. Rate = valid() [same_lattice(LYSO) AND frac>=.25 AND >=10 inliers].
Performance: xgandalf/ffbidx per-frame ms from their output files; GLINT measured here.

  python compare3.py [frames] [N]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CDIRS", "16384")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_fast import index_blind_fast, load, LYSO
from replica_gpu import index_known_gpu as rescue
from fftindex.multishot import consensus_cell, same_lattice

DEV = ("cuda" if torch.cuda.is_available() else "cpu")
TOL = 0.15


def parseM(path, transpose=True):
    """ffbidx prints rows=axes (needs .T -> columns=axes); xgandalf prints columns=axes
    (no transpose). Either way M columns are real-space axes a,b,c -> hkl=round(q@M)."""
    out = {}; ms = {}
    for ln in open(path):
        t = ln.split()
        if len(t) < 2:
            continue
        fid = int(t[0])
        if t[1] == "NONE":
            out[fid] = None; ms[fid] = float(t[-1]) if len(t) > 2 else np.nan
        else:
            M = np.array(list(map(float, t[1:10]))).reshape(3, 3)
            out[fid] = M.T if transpose else M
            ms[fid] = float(t[-1])
    return out, ms


def valid(M, q):
    if M is None or not same_lattice(M, LYSO):
        return False
    H = q @ M; inl = (np.abs(H - np.rint(H)).max(1) < TOL).sum()
    return inl / len(q) >= 0.25 and inl >= 10


def rate(Ms, frames):
    return sum(valid(Ms[i] if isinstance(Ms, dict) else Ms[i], q) for i, q in enumerate(frames))


if __name__ == "__main__":
    fr_path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(fr_path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]; n = len(frames)

    FB, FBms = parseM("xgandalf/ffbidx_120.txt", transpose=True)    # ffbidx: rows=axes
    XG, XGms = parseM("xgandalf/xg_blind120.txt", transpose=False)   # xgandalf: cols=axes

    # GLINT-① measured
    index_blind_fast(frames[0]); rescue(frames[0])
    if DEV == "cuda": torch.cuda.synchronize()
    t0 = time.time(); blind = [index_blind_fast(q) for q in frames]
    Mc, sup = consensus_cell([M for M in blind if M is not None])
    GL = []
    for M, q in zip(blind, frames):
        GL.append(M if (M is not None and same_lattice(M, Mc)) else rescue(q))
    if DEV == "cuda": torch.cuda.synchronize()
    gl_ms = 1e3 * (time.time() - t0) / n

    g = sum(valid(GL[i], q) for i, q in enumerate(frames))
    f = sum(valid(FB.get(i), q) for i, q in enumerate(frames))
    x = sum(valid(XG.get(i), q) for i, q in enumerate(frames))
    fbm = np.median([FBms[i] for i in range(1, n) if i in FBms])      # skip warmup frame 0
    xgm = np.median([XGms[i] for i in range(n) if i in XGms])

    print(f"\n=== 3-way on {n} sparse cxidb frames (same gate) ===\n")
    print(f"{'indexer':14}{'mode':12}{'indexed':>12}{'ms/frame':>12}{'frames/s':>12}")
    print(f"{'-'*62}")
    print(f"{'GLINT-①':14}{'BLIND':12}{f'{g}/{n} ({100*g//n}%)':>12}{gl_ms:>12.1f}{1000/gl_ms:>12.1f}")
    print(f"{'xgandalf':14}{'BLIND':12}{f'{x}/{n} ({100*x//n}%)':>12}{xgm:>12.0f}{1000/xgm:>12.3f}")
    print(f"{'ffbidx':14}{'known-cell':12}{f'{f}/{n} ({100*f//n}%)':>12}{fbm:>12.1f}{1000/fbm:>12.1f}")
    print(f"\n  GLINT speedup vs xgandalf (both BLIND): {xgm/gl_ms:.0f}x")
    print(f"  GLINT known-cell mode (replica_gpu): 16.5 ms/frame (60 f/s) vs ffbidx 4.4 ms")
