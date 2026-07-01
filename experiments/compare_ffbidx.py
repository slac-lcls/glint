"""Per-frame GLINT-① vs ffbidx(known-cell) comparison on the same 120 cxidb frames.
Answers: (1) agreement matrix both/onlyGLINT/onlyFFBIDX/neither; (2) do the two
solutions describe the SAME orientation -> is there a single integer unimodular U with
hkl_ffbidx = U @ hkl_glint on the shared inlier spots; (3) what fraction of shared spots
get the SAME Miller index after that relabeling (=indices match).

  python compare_ffbidx.py [frames] [ffbidx_out] [N]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CDIRS", "16384")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
from glint_fast import index_blind_fast, load, LYSO
from replica_v2 import index_known as replica_rescue
from glint.multishot import consensus_cell, same_lattice

TOL = 0.15


def parse_ffbidx(path):
    out = {}
    for ln in open(path):
        t = ln.split()
        if len(t) < 2:
            continue
        fid = int(t[0])
        if t[1] == "NONE":
            out[fid] = None
        else:
            v = np.array(list(map(float, t[1:10])))           # a,b,c as 3 row-vectors
            out[fid] = v.reshape(3, 3).T                       # columns = real axes a,b,c
    return out


def inliers(M, q):
    if M is None:
        return None, None
    H = q @ M; hkl = np.rint(H); inl = np.abs(H - hkl).max(1) < TOL
    return hkl.astype(int), inl


def valid(M, q):
    if M is None or not same_lattice(M, LYSO):
        return False
    _, inl = inliers(M, q)
    return inl.sum() / len(q) >= 0.25 and inl.sum() >= 10


def index_agreement(Mg, Mf, q):
    """integer U with hkl_f = U @ hkl_g on shared inliers; return (match_frac, detU, nshared)."""
    hg, ig = inliers(Mg, q); hf, iff = inliers(Mf, q)
    sh = ig & iff
    if sh.sum() < 6:
        return None
    Hg = hg[sh].astype(float); Hf = hf[sh].astype(float)       # (n,3)
    # solve U (3x3): Hf = Hg @ U^T  ->  U^T = lstsq(Hg, Hf)
    UT, *_ = np.linalg.lstsq(Hg, Hf, rcond=None)
    U = np.rint(UT.T).astype(int)
    pred = (U @ hg[sh].T).T                                     # mapped glint indices
    match = np.all(pred == hf[sh], axis=1).mean()
    return match, int(round(np.linalg.det(U))), int(sh.sum())


if __name__ == "__main__":
    fr_path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    fb_path = sys.argv[2] if len(sys.argv) > 2 else "xgandalf/ffbidx_120.txt"
    frames = [q for q in load(fr_path) if len(q) >= 6]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else len(frames)
    frames = frames[:N]; n = len(frames)
    FB = parse_ffbidx(fb_path)

    # GLINT-① per-frame M (blind fast -> consensus -> rescue failures)
    index_blind_fast(frames[0])
    blind = [index_blind_fast(q) for q in frames]
    Mc, sup = consensus_cell([M for M in blind if M is not None])
    print(f"consensus cell {np.round(np.sort(np.linalg.norm(Mc,axis=0)),1)} support {sup} lyso={same_lattice(Mc,LYSO)}")
    Mg = []; origin = []
    for M, q in zip(blind, frames):
        if M is not None and same_lattice(M, Mc):
            Mg.append(M); origin.append("blind")
        else:
            Mg.append(replica_rescue(q)); origin.append("rescue")

    both = onlyG = onlyF = neither = 0
    matches = []; dets = []; consistent = 0; disagree = 0
    blind_cons = blind_both = 0                                # independent-agreement subset
    for i, q in enumerate(frames):
        g = valid(Mg[i], q); f = valid(FB.get(i), q)
        if g and f:
            both += 1
            r = index_agreement(Mg[i], FB[i], q)
            if r is not None:
                mf, dU, ns = r; matches.append(mf); dets.append(dU)
                ok = mf >= 0.9 and abs(dU) == 1
                consistent += ok; disagree += (not ok)
                if origin[i] == "blind":
                    blind_both += 1; blind_cons += ok
        elif g:
            onlyG += 1
        elif f:
            onlyF += 1
        else:
            neither += 1
    print(f"\n=== GLINT-① vs ffbidx(known)  N={n} ===")
    print(f"  both solved      : {both}/{n}")
    print(f"  only GLINT        : {onlyG}/{n}")
    print(f"  only ffbidx       : {onlyF}/{n}")
    print(f"  neither           : {neither}/{n}")
    print(f"  GLINT total solved: {both+onlyG}/{n} ({100*(both+onlyG)//n}%)   "
          f"ffbidx total: {both+onlyF}/{n} ({100*(both+onlyF)//n}%)")
    if matches:
        m = np.array(matches)
        print(f"\n  -- index agreement on {len(m)} both-solved frames --")
        print(f"  median shared-spot index match: {100*np.median(m):.0f}%   mean: {100*m.mean():.0f}%")
        print(f"  CONSISTENT (match>=90% & |detU|=1): {consistent}/{len(m)}   genuine DISAGREE: {disagree}/{len(m)}")
        print(f"  relabel det(U) distribution: {dict(zip(*np.unique(dets, return_counts=True)))}  (-1 = handedness/inversion convention)")
        print(f"  INDEPENDENT subset (GLINT solved by BLIND, not the ffbidx-replica rescue):")
        print(f"    {blind_cons}/{blind_both} blind-solved frames agree with ffbidx (consistent indexing)")
