"""Faithful-er ffbidx replica: add the MISSING module — the ROTATION/ANGLE search.
ffbidx finds candidate vectors then samples rotation space (num_angle_points) to ORIENT
the cell; my v0/tuned skipped this (took a_|_c, b=c x a from independent axis search).
Here: for each c-axis candidate, sweep a around the c-perpendicular plane through many
angles, score each oriented cell, refine the best (ifss). Also use ffbidx's real
objective trim (triml=0.001) and denser c-direction sampling.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, "/Users/smarches/git/fftindex")
sys.path.insert(0, "/Users/smarches/git/fftindex/experiments")
import replica_v0 as R
from replica_v0 import fib_halfsphere, refine_vec, LA, LC
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice

R.TRIML = 0.001                                          # ffbidx's real lower trim (was 0.05)
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
DIRS = fib_halfsphere(int(os.environ.get("CDIRS", "8000")))
NANG = int(os.environ.get("NANG", "360"))               # rotation-search angle points
NC = int(os.environ.get("NC", "16"))                    # top c-axis candidates


def c_candidates(Q, n):
    inl, sub = R.objective(LC * DIRS, Q)
    idx = np.lexsort((sub, -inl))[:120]
    ref = refine_vec((LC * DIRS)[idx], Q, steps=30)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True) * LC
    inl2, sub2 = R.objective(ref, Q)
    order = np.lexsort((sub2, -inl2)); out = []
    for j in order:
        if all(abs((ref[j] / LC) @ (o / LC)) < 0.985 for o in out):
            out.append(ref[j])
        if len(out) >= n:
            break
    return out


def index_known(Q):
    Q = np.asarray(Q, float)
    if len(Q) < 6:
        return None
    best = None; best_s = None
    th = np.linspace(0, np.pi, NANG, endpoint=False)      # a-axis is 2-fold ambiguous -> [0,pi)
    ca, sa = np.cos(th), np.sin(th)
    for c in c_candidates(Q, NC):
        cn = c / np.linalg.norm(c)
        # orthonormal basis in the plane perpendicular to c
        tmp = np.array([1.0, 0, 0]) if abs(cn[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(cn, tmp); u /= np.linalg.norm(u); v = np.cross(cn, u)
        A = LA * (ca[:, None] * u[None] + sa[:, None] * v[None])   # (NANG,3) a-axis sweep
        # score each oriented cell by inlier count (M2), pick top few to ifss-refine
        inl, sub = R.objective(A, Q)                     # objective on full-length a-vector
        for k in np.lexsort((sub, -inl))[:8]:
            a = A[k]; b = np.cross(c, a); b = b / np.linalg.norm(b) * LA
            M = np.column_stack([a, b, c])
            if abs(np.linalg.det(M)) < 1e3:
                continue
            Mr, score = R.ifss(M, Q, thr0=0.30, contract=0.82, max_iter=20, min_thr=0.02)
            if best_s is None or score > best_s:
                best_s = score; best = Mr
    return best


def load(p):
    fr = []; L = open(p).read().split("\n"); i = 0
    while i < len(L):
        if not L[i].startswith("FRAME"): i += 1; continue
        _, fid, n = L[i].split(); n = int(n)
        fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
    return fr


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/frames_cxidb.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]
    f25 = fN = n = 0; t0 = time.time()
    for q in frames:
        n += 1
        M = index_known(q)
        if M is None or not same_lattice(M, LYSO):
            continue
        m = int((np.abs(q @ M - np.rint(q @ M)).max(1) < 0.15).sum())
        f25 += m / len(q) >= 0.25; fN += m >= 10
    print(f"ffbidx replica v2 (+ROTATION search NANG={NANG}, triml=0.001, CDIRS={len(DIRS)})  "
          f"N={n}  {1e3*(time.time()-t0)/n:.0f} ms/frame")
    print(f"  frac>=.25: {f25}/{n} ({100*f25/n:.0f}%)   >=10refl: {fN}/{n} ({100*fN/n:.0f}%)")
    print(f"  prior: v0 42/68, tuned 59/79 ; real ffbidx 72/84")
