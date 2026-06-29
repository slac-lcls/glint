"""Replica v0 of ffbidx (known-cell), decoded from its source. Modular so modules 2
(objective) and 6 (score) can later be swapped for photon-weighted / weak-peak
versions. Oracle = ffbidx (100% on 60-set, 98% on 120-set, Buerger metric).

Modules:
  1 SAMPLE   Fibonacci half-sphere dirs x cell-prior lengths -> candidate vectors
  2 OBJECTIVE per vector v over spots q: d=dist2int(v.q); inliers=#(d<trimh);
             sub=mean log2(clip(d,triml,trimh)+delta)   [trimmed-log2-dist2int]
  3 REFINE   light gradient ascent of Re F(v)=sum cos(2pi v.q) (continuous maxima)
  4 ASSEMBLE tetragonal: top c-axis (len lc) + top a-axis (len la) with a _|_ c,
             b = norm(c x a)*la  -> cell M (columns a,b,c)
  5 ifss     iterative LSQ to inliers, threshold *0.8 contraction
  6 SCORE    inlier count at final threshold; best cell = max
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, "/Users/smarches/git/fftindex")
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice
from diffgrid import load

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
LA, LC = 79.02, 37.98
TRIML, TRIMH, DELTA = 0.05, 0.30, 0.10


def fib_halfsphere(D):
    i = np.arange(D); phi = np.pi * (3 - np.sqrt(5)) * i
    z = 1.0 - (i + 0.5) / D
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def objective(V, Q):
    """module 2: inlier count + trimmed-log2 sub-score per candidate vector."""
    proj = V @ Q.T
    d = np.abs(proj - np.rint(proj))
    inl = (d < TRIMH).sum(1)
    sub = np.log2(np.clip(d, TRIML, TRIMH) + DELTA).mean(1)
    return inl, sub


def refine_vec(V, Q, steps=8, lr=None):
    """module 3: ascend Re F(v)=sum cos(2pi v.q); grad = -2pi sum q sin(2pi v.q)."""
    qmax = np.linalg.norm(Q, axis=1).max()
    if lr is None:
        lr = 1.0 / (4 * np.pi ** 2 * len(Q) * qmax ** 2)
    V = V.copy()
    for _ in range(steps):
        th = 2 * np.pi * (V @ Q.T)
        grad = 2 * np.pi * (np.sin(th) @ Q)            # d/dv (-ReF) ; ascend ReF -> -grad
        V = V - lr * grad                              # minimize -ReF
    return V


def ifss(M, Q, thr0=0.25, contract=0.8, max_iter=10, min_thr=0.03, score_thr=0.15):
    """module 5+6: iterative LSQ fit to inliers (contracting threshold), then score
    like ffbidx: main = inlier count at a LOOSE score_thr (tight thresholds collapse
    to noise on real frames), sub = mean trimmed-log2 residual (tiebreak)."""
    thr = thr0
    for _ in range(max_iter):
        H = Q @ M
        hkl = np.rint(H)
        res = np.abs(H - hkl).max(1)
        inl = res < thr
        if inl.sum() < 6:
            break
        try:
            M = np.linalg.lstsq(Q[inl], hkl[inl], rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        thr = max(thr * contract, min_thr)
    H = Q @ M; dd = np.abs(H - np.rint(H))
    main = int((dd.max(1) < score_thr).sum())
    sub = float(np.log2(np.clip(dd, TRIML, TRIMH) + DELTA).mean())
    return M, (main, -sub)


DIRS = fib_halfsphere(3000)


def candidates(vec, L, Q, n_top, n_refine=120):
    """module 1+3+2: raw-score -> refine top n_refine (ROPT) -> rescale to length L
    -> re-score -> de-duplicated top n_top by refined inlier count."""
    inl, sub = objective(vec, Q)
    idx = np.lexsort((sub, -inl))[:n_refine]
    ref = refine_vec(vec[idx], Q)                      # robust optimization (continuous maxima)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True) * L   # enforce known length
    inl2, sub2 = objective(ref, Q)
    order = np.lexsort((sub2, -inl2))
    out = []
    for j in order:
        d = ref[j] / L
        if all(abs(d @ (o / L)) < 0.985 for o in out):
            out.append(ref[j])
        if len(out) >= n_top:
            break
    return out


def index_known(Q, n_top=20):
    Q = np.asarray(Q, float)
    if len(Q) < 6:
        return None
    Acand = candidates(LA * DIRS, LA, Q, n_top)
    Ccand = candidates(LC * DIRS, LC, Q, n_top)
    best = None; best_s = None
    for c in Ccand:
        cn = c / LC
        for a in Acand:
            if abs((a / LA) @ cn) > 0.12:                          # need a _|_ c
                continue
            b = np.cross(c, a); b = b / np.linalg.norm(b) * LA
            M = np.column_stack([a, b, c])
            if abs(np.linalg.det(M)) < 1e3:
                continue
            Mr, score = ifss(M, Q)
            if best_s is None or score > best_s:
                best_s = score; best = Mr
    return best


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/frames_cxidb.txt"
    frames = load(path)
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:lim]
    ok = n = 0; t0 = time.time()
    for q in frames:
        if len(q) < 6:
            continue
        n += 1
        M = index_known(q)
        ok += M is not None and same_lattice(M, LYSO)
    print(f"{path}  Replica v0 (known-cell): {ok}/{n} ({100*ok/n:.0f}%)   {1e3*(time.time()-t0)/n:.0f} ms/frame CPU")
