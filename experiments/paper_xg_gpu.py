"""XGANDALF (paper) blind indexer, torch/GPU port. The ascent of the weighted
proximity objective over ~50k starts is pure matmul -> GPU-ideal.

  f(t) = sum_i w_i c(q_i . t),  c_a=cos(2pi x) (climb), c_f=cos^2(pi x) (sharpen)
  tolerance: exclude node if dist2int(q_i.t) > tol      w_i = 1/|q_i|  (or photon-weighted)
  ascend ~50k starts (Fibonacci dirs x length shells) with normalized-grad + momentum
  (approximates the paper's extended-GD zigzag mitigation); rank maxima by f VALUE;
  assemble top vectors -> ifss -> most-inliers then smaller cell (anti-supercell).

device auto (cuda/mps/cpu).  `bench <frames> [lim]` to benchmark.
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
from fftindex.lattice import buerger_reduce, cell_to_Ar
from fftindex.multishot import same_lattice
from replica_v0 import ifss

PI = np.pi
DEV = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
XGDIR = int(os.environ.get("XGDIR", "2200"))                 # Fibonacci directions
XGDL = float(os.environ.get("XGDL", "3.0"))                  # length-shell spacing (A)
XGSTEPS = int(os.environ.get("XGSTEPS", "80"))               # M3 ascent steps (GLINT cut 80->8)


def fib(D):
    i = np.arange(D); phi = np.pi * (3 - np.sqrt(5)) * i
    z = 1.0 - (i + 0.5) / D
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _starts(n_dir=2200, lo=30.0, hi=126.0, dl=3.0):
    D = torch.as_tensor(fib(n_dir), dtype=torch.float32)
    Ls = torch.arange(lo, hi, dl)
    return torch.cat([L * D for L in Ls], 0)                 # (M,3)


def obj_grad(T, Q, w, tol=0.18, sharp=False):
    proj = T @ Q.t()
    ad = torch.abs(proj - torch.round(proj))
    mask = (ad < tol).to(T.dtype)
    if sharp:
        c = torch.cos(PI * proj) ** 2
        dc = -PI * torch.sin(2 * PI * proj)
    else:
        c = torch.cos(2 * PI * proj)
        dc = -2 * PI * torch.sin(2 * PI * proj)
    wm = w.unsqueeze(0) * mask
    f = (c * wm).sum(1)
    g = (dc * wm) @ Q
    return f, g


def ascend(T, Q, w, qmax, steps=80, tol=0.18, sharp_last=25, mom=0.5):
    step0 = 0.25 / qmax
    vel = torch.zeros_like(T)
    for s in range(steps):
        lr = step0 * (1 - 0.7 * s / steps)
        _, g = obj_grad(T, Q, w, tol, sharp=(s >= steps - sharp_last))
        step = g / (g.norm(dim=1, keepdim=True) + 1e-12)
        vel = mom * vel + step                               # momentum ~ zigzag damping
        T = T + lr * vel
    return T


def cluster(Tn, fn, tol=2.0, keep=44):
    s = np.where(Tn[:, 0] != 0, np.sign(Tn[:, 0]), 1.0); Tc = Tn * s[:, None]
    order = np.argsort(-fn); seen = {}; vecs = []
    for j in order:
        if np.linalg.norm(Tc[j]) < 20: continue
        key = tuple(np.round(Tc[j] / tol).astype(int))
        if key in seen: continue
        seen[key] = 1; vecs.append(Tc[j])
        if len(vecs) >= keep: break
    return np.array(vecs)


STARTS = _starts(n_dir=XGDIR, dl=XGDL).to(DEV)


def index_blind(q, n_top=30, weight="invq"):
    q = np.asarray(q, float)
    if len(q) < 6:
        return None
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV)
    qn = Q.norm(dim=1)
    qmax = float(qn.max())
    w = (1.0 / qn) if weight == "invq" else torch.ones_like(qn)
    T = ascend(STARTS.clone(), Q, w, qmax, steps=XGSTEPS)
    f, _ = obj_grad(T, Q, w, sharp=True)
    cands = cluster(T.cpu().numpy(), f.cpu().numpy())[:n_top]
    if len(cands) < 3:
        return None
    best = None; bk = None
    for tri in itertools.combinations(range(len(cands)), 3):
        M0 = cands[list(tri)].T
        sc = np.prod([np.linalg.norm(cands[t]) for t in tri])
        if sc <= 0 or abs(np.linalg.det(M0)) < 0.1 * sc:
            continue
        # residual threshold annealing (TORO): tighter, more iters -> cleaner defect
        M, _ = ifss(M0, q, thr0=0.25, contract=0.85, max_iter=15, min_thr=0.02)
        H = q @ M; dd = np.abs(H - np.rint(H)); inl = dd.max(1) < 0.15
        ni = int(inl.sum())
        if ni < max(8, 0.25 * len(q)):
            continue
        defect = float(dd[inl].mean())                       # mean fractional defect (blind discriminator)
        key = (-defect,)                                     # tightest fit wins (NOT most inliers)
        if bk is None or key > bk:
            bk = key; best = buerger_reduce(M)
    return best


def load(p):
    fr = []; L = open(p).read().split("\n"); i = 0
    while i < len(L):
        if not L[i].startswith("FRAME"): i += 1; continue
        _, fid, n = L[i].split(); n = int(n)
        fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
    return fr


if __name__ == "__main__":
    print(f"device={DEV}  starts={STARTS.shape[0]}")
    path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/frames_cxidb.txt"
    frames = load(path)
    lim = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else len(frames)
    frames = frames[:lim]
    ok = n = 0; t0 = time.time()
    for q in frames:
        if len(q) < 6: continue
        n += 1; M = index_blind(q); ok += M is not None and same_lattice(M, LYSO)
    print(f"{path}  XGANDALF-paper GPU BLIND: {ok}/{n} ({100*ok/n:.0f}%)  {1e3*(time.time()-t0)/n:.0f} ms/frame")
