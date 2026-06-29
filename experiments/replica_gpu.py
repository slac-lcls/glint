"""GPU known-cell indexer = faithful torch port of replica_v2.index_known (the ① rescue
engine), closing the per-frame gap to ffbidx. Same algorithm: GPU c-axis candidate search
(objective + gradient refine over CDIRS dirs) -> per c-candidate sweep NANG a-orientations
-> batched ifss (anneal_batch_t) on the top oriented cells -> pick best by ifss score.
The numpy replica is ~127 ms/frame; this targets a few ms on the A100.

  python replica_gpu.py [frames.txt] [N]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_fast import anneal_batch_t
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice

DEV = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
LA, LC = 79.02, 37.98
TRIML, TRIMH, DELTA = 0.05, 0.30, 0.10
CDIRS = int(os.environ.get("CDIRS", "16384"))
NANG = int(os.environ.get("NANG", "360"))
NC = int(os.environ.get("NC", "16"))
PI = np.pi


def _fib_halfsphere(D):
    i = np.arange(D); phi = PI * (3 - np.sqrt(5)) * i
    z = 1.0 - (i + 0.5) / D; r = np.sqrt(np.clip(1 - z * z, 0, 1))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


DIRS = torch.as_tensor(_fib_halfsphere(CDIRS), dtype=torch.float64, device=DEV)
TH = torch.as_tensor(np.linspace(0, PI, NANG, endpoint=False), dtype=torch.float64, device=DEV)
CA, SA = torch.cos(TH), torch.sin(TH)


def objective_t(V, Q):
    """replica module 2 (torch): inlier count + trimmed-log2 sub-score. V:(B,3) Q:(P,3)."""
    proj = V @ Q.T
    d = torch.abs(proj - torch.round(proj))
    inl = (d < TRIMH).sum(1)
    sub = torch.log2(torch.clamp(d, TRIML, TRIMH) + DELTA).mean(1)
    return inl, sub


def refine_vec_t(V, Q, steps=30):
    qmax = float(Q.norm(dim=1).max())
    lr = 1.0 / (4 * PI ** 2 * len(Q) * qmax ** 2)
    for _ in range(steps):
        th = 2 * PI * (V @ Q.T)
        V = V - lr * (2 * PI * (torch.sin(th) @ Q))
    return V


def c_candidates_t(Q):
    inl, sub = objective_t(LC * DIRS, Q)
    key = inl.double() * 100.0 - sub                      # max inl, then min sub
    idx = torch.argsort(key, descending=True)[:120]
    ref = refine_vec_t((LC * DIRS)[idx], Q, steps=30)
    ref = ref / ref.norm(dim=1, keepdim=True) * LC
    inl2, sub2 = objective_t(ref, Q)
    order = torch.argsort(inl2.double() * 100.0 - sub2, descending=True)
    refc = ref.cpu().numpy(); out = []
    for j in order.tolist():
        d = refc[j] / LC
        if all(abs(d @ (o / LC)) < 0.985 for o in out):
            out.append(refc[j])
        if len(out) >= NC:
            break
    return torch.as_tensor(np.array(out), dtype=torch.float64, device=DEV) if out else None


def index_known_gpu(q, topa=8):
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float64, device=DEV)
    if len(Q) < 6:
        return None
    C = c_candidates_t(Q)                                  # (NC,3)
    if C is None:
        return None
    # per c-candidate: orthonormal basis u,v perp c; sweep a = LA*(ca*u + sa*v)
    cn = C / C.norm(dim=1, keepdim=True)
    tmp = torch.where(cn[:, :1].abs() < 0.9,
                      torch.tensor([1.0, 0, 0], dtype=torch.float64, device=DEV),
                      torch.tensor([0, 1.0, 0], dtype=torch.float64, device=DEV))
    u = torch.cross(cn, tmp, dim=1); u = u / u.norm(dim=1, keepdim=True)
    v = torch.cross(cn, u, dim=1)
    A = LA * (CA[None, :, None] * u[:, None, :] + SA[None, :, None] * v[:, None, :])  # (NC,NANG,3)
    inl, _ = objective_t(A.reshape(-1, 3), Q)
    inl = inl.reshape(len(C), NANG)
    ta = min(topa, NANG)
    topi = torch.topk(inl, ta, dim=1).indices                                         # (NC,ta)
    a_sel = torch.gather(A, 1, topi[:, :, None].expand(-1, -1, 3))                     # (NC,ta,3)
    c_exp = C[:, None, :].expand(-1, ta, -1)                                           # (NC,ta,3)
    b = torch.cross(c_exp, a_sel, dim=2); b = b / b.norm(dim=2, keepdim=True) * LA
    M0 = torch.stack([a_sel, b, c_exp], dim=3).reshape(-1, 3, 3)                       # (NC*ta,3,3) cols a,b,c
    det = torch.abs(torch.linalg.det(M0))
    M0 = M0[det >= 1e3]
    if len(M0) == 0:
        return None
    Mt = anneal_batch_t(M0, Q, thr0=0.30, contract=0.82, max_iter=20, min_thr=0.02)    # ifss
    H = torch.einsum('pc,bcd->bpd', Q, Mt); dd = torch.abs(H - torch.round(H))
    main = (dd.amax(2) < 0.15).sum(1)                                                  # ffbidx score_thr
    sub = torch.log2(torch.clamp(dd, TRIML, TRIMH) + DELTA).mean((1, 2))
    key = main.double() * 1000.0 - sub
    return Mt[int(torch.argmax(key))].cpu().numpy()


def load(p):
    fr = []; L = open(p).read().split("\n"); i = 0
    while i < len(L):
        if not L[i].startswith("FRAME"): i += 1; continue
        _, fid, n = L[i].split(); n = int(n)
        fr.append(np.array([[float(x) for x in L[i + 1 + j].split()] for j in range(n)])); i += 1 + n
    return fr


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]; n = len(frames)
    index_known_gpu(frames[0])                              # warmup
    if DEV == "cuda": torch.cuda.synchronize()
    f25 = fN = 0; t0 = time.time()
    for q in frames:
        M = index_known_gpu(q)
        if M is None or not same_lattice(M, LYSO):
            continue
        m = int((np.abs(q @ M - np.rint(q @ M)).max(1) < 0.15).sum())
        f25 += m / len(q) >= 0.25; fN += m >= 10
    if DEV == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"replica_GPU (known-cell)  device={DEV}  N={n}  {1e3*dt/n:.1f} ms/frame  ({n/dt:.0f} f/s)")
    print(f"  frac>=.25: {f25}/{n} ({100*f25//n}%)   >=10refl: {fN}/{n} ({100*fN//n}%)")
    print(f"  numpy replica baseline: ~127 ms/frame ; ffbidx ~4.4 ms/frame")
