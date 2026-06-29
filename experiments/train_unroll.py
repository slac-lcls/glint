"""Algorithm unrolling POC: unroll the M3 candidate ascent (refine_vec) and LEARN its
per-iteration step schedule, to match the 80-step hand-tuned front-end in far fewer steps
(the open throughput lever; front-end = 94% of the pipeline). Self-supervised: the M2
objective IS the loss (no ground-truth labels) -- learn a K-step optimizer that reaches the
same converged maxima as 80 hand-tuned steps. Tiny parameter count (K step sizes + momentum)
=> no overfit. Trains on a few frames, evaluates blind rate on all 120.

  python train_unroll.py [frames] [K] [iters]
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_index import objective, fib_sphere, distinct_maxima, invq_weight, buerger_reduce, primitivize, DEV
from glint_fast import anneal_batch_t, load, matched
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
PI = np.pi
NDIR_TRAIN = 1500
NDIR_EVAL = 2200
LENS = np.arange(30.0, 126.0, 3.0)


def make_starts(ndir):
    D = torch.as_tensor(fib_sphere(ndir), dtype=torch.float32)
    return torch.cat([float(L) * D for L in LENS], 0).to(DEV)


def unrolled_ascent(T0, Q, w, qmax, theta, mom, sharp_last=3):
    """K-step ascent with LEARNABLE per-step lr = softplus(theta_k)/qmax and momentum mom."""
    K = theta.shape[0]
    lr = torch.nn.functional.softplus(theta) / qmax
    vel = torch.zeros_like(T0); T = T0
    for s in range(K):
        _, g = objective(T, Q, w, tol=0.18, sharp=(s >= K - sharp_last))
        vel = torch.sigmoid(mom) * vel + g / (g.norm(dim=1, keepdim=True) + 1e-12)
        T = T + lr[s] * vel
    return T


def hand_schedule(K):
    """the current refine_vec schedule, sampled to K steps, as inverse-softplus init."""
    s = np.arange(K)
    vals = 0.25 * (1 - 0.7 * s / K)
    return np.log(np.expm1(np.clip(vals, 1e-3, None)))            # inverse softplus


def cands_from(T, Q, w, ntop=30):
    f, _ = objective(T, Q, w, sharp=True)
    return distinct_maxima(T.cpu().numpy(), f.cpu().numpy())[:ntop]


def blind_from_cands(cands, q, Q):
    if len(cands) < 3:
        return None
    nrm = np.linalg.norm(cands, axis=1)
    tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
    M0 = np.transpose(cands[tris], (0, 2, 1))
    sc = nrm[tris].prod(1); det = np.abs(np.linalg.det(M0)); M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    if len(M0) == 0:
        return None
    Qd = Q.double(); Mt = anneal_batch_t(torch.as_tensor(M0, dtype=torch.float64, device=DEV), Qd)
    H = torch.einsum('pc,bcd->bpd', Qd, Mt); dd = torch.abs(H - torch.round(H))
    dmax = dd.amax(2); inl = (dmax < 0.15).to(Qd.dtype); ni = inl.sum(1)
    md = (dd * inl[:, :, None]).sum((1, 2)) / (3 * ni).clamp(min=1)
    cov = (ni >= 0.30 * len(q)).to(Qd.dtype)
    key = torch.where(ni >= 8, cov * 1e3 - md, torch.full_like(md, -1e9))
    b = int(torch.argmax(key))
    return primitivize(buerger_reduce(Mt[b].cpu().numpy()), q) if float(key[b]) > -1e8 else None


def eval_rate(frames, starts, theta, mom, K, label):
    g = np.zeros(2, int); sl = 0; t0 = time.time()
    for q in frames:
        Q = torch.as_tensor(q, dtype=torch.float32, device=DEV); w = invq_weight(Q)
        qmax = float(Q.norm(dim=1).max())
        with torch.no_grad():
            T = unrolled_ascent(starts.clone(), Q, w, qmax, theta, mom)
        M = blind_from_cands(cands_from(T, Q, w), q, Q)
        m = matched(M, q) if M is not None else 0
        ok = M is not None and same_lattice(M, LYSO)
        sl += ok; g[0] += ok and m / len(q) >= 0.25; g[1] += ok and m >= 10
    if DEV == "cuda": torch.cuda.synchronize()
    print(f"  [{label}] K={K}  same_lattice {sl}/{len(frames)} ({100*sl//len(frames)}%)  "
          f"frac {g[0]} ({100*g[0]//len(frames)}%)  >=10refl {g[1]} ({100*g[1]//len(frames)}%)  "
          f"{1e3*(time.time()-t0)/len(frames):.0f} ms/frame")
    return sl


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    frames = [q for q in load(path) if len(q) >= 6]
    tr = frames[:16]                                              # small train batch (few params, self-sup)
    St = make_starts(NDIR_TRAIN); Se = make_starts(NDIR_EVAL)

    theta = torch.tensor(hand_schedule(K), dtype=torch.float32, device=DEV, requires_grad=True)
    mom = torch.tensor(0.0, dtype=torch.float32, device=DEV, requires_grad=True)   # sigmoid(0)=0.5
    opt = torch.optim.Adam([theta, mom], lr=0.05)
    Qs = [(torch.as_tensor(q, dtype=torch.float32, device=DEV)) for q in tr]
    ws = [invq_weight(Q) for Q in Qs]; qmaxs = [float(Q.norm(dim=1).max()) for Q in Qs]

    print(f"training unrolled ascent: K={K}, {iters} iters, {len(tr)} frames, {St.shape[0]} starts ...")
    for it in range(iters):
        opt.zero_grad(); loss = 0.0
        for Q, w, qm in zip(Qs, ws, qmaxs):
            T = unrolled_ascent(St.clone(), Q, w, qm, theta, mom)
            f, _ = objective(T, Q, w, sharp=True)
            loss = loss - torch.topk(f, 40).values.mean()        # maximize top-40 converged maxima
        loss = loss / len(Qs); loss.backward(); opt.step()
        if it % 30 == 0 or it == iters - 1:
            print(f"  it {it:3d}  loss {loss.item():.3f}  mom {torch.sigmoid(mom).item():.2f}")

    theta_hand = torch.tensor(hand_schedule(K), dtype=torch.float32, device=DEV)
    theta80 = torch.tensor(hand_schedule(80), dtype=torch.float32, device=DEV)
    print("\nEVAL on all", len(frames), "frames:")
    eval_rate(frames, Se, theta80, torch.tensor(0.0, device=DEV), 80, "hand K=80 (baseline)")
    eval_rate(frames, Se, theta_hand, torch.tensor(0.0, device=DEV), K, f"hand K={K} (truncated)")
    eval_rate(frames, Se, theta.detach(), mom.detach(), K, f"LEARNED K={K} (unrolled)")
    print(f"  learned lr schedule: {torch.nn.functional.softplus(theta).detach().cpu().numpy().round(3)}")
