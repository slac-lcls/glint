"""User idea: a REVERSE/Chamfer selection cost. Instead of forward residual |q.M - round|
(gameable by spurious tight cells that index few spots), score a guessed lattice by
predicting its reciprocal nodes and measuring how well they match OBSERVED peaks. On a
single shot only the Ewald-sphere slice is observed, so we FIT the Ewald sphere from the
spots (q.k = -|q|^2/2) and restrict predicted nodes to a band near it. Then:
  reverse recall = frac of near-Ewald predicted nodes (|q|<=qmax) that have an observed
                   spot within tol  -> spurious dense cells predict many unmatched nodes.
A/B the SELECTION: cover-scorer pick vs reverse-recall pick vs combined, same candidates.

  STAGE=fit|select  python reverse_cost.py [frames] [N]
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_index import objective, refine_vec, distinct_maxima, invq_weight, buerger_reduce, primitivize, STARTS, DEV
from glint_fast import anneal_batch_t, load, matched
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
STEPS = int(os.environ.get("STEPS", "8"))
TOLABS = float(os.environ.get("TOLABS", "0.004"))     # spot-match dist in q-units (A^-1)
BAND = float(os.environ.get("BAND", "0.006"))         # Ewald-sphere thickness (A^-1)


def fit_ewald(q):
    """k_in from q.k = -|q|^2/2 (least squares). Returns k (3,), |k|=1/lambda."""
    b = -0.5 * (q ** 2).sum(1)
    k, *_ = np.linalg.lstsq(q, b, rcond=None)
    return k


def annealed_cells(qf):
    """front-end -> triplets -> batched anneal; return (Mt list, cover_key array)."""
    q = np.asarray(qf, float)
    Q = torch.as_tensor(q, dtype=torch.float32, device=DEV); w = invq_weight(Q)
    qmax = float(Q.norm(dim=1).max())
    T = refine_vec(STARTS.clone(), Q, w, qmax, steps=STEPS)
    f, _ = objective(T, Q, w, sharp=True)
    cands = distinct_maxima(T.cpu().numpy(), f.cpu().numpy())[:30]
    if len(cands) < 3:
        return None, None, None
    nrm = np.linalg.norm(cands, axis=1)
    tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
    M0 = np.transpose(cands[tris], (0, 2, 1)); sc = nrm[tris].prod(1); det = np.abs(np.linalg.det(M0))
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    if len(M0) == 0:
        return None, None, None
    Qd = Q.double(); Mt = anneal_batch_t(torch.as_tensor(M0, dtype=torch.float64, device=DEV), Qd)
    H = torch.einsum('pc,bcd->bpd', Qd, Mt); dd = torch.abs(H - torch.round(H))
    dmax = dd.amax(2); inl = (dmax < 0.15).to(Qd.dtype); ni = inl.sum(1)
    md = (dd * inl[:, :, None]).sum((1, 2)) / (3 * ni).clamp(min=1)
    cov = (ni >= 0.30 * len(q)).to(Qd.dtype)
    cover_key = torch.where(ni >= 8, cov * 1e3 - md, torch.full_like(md, -1e9)).cpu().numpy()
    return Mt.cpu().numpy(), cover_key, ni.cpu().numpy()


def reverse_recall(M, q, k, qmax):
    if abs(np.linalg.det(M)) < 1e3:
        return (0, 1)
    Minv = np.linalg.inv(M)
    H0 = np.rint(q @ M).astype(int)
    lo = H0.min(0) - 2; hi = H0.max(0) + 2
    if np.prod(hi - lo + 1) > 200000:
        return (0, 1)
    gh = [np.arange(lo[i], hi[i] + 1) for i in range(3)]
    H = np.stack(np.meshgrid(*gh, indexing="ij"), -1).reshape(-1, 3)
    qp = H @ Minv
    r = np.linalg.norm(qp, axis=1)
    qp = qp[(r > 1e-6) & (r <= qmax)]
    if len(qp) == 0:
        return (0, 1)
    dell = np.abs(np.linalg.norm(qp + k, axis=1) - np.linalg.norm(k))
    qpn = qp[dell < BAND]
    if len(qpn) < 5:
        return (0, 1)
    d = np.sqrt(((qpn[:, None, :] - q[None, :, :]) ** 2).sum(2)).min(1)
    matched = int((d < TOLABS).sum())
    return (matched, len(qpn))                                   # (matched near-Ewald nodes, total)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]; n = len(frames)
    stage = os.environ.get("STAGE", "select")

    if stage == "fit":
        ls = []
        for q in frames:
            q = np.asarray(q, float); k = fit_ewald(q)
            resid = np.abs(q @ k + 0.5 * (q ** 2).sum(1))
            ls.append((1.0 / np.linalg.norm(k), resid.mean()))
        ls = np.array(ls)
        print(f"Ewald fit on {n} frames: lambda median={np.median(ls[:,0]):.3f} A  "
              f"spread[{np.percentile(ls[:,0],10):.3f},{np.percentile(ls[:,0],90):.3f}]")
        print(f"  fit residual median={np.median(ls[:,1]):.2e} (should be << |q|~0.3 if planar Ewald holds)")
        sys.exit()

    # selection A/B over several reverse-score forms
    def ok(M, q):
        return same_lattice(M, LYSO) and matched(M, q) / len(q) >= 0.25
    tot = {"cover": 0, "recall": 0, "count": 0, "precpen": 0}; have = 0
    annealed_cells(frames[0])
    for q in frames:
        q = np.asarray(q, float)
        Mt, ckey, ni = annealed_cells(q)
        if Mt is None:
            continue
        have += 1
        qmax = float(np.linalg.norm(q, axis=1).max()); k = fit_ewald(q)
        fwd = np.where(ni >= 10)[0]
        if len(fwd) == 0:
            continue
        fwd = fwd[np.argsort(-ni[fwd])[:15]]
        mn = np.array([reverse_recall(Mt[i], q, k, qmax) for i in fwd], float)   # (matched, nband)
        m, nb = mn[:, 0], np.maximum(mn[:, 1], 1)
        scores = {"recall": m / nb, "count": m, "precpen": 1.5 * m - 0.5 * nb}
        tot["cover"] += ok(primitivize(buerger_reduce(Mt[int(np.argmax(ckey))]), q), q)
        for nm, sc in scores.items():
            b = fwd[int(np.argmax(sc))]
            tot[nm] += ok(primitivize(buerger_reduce(Mt[b]), q), q)
    print(f"selection A/B on {have}/{n} frames (TOLABS={TOLABS} BAND={BAND}):")
    for nm in ("cover", "recall", "count", "precpen"):
        print(f"  {nm:8}: {tot[nm]}/{n} ({100*tot[nm]//n}%)")
