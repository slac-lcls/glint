"""D3/D4: recover BOTH lattices in a multi-crystal shot by DEFLATE-AND-REINDEX.
QDIST already locks ONE lattice cleanly (95%); here we add: index -> remove that cell's
inlier spots (reciprocal-distance) -> re-index the residual -> repeat. Test on a controlled
two-crystal mixture (two known LYSO orientations + spurious) with ground-truth spot labels,
comparing single-pass vs deflate on 'fraction of frames where BOTH lattices are recovered'.

Run with QDIST on:  QDIST=1 QDTOL=0.004 python d34_deflate.py [K]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
os.environ.setdefault("QDIST", "1")
import numpy as np
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_fast import index_blind_fast, LYSO, QDTOL
from fftindex.multishot import same_lattice
from fat_ewald import sim_fat


def inliers(M, q):
    if M is None or len(q) == 0:
        return np.zeros(len(q), bool)
    r = q @ M - np.rint(q @ M)
    return np.linalg.norm(r @ np.linalg.pinv(M), axis=1) < QDTOL


def two_crystal(rng, frac2=1.0, spur=0.1):
    g1, M1 = sim_fat(0.001, spur=0.0, rng=rng)
    g2, M2 = sim_fat(0.001, spur=0.0, n_cap=max(6, int(frac2 * len(g1))), rng=rng)
    q = np.vstack([g1, g2])
    lab = np.array([1] * len(g1) + [2] * len(g2))
    if spur > 0:
        ns = int(round(spur * len(q))); qm = float(np.linalg.norm(q, axis=1).max())
        d = rng.normal(size=(ns, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        q = np.vstack([q, d * (qm * rng.random(ns) ** (1 / 3))[:, None]])
        lab = np.concatenate([lab, np.zeros(ns, int)])
    return q, lab


def deflate(q, max_lat=4, min_new=15, min_cov=0.20):
    """Index, strip a cell's NEW residual inliers, repeat. Stop when a new cell fails to
    explain enough FRESH spots (a spurious cell only re-indexes leftover noise)."""
    cells = []; taken = np.zeros(len(q), bool)
    for _ in range(max_lat):
        ridx = np.where(~taken)[0]
        if len(ridx) < 6:
            break
        resid = q[ridx]
        M = index_blind_fast(resid)
        if M is None:
            break
        new = inliers(M, resid)                   # inliers among the RESIDUAL only
        if new.sum() < min_new or new.mean() < min_cov:
            break                                 # not a real new lattice -> stop
        cells.append(M); taken[ridx[new]] = True
    return cells


def recovered(cells, g):                          # best single-cell recall over a lattice's spots
    return max((inliers(M, g).mean() for M in cells), default=0.0)


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.default_rng(0)
    q0, _ = two_crystal(rng); index_blind_fast(q0)            # warmup
    print(f"D3/D4 deflate stopping-criterion sweep  QDIST QDTOL={QDTOL}  K={K}  (truth = 2 lattices)")
    print(f"{'min_new':>7} {'min_cov':>7} {'frac2':>6} {'BOTH':>6} {'med#lat':>8} {'>2lat':>6}")
    for min_new, min_cov in ((10, 0.15), (15, 0.20), (20, 0.25)):
        for frac2 in (0.5, 1.0):
            rng = np.random.default_rng(1)
            db = 0; nl = []; over = 0
            for _ in range(K):
                q, lab = two_crystal(rng, frac2=frac2)
                g1, g2 = q[lab == 1], q[lab == 2]
                cells = deflate(q, min_new=min_new, min_cov=min_cov)
                nl.append(len(cells)); over += len(cells) > 2
                d1 = recovered(cells, g1); d2 = recovered(cells, g2)
                db += (d1 > 0.5 and d2 > 0.5)
            print(f"{min_new:7d} {min_cov:7.2f} {frac2:6.1f} {100*db//K:5d}% "
                  f"{int(np.median(nl)):8d} {100*over//K:5d}%")
