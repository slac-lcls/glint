"""Does a FAT Ewald slice relax the single-shot indexing degeneracy? (user idea).
A single monochromatic shot samples a ~2D slice of 3D reciprocal space -> the difference
cloud is rank-deficient (sigma ~ [1,0.88,0.16], third axis barely constrained), which is
why blind single-shot indexing and the reverse/Chamfer cost struggle. FEL bandwidth
(dlam/lam ~ 0.1-2%+) and small crystals / mosaicity THICKEN the slice into a 3D shell.
Here we simulate lysozyme shots keeping ALL nodes with excitation error |eps|<tol (tol =
slice half-thickness, ~ dlam/lam * |k0|) and sweep tol -> measure (a) degeneracy sigma3/
sigma1 of the difference cloud, (b) blind indexing rate.

  python fat_ewald.py [N_per_tol]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from fftindex.lattice import cell_to_Ar, Ar_to_Br, random_rotation
from glint_fast import index_blind_fast, LYSO, matched
from fftindex.multishot import same_lattice

CELL = (79.02, 79.02, 37.98, 90, 90, 90)
LAM = 1.322                                   # A (cxidb-17 beam)
K0MAG = 1.0 / LAM
DMIN = float(os.environ.get("DMIN", "5.0"))   # lower res -> flatter slice -> degenerate regime


NCAP = int(os.environ.get("NCAP", "300"))


def sim_fat(tol, n_cap=NCAP, spur=0.30, jitter=5e-4, rng=None):
    Ar = cell_to_Ar(*CELL); Br = Ar_to_Br(Ar); R = random_rotation(rng)
    M = R @ Ar                                # primitive P
    qmax = 1.0 / DMIN
    hmax = int(np.ceil(qmax * max(np.linalg.norm(Ar, axis=0)))) + 1
    rh = np.arange(-hmax, hmax + 1)
    H = np.array(np.meshgrid(rh, rh, rh, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, 1)]
    g = (R @ Br @ H.T).T
    ok = np.linalg.norm(g, axis=1) <= qmax
    g = g[ok]
    k0 = np.array([0.0, 0.0, K0MAG])
    eps = np.linalg.norm(k0 + g, axis=1) - K0MAG
    g = g[np.abs(eps) < tol]                  # FAT slice: keep ALL within tol
    if len(g) > n_cap:
        g = g[rng.choice(len(g), n_cap, replace=False)]
    if jitter > 0:
        g = g + rng.normal(0, jitter, g.shape)
    if spur > 0 and len(g):                   # spurious spots in the resolution shell
        ns = int(round(spur * len(g)))
        d = rng.normal(size=(ns, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        g = np.vstack([g, d * (qmax * rng.random(ns) ** (1 / 3))[:, None]])
    return g, M


def degeneracy(g):
    """sigma3/sigma1 of the difference cloud: ~0 for a thin 2D slice, ->1 when 3D."""
    n = len(g)
    if n < 4:
        return 0.0
    i, j = np.triu_indices(n, 1)
    d = g[i] - g[j]
    s = np.linalg.svd(d, compute_uv=False)
    return float(s[2] / s[0])


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(0)
    g0, _ = sim_fat(0.01, rng=rng); index_blind_fast(g0)       # warmup
    print(f"{'tol(1/A)':>9} {'~dlam/lam':>10} {'med_spots':>10} {'deg sig3/sig1':>14} {'blind rate':>12}")
    print(f"  (DMIN={DMIN} A, qmax={1/DMIN:.3f}, curvature term qmax^2/2|k0|={(1/DMIN)**2/(2*K0MAG):.4f})")
    for tol in (0.0005, 0.001, 0.002, 0.004, 0.008, 0.015, 0.030):
        rng = np.random.default_rng(100)
        degs = []; nsp = []; ok = 0; tried = 0
        for _ in range(M):
            g, Mt = sim_fat(tol, rng=rng)
            if len(g) < 8:
                continue
            tried += 1; nsp.append(len(g)); degs.append(degeneracy(g))
            Mi = index_blind_fast(g)
            ok += Mi is not None and same_lattice(Mi, LYSO)
        bw = tol * LAM                                          # ~ dlam/lam
        print(f"{tol:9.3f} {100*bw:9.1f}% {int(np.median(nsp)):>10} "
              f"{np.median(degs):>14.3f} {ok}/{tried} ({100*ok//max(tried,1)}%)")
