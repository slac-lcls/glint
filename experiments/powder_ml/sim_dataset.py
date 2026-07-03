"""Simulated labelled powder patterns for the powder autoencoder + cell-regression head.

Each sample: a random unit cell (one of the 7 crystal systems) -> its distinct d-spacings -> a fixed-length
1-D I(q) profile (pseudo-Voigt peaks on a smooth background). Labels = (cell a,b,c,alpha,beta,gamma;
crystal-system index 0-6; centering). Reuses the classical indexer's metric/centering machinery
(radial_integration/powder_index.py) so the "learned indexer" trains on exactly the physics the classical
seed-and-verify solves. Clean profiles are returned; the training loop adds noise on the fly (denoising AE).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from powder_index import cell_to_metric, _centering_ok  # noqa: E402

SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
QMIN, QMAX, NBINS = 0.5, 6.0, 1024                 # physics q (A^-1); d = 2*pi/q -> ~1.05..12.6 A
QGRID = np.linspace(QMIN, QMAX, NBINS)


def random_cell(system, rng):
    """A random cell (a,b,c,alpha,beta,gamma) + a plausible centering for the given system."""
    e = lambda lo=3.0, hi=16.0: rng.uniform(lo, hi)  # noqa: E731
    if system == "cubic":
        a = e(); cell = (a, a, a, 90, 90, 90); cen = rng.choice(["P", "I", "F"])
    elif system == "tetragonal":
        a = e(); c = e(); cell = (a, a, c, 90, 90, 90); cen = rng.choice(["P", "I"])
    elif system == "hexagonal":
        a = e(); c = e(); cell = (a, a, c, 90, 90, 120); cen = "P"
    elif system == "trigonal":                       # rhombohedral setting: a=b=c, equal oblique angles
        a = e(); ang = rng.uniform(60, 110); cell = (a, a, a, ang, ang, ang); cen = "P"
    elif system == "orthorhombic":
        cell = (e(), e(), e(), 90, 90, 90); cen = rng.choice(["P", "I", "F", "C"])
    elif system == "monoclinic":
        cell = (e(), e(), e(), 90, rng.uniform(95, 125), 90); cen = rng.choice(["P", "C"])
    else:                                            # triclinic
        cell = (e(), e(), e(), rng.uniform(72, 108), rng.uniform(72, 108), rng.uniform(72, 108)); cen = "P"
    return tuple(float(x) for x in cell), cen


def d_list(cell, centering, imax=6, nmax=60):
    """Distinct d-spacings (A) whose physics-q lies in [QMIN,QMAX], lattice-allowed, sorted (strongest low-q)."""
    A = cell_to_metric(*cell)
    rng = np.arange(-imax, imax + 1)
    H = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    H = H[np.any(H != 0, 1)]
    H = H[np.asarray(_centering_ok(H, centering))]
    Q = np.einsum("ni,ij,nj->n", H, A, H)            # 1/d^2
    Q = Q[Q > 0]
    d = 1.0 / np.sqrt(np.unique(np.round(Q, 6)))
    q = 2 * np.pi / d
    q = np.sort(q[(q >= QMIN) & (q <= QMAX)])
    return q[:nmax]


def profile(cell, centering, rng):
    """Clean I(q) profile (peaks + smooth background), max-normalised to 1."""
    qpk = d_list(cell, centering)
    y = np.zeros(NBINS)
    sig = rng.uniform(0.012, 0.035)                  # peak width (A^-1): instrument + broadening
    for i, q0 in enumerate(qpk):
        amp = rng.uniform(0.3, 1.0) * (1.0 + 1.5 * np.exp(-i / 8.0))   # low-q peaks a bit stronger
        y += amp * np.exp(-((QGRID - q0) ** 2) / (2 * sig ** 2))
    bg = rng.uniform(0.05, 0.3) * np.exp(-QGRID / rng.uniform(1.0, 3.0)) + rng.uniform(0.0, 0.05)
    y = y + bg
    return (y / (y.max() + 1e-9)).astype(np.float32), qpk.astype(np.float32)


def make_dataset(n, seed=0):
    """-> X (n,NBINS) clean profiles, cells (n,6), systems (n,), centerings (n,) [str]."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, NBINS), np.float32); cells = np.zeros((n, 6), np.float32)
    sysid = np.zeros(n, np.int64); cens = []
    for i in range(n):
        s = rng.integers(0, len(SYSTEMS)); system = SYSTEMS[s]
        cell, cen = random_cell(system, rng)
        y, qpk = profile(cell, cen, rng)
        if len(qpk) < 4:                             # too few lines in range -> resample
            while len(qpk) < 4:
                cell, cen = random_cell(system, rng); y, qpk = profile(cell, cen, rng)
        X[i] = y; cells[i] = cell; sysid[i] = s; cens.append(cen)
    return X, cells, sysid, np.array(cens)


# cell-parameter normalisation for the regression head (edges ~[3,16] A, angles ~[60,125] deg)
CELL_LO = np.array([3, 3, 3, 60, 60, 60], np.float32)
CELL_HI = np.array([16, 16, 16, 125, 125, 125], np.float32)


def norm_cell(c):
    return (np.asarray(c, np.float32) - CELL_LO) / (CELL_HI - CELL_LO)


def denorm_cell(cn):
    return np.asarray(cn, np.float32) * (CELL_HI - CELL_LO) + CELL_LO


if __name__ == "__main__":
    X, cells, sysid, cens = make_dataset(2000, seed=1)
    print("dataset:", X.shape, "systems:", np.bincount(sysid),
          "mean peaks/pattern:", float(np.mean((X > 0.2).sum(1))))
