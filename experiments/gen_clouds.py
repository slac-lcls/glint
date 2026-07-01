"""Generate shared reciprocal-lattice point clouds for the GLINT-vs-DIALS/phenix head-to-head.
Same lyso lattice, density sweep (still slice -> oscillation wedges -> full rotation). Saved as npz so
the GLINT (pytorch/GPU env) and DIALS/labelit (dials.python env) runners index the IDENTICAL clouds.

  python gen_clouds.py   ->  /pscratch/sd/s/smarches/glint_real/clouds.npz
"""
import numpy as np

CELL = np.array([79.0, 79.0, 38.0]); LAM = 1.0; DMIN = 3.0
QMAX = 1.0 / DMIN
B = np.diag(1.0 / CELL)
REPS = 8
WEDGES = [("still", 0.0), ("wedge5", 5.0), ("wedge20", 20.0), ("wedge60", 60.0), ("full", 180.0)]


def rot(ax, th):
    ax = ax / np.linalg.norm(ax); c, s = np.cos(th), np.sin(th); x, y, z = ax
    return np.array([[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])


H = (np.ceil(QMAX * CELL) + 1).astype(int)
HKL = np.mgrid[-H[0]:H[0]+1, -H[1]:H[1]+1, -H[2]:H[2]+1].reshape(3, -1).T
HKL = HKL[np.any(HKL != 0, axis=1)]


def observed(A, W, axis):
    q0 = HKL @ A.T; m = np.linalg.norm(q0, axis=1) <= QMAX; q0, hk = q0[m], HKL[m]
    tol = 0.004
    if W == 0.0:
        obs = np.abs(q0[:, 2] + 0.5 * LAM * np.einsum("ij,ij->i", q0, q0)) < tol
        return hk[obs] @ A.T
    phis = np.deg2rad(np.linspace(0, W, max(2, int(W * 2))))
    obs = np.zeros(len(q0), bool)
    for ph in phis:
        qp = q0 @ rot(axis, ph).T
        obs |= np.abs(qp[:, 2] + 0.5 * LAM * np.einsum("ij,ij->i", qp, qp)) < tol
    return hk[obs] @ A.T


rng = np.random.default_rng(0)
out = {"cell": CELL, "qmax": QMAX, "reps": REPS, "regimes": np.array([w[0] for w in WEDGES])}
for label, W in WEDGES:
    for r in range(REPS):
        A = rot(rng.normal(size=3), rng.uniform(0, np.pi)) @ B
        out[f"{label}_{r}"] = observed(A, W, rng.normal(size=3)).astype(np.float64)
np.savez("/pscratch/sd/s/smarches/glint_real/clouds.npz", **out)
print("saved clouds.npz:", {w[0]: int(np.median([len(out[f'{w[0]}_{r}']) for r in range(REPS)])) for w in WEDGES})
