"""Flip / flat-Ewald indexing ambiguity — synthetic demonstration.

A still-frame orientation R and its reflection through the detector plane R' = Z.R (Z=diag(1,1,-1), the
crystal "flipped back toward the beam") map the reciprocal lattice to the SAME detector (x,y) positions;
only the Ewald-sphere CURVATURE (the small q_z of each excited reflection) distinguishes them. GLINT's
objective is position-only (inliers on |v.q - round|), so it should score the two ~identically as the
sphere flattens -- and then the STRUCTURE-FACTOR AMPLITUDES must break the tie.

We sweep curvature (LAM = 1/Ewald-radius, in the scaled units of basin_char) and report, on the TRUE data:
  inl_true  : inlier fraction of the true lattice basis V = inv(A)          (==1 by construction)
  inl_flip  : inlier fraction of the flipped basis V' = inv(Z.A) = V.Z      (-> 1 as sphere flattens)
  corr_true : correlation of predicted vs observed intensities, true index assignment  (==1)
  corr_flip : same for the flipped assignment                                (stays ~0: amplitudes decide)
Pure numpy; no GPU / no data needed."""
import numpy as np

QMAX = 1 / 3.0
TOL = 0.18
CELL = np.array([79.0, 79.0, 38.0])
Z = np.diag([1.0, 1.0, -1.0])


def rot(ax, th):
    ax = ax / np.linalg.norm(ax); c, s = np.cos(th), np.sin(th); x, y, z = ax
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def F2(hkl):
    """A deterministic pseudo-structure |F(hkl)|^2 > 0 (distinct per reflection; stands in for real |F|)."""
    h = hkl.astype(float)
    k1 = np.array([0.7, 1.9, 1.1]); k2 = np.array([2.3, 0.5, 1.7]); k3 = np.array([1.3, 2.7, 0.9])
    v = np.sin(h @ k1) + 0.8 * np.cos(h @ k2) + 0.6 * np.sin(h @ k3 + 1.0)
    return v * v + 0.05


def excited(A, LAM):
    R = np.array([int(np.ceil(QMAX * c)) + 1 for c in CELL])
    H = np.mgrid[-R[0]:R[0] + 1, -R[1]:R[1] + 1, -R[2]:R[2] + 1].reshape(3, -1).T.astype(float)
    H = H[np.any(H != 0, 1)]
    q = H @ A.T
    m = np.linalg.norm(q, axis=1) <= QMAX
    q, H = q[m], H[m]
    on = np.abs(q[:, 2] + 0.5 * LAM * np.einsum('ij,ij->i', q, q)) < 0.004      # thin Ewald shell
    return q[on], H[on]


def inl(V, q):
    h = q @ V.T
    return float(np.mean(np.all(np.abs(h - np.round(h)) < TOL, axis=1))), np.round(q @ V.T)


def corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


rng = np.random.default_rng(0)
print("cell=%s  QMAX=%.3f  TOL=%.2f   (LAM = curvature; small = flat Ewald)" % (tuple(CELL), QMAX, TOL))
print("%6s %7s %9s %9s %9s %9s" % ("LAM", "nspot", "inl_true", "inl_flip", "corr_true", "corr_flip"))
for LAM in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    it = ifl = ct = cf = 0.0; ns = 0; nrep = 12
    acc = np.zeros(4)
    for _ in range(nrep):
        A = rot(rng.normal(size=3), rng.uniform(0, np.pi)) @ np.diag(1 / CELL)
        q, H = excited(A, LAM)
        if len(q) < 20:
            continue
        V = np.linalg.inv(A)
        it_, hkl_t = inl(V, q)                       # true basis on true data
        qf = q * np.array([1, 1, -1])                # z-flipped data == flipped orientation on true data
        ifl_, _ = inl(V, qf)
        hkl_f = np.round(qf @ V.T)                    # index assignment the flip would give
        Iobs = F2(H)                                  # observed intensities follow the TRUE hkl
        acc += [it_, ifl_, corr(Iobs, F2(hkl_t)), corr(Iobs, F2(hkl_f))]
        ns += len(q); nrep_ok = True
    n = max(1, sum(1 for _ in range(nrep)))           # nrep constant; frames all pass at these LAM
    acc /= nrep
    print("%6.2f %7d %9.2f %9.2f %9.2f %9.2f" % (LAM, ns // nrep, acc[0], acc[1], acc[2], acc[3]))
print("\nreading: inl_flip -> inl_true as LAM->0 (position degeneracy on a flat Ewald sphere);")
print("corr_flip stays ~0 while corr_true=1 -> the structure-factor amplitudes break the tie at any curvature.")
