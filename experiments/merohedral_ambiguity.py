"""Curvature-INDEPENDENT indexing ambiguity (the complement of flip_ambiguity.py).

When the LATTICE has higher symmetry than the STRUCTURE, an operator R_amb in the lattice point group but
not the structure's maps every reciprocal-lattice point to another lattice point -> the two indexings give
IDENTICAL spot positions at EVERY Ewald curvature (it's a lattice symmetry, not a flat-Ewald effect). So
position can NEVER decide (inl_amb = 1.0 always); only the structure-factor AMPLITUDES break it (Brehm-
Diederichs / ambigator, resolved at merge). Here: a pseudo-cubic P lattice (a=b=c) with the 4-fold about z
as the ambiguity operator, and an asymmetric pseudo-structure. Contrast with the geometric flip, whose
degeneracy only appears as the sphere flattens. Pure numpy."""
import numpy as np

QMAX = 1 / 3.0
TOL = 0.18
A0 = 60.0                                        # a=b=c -> the 4-fold about z is a lattice symmetry
OP = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)     # 90 deg about z (maps the cubic lattice to itself)
CELL = np.array([A0, A0, A0])


def rot(ax, th):
    ax = ax / np.linalg.norm(ax); c, s = np.cos(th), np.sin(th); x, y, z = ax
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def F2(hkl):
    h = hkl.astype(float)
    k1 = np.array([0.7, 1.9, 1.1]); k2 = np.array([2.3, 0.5, 1.7]); k3 = np.array([1.3, 2.7, 0.9])
    v = np.sin(h @ k1) + 0.8 * np.cos(h @ k2) + 0.6 * np.sin(h @ k3 + 1.0)
    return v * v + 0.05


def excited(A, LAM):
    R = int(np.ceil(QMAX * A0)) + 1
    H = np.mgrid[-R:R + 1, -R:R + 1, -R:R + 1].reshape(3, -1).T.astype(float)
    H = H[np.any(H != 0, 1)]
    q = H @ A.T
    m = np.linalg.norm(q, axis=1) <= QMAX
    q, H = q[m], H[m]
    on = np.abs(q[:, 2] + 0.5 * LAM * np.einsum('ij,ij->i', q, q)) < 0.004
    return q[on], H[on]


def inl(V, q):
    h = q @ V.T
    return float(np.mean(np.all(np.abs(h - np.round(h)) < TOL, axis=1)))


def corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


rng = np.random.default_rng(0)
print("pseudo-cubic a=b=c=%.0f  op = 4-fold about z (lattice symmetry, not structure symmetry)" % A0)
print("%6s %7s %9s %9s %9s %9s" % ("LAM", "nspot", "inl_true", "inl_amb", "corr_true", "corr_amb"))
for LAM in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    acc = np.zeros(4); ns = 0; nrep = 12
    for _ in range(nrep):
        A = rot(rng.normal(size=3), rng.uniform(0, np.pi)) @ np.diag(1 / CELL)
        q, H = excited(A, LAM)
        if len(q) < 20:
            continue
        V = np.linalg.inv(A)
        Hamb = (OP @ H.T).T                          # the ambiguous (reindexed) assignment
        Iobs = F2(H)                                 # observed intensities follow the TRUE hkl
        acc += [inl(V, q), inl(V, q),                # positions identical under a lattice symmetry
                corr(Iobs, F2(H)), corr(Iobs, F2(Hamb))]
        ns += len(q)
    acc /= nrep
    print("%6.2f %7d %9.2f %9.2f %9.2f %9.2f" % (LAM, ns // nrep, acc[0], acc[1], acc[2], acc[3]))
print("\nreading: inl_amb = inl_true = 1.0 at EVERY curvature (a lattice symmetry, not a flat-Ewald flip);")
print("corr_amb stays low & flat -> only the amplitudes decide, at all curvatures (Brehm-Diederichs regime).")
