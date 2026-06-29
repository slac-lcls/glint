"""Unit-cell / lattice conventions.

Crystallographic convention, no 2*pi factor:

    Ar  = real-space basis, columns (a, b, c)           [Angstrom]
    Br  = reciprocal basis,  columns (a*, b*, c*)        [1/Angstrom]
        = inv(Ar).T          so that  Ar.T @ Br = I  (a*.a = 1, a*.b = 0, ...)

A reciprocal-lattice node h = (h,k,l) sits at  g = Br @ h.
With a crystal orientation R in SO(3) the *observed* node is  g = R @ Br @ h.

Key identity used throughout the indexer
-----------------------------------------
Let  M = R @ Ar  (columns are the rotated *real* lattice vectors R a, R b, R c).
Then the observed reciprocal vectors satisfy

    g = M^{-T} h          (since (R Ar)^{-T} = R Br)
    h = round(M.T @ g)    (recover Miller indices once M is known)

So indexing == finding M, and the FFT of the observed point cloud {g_i}
peaks at the columns of M and their integer combinations (the direct lattice).
"""

from __future__ import annotations

import numpy as np


def cell_to_Ar(a, b, c, alpha, beta, gamma):
    """Cell parameters (lengths in A, angles in degrees) -> real basis Ar (columns a,b,c).

    Standard crystallographic setup: a along x, b in the xy-plane.
    """
    al, be, ga = np.radians([alpha, beta, gamma])
    ca, cb, cg = np.cos([al, be, ga])
    sg = np.sin(ga)
    # volume factor
    vfac = np.sqrt(max(1.0 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg, 1e-12))
    av = np.array([a, 0.0, 0.0])
    bv = np.array([b * cg, b * sg, 0.0])
    cv = np.array([c * cb, c * (ca - cb * cg) / sg, c * vfac / sg])
    return np.column_stack([av, bv, cv])


def Ar_to_Br(Ar):
    """Reciprocal basis (columns a*,b*,c*) from real basis."""
    return np.linalg.inv(Ar).T


def cell_params(Ar):
    """(a,b,c,alpha,beta,gamma) from a real basis matrix (columns a,b,c)."""
    a, b, c = Ar.T
    la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))
    ang = lambda u, v: np.degrees(
        np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    )
    return (la, lb, lc, ang(b, c), ang(a, c), ang(a, b))


_BR = np.arange(-3, 4)
_BN = np.array(np.meshgrid(_BR, _BR, _BR, indexing="ij")).reshape(3, -1).T
_BN = _BN[np.any(_BN != 0, axis=1)]


def buerger_reduce(M):
    """Reduced basis = 3 shortest independent lattice vectors (columns).

    Canonicalizes a basis up to point group, so a non-primitive expression of a
    lattice (e.g. 'true c-axis + true a-axis + a face DIAGONAL' that fits a couple
    extra spots) collapses onto the true primitive cell. Returns the reduced 3x3,
    or the input unchanged if the lattice vectors are degenerate (collinear/coplanar
    within the search box)."""
    M = np.asarray(M, float)
    V = (M @ _BN.T).T
    L = np.linalg.norm(V, axis=1)
    o = np.argsort(L); V, L = V[o], L[o]
    v1 = V[0]; v2 = None
    for k in range(1, len(V)):
        if np.linalg.norm(np.cross(V[k], v1)) > 1e-3 * L[k] * L[0]:
            v2 = V[k]; break
    if v2 is None:
        return M
    nrm = np.cross(v1, v2); v3 = None
    for k in range(1, len(V)):
        if abs(V[k] @ nrm) > 1e-3 * L[k] * np.linalg.norm(nrm):
            v3 = V[k]; break
    if v3 is None:
        return M
    return np.column_stack([v1, v2, v3])


def reduced_params(M):
    """(sorted axis lengths, sorted |cos angle|) of the Buerger-reduced cell --
    a rotation- and basis-invariant fingerprint that respects degeneracy and volume."""
    R = buerger_reduce(M); cols = [R[:, i] for i in range(3)]
    lens = np.array([np.linalg.norm(c) for c in cols])
    cs = np.array([abs(cols[0] @ cols[1]) / (lens[0] * lens[1]),
                   abs(cols[0] @ cols[2]) / (lens[0] * lens[2]),
                   abs(cols[1] @ cols[2]) / (lens[1] * lens[2])])
    return np.sort(lens), np.sort(cs)


def random_rotation(rng):
    """Uniform random rotation matrix in SO(3) (Shoemake)."""
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
    )
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


SYMMETRIES = ["cubic", "tetragonal", "orthorhombic", "hexagonal",
              "trigonal", "monoclinic", "triclinic"]


def random_cell(rng, symmetry="random", lo=30.0, hi=120.0):
    """Random cell parameters (a,b,c,alpha,beta,gamma) of a given symmetry class.

    'monoclinic' and 'triclinic' are the sheared cases. 'random' picks a class
    uniformly. Triclinic angles are redrawn until the cell volume is non-degenerate.
    """
    if symmetry == "random":
        symmetry = SYMMETRIES[rng.integers(len(SYMMETRIES))]
    L = lambda: rng.uniform(lo, hi)
    if symmetry == "cubic":
        a = L(); return (a, a, a, 90.0, 90.0, 90.0)
    if symmetry == "tetragonal":
        return (lambda a: (a, a, L(), 90.0, 90.0, 90.0))(L())
    if symmetry == "orthorhombic":
        return (L(), L(), L(), 90.0, 90.0, 90.0)
    if symmetry == "hexagonal":
        return (lambda a: (a, a, L(), 90.0, 90.0, 120.0))(L())
    if symmetry == "trigonal":                       # rhombohedral setting
        a = L(); ang = rng.uniform(70.0, 110.0); return (a, a, a, ang, ang, ang)
    if symmetry == "monoclinic":
        return (L(), L(), L(), 90.0, rng.uniform(95.0, 125.0), 90.0)
    if symmetry == "triclinic":
        while True:
            al, be, ga = rng.uniform(70.0, 110.0, 3)
            ca, cb, cg = np.cos(np.radians([al, be, ga]))
            if 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg > 0.1:
                return (L(), L(), L(), al, be, ga)
    raise ValueError(f"unknown symmetry: {symmetry}")
