"""Synthetic single-shot SFX forward model (monochromatic).

We work directly in reciprocal space -- the right altitude for an indexer
prototype -- so we skip detector pixels for v0 and emit observed reciprocal
vectors g_i directly, together with ground-truth (M, h_i).

Knobs (the sparse-regime stress axes):
    n_target       desired number of observed spots (sets Ewald tolerance)
    pos_sigma      Gaussian jitter on each g_i [1/A] (measurement noise)
    frac_spurious  fraction of extra random spots not on the lattice
    frac_missing   fraction of true spots randomly dropped
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lattice import Ar_to_Br, cell_to_Ar, random_rotation

# Centering: primitive basis = conventional @ P (columns = primitive translations in
# conventional coords), and the systematic-absence rule for which conventional hkl are
# present. The indexer sees the *primitive* reciprocal lattice, so ground-truth M uses P.
_PRIMITIVE = {
    "P": np.eye(3),
    "I": np.array([[-.5, .5, .5], [.5, -.5, .5], [.5, .5, -.5]]),   # body (BCC)
    "F": np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),         # face (FCC)
    "C": np.array([[.5, -.5, 0], [.5, .5, 0], [0, 0, 1.]]),         # base
}


def _allowed(H, centering):
    h, k, l = H[:, 0], H[:, 1], H[:, 2]
    c = centering.upper()
    if c == "P":
        return np.ones(len(H), bool)
    if c == "I":
        return (h + k + l) % 2 == 0
    if c == "F":
        return (h % 2 == k % 2) & (k % 2 == l % 2)
    if c == "C":
        return (h + k) % 2 == 0
    raise ValueError(f"centering {centering!r} not supported (P/I/F/C)")


@dataclass
class Shot:
    g: np.ndarray            # (n,3) observed reciprocal vectors [1/A]
    M: np.ndarray            # (3,3) ground-truth rotated real basis (cols Ra,Rb,Rc)
    R: np.ndarray            # (3,3) orientation
    Ar: np.ndarray           # (3,3) real basis (unrotated)
    hkl_true: np.ndarray     # (n,3) int Miller index per spot (NaN-row for spurious)
    is_spurious: np.ndarray  # (n,) bool
    meta: dict = field(default_factory=dict)


def simulate_shot(
    cell=(78.0, 78.0, 38.0, 90.0, 90.0, 120.0),
    wavelength=1.3,          # Angstrom
    dmin=3.0,                # resolution limit [A] -> |g| <= 1/dmin
    n_target=40,
    pos_sigma=0.0,
    frac_spurious=0.0,
    frac_missing=0.0,
    centering="P",
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng
    Ar = cell_to_Ar(*cell)
    Br = Ar_to_Br(Ar)
    R = random_rotation(rng)
    # ground-truth lattice the indexer should recover = the PRIMITIVE lattice (the
    # actual lattice the observed spots form, even for centered Bravais types).
    M = R @ Ar @ _PRIMITIVE[centering.upper()]

    qmax = 1.0 / dmin
    # enumerate candidate hkl whose |g| <= qmax, then drop centering-forbidden ones
    hmax = int(np.ceil(qmax * max(np.linalg.norm(Ar, axis=0)))) + 1
    rng_h = np.arange(-hmax, hmax + 1)
    H = np.array(np.meshgrid(rng_h, rng_h, rng_h, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, axis=1)]
    H = H[_allowed(H, centering)]
    g_all = (R @ Br @ H.T).T
    res_ok = np.linalg.norm(g_all, axis=1) <= qmax
    H, g_all = H[res_ok], g_all[res_ok]

    # Ewald sphere (monochromatic): incident wavevector along +z, |k0| = 1/lambda.
    # Excitation error for node g: eps = |k0 + g| - |k0|. Observed if |eps| < tol.
    k0 = np.array([0.0, 0.0, 1.0 / wavelength])
    eps = np.linalg.norm(k0 + g_all, axis=1) - np.linalg.norm(k0)
    # choose tol to hit ~n_target spots
    order = np.argsort(np.abs(eps))
    n_keep = min(n_target, len(order))
    sel = order[:n_keep]
    H_obs, g_obs = H[sel], g_all[sel]

    # missing spots
    if frac_missing > 0 and len(g_obs) > 0:
        keep = rng.random(len(g_obs)) >= frac_missing
        H_obs, g_obs = H_obs[keep], g_obs[keep]

    # measurement jitter
    if pos_sigma > 0:
        g_obs = g_obs + rng.normal(0, pos_sigma, g_obs.shape)

    is_spur = np.zeros(len(g_obs), dtype=bool)
    hkl = H_obs.astype(float)

    # spurious spots: random points inside the resolution shell
    if frac_spurious > 0:
        n_spur = int(round(frac_spurious * len(g_obs)))
        if n_spur > 0:
            dirs = rng.normal(size=(n_spur, 3))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            mags = qmax * rng.random(n_spur) ** (1 / 3)
            g_spur = dirs * mags[:, None]
            g_obs = np.vstack([g_obs, g_spur])
            hkl = np.vstack([hkl, np.full((n_spur, 3), np.nan)])
            is_spur = np.concatenate([is_spur, np.ones(n_spur, dtype=bool)])

    # shuffle so spurious aren't all at the end
    perm = rng.permutation(len(g_obs))
    return Shot(
        g=g_obs[perm],
        M=M,
        R=R,
        Ar=Ar,
        hkl_true=hkl[perm],
        is_spurious=is_spur[perm],
        meta=dict(cell=cell, wavelength=wavelength, dmin=dmin, qmax=qmax,
                  centering=centering.upper()),
    )
