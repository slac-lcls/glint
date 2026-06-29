"""Try 2: break the cubic |G|-alias with rotation-invariant streak-PAIR features.

|G| alone leaves the CBXD consensus cell under-determined (a cubic 29^3 aliases the true
16/21/25; cbxd_multishot.py). But the angle between two reflections is ROTATION-INVARIANT
(q_i . q_j = G_i . G_j, since the per-shot orientation R preserves dot products). So pooling
the per-shot pair features (|G_a|, |G_b|, cos angle) over many random-orientation patterns
samples the reciprocal METRIC TENSOR, not just |G| magnitudes -- which distinguishes
orthorhombic from cubic. Per-streak direction is biased ~0.015 1/A (tens of deg at low |G|),
so we use only higher-|G| pairs where the relative bias is small, and rely on pooling to
average the per-measurement angle error.

  python cbxd_angles.py
"""
import sys
import numpy as np
from scipy.spatial import cKDTree
from cbxd_joint import K, simulate, rand_rot

_RH = np.arange(-8, 9)
_H = np.array(np.meshgrid(_RH, _RH, _RH, indexing="ij")).reshape(3, -1).T
_H = _H[np.any(_H != 0, 1)]


def predicted_features(cell, gmin, gmax):
    Bm = np.diag([1.0 / cell[0], 1.0 / cell[1], 1.0 / cell[2]])
    G = (Bm @ _H.T).T
    g = np.linalg.norm(G, axis=1)
    keep = (g >= gmin) & (g <= gmax)
    G, g = G[keep], g[keep]
    i, j = np.triu_indices(len(G), 1)
    ga = np.minimum(g[i], g[j]); gb = np.maximum(g[i], g[j])
    cos = (G[i] * G[j]).sum(1) / (g[i] * g[j])
    return np.column_stack([ga, gb, cos])


def observed_features(nshot, noise, rng, gmin, gmax):
    feats = []
    for _ in range(nshot):
        _, _, cents = simulate(rand_rot(rng), rng, noise)
        q = cents - np.array([0.0, 0.0, K])
        g = np.linalg.norm(q, axis=1)
        m = (g >= gmin) & (g <= gmax)
        q, g = q[m], g[m]
        if len(q) < 2:
            continue
        i, j = np.triu_indices(len(q), 1)
        ga = np.minimum(g[i], g[j]); gb = np.maximum(g[i], g[j])
        cos = (q[i] * q[j]).sum(1) / (g[i] * g[j])
        feats.append(np.column_stack([ga, gb, cos]))
    return np.vstack(feats)


def fit_score(obs, pred, wg=1 / 0.004, wc=1 / 0.04):
    """median nearest-neighbour distance of observed pair-features to predicted (scaled)."""
    O = obs * [wg, wg, wc]
    P = pred * [wg, wg, wc]
    d, _ = cKDTree(P).query(O)
    return float(np.median(d))


if __name__ == "__main__":
    noise = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-4
    gmin, gmax = 0.13, 0.30
    rng = np.random.default_rng(1)
    obs = observed_features(120, noise, rng, gmin, gmax)
    print(f"try2 pair-angle discrimination  noise={noise:.0e}  {len(obs)} observed pair-features "
          f"(|G| in [{gmin},{gmax}])")
    cells = [("ortho-true   (16,21,25)", (16., 21., 25.)),
             ("cubic-alias  (29,29,29)", (29., 29., 29.)),
             ("cubic-alias  (29.4^3)  ", (29.4, 29.4, 29.4)),
             ("tetragonal   (16,16,25)", (16., 16., 25.)),
             ("ortho-near   (16,21,24)", (16., 21., 24.)),
             ("ortho-swap   (15,22,25)", (15., 22., 25.))]
    for name, cell in cells:
        s = fit_score(obs, predicted_features(cell, gmin, gmax))
        print(f"  {name}: fit-score = {s:.3f}   (lower = better)")
