"""Blind CBXD orientation by RANSAC over streak pairs + multistart joint refine.

Lesson so far (cbxd_joint.py): the joint Kossel-overlay OBJECTIVE is robust (97% indexed,
0 spurious at 2e-4 noise at the true orientation) but its basin is ~0.2 deg, so random /
centroid / annealed-NM seeders only land ~30-60% of the time. Per-streak plane fits give a
noisy G DIRECTION (curvature ~ noise), but the streak's |G| (= |centroid backprojection|)
is robust. So seed by RANSAC:
  1. backproject each streak centroid -> q = cent - k0 (rough G; robust |q|).
  2. assign each streak to candidate hkl SHELLS of the known cell by |G| match.
  3. sample streak PAIRS (i,j) + an hkl guess each; keep only pairs whose observed angle
     matches the cell-fixed model angle (the pair-angle constraint).
  4. solve R from the two direction correspondences (triad / two-vector Kabsch).
  5. score R with the robust joint overlay; keep the best few; multistart-refine.

  python cbxd_ransac.py [ncry]
"""
import sys
import numpy as np
from cbxd_joint import (B, HS, K, simulate, score, refine, rand_rot)

GNODES = (B @ HS.T).T                       # crystal-frame G for each hkl  (Nn,3)
GMAG = np.linalg.norm(GNODES, axis=1)


def triad(u1, u2):
    """Orthonormal frame (columns e1,e2,e3): e1 along u1, e2 in-plane, e3 = e1 x e2."""
    e1 = u1 / np.linalg.norm(u1)
    e2 = u2 - (u2 @ e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-9:
        return None
    e2 /= n2
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def rot_from_two(u1, u2, v1, v2):
    """Proper rotation R with R u1 || v1 and span(u1,u2) -> span(v1,v2)."""
    Fu = triad(u1, u2)
    Fv = triad(v1, v2)
    if Fu is None or Fv is None:
        return None
    return Fv @ Fu.T


def ransac_seed(kobs, cents, rng, n_iter=6000, gtol=0.012, atol=np.radians(2.0),
                keep=12, tol_seed=0.004):
    q = cents - np.array([0.0, 0.0, K])                 # rough G per streak
    qmag = np.linalg.norm(q, axis=1)
    cand = [np.where(np.abs(GMAG - m) < gtol)[0] for m in qmag]   # hkl shells per streak
    ns = len(q)
    usable = [k for k in range(ns) if len(cand[k]) > 0]
    if len(usable) < 2:
        return rand_rot(rng)
    hyps = []
    for _ in range(n_iter):
        i, j = rng.choice(usable, 2, replace=False)
        cos_obs = q[i] @ q[j] / (qmag[i] * qmag[j])
        obs_ang = np.arccos(np.clip(cos_obs, -1, 1))
        hi = cand[i][rng.integers(len(cand[i]))]
        hj = cand[j][rng.integers(len(cand[j]))]
        mod_ang = np.arccos(np.clip(GNODES[hi] @ GNODES[hj] / (GMAG[hi] * GMAG[hj]), -1, 1))
        if abs(obs_ang - mod_ang) > atol:
            continue
        R = rot_from_two(GNODES[hi], GNODES[hj], q[i], q[j])
        if R is None:
            continue
        hyps.append((score(R, kobs, tol_seed), R))
    if not hyps:
        return rand_rot(rng)
    hyps.sort(key=lambda t: -t[0])
    bestR, bs = hyps[0][1], -1
    for _, R0 in hyps[:keep]:
        R = refine(kobs, R0, rng)
        s = score(R, kobs, 0.0025)
        if s > bs:
            bs, bestR = s, R
    return bestR


if __name__ == "__main__":
    ncry = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"RANSAC streak-pair seeder + multistart joint refine  nodes={len(HS)}")
    print(f"{'noise(1/A)':>11} {'blind success':>14} {'median real idx':>16}")
    for noise in (0.0, 1e-4, 2e-4):
        rng = np.random.default_rng(3)
        ok = 0
        fr = []
        for _ in range(ncry):
            Rt = rand_rot(rng)
            kobs, lab, cents = simulate(Rt, rng, noise)
            Rh = ransac_seed(kobs, cents, rng)
            idx = score(Rh, kobs, 0.0025, ret_mask=True)
            fr.append(idx[lab].mean())
            ok += idx[lab].mean() > 0.7
        print(f"{noise:11.1e} {f'{ok}/{ncry}':>14} {100*np.median(fr):>15.0f}%")
