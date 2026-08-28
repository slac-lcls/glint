"""Step 2 of the ridge_moments follow-up: does the validated per-streak tangent (tangent_validate.py
-- recoverable to ~1-2 deg using RAW, un-thinned streak points) actually help the accumulator tell
the true orientation apart from wrong ones? Tested specifically on the two crystals that failed to
solve in the big cbxd_hough_gpu.py validation run (crystal #4 and #7, seed=2, noise=2e-4) -- the
"reflection-starved corner" Stefano's issue #9 predicts the tangent should rescue.

Score augmentation: for each streak, fit a representative point + tangent from RAW points (as
validated). For a candidate orientation R, find the nearest lattice node's predicted tangent at
that point (Ghat x (p - G/2), same closed form as the ground truth) and add a bonus if it aligns
with the observed tangent within tol_ang_deg (abs cosine, since tangent has a sign ambiguity).

  python tangent_score_test.py
"""
import sys

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import B, HS, rand_rot, score, simulate

from tangent_validate import _simulate_grouped, analytic_tangent, local_tangent_pca

NOISE = 2e-4
SEED = 2
HARD_CRYSTALS = (4, 7)                              # from the 20-crystal cbxd_hough_gpu.py run


def get_crystal(idx, noise=NOISE, seed=SEED):
    """Replay the same rand_rot/simulate sequence cbxd_hough_gpu.py's run() used, to land on the
    exact same crystal orientation as the failing case."""
    rng = np.random.default_rng(seed)
    for c in range(idx + 1):
        Rt = rand_rot(rng)
        kobs, lab, cents = simulate(Rt, rng, noise)
    return Rt, kobs, lab


def streak_reprs_and_tangents(Rt, noise, rng):
    """Independent noisy raw-point realization of the same crystal's streaks, purely to extract a
    per-streak representative point + observed tangent (thin=1, per tangent_validate.py's result)."""
    streaks = _simulate_grouped(Rt, rng, noise, thin=1)
    reprs, tangents = [], []
    for s in streaks:
        clean, noisy = s["clean"], s["noisy"]
        if len(clean) < 3:
            continue
        i = len(clean) // 2
        t = local_tangent_pca(noisy, i, halfwin=1)
        if t is not None:
            reprs.append(noisy[i])
            tangents.append(t)
    return np.array(reprs), np.array(tangents)


def tangent_bonus(R, reprs, tangents, tol, tol_ang_deg=15.0, bonus=1.0):
    """For each streak (repr point + observed tangent), find its BEST-MATCHING lattice node under R
    by the same Bragg-plane residual score() uses (NOT raw Euclidean distance to G -- a point near
    the circle is close to G/2 within the plane perp to G, not close to G itself). Only credit the
    tangent bonus if that match is already within the point tolerance `tol` -- the tangent is extra
    evidence for an already-plausible correspondence, not an independent vote that can inflate a
    wrong orientation on its own."""
    if len(reprs) == 0:
        return 0.0
    G = (R @ (B @ HS.T)).T                          # (Nn,3), all candidate lattice vectors under R
    Gn = np.linalg.norm(G, axis=1)
    Ghat = G / Gn[:, None]
    total = 0.0
    for p, t_obs in zip(reprs, tangents):
        resid = np.abs(p @ Ghat.T - Gn / 2.0)         # Bragg-plane residual to every node
        j = np.argmin(resid)
        if resid[j] >= tol:                           # gate: only an already-plausible point match
            continue
        t_pred = analytic_tangent(p, G[j])
        if t_pred is None:
            continue
        cos = abs(t_pred @ t_obs)
        ang = np.degrees(np.arccos(np.clip(cos, 0, 1)))
        if ang < tol_ang_deg:
            total += bonus
    return total


def margin_test(idx, n_random=2000, tol=0.0025, tol_ang_deg=15.0, bonus=1.0):
    Rt, kobs, lab = get_crystal(idx)
    rng = np.random.default_rng(1000 + idx)
    reprs, tangents = streak_reprs_and_tangents(Rt, NOISE, rng)

    s_true = score(Rt, kobs, tol)
    tb_true = tangent_bonus(Rt, reprs, tangents, tol, tol_ang_deg, bonus)

    rng2 = np.random.default_rng(2000 + idx)
    best_rand, best_rand_t = -1, -1
    for _ in range(n_random):
        R = rand_rot(rng2)
        s = score(R, kobs, tol)
        tb = tangent_bonus(R, reprs, tangents, tol, tol_ang_deg, bonus)
        best_rand = max(best_rand, s)
        best_rand_t = max(best_rand_t, s + tb)

    print(f"crystal #{idx}  n_streaks_with_tangent={len(reprs)}  n_kobs={len(kobs)}")
    print(f"  points-only:      true={s_true:6.1f}   best-of-{n_random}-random={best_rand:6.1f}   "
          f"margin={s_true - best_rand:+.1f}")
    print(f"  points+tangent:   true={s_true + tb_true:6.1f}   "
          f"best-of-{n_random}-random={best_rand_t:6.1f}   "
          f"margin={(s_true + tb_true) - best_rand_t:+.1f}")


if __name__ == "__main__":
    for idx in HARD_CRYSTALS:
        margin_test(idx)
        print()
