"""Arm 1 of the three-arm sweep (issue #9): "peakfinder + COM -> standard point-indexer" -- the
simple baseline that never touches arc/tangent data, only each streak's centroid. This is the
arm the sweep expects the accumulator (arm 2/3) to have to beat in the reflection-starved corner,
and the one that's supposed to "catch up" as streaks get sparse/short.

Reuses simulate()'s own `cents` (streak centroids, already computed) and cbxd_joint.cent_score
(the existing parallel-beam-equivalent point-distance objective) for evaluation/final-pick, adding
only what was missing: a GPU-batched coarse search over candidate orientations (com_score_batch_gpu,
same shape as cbxd_hough_gpu.score_batch_gpu) and a CENTROID-ONLY local refine (refine_com) --
cbxd_joint.refine() anneals against the full arc (soft_score(kobs,...)), which would leak arc
information into a baseline that's supposed to not have it.
"""
import sys

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, "..")
from cbxd_joint import B, HS, K, cent_score, rand_rot, rotvec

from cbxd_hough_gpu import DEVICE, rand_rot_batch

_GB = B @ HS.T                              # (3,Nn)


def _com_score_chunk(Rc_np, q_t, GB_t, tol, device):
    """One chunk's worth of com_score_batch_gpu, split out so it can be retried at a smaller
    chunk size on OOM (this is a shared, multi-tenant GPU node -- free memory fluctuates with
    what OTHER users' jobs are doing, not something a fixed chunk size can assume away)."""
    Rc = torch.as_tensor(Rc_np, dtype=torch.float32, device=device)        # (C,3,3)
    G = torch.einsum('cij,jn->cni', Rc, GB_t)                              # (C,Nn,3)
    G2 = (G ** 2).sum(-1)                                                  # (C,Nn)
    qG = torch.einsum('pk,cnk->cpn', q_t, G)                               # (C,Nq,Nn)
    q2 = (q_t ** 2).sum(1)                                                 # (Nq,)
    d2 = q2.view(1, -1, 1) + G2.unsqueeze(1) - 2 * qG                      # (C,Nq,Nn)
    dmin2 = d2.min(dim=2).values                                           # (C,Nq)
    return (dmin2 < tol * tol).sum(dim=1).cpu().numpy()


def com_score_batch_gpu(Rs, q, tol, r_chunk=5000, device=DEVICE, min_chunk=64):
    """GPU-batched cbxd_joint.cent_score over many candidate orientations. q: (Nq,3) centroid
    backprojections (cents - k0). Returns (Nr,) int counts.

    |q-G|^2 = |q|^2 + |G|^2 - 2 q.G expanded via matmul, so the running tensor stays (C,Nq,Nn) --
    NOT (C,Nq,Nn,3), which OOM'd here at r_chunk=20000 (a 4-D difference tensor is 3x the memory
    for no reason, the same mistake score_batch_gpu/tangent_bonus_batch_gpu avoid elsewhere).

    On CUDA OOM (shared GPU, contention from other users' jobs), halves the chunk for that span
    and retries rather than failing outright."""
    Nr = len(Rs)
    q_t = torch.as_tensor(q, dtype=torch.float32, device=device)          # (Nq,3)
    GB_t = torch.as_tensor(_GB, dtype=torch.float32, device=device)       # (3,Nn)
    counts = np.empty(Nr, dtype=np.int64)
    i0 = 0
    chunk = r_chunk
    while i0 < Nr:
        c = min(chunk, Nr - i0)
        try:
            counts[i0:i0 + c] = _com_score_chunk(Rs[i0:i0 + c], q_t, GB_t, tol, device)
            i0 += c
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= min_chunk:
                raise
            chunk = max(chunk // 2, min_chunk)
    return counts


def soft_cent_score(R, q, sigma):
    """Smooth centroid-only objective (the arm-1 analog of cbxd_joint.soft_score)."""
    G = (R @ (B @ HS.T)).T
    d2 = ((q[:, None, :] - G[None, :, :]) ** 2).sum(2).min(1)
    return float(np.exp(-d2 / (2 * sigma * sigma)).sum())


def refine_com(q, R0, rng):
    """Centroid-only local refine -- same deterministic-annealing shape as cbxd_joint.refine(),
    but the objective never sees arc points, only centroids (q = cents - k0)."""
    bR = R0
    for sigma in (0.06, 0.035, 0.02, 0.012, 0.007, 0.004, 0.0025, 0.0015):
        res = minimize(lambda w: -soft_cent_score(bR @ rotvec(w), q, sigma), np.zeros(3),
                       method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 300})
        bR = bR @ rotvec(res.x)
    return bR


def com_seed_index(cents, rng, n_coarse=1_000_000, tol_c=0.03, keep=10, r_chunk=20000):
    """Full arm-1 pipeline: batched coarse centroid search -> centroid-only refine -> best-by-
    centroid-score among the top `keep`. Never touches arc points."""
    q = cents - np.array([0.0, 0.0, K])
    Rs = rand_rot_batch(rng, n_coarse)
    counts = com_score_batch_gpu(Rs, q, tol_c, r_chunk=r_chunk)
    top = np.argsort(counts)[::-1][:keep]
    best_R, best_s = None, -1
    for i in top:
        R = refine_com(q, Rs[i], rng)
        s = cent_score(R, q, 0.0025)
        if s > best_s:
            best_s, best_R = s, R
    return best_R


def _golden_check(rng):
    q = rng.normal(size=(15, 3)) * 0.2
    Rs = np.stack([rand_rot(rng) for _ in range(10)])
    ref = np.array([cent_score(R, q, 0.03) for R in Rs])
    got = com_score_batch_gpu(Rs, q, 0.03, r_chunk=5)
    assert np.array_equal(ref, got), f"com_score_batch_gpu mismatch: ref={ref} got={got}"
    print(f"golden check OK (device={DEVICE}): counts={got}")


if __name__ == "__main__":
    _golden_check(np.random.default_rng(0))
