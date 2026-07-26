"""CBXD deliverable 2, arm 3: the GPU arc-Hough accumulator (cbxd_hough_gpu.py) + the per-streak
TANGENT vote (github.com/slac-lcls/glint issue #9). Validated in two steps first:

  tangent_validate.py   -- tangent IS recoverable from simulate()'s streaks (median 1-2 deg error),
                            but only using RAW (un-thinned) points -- simulate()'s own ::6
                            decimation leaves a median of 1 pt/streak, nothing to fit.
  tangent_score_test.py -- a tangent bonus, gated behind an already-passing point match (same
                            Bragg-plane residual/tolerance score() uses -- NOT raw distance to G,
                            which let wrong orientations rack up spurious credit in the first,
                            buggy pass), widens the true-vs-best-random SCALAR score margin 45-58%
                            on the two crystals that failed the blind search in cbxd_hough_gpu.py.

This file GPU-batches that bonus (tangent_bonus_batch_gpu, same chunk-over-candidates shape as
score_batch_gpu) and wires it into the coarse-seeding stage: candidates are ranked by
points-vote + tangent-bonus, then the usual refine()/score() tail (points-only -- refine's soft
objective doesn't know about tangents) picks the winner among the top-K.

  python cbxd_hough_tangent.py [n_coarse]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "..")
from cbxd_joint import B, HS, rand_rot, refine, score, simulate

from cbxd_hough_gpu import DEVICE, rand_rot_batch, score_batch_gpu
from tangent_validate import _simulate_grouped, analytic_tangent, local_tangent_pca
from tangent_score_test import get_crystal

_GB = B @ HS.T                              # (3,Nn), same as cbxd_hough_gpu.py


def streak_reprs_and_tangents(Rt, noise, rng):
    """Per-streak representative point + observed tangent, from RAW (thin=1) points -- the only
    setting tangent_validate.py found the tangent recoverable at all."""
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


def _tangent_bonus_chunk(Rc_np, reprs_t, tang_t, GB_t, tol, cos_tol, bonus, device):
    Rc = torch.as_tensor(Rc_np, dtype=torch.float32, device=device)  # (C,3,3)
    G = torch.einsum('cij,jn->cni', Rc, GB_t)                # (C,Nn,3)
    Gn = G.norm(dim=2)                                        # (C,Nn)
    Ghat = G / Gn.unsqueeze(-1)
    dot = torch.einsum('sk,cnk->csn', reprs_t, Ghat)          # (C,Ns,Nn)
    resid = (dot - (Gn / 2).unsqueeze(1)).abs()                # (C,Ns,Nn)
    best_resid, best_j = resid.min(dim=2)                      # (C,Ns)
    gate = best_resid < tol
    Gbest = torch.gather(G, 1, best_j.unsqueeze(-1).expand(-1, -1, 3))   # (C,Ns,3)
    r = reprs_t.unsqueeze(0) - Gbest / 2.0
    Gbest_hat = Gbest / Gbest.norm(dim=2, keepdim=True)
    t_pred = torch.cross(Gbest_hat, r, dim=2)
    t_pred = t_pred / t_pred.norm(dim=2, keepdim=True).clamp_min(1e-12)
    cos = (t_pred * tang_t.unsqueeze(0)).sum(dim=2).abs()
    match = gate & (cos > cos_tol)
    return match.float().sum(dim=1).cpu().numpy() * bonus


def tangent_bonus_batch_gpu(Rs, reprs, tangents, tol, tol_ang_deg=15.0, bonus=1.0,
                            r_chunk=5000, device=DEVICE, min_chunk=64):
    """GPU-batched cbxd_hough_tangent bonus over many candidate orientations. reprs/tangents:
    (Ns,3). Returns (Nr,) float array. Ns is small (a few dozen streaks) so, unlike
    score_batch_gpu, no node-axis chunking is needed -- (r_chunk, Ns, Nn) fits comfortably.

    On CUDA OOM (shared GPU, contention from other users' jobs), halves r_chunk for that span
    and retries rather than failing outright."""
    Nr = len(Rs)
    if len(reprs) == 0:
        return np.zeros(Nr, dtype=np.float64)
    reprs_t = torch.as_tensor(reprs, dtype=torch.float32, device=device)     # (Ns,3)
    tang_t = torch.as_tensor(tangents, dtype=torch.float32, device=device)   # (Ns,3)
    GB_t = torch.as_tensor(_GB, dtype=torch.float32, device=device)          # (3,Nn)
    cos_tol = float(np.cos(np.radians(tol_ang_deg)))
    out = np.zeros(Nr, dtype=np.float64)
    i0 = 0
    chunk = r_chunk
    while i0 < Nr:
        c = min(chunk, Nr - i0)
        try:
            out[i0:i0 + c] = _tangent_bonus_chunk(Rs[i0:i0 + c], reprs_t, tang_t, GB_t, tol,
                                                  cos_tol, bonus, device)
            i0 += c
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= min_chunk:
                raise
            chunk = max(chunk // 2, min_chunk)
    return out


def hough_seed_index_tangent(kobs, reprs, tangents, rng, n_coarse=1_000_000, tol_c=0.03,
                             tol_ang_deg=15.0, tbonus=1.0, keep=10, **gpu_kw):
    """hough_seed_index (cbxd_hough_gpu.py), but the coarse ranking is points-vote + tangent-bonus
    instead of points-vote alone. Refine + final select stay points-only (unchanged from
    cbxd_joint.seed_index / hough_seed_index) -- the tangent's job is to help PICK which coarse
    candidates are worth the expensive refine, not to replace the point objective."""
    Rs = rand_rot_batch(rng, n_coarse)
    pt_counts = score_batch_gpu(Rs, kobs, tol_c, **gpu_kw)
    tb = tangent_bonus_batch_gpu(Rs, reprs, tangents, tol_c, tol_ang_deg, tbonus)
    combined = pt_counts + tb
    top = np.argsort(combined)[::-1][:keep]
    best_R, best_s = None, -1.0
    for i in top:
        R = refine(kobs, Rs[i], rng, iters=300)
        s = score(R, kobs, 0.0025)
        if s > best_s:
            best_s, best_R = s, R
    return best_R


def _golden_check(rng):
    """tangent_bonus_batch_gpu must agree with tangent_score_test.py's scalar tangent_bonus."""
    from tangent_score_test import tangent_bonus

    Rt = rand_rot(rng)
    kobs, lab, _ = simulate(Rt, rng, 2e-4)
    reprs, tangents = streak_reprs_and_tangents(Rt, 2e-4, rng)
    Rs = np.stack([rand_rot(rng) for _ in range(9)] + [Rt])
    ref = np.array([tangent_bonus(R, reprs, tangents, 0.03, 15.0, 1.0) for R in Rs])
    got = tangent_bonus_batch_gpu(Rs, reprs, tangents, 0.03, 15.0, 1.0, r_chunk=5)
    assert np.allclose(ref, got), f"tangent_bonus_batch_gpu mismatch: ref={ref} got={got}"
    print(f"golden check OK (device={DEVICE}): bonuses={got}")


NOISE_LEVELS = (1e-4, 2e-4)
SEED = 2                                   # same crystals as cbxd_hough_gpu.py


def run(n_coarse, ncry=6):
    print(f"hough_seed_index_tangent  device={DEVICE}  n_coarse={n_coarse}  ncry={ncry}  seed={SEED}")
    print(f"{'noise(1/A)':>11} {'wall/crystal(s)':>16} {'success':>9} {'median real idx':>16}")
    for noise in NOISE_LEVELS:
        ok = 0
        fr = []
        times = []
        for c in range(ncry):
            # get_crystal replays rand_rot+simulate only (no extra draws in between), so crystal
            # index c here is the SAME physical crystal as in cbxd_hough_gpu.py's run() -- needed
            # to check the two known-hard crystals (#4, #7) by the same index.
            Rt, kobs, lab = get_crystal(c, noise=noise, seed=SEED)
            rng_t = np.random.default_rng(10_000 + c)      # independent stream: tangent extraction
            reprs, tangents = streak_reprs_and_tangents(Rt, noise, rng_t)
            rng = np.random.default_rng(20_000 + c)        # independent stream: the search itself
            t0 = time.perf_counter()
            Rh = hough_seed_index_tangent(kobs, reprs, tangents, rng, n_coarse=n_coarse)
            times.append(time.perf_counter() - t0)
            idx = score(Rh, kobs, 0.0025, ret_mask=True)
            frac = idx[lab].mean()
            fr.append(frac)
            ok += frac > 0.7
            print(f"  noise={noise:.0e} crystal={c:2d} frac={frac:.2f}", flush=True)
        print(f"{noise:11.1e} {np.mean(times):16.2f} {f'{ok}/{ncry}':>9} {100*np.median(fr):15.0f}%")


if __name__ == "__main__":
    _golden_check(np.random.default_rng(0))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    ncry = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    run(n, ncry)
