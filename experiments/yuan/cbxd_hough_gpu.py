"""CBXD deliverable 2: the GPU arc-Hough accumulator, per research-plan-yuan.md Thrust A #2 --
"Build orientation voting over Kossel arcs on the GPU -- batched over candidate orientations x
streaks, the arc-curvature analog of index_blind_fast's cosine sum."

cbxd_batch_seed.py already proved the shape of this (batched-over-orientations, curvature-aware
via the full arc not just centroids) on CPU/numpy, chunked over candidates only -- and showed
that candidate DENSITY is the lever that matters (0/6 -> 4/6 blind success going 20k -> 200k
candidates, at ~68s/crystal on CPU). This file is that same accumulator on the GPU (torch),
so density can go another 1-2 orders of magnitude in the same wall-clock.

Memory shape: naively the accumulator is a dense (Nr, Np, Nn) vote tensor -- candidates x
streak-points x lattice-nodes -- which is far too big to materialize even in float32 once Nr
gets large (Nr=1e6, Np~150, Nn~822 -> ~120e9 elements). So this chunks over BOTH axes: an outer
loop over candidate-orientation chunks (r_chunk) and an inner loop over lattice-node chunks
(node_chunk), OR-reducing "explained" over nodes as it goes so only a (r_chunk, Np, node_chunk)
tile is ever live. Bound r_chunk * Np * node_chunk to the memory budget and both loops are cheap
(tens of chunks), not the 1e6-iteration Python loop this replaces.

  python cbxd_hough_gpu.py [n_coarse]
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "..")
from cbxd_joint import B, COSA, HS, K, SINA, rand_rot, refine, score, simulate

_GB = B @ HS.T                              # (3,Nn): lattice nodes, fixed crystal frame
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rand_rot_batch(rng, n):
    """Vectorized cbxd_joint.rand_rot: n uniform random rotations via unit quaternions."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _score_chunk(Rc_np, kobs_t, kx, ky, kz, GB_t, tol, node_chunk, device):
    Rc = torch.as_tensor(Rc_np, dtype=torch.float32, device=device)        # (C,3,3)
    C, Np, Nn = Rc.shape[0], kobs_t.shape[0], GB_t.shape[1]
    matched = torch.zeros(C, Np, dtype=torch.bool, device=device)
    for j0 in range(0, Nn, node_chunk):
        GBc = GB_t[:, j0:j0 + node_chunk]                          # (3,M)
        G = torch.einsum('cij,jm->cmi', Rc, GBc)                   # (C,M,3)
        Gn = G.norm(dim=2)                                          # (C,M)
        Ghat = G / Gn.unsqueeze(-1)
        dot = torch.einsum('pk,cmk->cpm', kobs_t, Ghat)             # (C,Np,M)
        resid = (dot - (Gn / 2).unsqueeze(1)).abs()
        kin_z = kz.view(1, -1, 1) - G[..., 2].unsqueeze(1)
        kin_x = kx.view(1, -1, 1) - G[..., 0].unsqueeze(1)
        kin_y = ky.view(1, -1, 1) - G[..., 1].unsqueeze(1)
        cone = (kin_z > K * COSA) & ((kin_x ** 2 + kin_y ** 2).sqrt() < K * SINA)
        matched |= ((resid < tol) & cone).any(dim=2)
    return matched.sum(dim=1).cpu().numpy()


def score_batch_gpu(Rs, kobs, tol, r_chunk=20000, node_chunk=64, device=DEVICE, min_chunk=64):
    """GPU arc-Hough accumulator: vote count per candidate orientation.

    Rs: (Nr,3,3). kobs: (Np,3) observed points (real + spurious). Returns (Nr,) int array,
    same definition as cbxd_joint.score(R, kobs, tol) for each R in Rs.

    On CUDA OOM (shared GPU, contention from other users' jobs), halves r_chunk for that span
    and retries rather than failing outright.
    """
    kobs_t = torch.as_tensor(kobs, dtype=torch.float32, device=device)
    GB_t = torch.as_tensor(_GB, dtype=torch.float32, device=device)
    kx, ky, kz = kobs_t[:, 0], kobs_t[:, 1], kobs_t[:, 2]
    Nr = len(Rs)
    counts = np.empty(Nr, dtype=np.int64)
    i0 = 0
    chunk = r_chunk
    while i0 < Nr:
        c = min(chunk, Nr - i0)
        try:
            counts[i0:i0 + c] = _score_chunk(Rs[i0:i0 + c], kobs_t, kx, ky, kz, GB_t, tol,
                                             node_chunk, device)
            i0 += c
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= min_chunk:
                raise
            chunk = max(chunk // 2, min_chunk)
    return counts


def hough_seed_index(kobs, rng, n_coarse=1_000_000, tol_c=0.03, keep=10, **gpu_kw):
    """Blind orientation search: GPU-accumulate votes over n_coarse candidates, refine+select
    the top `keep` exactly as seed_index / batch_seed_index do."""
    Rs = rand_rot_batch(rng, n_coarse)
    counts = score_batch_gpu(Rs, kobs, tol_c, **gpu_kw)
    top = np.argsort(counts)[::-1][:keep]
    best_R, best_s = None, -1.0
    for i in top:
        R = refine(kobs, Rs[i], rng, iters=300)
        s = score(R, kobs, 0.0025)
        if s > best_s:
            best_s, best_R = s, R
    return best_R


def _golden_check(rng):
    Rt = rand_rot(rng)
    kobs, lab, _ = simulate(Rt, rng, 2e-4)
    Rs = np.stack([rand_rot(rng) for _ in range(9)] + [Rt])
    ref = np.array([score(R, kobs, 0.03) for R in Rs])
    got = score_batch_gpu(Rs, kobs, 0.03, r_chunk=5, node_chunk=64)
    assert np.array_equal(ref, got), f"score_batch_gpu mismatch: ref={ref} got={got}"
    print(f"golden check OK (device={DEVICE}): counts={got}")


NOISE_LEVELS = (1e-4, 2e-4)
SEED = 2                                    # same crystals as cbxd_baseline.py / batch_seed.py


def run(n_coarse, ncry=6):
    """NOTE: crystal generation (rand_rot/simulate, driven by `rng`) and each crystal's search
    (hough_seed_index, driven by its own np.random.default_rng(SEARCH_SEED_BASE + c)) use
    INDEPENDENT rng streams. Earlier this used one shared rng for both -- hough_seed_index's
    millions of candidate draws left crystal N+1's orientation dependent on how much random state
    crystal N's SEARCH consumed, so "crystal #4" wasn't the same physical crystal across runs with
    different n_coarse. See experiments/yuan/generate_dataset.py's docstring for the full story;
    that + run_three_arms.py is the fixed, on-disk-dataset version of this same benchmark."""
    print(f"hough_seed_index  device={DEVICE}  n_coarse={n_coarse}  ncry={ncry}  seed={SEED}")
    print(f"{'noise(1/A)':>11} {'wall/crystal(s)':>16} {'success':>9} {'median real idx':>16}")
    for noise in NOISE_LEVELS:
        rng = np.random.default_rng(SEED)
        ok = 0
        fr = []
        times = []
        for c in range(ncry):
            Rt = rand_rot(rng)
            kobs, lab, cents = simulate(Rt, rng, noise)
            t0 = time.perf_counter()
            Rh = hough_seed_index(kobs, np.random.default_rng(SEED + 1000 + c), n_coarse=n_coarse)
            times.append(time.perf_counter() - t0)
            idx = score(Rh, kobs, 0.0025, ret_mask=True)
            fr.append(idx[lab].mean())
            ok += idx[lab].mean() > 0.7
        print(f"{noise:11.1e} {np.mean(times):16.2f} {f'{ok}/{ncry}':>9} {100*np.median(fr):15.0f}%")


if __name__ == "__main__":
    _golden_check(np.random.default_rng(0))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    ncry = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    run(n, ncry)
