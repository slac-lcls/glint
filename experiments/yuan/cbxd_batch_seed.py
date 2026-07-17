"""CBXD deliverable 2, step 1: batch the coarse blind-search stage over MANY candidate
orientations at once (numpy, chunked), instead of cbxd_joint.seed_index's 20k-iteration
Python loop calling the cheap centroid-only cent_score one rotation at a time.

Two changes from seed_index's coarse stage:
  1. Batched: score all n_coarse candidates together (in chunks, to bound memory) instead
     of one Python-level call per candidate -- removes per-call interpreter overhead.
  2. Curvature-aware: scores the FULL arc (kobs, the Bragg-plane + cone test cbxd_joint.score
     uses), not the centroid-only parallel-beam proxy (cent_score) -- so the coarse stage
     already uses the extra information CBXD streaks carry, instead of throwing it away and
     relying on refine() to recover it later.

Everything downstream (top-K -> refine -> best-by-score) is unchanged from seed_index, so a
win here isolates the coarse-stage change.

  python cbxd_batch_seed.py [ncry]
"""
import sys
import time

import numpy as np

sys.path.insert(0, "..")
from cbxd_joint import B, COSA, HS, K, SINA, rand_rot, refine, score, simulate

_GB = B @ HS.T                             # (3,Nn): lattice nodes in the fixed crystal frame


def score_batch(Rs, kobs, tol, chunk=200):
    """Vectorized cbxd_joint.score() over a batch of candidate rotations.

    Rs: (Nr,3,3). kobs: (Np,3) observed points (real + spurious). Returns (Nr,) int counts,
    identical in definition to [score(R, kobs, tol) for R in Rs] (bit-exact would need
    float64; here float32 for chunk-size headroom, see golden check in __main__).

    Chunked over Rs (not a dense (Nr,Np,Nn) tensor) to bound memory: each chunk holds a
    handful of (C,Np,Nn) float32 arrays. Kept at (C,Np,Nn) throughout, never (C,Np,Nn,3) --
    the cone test only needs per-component broadcasts (kin_x/kin_y/kin_z separately), not a
    full 3-vector difference tensor, which would cost a factor of 3 more memory for nothing.
    """
    Nr = len(Rs)
    kobs32 = kobs.astype(np.float32)
    kx, ky, kz = kobs32[:, 0], kobs32[:, 1], kobs32[:, 2]
    counts = np.empty(Nr, dtype=np.int64)
    for i0 in range(0, Nr, chunk):
        Rc = Rs[i0:i0 + chunk].astype(np.float32)                        # (C,3,3)
        G = np.einsum('cij,jn->cni', Rc, _GB.astype(np.float32))         # (C,Nn,3)
        Gn = np.linalg.norm(G, axis=2)                                   # (C,Nn)
        Ghat = G / Gn[..., None]
        dot = np.einsum('pk,cnk->cpn', kobs32, Ghat)                     # (C,Np,Nn)
        resid = np.abs(dot - Gn[:, None, :] / 2.0)
        kin_z = kz[None, :, None] - G[:, None, :, 2]
        kin_x = kx[None, :, None] - G[:, None, :, 0]
        kin_y = ky[None, :, None] - G[:, None, :, 1]
        cone = (kin_z > K * COSA) & (np.hypot(kin_x, kin_y) < K * SINA)
        match = (resid < tol) & cone
        counts[i0:i0 + len(Rc)] = match.any(axis=2).sum(axis=1)
    return counts


def batch_seed_index(kobs, rng, n_coarse=20000, tol_c=0.03, keep=10, chunk=200):
    """Curvature-aware replacement for seed_index's coarse stage: score ALL n_coarse random
    candidates against the full arc via score_batch, then refine+select the top `keep` exactly
    as seed_index does. `cents` is no longer needed -- score_batch reads kobs directly.
    """
    Rs = np.stack([rand_rot(rng) for _ in range(n_coarse)])
    counts = score_batch(Rs, kobs, tol_c, chunk=chunk)
    top = np.argsort(counts)[::-1][:keep]
    best_R, best_s = None, -1.0
    for i in top:
        R = refine(kobs, Rs[i], rng, iters=300)
        s = score(R, kobs, 0.0025)
        if s > best_s:
            best_s, best_R = s, R
    return best_R


def _golden_check(rng):
    """score_batch must agree with the scalar score() it's replacing."""
    Rt = rand_rot(rng)
    kobs, lab, _ = simulate(Rt, rng, 2e-4)
    Rs = np.stack([rand_rot(rng) for _ in range(9)] + [Rt])
    ref = np.array([score(R, kobs, 0.03) for R in Rs])
    got = score_batch(Rs, kobs, 0.03)
    assert np.array_equal(ref, got), f"score_batch mismatch: ref={ref} got={got}"
    print(f"golden check OK  (10 rotations incl. true R, tol=0.03): counts={got}")


NOISE_LEVELS = (1e-4, 2e-4)
SEED = 2                                   # same as cbxd_baseline.py, same crystals


def run(ncry):
    """NOTE: crystal generation and each crystal's search use INDEPENDENT rng streams (see
    experiments/yuan/generate_dataset.py's docstring) -- batch_seed_index's own internal
    candidate draws must not perturb the next crystal's orientation."""
    print(f"batch_seed_index (batched full-arc coarse search -> refine top-10)")
    print(f"ncry={ncry}  seed={SEED}")
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
            Rh = batch_seed_index(kobs, np.random.default_rng(SEED + 1000 + c))
            times.append(time.perf_counter() - t0)
            idx = score(Rh, kobs, 0.0025, ret_mask=True)
            fr.append(idx[lab].mean())
            ok += idx[lab].mean() > 0.7
        print(f"{noise:11.1e} {np.mean(times):16.2f} {f'{ok}/{ncry}':>9} "
              f"{100*np.median(fr):15.0f}%")


if __name__ == "__main__":
    _golden_check(np.random.default_rng(0))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run(n)
