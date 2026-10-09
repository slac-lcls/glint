"""Rescoring index_blind_nbest's own broad candidate pool by peak-count coverage, instead of
trusting its internal M4 ranking, recovers the true cell far more often at the noise levels where
production's own top-1 rate is weakest.

Motivation: index_blind_nbest(q, N) already computes and internally ranks many more candidate
cells than it returns -- its own docstring notes "the true cell is often a reachable-but-not-top-1
hypothesis that single-pass discards." This script asks: if we just ask for a LARGER N (so the
true-but-not-top-1 candidate is actually returned) and rescore the returned list by something as
simple as "how many observed peaks does this candidate explain within our own indexing tolerance,"
does the true cell rank higher than under our own M4 score?

count_explained_peaks(cell, q, tau) is deliberately the cheapest possible score: round every
observed peak to its nearest integer Miller index under the candidate cell, count how many land
within TOL (our own M2/M3 inlier tolerance). No orientation search, no new candidate generation,
no dependency beyond index_blind_nbest itself -- a candidate rescoring of an existing pool, not a
new indexer.

One important negative result folded in below: residual/fit-quality scores tried ALONE (mean
inlier distance, RMSD among explained peaks) are considerably worse than peak-count alone -- they
are gamed by candidates that fit a handful of points very well while explaining almost nothing.
Coverage should gate residual, not the reverse (consistent with our own SCORER's cover_flag gate
in glint_fast.py already doing this for the M4 stage itself).

  python rescore_by_peak_count.py                 # runs the sigma sweep below
"""
import sys

import numpy as np

sys.path.insert(0, "..")
sys.path.insert(0, "../..")

from glint import glint_fast as gf
from glint.simulate import simulate_shot
from glint.multishot import same_lattice

CELL = (79.02, 79.02, 37.98, 90, 90, 90)
WAVELENGTH = 1.3
DMIN = 1.8
K_TARGET = 100


def count_explained_peaks(cell, q, tau=0.18):
    """Coverage score: number of observed peaks explained (nearest-integer Miller index within
    tau) by this candidate cell. Higher is better. Returns 0 for a numerically degenerate cell
    (near-singular, can slip through index_blind_nbest's own determinant filter for a
    near-coplanar M4 triplet) rather than raising."""
    if not np.isfinite(cell).all() or np.linalg.cond(cell) > 1e8:
        return 0
    u = q @ cell
    h = np.rint(u)
    return int(np.sum(np.linalg.norm(u - h, axis=1) < tau))


def rank_of_true(cells, is_true_fn, score_fn):
    """1-indexed rank of the true cell under score_fn (higher-is-better), or None if absent."""
    scored = sorted(((score_fn(c), is_true_fn(c)) for c, _ in cells), reverse=True)
    for rank, (s, is_true) in enumerate(scored, start=1):
        if is_true:
            return rank
    return None


def main():
    sigmas = [0.0, 0.001, 0.002, 0.0025, 0.003]
    n_frames = 15
    n_candidates = 150  # ask index_blind_nbest for far more than production's default top-3
    seed = 41

    warm = simulate_shot(cell=CELL, wavelength=WAVELENGTH, dmin=DMIN, n_target=K_TARGET,
                          pos_sigma=0.002, rng=np.random.default_rng(999))
    _ = gf.index_blind_nbest(warm.g, N=5)  # CUDA warm-up, excluded from timing

    print("Rescoring index_blind_nbest's own broad pool by peak-count coverage\n"
          f"n_frames={n_frames}  N={n_candidates}  seed={seed}\n")

    for sigma in sigmas:
        rng = np.random.default_rng(seed)
        prod_ranks, cov_ranks = [], []
        n_present = 0
        for _ in range(n_frames):
            shot = simulate_shot(cell=CELL, wavelength=WAVELENGTH, dmin=DMIN,
                                  n_target=K_TARGET, pos_sigma=sigma, rng=rng)
            q = shot.g
            cells = gf.index_blind_nbest(q, N=n_candidates)
            if not cells:
                continue
            is_true = lambda c: same_lattice(c, gf.LYSO)  # noqa: E731
            if not any(is_true(c) for c, s in cells):
                continue
            n_present += 1

            # production's own M4 ranking is just the list's own order (already sorted by score)
            prod_rank = next(i for i, (c, s) in enumerate(cells, start=1) if is_true(c))
            prod_ranks.append(prod_rank)

            r = rank_of_true(cells, is_true, lambda c: count_explained_peaks(c, q))
            cov_ranks.append(r)

        print(f"sigma={sigma}: {n_present}/{n_frames} frames had the true cell present in the "
              f"N={n_candidates} pool")
        if n_present == 0:
            print("  (candidate-GENERATION failure at this sigma -- rescoring cannot help)\n")
            continue

        def summarize(ranks, label):
            ranks = np.array(ranks)
            p1 = np.mean(ranks == 1) * 100
            p3 = np.mean(ranks <= 3) * 100
            print(f"  {label:22s} P(rank=1)={p1:5.1f}%  P(rank<=3)={p3:5.1f}%  "
                  f"median rank={np.median(ranks):.1f}")

        summarize(prod_ranks, "production's own rank")
        summarize(cov_ranks, "peak-count rescoring")
        print()


if __name__ == "__main__":
    main()
