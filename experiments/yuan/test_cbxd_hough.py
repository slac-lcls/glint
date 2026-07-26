"""Tests for the CBXD blind-orientation search (deliverable 2: GPU arc-Hough accumulator).

Correctness: score_batch / score_batch_gpu must agree exactly with cbxd_joint.score() -- they
are meant to be the same objective, just batched over candidate orientations. Regression: the
accumulator-seeded blind search must not be worse than the original seed_index on a frozen
crystal (and in practice is much better -- the canonical frozen-dataset number is 30/40 (75%)
for this arm vs the original serial seed_index's 0/6 on a small smoke-test regime; see
cbxd_hough_gpu.py's module docstring / PR #10 for the full four-arm comparison).

  python -m pytest test_cbxd_hough.py -v
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "..")
from cbxd_joint import rand_rot, score, seed_index, simulate

from cbxd_batch_seed import score_batch

try:
    from cbxd_hough_gpu import hough_seed_index, score_batch_gpu
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


def _sample_crystal(seed):
    rng = np.random.default_rng(seed)
    Rt = rand_rot(rng)
    kobs, lab, cents = simulate(Rt, rng, 2e-4)
    return kobs, lab, cents


def test_score_batch_matches_scalar():
    kobs, _, _ = _sample_crystal(0)
    rng = np.random.default_rng(1)
    Rs = np.stack([rand_rot(rng) for _ in range(10)])
    ref = np.array([score(R, kobs, 0.03) for R in Rs])
    got = score_batch(Rs, kobs, 0.03)
    assert np.array_equal(ref, got)


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
def test_score_batch_gpu_matches_scalar():
    kobs, _, _ = _sample_crystal(0)
    rng = np.random.default_rng(1)
    Rs = np.stack([rand_rot(rng) for _ in range(10)])
    ref = np.array([score(R, kobs, 0.03) for R in Rs])
    got = score_batch_gpu(Rs, kobs, 0.03, r_chunk=5, node_chunk=64)
    assert np.array_equal(ref, got)


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
def test_hough_seed_index_beats_seed_index():
    """On a frozen crystal, the GPU-accumulator-seeded search must recover at least as much
    of the true lattice as the original centroid-seeded seed_index (usually far more -- the
    canonical frozen-dataset number is 30/40 (75%), see cbxd_hough_gpu.py's module docstring)."""
    kobs, lab, cents = _sample_crystal(2)

    R_old = seed_index(kobs, cents, np.random.default_rng(2))
    frac_old = score(R_old, kobs, 0.0025, ret_mask=True)[lab].mean()

    R_new = hough_seed_index(kobs, np.random.default_rng(2), n_coarse=500_000)
    frac_new = score(R_new, kobs, 0.0025, ret_mask=True)[lab].mean()

    assert frac_new >= frac_old - 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
