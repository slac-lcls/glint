"""Tests for arm 1 (COM-only baseline, cbxd_arm1_com.py) and arm 3 (accumulator + tangent,
cbxd_hough_tangent.py) from the three-arm sweep (github.com/slac-lcls/glint issue #9).

Correctness: each arm's GPU-batched objective must agree exactly with its scalar reference
(com_score_batch_gpu vs cent_score; tangent_bonus_batch_gpu vs tangent_bonus). Smoke: each arm's
full seed_index pipeline must run end-to-end without error and return a valid rotation, at a
candidate count small enough to run quickly (this is NOT a claim about blind success rate --
that's what cbxd_sweep.py / cbxd_hough_tangent.py's own benchmarks are for).

  python -m pytest test_cbxd_arms.py -v
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "..")
from cbxd_joint import cent_score, rand_rot, simulate

try:
    from cbxd_arm1_com import com_score_batch_gpu, com_seed_index
    from cbxd_hough_tangent import (streak_reprs_and_tangents, tangent_bonus_batch_gpu,
                                    hough_seed_index_tangent)
    from tangent_score_test import tangent_bonus
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


def _sample_crystal(seed, noise=2e-4):
    rng = np.random.default_rng(seed)
    Rt = rand_rot(rng)
    kobs, lab, cents = simulate(Rt, rng, noise)
    return Rt, kobs, lab, cents


def _is_rotation(R):
    return (R is not None and R.shape == (3, 3)
            and np.allclose(R @ R.T, np.eye(3), atol=1e-4)
            and abs(np.linalg.det(R) - 1.0) < 1e-3)


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
def test_com_score_batch_gpu_matches_scalar():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(15, 3)) * 0.2
    Rs = np.stack([rand_rot(rng) for _ in range(10)])
    ref = np.array([cent_score(R, q, 0.03) for R in Rs])
    got = com_score_batch_gpu(Rs, q, 0.03, r_chunk=5)
    assert np.array_equal(ref, got)


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
def test_tangent_bonus_batch_gpu_matches_scalar():
    rng = np.random.default_rng(0)
    Rt, kobs, lab, cents = _sample_crystal(0)
    reprs, tangents = streak_reprs_and_tangents(Rt, 2e-4, rng)
    Rs = np.stack([rand_rot(rng) for _ in range(9)] + [Rt])
    ref = np.array([tangent_bonus(R, reprs, tangents, 0.03, 15.0, 1.0) for R in Rs])
    got = tangent_bonus_batch_gpu(Rs, reprs, tangents, 0.03, 15.0, 1.0, r_chunk=5)
    assert np.allclose(ref, got)


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
def test_com_seed_index_smoke():
    """Arm 1's full pipeline runs end-to-end and returns a valid rotation -- not a claim about
    accuracy (COM is the weak baseline; see cbxd_sweep.py for where it does/doesn't compete)."""
    _, kobs, lab, cents = _sample_crystal(2)
    R = com_seed_index(cents, np.random.default_rng(2), n_coarse=50_000)
    assert _is_rotation(R)


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
def test_hough_seed_index_tangent_smoke():
    """Arm 3's full pipeline (accumulator + tangent bonus) runs end-to-end and returns a valid
    rotation that indexes a decent fraction of the true streak points."""
    from cbxd_joint import score

    Rt, kobs, lab, cents = _sample_crystal(2)
    rng = np.random.default_rng(2)
    reprs, tangents = streak_reprs_and_tangents(Rt, 2e-4, rng)
    R = hough_seed_index_tangent(kobs, reprs, tangents, np.random.default_rng(2), n_coarse=500_000)
    assert _is_rotation(R)
    frac = score(R, kobs, 0.0025, ret_mask=True)[lab].mean()
    assert frac > 0.3                      # loose bound at reduced n_coarse -- a smoke test, not
                                            # the accuracy benchmark (cbxd_hough_tangent.py's own)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
