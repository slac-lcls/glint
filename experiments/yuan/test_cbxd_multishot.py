"""Tests for CBXD deliverable 4 (issue #12): cross-frame consensus.

The core claim generate_dataset_multishot.py and run_multishot.py rely on: pooling M shots of
the SAME crystal into one accumulator run is mathematically exact for the vote-count objective,
not an approximation -- score() just counts matched points, so concatenating point clouds and
summing per-shot scores must agree exactly.

  python -m pytest test_cbxd_multishot.py -v
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "..")
from cbxd_joint import rand_rot, score, simulate

from generate_dataset_multishot import NA, pooled, load_dataset

from generate_dataset_grid import load_cell


def test_multishot_pooling_exact():
    """score(R, concat(kobs_1..kobs_M), tol) == sum_m score(R, kobs_m, tol) -- pooling shots is
    exact accumulator-stacking, the same operation as pooling streaks within one shot."""
    rng = np.random.default_rng(0)
    Rt = rand_rot(rng)
    shots = [simulate(Rt, rng, 2e-4) for _ in range(5)]
    tol = 0.0025

    summed = sum(score(Rt, kobs, tol) for kobs, _, _ in shots)
    kobs_pool = np.vstack([s[0] for s in shots])
    pooled_score = score(Rt, kobs_pool, tol)
    assert pooled_score == summed

    # must also hold for an arbitrary (wrong) candidate orientation, not just the truth
    R_wrong = rand_rot(rng)
    summed_wrong = sum(score(R_wrong, kobs, tol) for kobs, _, _ in shots)
    assert score(R_wrong, kobs_pool, tol) == summed_wrong


def test_pooled_helper_matches_manual_concat():
    """generate_dataset_multishot.pooled() must match a manual vstack/concatenate of the same
    shots, and pooled(c, m) must be a strict prefix of pooled(c, m+1) (nested, not resampled)."""
    rng = np.random.default_rng(1)
    Rt = rand_rot(rng)
    shots = [simulate(Rt, rng, 2e-4) for _ in range(4)]
    crystal = dict(Rt=Rt, shots=shots)

    kobs2, lab2, cents2 = pooled(crystal, 2)
    assert np.array_equal(kobs2, np.vstack([shots[0][0], shots[1][0]]))
    assert np.array_equal(lab2, np.concatenate([shots[0][1], shots[1][1]]))

    kobs3, _, _ = pooled(crystal, 3)
    assert np.array_equal(kobs3[:len(kobs2)], kobs2)                    # M=2 is a strict prefix of M=3
    assert len(kobs3) > len(kobs2)


def test_shot_zero_matches_grid_cell():
    """generate_dataset_multishot.py must reuse deliverable 3's exact crystals, not re-derive a
    look-alike set: shot 0 of every multishot crystal must be BYTE-IDENTICAL (same Rt, same
    kobs/lab/cents) to the corresponding cached data/grid cell at (NA=0.028, that noise, i)."""
    noise = 2e-4
    ms_ds = load_dataset(noise, n=3)          # just the first few crystals -- cheap
    grid_ds = load_cell(NA, noise, n=3)
    for i in range(3):
        assert np.array_equal(ms_ds[i]["Rt"], grid_ds[i]["Rt"])
        kobs0, lab0, cents0 = ms_ds[i]["shots"][0]
        assert np.array_equal(kobs0, grid_ds[i]["kobs"])
        assert np.array_equal(lab0, grid_ds[i]["lab"])
        assert np.array_equal(cents0, grid_ds[i]["cents"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
