#!/usr/bin/env python3
"""Geometry QA check in the idle-GPU reference validation (glint/gpu_pool.py).

The check radially integrates a reference frame and correlates the profile against a
baseline; a detector that has shifted since the baseline gives a lower correlation.

These tests exist because the first version of the check never ran: it handed the
integrator a peak list where an image was expected, mishandled its (q, I) tuple return,
and swallowed the resulting exception -- so it reported nothing and passed everything.
Hence the assertions below: the check must FIRE (produce a correlation), must DROP on a
shifted geometry, and a check that errors must NOT read as healthy.

CPU-only (numpy + the drp-benchmarks RadialIntegrator); no GPU or torch required.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser('~/git/glint-claude'))
from glint.gpu_pool import GPUPool, _profile_corr  # noqa: E402

H = W = 96
NBIN = 100


def _qmap(cx, cy):
    """Per-pixel |q|, standing in for a detector geometry with beam centre (cx, cy)."""
    yy, xx = np.mgrid[0:H, 0:W]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * 0.002


def _rings(cx, cy, radii=(12.0, 21.0, 33.0, 44.0), width=1.6, seed=0):
    """A frame with concentric Bragg-like rings about (cx, cy), plus Poisson background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = rng.poisson(20.0, size=(H, W)).astype(float)
    for i, r0 in enumerate(radii):
        img += (900.0 / (i + 1)) * np.exp(-0.5 * ((r - r0) / width) ** 2)
    return img


def _pool(cx=W / 2, cy=H / 2):
    """A pool whose integrator is built on the geometry centred at (cx, cy)."""
    pool = GPUPool(n_gpus=1, use_threading=False,
                   reference_cell=np.eye(3), detector_geometry=_qmap(cx, cy))
    assert pool._integrator is not None, "RadialIntegrator unavailable -- is drp-benchmarks present?"
    return pool


def _validate(pool, image):
    """Run only the geometry check (other checks need torch/an indexer)."""
    return pool.validate_reference(np.zeros((0, 3)), image=image)['geometry']


def test_check_actually_fires():
    """A configured check must produce a correlation -- the original bug was silence."""
    pool = _pool()
    seeded = _validate(pool, _rings(W / 2, H / 2, seed=1))
    assert seeded['stability'] == 'seeded', seeded
    assert 'error' not in seeded, seeded

    second = _validate(pool, _rings(W / 2, H / 2, seed=2))
    assert 'profile_correlation' in second, f"check did not fire: {second}"
    assert second.get('error') is None, second


def test_same_geometry_is_stable():
    """Same detector, different noise -> the rings still line up."""
    pool = _pool()
    _validate(pool, _rings(W / 2, H / 2, seed=1))
    r = _validate(pool, _rings(W / 2, H / 2, seed=2))
    assert r['profile_correlation'] > 0.95, r
    assert r['stability'] == 'stable', r


def test_shifted_geometry_drifts():
    """Detector shifted since the baseline -> rings fall in different q bins -> drift."""
    pool = _pool()
    _validate(pool, _rings(W / 2, H / 2, seed=1))
    r = _validate(pool, _rings(W / 2 + 9.0, H / 2 - 6.0, seed=2))
    assert r['profile_correlation'] < 0.90, f"drift not detected: {r}"
    assert r['stability'] == 'drift', r


def test_drift_fails_overall_validation():
    """A drifted geometry must sink all_agree, not just annotate it."""
    pool = _pool()
    pool.validate_reference(np.zeros((0, 3)), image=_rings(W / 2, H / 2, seed=1))
    res = pool.validate_reference(np.zeros((0, 3)),
                                  image=_rings(W / 2 + 9.0, H / 2 - 6.0, seed=2))
    assert res['geometry']['stability'] == 'drift', res['geometry']
    assert res['validation']['all_agree'] is False, res['validation']


def test_broken_check_is_not_healthy():
    """A check that cannot run must report 'error' and fail validation, not pass quietly."""
    pool = _pool()
    pool._integrator = None
    pool._geom_init_error = 'simulated integrator failure'
    res = pool.validate_reference(np.zeros((0, 3)), image=_rings(W / 2, H / 2))
    assert res['geometry']['stability'] == 'error', res['geometry']
    assert res['validation']['all_agree'] is False, res['validation']
    assert pool.reference_stats()['geometry_error_count'] == 1, pool.reference_stats()


def test_no_image_skips_cleanly():
    """Geometry check is opt-in: without an image it is skipped, not errored."""
    pool = _pool()
    geom = pool.validate_reference(np.zeros((0, 3)))['geometry']
    assert geom == {}, geom


def test_stats_report_drift():
    pool = _pool()
    _validate(pool, _rings(W / 2, H / 2, seed=1))
    _validate(pool, _rings(W / 2, H / 2, seed=2))
    _validate(pool, _rings(W / 2 + 9.0, H / 2 - 6.0, seed=3))
    s = pool.reference_stats()
    assert s['geometry_drift_count'] == 1, s
    assert s['geometry_stable_count'] == 1, s
    assert s['geometry_profile_correlation_min'] < 0.90, s


def test_profile_corr_handles_empty_bins():
    """Integrator returns NaN for unpopulated bins; compare only where both are finite."""
    a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    assert abs(_profile_corr(a, a.copy()) - 1.0) < 1e-12
    flat = np.ones(5)
    assert _profile_corr(flat, flat) == 0.0          # no variance -> no information
    assert _profile_corr(np.full(5, np.nan), a) == 0.0


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                                  # noqa: BLE001
            fails += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print("all geometry-QA tests passed" if not fails else f"{fails} test(s) FAILED")
    sys.exit(1 if fails else 0)
