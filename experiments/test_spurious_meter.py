"""CPU tests for the spurious-peak meters (glint.spurious_meter) -- pure numpy, no GPU/torch.

`pytest experiments/test_spurious_meter.py`, or `python experiments/test_spurious_meter.py` for a
PASS/FAIL loop. Cell convention: hkl = q @ M, q = hkl @ inv(M).
"""
import numpy as np

from glint.spurious_meter import (finder_spurious, unindexed_fraction, null_margin,
                                  run_orphan_rate, frame_report)

SEED = 20260727
M_TRUE = np.diag([40.0, 55.0, 70.0])          # orthorhombic P, distinct axes


def _lattice_q(M, rng, n=120, hmax=6):
    """n reciprocal peaks on the lattice of M (random integer nodes in a shell, sub-tol jitter)."""
    Minv = np.linalg.inv(M)
    H = rng.integers(-hmax, hmax + 1, size=(n * 4, 3)).astype(float)
    H = H[np.any(H != 0, axis=1)]
    q = H @ Minv
    qn = np.sqrt((q * q).sum(1))
    q = q[qn <= np.quantile(qn, 0.5)][:n]
    return q + rng.normal(0.0, 0.0008, q.shape)


def _random_q(rng, n, scale):
    """n off-lattice scatter peaks in the same |q| range."""
    v = rng.normal(0.0, scale, (n, 3))
    return v


# --------------------------------------------------------------------------------- finder confidence
def test_finder_spurious_counts():
    assert finder_spurious(np.ones(50))["expected_spurious"] == 0.0
    r = finder_spurious(np.array([1.0, 0.5, 0.0, 0.5]))
    assert abs(r["expected_spurious"] - 2.0) < 1e-9 and abs(r["spurious_frac"] - 0.5) < 1e-9
    assert finder_spurious(np.array([]))["n"] == 0


# --------------------------------------------------------------------------------- unindexed fraction
def test_unindexed_fraction():
    rng = np.random.default_rng(SEED)
    q = _lattice_q(M_TRUE, rng, n=100)
    assert unindexed_fraction(q, M_TRUE)["unindexed_frac"] < 0.02      # all on-lattice
    scale = float(np.sqrt((q * q).sum(1)).mean())
    q2 = np.vstack([q, _random_q(rng, 100, scale)])                   # half scatter
    uf = unindexed_fraction(q2, M_TRUE)["unindexed_frac"]
    assert 0.35 < uf < 0.65, uf


# --------------------------------------------------------------------------------- the ceiling meter
def test_null_margin_clean_frame_clears():
    rng = np.random.default_rng(SEED + 1)
    q = _lattice_q(M_TRUE, rng, n=140)
    r = null_margin(q, M_TRUE, n_null=200, K=5_000_000, rng=rng)
    assert r["s_true"] > r["null_mean"] + 5 * r["null_std"], r        # true fit towers over the null
    assert r["z"] > 8.0 and r["wall"] is False and r["wall_z"] is False, r


def test_null_margin_spurious_frame_walls():
    rng = np.random.default_rng(SEED + 2)
    # a few real peaks buried in mostly scatter: the true orientation barely beats a random one
    q = np.vstack([_lattice_q(M_TRUE, rng, n=8), _random_q(rng, 160, 0.02)])
    r = null_margin(q, M_TRUE, n_null=200, K=5_000_000, rng=rng)
    assert r["wall"] is True, r                                        # does not clear the evt floor
    # and it is far less separated than the clean frame (z ~ single digits, not tens)
    assert r["z"] < 8.0, r


def test_null_margin_empty():
    r = null_margin(np.zeros((0, 3)), M_TRUE)
    assert r["s_true"] == 0 and r["wall_z"] is True


# ------------------------------------------------------------------------------ cross-frame orphans
def test_run_orphan_rate():
    rng = np.random.default_rng(SEED + 3)
    clean = [_lattice_q(M_TRUE, rng, n=80) for _ in range(6)]
    assert run_orphan_rate(clean, M_TRUE)["orphan_frac"] < 0.03
    scale = float(np.sqrt((clean[0] ** 2).sum(1)).mean())
    dirty = [np.vstack([_lattice_q(M_TRUE, rng, n=60), _random_q(rng, 60, scale)]) for _ in range(6)]
    assert run_orphan_rate(dirty, M_TRUE)["orphan_frac"] > 0.35


# ----------------------------------------------------------------------------------- combined report
def test_frame_report_combines():
    rng = np.random.default_rng(SEED + 4)
    q = _lattice_q(M_TRUE, rng, n=100)
    rep = frame_report(q, M=M_TRUE, p_peak=np.full(len(q), 0.9), n_null=100, K=1_000_000, rng=rng)
    assert "finder" in rep and "unindexed" in rep and "null" in rep
    assert rep["null"]["wall"] is False


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    raise SystemExit(1 if fails else 0)
