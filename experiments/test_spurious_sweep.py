"""CPU tests for the spurious-ceiling sweep (glint.spurious_sweep) -- pure numpy.

`pytest experiments/test_spurious_sweep.py` or `python experiments/test_spurious_sweep.py`.
"""
import numpy as np

from glint.spurious_sweep import sweep, classify, format_sweep

SEED = 20260727
M_TRUE = np.diag([40.0, 55.0, 70.0])


def _latq(rng, n=120, hmax=6):
    Minv = np.linalg.inv(M_TRUE)
    H = rng.integers(-hmax, hmax + 1, (n * 4, 3)).astype(float)
    H = H[np.any(H != 0, 1)]
    q = H @ Minv
    qn = np.sqrt((q * q).sum(1))
    q = q[qn <= np.quantile(qn, 0.5)][:n]
    return q + rng.normal(0.0, 8e-4, q.shape)


def _scat(rng, n, s):
    return rng.normal(0.0, s, (n, 3))


def test_sweep_three_way_split():
    """Robust invariants: the clear end and blank end classify cleanly, buried frames never read as
    'clear', and wall_rate is the wall fraction over non-blank frames. (That the wall class is reachable
    at all is proven in test_spurious_meter::test_null_margin_spurious_frame_walls -- the middle band is a
    genuine knife-edge, so we don't assert an exact wall count here.)"""
    rng = np.random.default_rng(SEED)
    scale = float(np.sqrt((_latq(rng) ** 2).sum(1)).mean())
    frames = []
    frames += [_latq(rng, 140) for _ in range(5)]                        # clear (strong real signal)
    frames += [np.vstack([_latq(rng, 6), _scat(rng, 180, 0.02)]) for _ in range(4)]  # buried (wall/blank)
    frames += [_scat(rng, 3, scale) for _ in range(3)]                   # blank (too few peaks)
    res = sweep(frames, M_TRUE, K=5_000_000, n_null=128, seed=1)
    c = res["counts"]
    assert res["n_frames"] == 12
    assert c["clear"] == 5, c                                            # strong frames all index
    assert c["blank"] >= 3, c                                           # the 3 tiny frames are blank
    assert c["clear"] + c["wall"] + c["blank"] == 12
    # none of the buried/blank frames masquerade as indexable
    assert all(r["cls"] != "clear" for r in res["per_frame"][5:]), [r["cls"] for r in res["per_frame"][5:]]
    indexable = c["wall"] + c["clear"]
    assert abs(res["wall_rate"] - (c["wall"] / indexable if indexable else 0.0)) < 1e-9


def test_classify_boundaries():
    assert classify(dict(n=3, s_true=0, null_mean=0.0, null_std=1.0, wall=None, wall_z=True), 6) == "blank"
    assert classify(dict(n=50, s_true=100, null_mean=4.0, null_std=2.0, wall=False, wall_z=False), 6) == "clear"
    assert classify(dict(n=50, s_true=10, null_mean=4.0, null_std=2.0, wall=True, wall_z=True), 6) == "wall"


def test_format_runs():
    rng = np.random.default_rng(SEED + 2)
    res = sweep([_latq(rng, 120) for _ in range(3)], M_TRUE, n_null=64, seed=2)
    s = format_sweep(res)
    assert "frames: 3" in s and "wall-rate" in s


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
