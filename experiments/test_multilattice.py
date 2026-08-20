"""CPU test for glint.multilattice.deflate_peaks (the double-hit / second-lattice residual)."""
import numpy as np
from glint.multilattice import deflate_peaks


def test_deflate_peaks():
    """Two crystals in one shot: deflating lattice A leaves ~exactly lattice B's peaks."""
    rng = np.random.default_rng(0)
    M = np.diag([1 / 40.0, 1 / 50.0, 1 / 60.0])                    # cell A: hkl = q @ M
    Minv = np.linalg.inv(M)
    hklA = rng.integers(-5, 6, (70, 3)).astype(float); hklA = hklA[np.abs(hklA).sum(1) > 0]
    qA = hklA @ Minv + rng.normal(0, 0.001, (len(hklA), 3))       # A's peaks (+noise within tol)
    th = 0.7                                                       # a DIFFERENT lattice B (rotated + rescaled)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    M2 = R @ np.diag([1 / 44.0, 1 / 53.0, 1 / 66.0])
    hklB = rng.integers(-5, 6, (45, 3)).astype(float); hklB = hklB[np.abs(hklB).sum(1) > 0]
    qB = hklB @ np.linalg.inv(M2)

    resid = deflate_peaks(np.vstack([qA, qB]), M, tol=0.15)
    assert abs(len(resid) - len(qB)) <= 3, (len(resid), len(qB))   # A removed, ~all of B kept
    # none of A survives: every residual peak is non-integer under M
    hf = resid @ M
    assert (np.abs(hf - np.round(hf)).max(1) >= 0.15).all()
    return len(resid), len(qB)


def test_deflate_empty():
    assert deflate_peaks(np.zeros((0, 3)), np.eye(3)).shape == (0, 3)


if __name__ == "__main__":
    ok = 0
    for t in (test_deflate_peaks, test_deflate_empty):
        try:
            t(); ok += 1; print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"{ok}/2 passed")
    raise SystemExit(0 if ok == 2 else 1)
