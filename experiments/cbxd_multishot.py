"""Multi-shot CBXD blind CELL by |G| consensus (powder indexing of pooled streak |G|).

Single-pattern blind ORIENTATION is search-landscape-pathological (cbxd_ransac.py), but the
per-streak |G| (= |centroid backprojection|, q = cent - k0) is a robust SCALAR. Pooled over
many random-orientation patterns the |G| histogram peaks at the reflection shells |B hkl|:
the per-shot ~0.015 1/A error is random, not systematic, so the pooled peaks are sharp
(~5e-4 1/A). Powder-index that spectrum to recover the cell -- the CBXD analog of the
mono-(1) consensus cell. (Orthorhombic demo: the three smallest shells are the reciprocal
axes, since every axis |G| is shorter than any 2-index combination.)

  python cbxd_multishot.py
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from cbxd_joint import B, HS, K, simulate, rand_rot

TRUE = np.array([25.0, 21.0, 16.0])                     # sorted-descending true cell (A)
GVEC = (B @ HS.T).T
GMAX = float(np.linalg.norm(GVEC, axis=1).max())


def pooled_G(nshot, noise, rng):
    qs = []
    for _ in range(nshot):
        Rt = rand_rot(rng)
        _, _, cents = simulate(Rt, rng, noise)
        qs.append(np.linalg.norm(cents - np.array([0.0, 0.0, K]), axis=1))
    return np.concatenate(qs)


def shell_peaks(g, min_count=3, smooth=3.0, dq=0.0005):
    bins = int(GMAX / dq)
    hist, edges = np.histogram(g, bins=bins, range=(0.0, GMAX))
    c = 0.5 * (edges[:-1] + edges[1:])
    hs = gaussian_filter1d(hist.astype(float), smooth)
    pk, _ = find_peaks(hs, height=min_count, distance=4)
    return c[pk], hs[pk]


def refine_orthorhombic(shells, cell0, tol=0.004):
    """Assign each shell to the nearest predicted hkl, then LSQ-refine A=1/a^2,B,C."""
    a, b, cc = cell0
    A = np.array([1 / a**2, 1 / b**2, 1 / cc**2])
    rh = np.arange(-6, 7)
    H = np.array(np.meshgrid(rh, rh, rh, indexing="ij")).reshape(3, -1).T
    H = H[np.any(H != 0, 1)]
    for _ in range(8):
        pred2 = H**2 @ A                                # |G|^2 per hkl
        pred = np.sqrt(pred2)
        rows = []
        rhs = []
        for s in shells:
            k = np.argmin(np.abs(pred - s))
            if abs(pred[k] - s) < tol:
                rows.append(H[k]**2)
                rhs.append(s * s)
        if len(rows) < 3:
            break
        A, *_ = np.linalg.lstsq(np.array(rows, float), np.array(rhs), rcond=None)
        A = np.abs(A)
    return 1.0 / np.sqrt(A)


_RH = np.arange(-8, 9)
_H = np.array(np.meshgrid(_RH, _RH, _RH, indexing="ij")).reshape(3, -1).T
_H = _H[np.any(_H != 0, 1)]
_H2 = (_H ** 2).astype(float)


def fom_MN(shells, cell, Nmax=20):
    """de Wolff M_N figure of merit: Q_N / (2 <eps> N_calc), where N_calc = # distinct
    predicted lines below the Nth observed line. Over-predicting (too-small) cells inflate
    N_calc and score low, breaking the powder degeneracy."""
    Qobs = np.sort(np.asarray(shells) ** 2)
    N = min(Nmax, len(Qobs))
    QN = Qobs[N - 1]
    A = np.array([1.0 / cell[0] ** 2, 1.0 / cell[1] ** 2, 1.0 / cell[2] ** 2])
    Q = _H2 @ A
    Q = np.unique(np.round(Q[Q <= QN + 1e-9], 6))
    if Q.size == 0:
        return 0.0, 0
    eps = np.abs(Qobs[:N][:, None] - Q[None, :]).min(1).mean()
    eps = max(eps, 1e-6)
    return QN / (2.0 * eps * Q.size), Q.size


def _grid_best(shells, a_vals):
    best = (-1.0, None)
    for ia in range(len(a_vals)):
        for ib in range(ia, len(a_vals)):              # a <= b
            for ic in range(ib, len(a_vals)):          # b <= c
                cell = (a_vals[ia], a_vals[ib], a_vals[ic])
                f, _ = fom_MN(shells, cell)
                if f > best[0]:
                    best = (f, cell)
    return best


def search_orthorhombic(shells, amin=11.0, amax=33.0):
    f, c0 = _grid_best(shells, np.arange(amin, amax, 0.75))          # coarse
    if c0 is None:
        return -1.0, None
    fine = []
    for x in c0:                                                     # fine, around coarse best
        fine.append(np.arange(x - 1.2, x + 1.2, 0.15))
    a_vals = np.unique(np.round(np.concatenate(fine), 3))
    return _grid_best(shells, a_vals)


def index_powder(shells):
    if len(shells) < 5:
        return None, None, 0.0
    fom, cell0 = search_orthorhombic(shells)
    if cell0 is None:
        return None, None, 0.0
    cell = np.sort(refine_orthorhombic(shells, cell0))[::-1]
    return cell, np.sort(cell0)[::-1], fom


if __name__ == "__main__":
    noise = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-4
    print(f"multi-shot CBXD blind cell via pooled-|G| powder  noise={noise:.0e}")
    print(f"true cell (desc) = {TRUE}   true axes |G*| = {np.sort(1/TRUE)}")
    print(f"{'nshot':>6} {'#shells':>8} {'M20':>7} {'cell0 grid':>22} {'cell refined':>22} {'maxerr%':>8}")
    for nshot in (10, 20, 40, 80, 160):
        rng = np.random.default_rng(5)
        g = pooled_G(nshot, noise, rng)
        sh, _ = shell_peaks(g)
        cell, cell0, fom = index_powder(sh)
        if cell is None:
            print(f"{nshot:6d} {len(sh):8d}   (too few shells)")
            continue
        err = 100 * np.max(np.abs(cell - TRUE) / TRUE)
        print(f"{nshot:6d} {len(sh):8d} {fom:7.1f} "
              f"{str(np.round(cell0,2)):>22} {str(np.round(cell,2)):>22} {err:8.1f}")
