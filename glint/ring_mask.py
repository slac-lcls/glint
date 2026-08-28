"""Powder-ring focus mask for KNOWN-CELL peak-finding (throughput / blank-veto lever).

A Bragg peak can only lie at |q|=|B hkl| -- on a powder shell -- so with the cell known, a per-PIXEL mask of
"on an allowed shell" (optionally only the LOW-order shells) lets peakfinder_v4 search just the ring annuli.
Restricted to the low-order shells the annuli are a small INNER region of the detector, which is the win: not
a faster indexer (index_known is overhead-bound, flat in peak count) but a cheap SCAN + a blank veto -- an
inner-ring scan that yields too few peaks is a blank, skip the ~17 ms known-cell index. See the
reliability-selection study (memory glint-reliability-selection): known-cell indexing needs ~10 clean
low-order peaks.

Hook: ring_qmask returns a (H,W) bool that is AND-ed into peakfinder_v4's `good` mask (the finder already
gates search to `good`). Caveat: shell density ~ cell x resolution -- the FULL shell set is dense for a large
protein at high resolution (weak filter); only the LOW-order shells (small qlow) are sparse, so pass qlow.
Systematic absences are ignored (the mask is a conservative superset; real peaks are a subset); add centering
conditions from powder_index for a tighter mask.
"""
import numpy as np
from glint.lute_bridge import peaks_to_q


def cell_to_reciprocal(cell6):
    """(a,b,c,alpha,beta,gamma) [A, deg] -> reciprocal basis B (3x3, columns a*,b*,c*); |q(hkl)|=|B @ hkl| (1/d)."""
    a, b, c, al, be, ga = (float(x) for x in cell6)
    al, be, ga = np.radians([al, be, ga])
    bx, by = b * np.cos(ga), b * np.sin(ga)
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = c * np.sqrt(max(1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
                         + 2 * np.cos(al) * np.cos(be) * np.cos(ga), 0.0)) / np.sin(ga)
    A = np.array([[a, bx, cx], [0.0, by, cy], [0.0, 0.0, cz]])    # direct basis, columns a,b,c
    return np.linalg.inv(A).T                                     # reciprocal basis, columns a*,b*,c*


def allowed_shells(cell6, qmax):
    """Sorted unique powder-shell radii |q|=|B hkl| in (0, qmax] [1/A], ignoring systematic absences."""
    B = cell_to_reciprocal(cell6)
    Hb = (np.ceil(qmax * np.asarray([float(x) for x in cell6[:3]])) + 1).astype(int)
    hkl = np.mgrid[-Hb[0]:Hb[0]+1, -Hb[1]:Hb[1]+1, -Hb[2]:Hb[2]+1].reshape(3, -1).T
    hkl = hkl[np.any(hkl != 0, 1)]
    q = np.linalg.norm(hkl @ B.T, axis=1)
    return np.unique(np.round(q[(q > 1e-6) & (q <= qmax)], 4))


def ring_qmask(panels, clen_m, wavelength_A, cell6, shape, qlow=None, tol=0.003, qmax=0.6):
    """Per-pixel Bragg-ring mask (True = pixel on an allowed shell). shape=(H, W) of the detector array.

    Built with the SAME peaks_to_q geometry as the peak bridge, so mask and peaks are in exact registration.
    qlow: keep only shells with |q| <= qlow (low-order rings -- the sparse, spatially-inner, robust regime).
    tol: annulus half-width [1/A], >= the peak-position error. Off-panel pixels (NaN q) are False. Built ONCE
    per run (geometry ~fixed; per-event lambda jitter is absorbed by tol)."""
    H, W = int(shape[0]), int(shape[1])
    ss, fs = np.mgrid[0:H, 0:W]
    qpx = peaks_to_q(fs.ravel().astype(float), ss.ravel().astype(float), panels, clen_m, wavelength_A)
    r = np.linalg.norm(qpx, axis=1)                              # per-pixel |q|; NaN off-panel
    shells = allowed_shells(cell6, qmax)
    if qlow:
        shells = shells[shells <= float(qlow)]
    if len(shells) == 0:
        return np.zeros((H, W), bool)
    fin = np.isfinite(r); d = np.full(r.shape, np.inf)
    idx = np.clip(np.searchsorted(shells, r[fin]), 0, len(shells) - 1)
    lo = np.clip(idx - 1, 0, len(shells) - 1)
    d[fin] = np.minimum(np.abs(r[fin] - shells[idx]), np.abs(r[fin] - shells[lo]))
    mask = d < tol
    if qlow:
        mask &= fin & (r <= float(qlow))
    return mask.reshape(H, W)
