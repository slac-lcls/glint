"""Regenerate common_mode_golden.npz from REAL psana. Run where psana is importable.

CASES and make_inputs are lifted verbatim from test_gpu_calib_cm.py so the generator cannot drift
from the test; the test's own staleness check re-verifies this whenever psana is present.
"""
import os
import numpy as np
from Detector.UtilsCommonMode import (
    common_mode_rows, common_mode_cols, common_mode_2d,
    common_mode_rows_hsplit_nbanks, common_mode_2d_hsplit_nbanks,
)


def psana_reference(arrf, gmask, mode, cormax, npixmin):
    a = arrf.copy(); hrows = 176
    for s in range(a.shape[0]):
        if mode & 4:
            common_mode_2d_hsplit_nbanks(a[s, :hrows, :], mask=gmask[s, :hrows, :], nbanks=8, cormax=cormax, npix_min=npixmin)
            common_mode_2d_hsplit_nbanks(a[s, hrows:, :], mask=gmask[s, hrows:, :], nbanks=8, cormax=cormax, npix_min=npixmin)
        if mode & 1:
            common_mode_rows_hsplit_nbanks(a[s, ], mask=gmask[s, ], nbanks=8, cormax=cormax, npix_min=npixmin)
        if mode & 2:
            common_mode_cols(a[s, :hrows, :], mask=gmask[s, :hrows, :], cormax=cormax, npix_min=npixmin)
            common_mode_cols(a[s, hrows:, :], mask=gmask[s, hrows:, :], cormax=cormax, npix_min=npixmin)
    return a


CASES = [(mode, t, cormax, frac)
         for mode in (2, 1, 4, 3, 7)
         for t, (cormax, frac) in enumerate([(10.0, 0.95), (10.0, 0.5), (100.0, 0.99),
                                             (10.0, 0.03), (1e9, 0.8)])]


def make_inputs(nseg, ncol):
    """Regenerate every case's arrays from the seed, in the order the golden file was built."""
    rng = np.random.RandomState(0)          # stream-stable across numpy releases
    out = []
    for mode, t, cormax, frac in CASES:
        arrf = (rng.normal(0, 6, (nseg, 352, ncol))).astype(np.float32)
        arrf[:, :, ::37] += 25.0                     # big offsets so the cormax veto is exercised
        gmask = (rng.random_sample((nseg, 352, ncol)) < frac).astype(np.uint8)
        out.append((f"corr_{mode}_{t}", arrf, gmask, mode, cormax))
    arrf = rng.normal(0, 3, (1, 352, ncol)).astype(np.float32)
    gmask = np.zeros((1, 352, ncol), np.uint8)
    gmask[0, :10, 0] = 1        # exactly 10 good -> npix > 10 is False -> no correction
    gmask[0, :11, 1] = 1        # exactly 11 good -> corrected
    out.append(("corr_boundary", arrf, gmask, 2, 10.0))
    return out



out = {}
for key, arrf, gmask, mode, cormax in make_inputs(1, 64):
    out[key] = (psana_reference(arrf, gmask, mode, cormax, 10) - arrf).astype(np.float32)
# beside this script, so regeneration updates the file the test actually loads no matter
# which directory it is invoked from
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "common_mode_golden.npz")
np.savez_compressed(OUT, psana_release="ana-4.0.59-py3-minipytorch", **out)
print("wrote", OUT, "cases:", len(out), " bytes:", os.path.getsize(OUT))
