"""Shared case definitions for the GpuCalibrator common-mode check.

Single source of truth for the test (test_gpu_calib_cm.py) and the golden-file generator
(make_common_mode_golden.py): the case grid, the input construction and the psana reference
driver live here and only here, so the two consumers cannot drift.

GOLDEN GEOMETRY: (1, 352, 192), and the width is load-bearing. With nbanks=8 a rows-mode group
is one row of one bank = ncol/8 pixels, and both psana and the GPU port apply a correction only
when a group has MORE THAN npix_min=10 good pixels. At the previous width (64 -> 8-pixel groups)
the rows correction could never fire, so mode&1 was compared against identically-zero goldens
and CI could not catch a broken rows block. At 192 (24-pixel groups) rows corrections are real
for good-fractions >= 0.5, and both the test and the generator assert per-mode-bit nonzero
corrections so the gap cannot silently return.
"""
import numpy as np

GOLDEN_NSEG, GOLDEN_NCOL = 1, 192

CASES = [(mode, t, cormax, frac)
         for mode in (2, 1, 4, 3, 7)
         for t, (cormax, frac) in enumerate([(10.0, 0.95), (10.0, 0.5), (100.0, 0.99),
                                             (10.0, 0.03), (1e9, 0.8)])]


def make_inputs(nseg, ncol):
    """Regenerate every case's arrays from the seed, in the order the golden file is built.

    Legacy RandomState, not default_rng: numpy guarantees stream compatibility across releases
    for the former and explicitly does not for Generator distributions, and the goldens are
    only valid for the exact arrays that produced them."""
    rng = np.random.RandomState(0)
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


def assert_modes_fire(corrs, tol=1e-6):
    """corrs: {key: correction array}. Fail loudly if any mode bit has NO case with a real,
    nonzero correction -- the exact silent regression the 64-column goldens shipped with."""
    fails = []
    for bit in (1, 2, 4):
        peak = max(float(np.abs(a).max())
                   for k, a in corrs.items()
                   if k.startswith("corr_") and k != "corr_boundary" and int(k.split("_")[1]) & bit)
        ok = peak > tol
        print(f"[meta ] mode bit {bit}: max |correction| across its cases = {peak:.3f} "
              f"{'OK' if ok else 'NEVER FIRES -- goldens check nothing for this mode'}")
        if not ok:
            fails.append(bit)
    return fails


def psana_reference(arrf, gmask, mode, cormax, npixmin):
    """calib_epix10ka_any's common-mode block: per segment, banks then rows then cols.
    Imports psana lazily so this module stays importable where psana is absent."""
    from Detector.UtilsCommonMode import (
        common_mode_rows_hsplit_nbanks, common_mode_cols, common_mode_2d_hsplit_nbanks,
    )
    a = arrf.copy(); hrows = 176
    for s in range(a.shape[0]):
        if mode & 4:
            common_mode_2d_hsplit_nbanks(a[s, :hrows, :], mask=gmask[s, :hrows, :], nbanks=8,
                                         cormax=cormax, npix_min=npixmin)
            common_mode_2d_hsplit_nbanks(a[s, hrows:, :], mask=gmask[s, hrows:, :], nbanks=8,
                                         cormax=cormax, npix_min=npixmin)
        if mode & 1:
            common_mode_rows_hsplit_nbanks(a[s, ], mask=gmask[s, ], nbanks=8,
                                           cormax=cormax, npix_min=npixmin)
        if mode & 2:
            common_mode_cols(a[s, :hrows, :], mask=gmask[s, :hrows, :],
                             cormax=cormax, npix_min=npixmin)
            common_mode_cols(a[s, hrows:, :], mask=gmask[s, hrows:, :],
                             cormax=cormax, npix_min=npixmin)
    return a
