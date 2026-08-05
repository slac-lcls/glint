"""Shared peak-find + q core for the xtc readers -- identical for psana1 (in-process) and psana2
(over the bridge). Only the psana data-access layer differs between the two adapters; everything about
turning a calibrated frame + per-pixel lab coords into reciprocal q lives here, once.

numpy + cupy only (both LCLS conda stacks have them); no psana, no torch, no glint package import.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HC_EV_A = 12398.419843320026

# Peak-finder thresholds. PeakFinderV4's own library defaults (min_pix=1, min_sig=0.0) accept a SINGLE
# pixel clearing SNR 8 with no noise floor -- on a real multi-panel detector that fires on blank frames,
# so every frame looks like a hit and the indexer is handed noise. Measured on mfxx49820 r0016
# (Epix10ka2M, 5632x384): the library defaults pass 100% of frames at min_peaks=6, where CrystFEL's
# peakfinder8 calls 28.3% hits. The values below reproduce peakfinder8 on that data -- same hit count
# (85/85 of 300 events), 94.1% frame-level overlap, and ALL 55 of the frames peakfinder8 went on to
# index. They are a sane starting point, NOT a universal truth: they were calibrated on one detector,
# so tune per detector via glint_xtc.py's --min-pix / --son-min / --thr-high / --thr-low.
PF_MIN_PIX = 3
PF_SON_MIN = 15.0
PF_THR_HIGH = 10.0
PF_THR_LOW = 5.0


def event_in_shard(i, rank, nranks):
    """Round-robin event ownership for MPI sharding: rank r owns global event i iff i % nranks == r.
    Round-robin (not contiguous blocks) load-balances when indexable frames cluster in time, and needs
    no up-front event count -- each rank walks the smd stream and does the expensive calib + peak-find
    only on its own events. The global index i is preserved into the .stream so per-rank chunks from
    different ranks never collide, and the partial streams merge by a plain chunk concatenation."""
    return nranks <= 1 or (i % nranks) == rank


def _load_by_path(name, filename):
    """Load glint/<filename> directly, bypassing the glint package __init__ (which would drag GLINT's
    numpy-1 modules into a numpy-2 worker). Self-contained: numpy/cupy, no torch, no glint."""
    import sys
    pf = Path(__file__).resolve().parent.parent.parent / "glint" / filename
    spec = importlib.util.spec_from_file_location(name, pf)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                  # so a sibling's flat `import name` resolves to this one
    spec.loader.exec_module(mod)
    return mod


def load_peakfinder_v4():
    return _load_by_path("_pf_v4_standalone", "peakfinder_v4.py")


def load_peakfinder8():
    """PeakFinder8 + its RadialIntegrator, loaded the same way.

    `radial` MUST be loaded and registered first: peakfinder8 falls back to a FLAT `import radial`
    when it is not imported as part of the package, which is exactly the path taken here."""
    _load_by_path("radial", "radial.py")
    return _load_by_path("_pf8_standalone", "peakfinder8.py")


def pixel_q(X, Y, Zc, kin, lam):
    """Per-pixel |q| (1/A) for the same geometry frame_q uses -- what PeakFinder8 bins on.

    PeakFinderV4 needs no geometry: its background is a local annulus in PIXEL space. peakfinder8
    estimates the background in RADIAL shells, so it needs a q (or r) value per pixel, and that is
    the whole reason `--peakfinder pf8` was unavailable until now -- glint had the finder but not a
    q map to hand it."""
    r = np.stack([X, Y, np.full(X.shape, Zc)], axis=-1)
    s = r / np.linalg.norm(r, axis=-1, keepdims=True)
    return np.linalg.norm((s - kin) / lam, axis=-1)


def prep_geometry(Xf, Yf, Zf, shape, good, zdist, *, min_pix=PF_MIN_PIX, son_min=PF_SON_MIN,
                  thr_high=PF_THR_HIGH, thr_low=PF_THR_LOW, peakfinder="v4", lam=None):
    """From per-pixel lab coords (any layout of total size nseg*H*W) build what frame_q needs:
        X, Y : (nseg,H,W) transverse positions in METRES (psana coords are um)
        Zc   : the sample-detector distance, sign taken from psana's nominal Z, magnitude from zdist
               -- psana per-pixel Z is nominal, so it is REPLACED, not trusted
        kin  : incident-beam unit vector [0,0,sign(Zc)] (PSANA frame; see the readers' caveats)
        finders : one PeakFinderV4 per panel (per-panel bad-pixel masks; a single shared mask would be
                  wrong for a multi-panel detector, and zeroing bad pixels biases the ring background)

    The peak-finder thresholds are passed EXPLICITLY rather than left at PeakFinderV4's library
    defaults -- see the PF_* constants above for why that distinction decides whether blank frames
    are counted as hits.
    """
    import cupy as cp
    PeakFinderV4 = load_peakfinder_v4().PeakFinderV4
    nseg, H, W = shape
    Xf = np.asarray(Xf, float); Yf = np.asarray(Yf, float); Zf = np.asarray(Zf, float)
    if Xf.size != nseg * H * W:
        raise ValueError(f"geometry/data size mismatch: coords {Xf.size} != {nseg}*{H}*{W} "
                         f"(the .geom/detector segment layout does not match the calib array)")
    X = Xf.reshape(nseg, H, W) * 1e-6
    Y = Yf.reshape(nseg, H, W) * 1e-6
    Zc = np.sign(np.nanmean(Zf)) * zdist
    kin = np.array([0.0, 0.0, np.sign(Zc)])
    good = np.asarray(good, bool).reshape(nseg, H, W) if good is not None else np.ones((nseg, H, W), bool)
    if peakfinder == "pf8":
        # PER-PANEL, matching the v4 arrangement above, because ndimage.label works on a 2-D array:
        # stacking the panels would let a peak straddle a panel seam and merge two crystals' spots.
        # The COST of that choice is that each panel's radial shells are estimated from its own pixels
        # only -- on a 16-panel detector that is ~1/16 the statistics per shell of a whole-detector
        # peakfinder8. It is the honest per-panel analogue, not a bit-match to CrystFEL's. Measure
        # before quoting a rate from it.
        if lam is None:
            raise ValueError("peakfinder='pf8' needs the wavelength (per-pixel q is q(lambda)); pass "
                             "lam=, or use --wavelength so the reader has one before geometry setup")
        PeakFinder8 = load_peakfinder8().PeakFinder8
        qmap = pixel_q(X, Y, Zc, kin, float(lam))               # (nseg,H,W) in 1/A
        finders = [PeakFinder8(cp.asarray(qmap[p]), mask=cp.asarray(good[p]), dtype=cp.float32,
                               min_pix=min_pix, min_snr=son_min, thr_snr=thr_high)
                   for p in range(nseg)]
    else:
        finders = [PeakFinderV4(cp.asarray(good[p]), dtype=cp.float32, min_pix=min_pix, son_min=son_min,
                                thr_high=thr_high, thr_low=thr_low) for p in range(nseg)]
    return X, Y, Zc, kin, finders


def frame_q(frame, finders, X, Y, Zc, kin, lam, min_peaks):
    """One calibrated (nseg,H,W) frame -> its reciprocal q-vectors (M,3) in 1/A, or an empty (0,3).
    Per-panel GPU peak-find; peak centroid rounded to the nearest pixel for the coord lookup; NaN
    rows (panel-gap pixels have NaN coords) dropped. Returns [] shape (0,3) if under min_peaks."""
    import cupy as cp
    nseg = len(finders)
    _, H, W = X.shape
    segs, ys, xs = [], [], []
    for p in range(nseg):
        pk = finders[p].find(cp.asarray(frame[p], cp.float32))
        x = cp.asnumpy(pk["x"]); y = cp.asnumpy(pk["y"])
        if x.size:
            segs.append(np.full(x.size, p)); xs.append(x); ys.append(y)
    if not xs:
        return np.empty((0, 3))
    seg = np.concatenate(segs)
    fs = np.rint(np.concatenate(xs)).astype(int)
    ss = np.rint(np.concatenate(ys)).astype(int)
    np.clip(fs, 0, W - 1, out=fs); np.clip(ss, 0, H - 1, out=ss)
    r = np.stack([X[seg, ss, fs], Y[seg, ss, fs], np.full(seg.size, Zc)], axis=1)
    s = r / np.linalg.norm(r, axis=1, keepdims=True)
    q = (s - kin) / lam
    q = q[np.isfinite(q).all(axis=1)]
    return q if len(q) >= min_peaks else np.empty((0, 3))
