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


def event_in_shard(i, rank, nranks):
    """Round-robin event ownership for MPI sharding: rank r owns global event i iff i % nranks == r.
    Round-robin (not contiguous blocks) load-balances when indexable frames cluster in time, and needs
    no up-front event count -- each rank walks the smd stream and does the expensive calib + peak-find
    only on its own events. The global index i is preserved into the .stream so per-rank chunks from
    different ranks never collide, and the partial streams merge by a plain chunk concatenation."""
    return nranks <= 1 or (i % nranks) == rank


def load_peakfinder_v4():
    """Import glint/peakfinder_v4.py directly, bypassing the glint package __init__ (which would drag
    GLINT's numpy-1 modules into a numpy-2 worker). Self-contained: numpy/cupy, no torch, no glint."""
    pf = Path(__file__).resolve().parent.parent.parent / "glint" / "peakfinder_v4.py"
    spec = importlib.util.spec_from_file_location("_pf_v4_standalone", pf)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prep_geometry(Xf, Yf, Zf, shape, good, zdist):
    """From per-pixel lab coords (any layout of total size nseg*H*W) build what frame_q needs:
        X, Y : (nseg,H,W) transverse positions in METRES (psana coords are um)
        Zc   : the sample-detector distance, sign taken from psana's nominal Z, magnitude from zdist
               -- psana per-pixel Z is nominal, so it is REPLACED, not trusted
        kin  : incident-beam unit vector [0,0,sign(Zc)] (PSANA frame; see the readers' caveats)
        finders : one PeakFinderV4 per panel (per-panel bad-pixel masks; a single shared mask would be
                  wrong for a multi-panel detector, and zeroing bad pixels biases the ring background)
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
    finders = [PeakFinderV4(cp.asarray(good[p]), dtype=cp.float32) for p in range(nseg)]
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
