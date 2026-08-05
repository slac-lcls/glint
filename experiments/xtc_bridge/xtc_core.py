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


class _PanelFinders:
    """One finder per panel -- the arrangement PeakFinderV4 has always used here.

    Correct by construction across seams (each panel is labelled on its own), and the only sane
    choice for a LATENCY-bound consumer: panels can be found independently as they arrive."""

    def __init__(self, finders):
        self.finders = finders

    def find_all(self, frame):
        """-> (seg, ss, fs) integer arrays of peak positions in panel-local coordinates."""
        import cupy as cp
        segs, ys, xs = [], [], []
        for p, f in enumerate(self.finders):
            pk = f.find(cp.asarray(frame[p], cp.float32))
            x = cp.asnumpy(pk["x"]); y = cp.asnumpy(pk["y"])
            if x.size:
                segs.append(np.full(x.size, p)); xs.append(x); ys.append(y)
        if not xs:
            return None
        return np.concatenate(segs), np.concatenate(ys), np.concatenate(xs)


class _StackedFinder:
    """ONE peakfinder8 over the whole detector, with the panel seams masked out.

    This is the arrangement peakfinder8 actually wants, and the reason it exists: its background is
    estimated in RADIAL SHELLS, so a shell must see every pixel at that |q| across the WHOLE
    detector. Per-panel, a 16-panel detector estimates each shell from ~1/16 the pixels -- which
    throws away the one property that distinguishes pf8 from v4's local annulus.

    Stacking is already correct for the shells (the radial operator bins by q VALUE, not by spatial
    position). The only hazard is `ndimage.label`, which would happily connect a component across the
    boundary between two panels that are not physically adjacent. So the panels are laid out with a
    ONE-ROW GAP between them and that gap is masked bad:

        rows [p*pitch, p*pitch+H)  = panel p          pitch = H + 1
        row   p*pitch+H            = separator, good=False

    peakfinder8 zeroes snr wherever `good` is false (`snr = where(self.good, ..., 0.0)`) and takes
    `cand = snr > thr`, so a masked row can never join a component -- it is an impassable barrier to
    labelling. A synthetic gap row is used rather than masking a real detector row so that no actual
    pixels are sacrificed; the cost is nseg extra rows of buffer."""

    def __init__(self, finder, nseg, H, pitch):
        self.finder = finder
        self.nseg, self.H, self.pitch = nseg, H, pitch
        self._buf = None

    def find_all(self, frame):
        import cupy as cp
        if self._buf is None:                       # preallocate once; refilled per frame
            self._buf = cp.zeros((self.nseg * self.pitch, frame.shape[-1]), cp.float32)
        for p in range(self.nseg):                  # separator rows stay 0 and are masked anyway
            self._buf[p * self.pitch:p * self.pitch + self.H] = cp.asarray(frame[p], cp.float32)
        pk = self.finder.find(self._buf)
        x = cp.asnumpy(pk["x"]); y = cp.asnumpy(pk["y"])
        if not x.size:
            return None
        seg = (y // self.pitch).astype(int)
        ss = y - seg * self.pitch                   # < H: separators are masked, so never a peak
        return seg, ss, x


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
    gmask_v4 = good
    if peakfinder in ("pf8", "pf8-panel"):
        # peakfinder8: RADIAL-shell background, the closer match to what CrystFEL/PeakFinderSFX run.
        # Needs a per-pixel q map, which is why this was unavailable until radial.py was vendored.
        if lam is None:
            raise ValueError("peakfinder=%r needs the wavelength (per-pixel q is q(lambda)); pass "
                             "lam=, or use --wavelength so the reader has one before geometry setup"
                             % peakfinder)
        PeakFinder8 = load_peakfinder8().PeakFinder8
        # THRESHOLD MAPPING -- the two finders do not spend the same numbers the same way, and the
        # similar parameter NAMES are a trap. v4 labels components on `grow = snr > thr_low`, keeps
        # those containing a `seed = snr > thr_high` LOCAL MAX, then cuts on son_min, the INTEGRATED
        # SNR sum(I-bg)/sqrt(sum sigma^2). pf8 labels on `cand = snr > thr_snr` and cuts min_snr
        # against the per-peak MAX PIXEL SNR; it has no integrated-SNR cut at all. So:
        #     thr_snr <- thr_low     which pixels form a component
        #     min_snr <- thr_high    the component must contain one bright pixel (v4's seed)
        #     son_min -> NO analogue; do not route it into min_snr
        # Passing thr_snr=thr_high with min_snr=son_min, as this did until 2026-08-05, stiffens both
        # cuts at once (extent 10 vs 5, brightness 15 vs 10) and min_pix compounds it, since pf8 then
        # counts only pixels above 10 where v4 counts above 5. Measured on mfxx49820 r0016 over 6000
        # events that cost pf8 22 peaks/frame against v4's 35, and 95.8% of btx's indexed frames
        # against v4's 99.9% -- a mis-mapping that reads exactly like a finder deficit.
        qmap = pixel_q(X, Y, Zc, kin, float(lam))               # (nseg,H,W) in 1/A
        gmask = good if good is not None else np.ones(shape, bool)
        if peakfinder == "pf8-panel":
            # STREAMING choice: independent per-panel finders, so panels can be processed as they
            # arrive and nothing waits for a whole detector. Accepts ~1/nseg the statistics per
            # radial shell -- for a latency-bound consumer that is the right trade.
            finders = _PanelFinders([
                PeakFinder8(cp.asarray(qmap[p]), mask=cp.asarray(gmask[p]), dtype=cp.float32,
                            min_pix=min_pix, min_snr=thr_high, thr_snr=thr_low)
                for p in range(nseg)])
        else:
            # OFFLINE default: ONE finder over the whole detector, seams masked. See _StackedFinder.
            pitch = H + 1
            qs = np.zeros((nseg * pitch, W)); gs = np.zeros((nseg * pitch, W), bool)
            for p in range(nseg):
                qs[p * pitch:p * pitch + H] = qmap[p]
                gs[p * pitch:p * pitch + H] = gmask[p]           # separator row left good=False
            # NaN q (panel-gap pixels) would poison the radial binning; mask them and give them a
            # finite q so RadialIntegrator's bin edges are not NaN-driven.
            bad = ~np.isfinite(qs)
            gs &= ~bad; qs[bad] = 0.0
            finders = _StackedFinder(
                PeakFinder8(cp.asarray(qs), mask=cp.asarray(gs), dtype=cp.float32,
                            min_pix=min_pix, min_snr=thr_high, thr_snr=thr_low),
                nseg, H, pitch)
    else:
        finders = _PanelFinders([
            PeakFinderV4(cp.asarray(gmask_v4[p]), dtype=cp.float32, min_pix=min_pix, son_min=son_min,
                         thr_high=thr_high, thr_low=thr_low) for p in range(nseg)])
    return X, Y, Zc, kin, finders


def frame_q(frame, finders, X, Y, Zc, kin, lam, min_peaks):
    """One calibrated (nseg,H,W) frame -> its reciprocal q-vectors (M,3) in 1/A, or an empty (0,3).

    `finders` is a _PanelFinders or a _StackedFinder; both expose find_all(frame) -> (seg, ss, fs) in
    panel-local coordinates, so the geometry lookup below is identical either way. Peak centroids are
    rounded to the nearest pixel for that lookup; NaN rows (panel-gap pixels have NaN coords) are
    dropped. Returns shape (0,3) if under min_peaks."""
    _, H, W = X.shape
    found = finders.find_all(frame)
    if found is None:
        return np.empty((0, 3))
    seg, ss_f, fs_f = found
    fs = np.rint(fs_f).astype(int)
    ss = np.rint(ss_f).astype(int)
    np.clip(fs, 0, W - 1, out=fs); np.clip(ss, 0, H - 1, out=ss)
    r = np.stack([X[seg, ss, fs], Y[seg, ss, fs], np.full(seg.size, Zc)], axis=1)
    s = r / np.linalg.norm(r, axis=1, keepdims=True)
    q = (s - kin) / lam
    q = q[np.isfinite(q).all(axis=1)]
    return q if len(q) >= min_peaks else np.empty((0, 3))
