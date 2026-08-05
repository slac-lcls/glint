"""det.calib on the GPU, from raw uint16 -- STATUS.md item 4.

`det.calib()` is 97.7% of a LUTE xtc event: 145.8 ms of 149.2 ms measured on mfxx49820 r0016
(Epix10ka2M, 2.16 Mpix), against 0.94 ms to advance the event stream and 0.86 ms for `det.raw`. It is
numpy on one core, and it is the whole cost of the offline route.

This module is that stage on the device. It is not a reimplementation from the documentation: every
step below was read out of psana's own `Detector/UtilsEpix10ka.py::calib_epix10ka_any` and
`Detector/UtilsCommonMode.py` in ana-4.0.58-py3, and the constants come from psana at setup, so the
arrays are the same ones `det.calib` uses. psana's pipeline, in order:

    arrf  = (raw & M14) - peds[gain_mode]        # M14 = 0x3fff, pedestal per gain mode
    arrf  = common_mode(arrf, gmask)             # in-place, banks then rows then cols
    out   = arrf * gfac[gain_mode] * mask        # gfac = 1/gain, mask = det.mask_total

THREE THINGS ARE EASY TO GET WRONG HERE, and getting them wrong is expensive rather than obvious.
Each cost a measured run to find:

1. GAIN IS A DIVISOR, NOT A MULTIPLIER. `det.gain()` for epix10ka holds gain in ADU/keV, and psana
   builds `store.gfac = divide_protected(np.ones_like(gain), gain)` -- one over it -- then multiplies
   by that. This is the opposite of CSPAD and epix100a, where `pixel_gain` holds a gain FACTOR in
   keV/ADU that is multiplied directly, and it is the opposite of what `Reader.cu` does. Multiplying
   by `det.gain()` here is wrong by gain squared, which is a factor of tens, not percent.

2. THE GAIN MODE IS NOT THE RAW HIGH BITS. See `gain_mode_planes`.

3. COMMON MODE IS PART OF det.calib, and it is on by default for this detector. See `_common_mode`.

STATE, 2026-08-05. Three earlier runs of validate_gpu_calib.py on mfxx49820 r0016, 40 events,
differing only in how the gain mode was resolved, and all three still MULTIPLYING by gain:

                          guessed (13,3)   DAQ (14,2)    psana decode
                          job 34275323     job 34275620  job 34275959
    det.calib (CPU)        142.84 ms        142.33 ms     144.12 ms
    this module              1.40 ms          1.37 ms       1.11 ms   (130x)
    per-pixel residual      4050 ADU         4035 ADU      1431 ADU
    peak-set Jaccard          35.7%            55.5%         58.6%

Fixing the decode cut the residual 2.8x and moved the Jaccard 3 points. The 1431 ADU that survived
was read at the time as the common-mode term. IT WAS NOT, and the arithmetic says so: psana caps
each common-mode offset at `cormax` ADU (10 by default for this detector) and applies it to a subset
of pixels, so common mode CANNOT move a frame by 1431 ADU RMS under any parameters. That residual is
the inverted gain of item 1 above. Common mode is a real but second-order term on top of it.

So the honest reading of the table is: it measures the speed of the GPU path and nothing about its
fidelity, because all three rows share a bug that dominates the fidelity columns. The numbers to
trust are the ones from the next run, with all three items above implemented. The arbiter is
validate_gpu_calib.py and the number to read is the peak-set Jaccard, not the residual: a bias that
shifts every pixel equally changes no peak, and it is the hit set that the rest of the pipeline sees.

DO NOT WIRE THIS INTO THE READER UNTIL THAT JACCARD IS MEASURED AND HIGH. 130x on 98% of the event
while losing a third of the real peaks is not a speedup anyone can use.
"""
from __future__ import annotations

import numpy as np

# psana's order in UtilsEpix10ka.GAIN_MODES.
#   0 FH   1 FM   2 FL   3 AHL-H   4 AML-M   5 AHL-L   6 AML-L
# psana common-mode-corrects H and M pixels only -- `grhm` in calib_epix10ka_any selects gr0, gr1,
# gr3, gr4 and nothing else. The three low-gain modes are left alone: a low-gain pixel is there
# because it saw a lot of signal, so its neighbourhood is a bad place to measure an offset.
HM_MODES = (0, 1, 3, 4)


def gain_mode_planes(det, run):
    """Per-pixel (pedestal, 1/gain, is-H-or-M) planes for data bit 14 CLEAR and SET.

    Epix10ka's gain mode is NOT the raw high bits alone. psana's `gain_maps_epix10ka_any` builds a
    per-pixel control word from the DETECTOR CONFIGURATION (test/mask/gain/ga bits and trbit, all
    run-constant) and then ORs in ONE bit from the data -- bit 14, shifted down to bit 5 -- before
    classifying into the seven modes. SLAC's own note on the readout mapping says the same thing from
    the hardware side: raw bit 14 separates low-gain from non-low, and FH vs FM is decided by trbit,
    which lives in the ASIC register and not in the data at all.

    That is why a 2-bit raw field cannot address 7 planes, and why the earlier `raw range k -> psana
    plane k` clamp lost a third of the peaks: for an auto-gain pixel the mode depends on the
    configuration, not on the raw value.

    Since the config half is constant for a run and the data half is a single bit, the whole decode
    collapses to TWO precomputed plane sets. This calls psana's function twice -- once with synthetic
    all-zero data, once with every pixel's bit 14 set -- so the classification is psana's rather than
    a reimplementation of it, and folds pedestal, gain factor and the H/M common-mode selection into
    those two cases. Per event the GPU then does one select per plane and no gather.
    """
    from Detector.UtilsEpix10ka import gain_maps_epix10ka_any, B14
    from Detector.GlobalUtils import divide_protected

    peds = det.pedestals(run)
    peds = None if peds is None else np.asarray(peds, dtype=np.float32)
    if peds is None or peds.ndim != 4:
        raise NotImplementedError(
            "gain-mode decode is implemented for the Epix10ka family (4-D pedestals, "
            f"got shape {None if peds is None else peds.shape})")
    nmodes = peds.shape[0]
    gain = det.gain(run)
    if gain is None:
        raise NotImplementedError("det.gain() is None -- det.calib would bail out here too")
    gain = np.asarray(gain, dtype=np.float32)
    if gain.shape != peds.shape:
        gain = np.broadcast_to(gain.reshape((1,) + gain.shape[-3:]), peds.shape)
    # THE DIVISION. psana: store.gfac = divide_protected(np.ones_like(gain), gain), then arrf*gfac.
    # divide_protected rather than 1/gain so a zero-gain pixel yields 0 instead of inf, as in psana.
    gfac = divide_protected(np.ones_like(gain), gain)

    shape = peds.shape[1:]
    zero = np.zeros(shape, dtype=np.uint16)
    ones = np.full(shape, B14, dtype=np.uint16)

    out = []
    for data in (zero, ones):                      # bit 14 clear, then set
        maps = gain_maps_epix10ka_any(det, data)
        if maps is None:
            raise NotImplementedError("gain_maps_epix10ka_any returned None -- no config for this "
                                      "detector; the GPU path cannot mirror det.calib without it")
        idx = np.zeros(shape, dtype=np.int8)
        covered = np.zeros(shape, dtype=bool)
        for k, m in enumerate(maps[:nmodes]):
            idx[m] = k
            covered |= m
        if not covered.all():                      # a pixel in no mode would silently take plane 0
            raise ValueError(f"{(~covered).sum()} pixels fall in no gain mode -- refusing to "
                             "calibrate them as mode 0")
        take = idx[None].astype(np.intp)
        hm = np.isin(idx, HM_MODES)                # psana's grhm, before the AND with det.mask_total
        out.append((np.take_along_axis(peds, take, 0)[0],
                    np.take_along_axis(gfac, take, 0)[0],
                    hm))
    return out[0], out[1], shape


class GpuCalibrator:
    """det.calib for epix10ka on the device: pedestal, common mode, gain, mask.

    Setup resolves the gain mode with psana's own decode (see `gain_mode_planes`) into two per-pixel
    plane sets -- bit 14 clear and bit 14 set -- and uploads them ONCE, together with
    `det.mask_total` and the run's common-mode parameters from `det.common_mode`. Six float planes
    plus two masks for an Epix10ka2M is about 60 MB; re-deriving them per event would cost more than
    the calibration saves.

    Pass `cmpars` to override the run's common-mode parameters, exactly as `det.calib(evt,
    cmpars=...)` does; `cmpars=(7, 0, 0, 0)` turns common mode off, which is the configuration the
    three runs in the module docstring measured.
    """

    DATA_MASK = np.uint16((1 << 14) - 1)      # psana's M14 = 0x3fff
    B14 = np.uint16(1 << 14)                  # psana's B14; the one per-event gain bit

    def __init__(self, det, run, cmpars=None, dtype=None):
        import cupy as cp
        self.cp = cp
        self.dt = dtype or cp.float32
        (p0, f0, h0), (p1, f1, h1), shape = gain_mode_planes(det, run)
        self.shape = shape

        cmp_ = det.common_mode(run) if cmpars is None else cmpars
        # psana: mode, cormax = int(cmp[1]), cmp[2]; npixmin = cmp[3] if len(cmp)>3 else 10.
        # cmp[0] is the algorithm number and calib_epix10ka_any ignores it -- alg 7 is the only one
        # this detector has. cmp None or mode 0 means the correction is off.
        if cmp_ is None:
            self.mode, self.cormax, self.npixmin = 0, 0.0, 10
        else:
            self.mode = int(cmp_[1])
            self.cormax = float(cmp_[2])
            self.npixmin = int(cmp_[3]) if len(cmp_) > 3 else 10
        self.cmpars = cmp_

        mask = det.mask_total(run)                 # what det.calib multiplies in at the end
        self._mask = None if mask is None else cp.asarray(np.asarray(mask, np.float32))
        gm = np.ones(shape, bool) if mask is None else np.asarray(mask).astype(bool)

        self._p0 = cp.asarray(p0, self.dt)
        self._p1 = cp.asarray(p1, self.dt)
        self._f0 = cp.asarray(f0, self.dt)
        self._f1 = cp.asarray(f1, self.dt)
        # psana: gmask = grhm & det.mask_total. Gain-mode dependent, so two planes like the rest.
        self._h0 = cp.asarray((h0 & gm).astype(np.uint8))
        self._h1 = cp.asarray((h1 & gm).astype(np.uint8))

        if self.mode and (len(shape) != 3 or shape[-2] % 2 or shape[-1] % 8):
            raise NotImplementedError(
                f"common mode is implemented for epix10ka panels (nseg,352,384), got {shape}; "
                "pass cmpars=(7,0,0,0) to run pedestal+gain only")

    def _seg_offset(self, v, g):
        """Masked median per group, with psana's two vetoes. Reduce axis must be LAST.

        psana computes `np.ma.median` over the good pixels of each group, zeroes it where the group
        has too few good pixels or the offset is too large, and subtracts it from the good pixels
        only. The median is the whole difficulty: it is a per-group sort, not a reduction that maps
        to `where`. Sorting with the masked entries replaced by +inf pushes them past the end of the
        good run, so element (k-1)//2 and element k//2 of the sorted group are the two middle good
        values for any k -- averaging them reproduces numpy's even-length median exactly, and for odd
        k the two indices coincide and the average is a no-op.
        """
        cp = self.cp
        s = cp.sort(cp.where(g.astype(bool), v, cp.inf), axis=-1)
        k = g.sum(axis=-1, dtype=cp.int32)                      # uint8 sums would wrap at 255
        lo = cp.maximum((k - 1) // 2, 0)[..., None]
        hi = (k // 2)[..., None]
        m = 0.5 * (cp.take_along_axis(s, lo, -1) + cp.take_along_axis(s, hi, -1))[..., 0]
        # k == 0 leaves m = +inf, which the cormax test rejects; `where` never multiplies it.
        return cp.where((k > self.npixmin) & (cp.abs(m) < self.cormax), m, self.dt(0))

    def _common_mode(self, a, g):
        """psana's algorithm 7 for epix10ka, in place, banks then rows then columns.

        The geometry is psana's, not the documentation's: an epix10ka panel is (352, 384), split
        top/bottom at row 176 into the two ASIC rows, and each half splits into 8 banks of (176, 48).
        `mode` is a bitmask -- +4 banks, +1 rows, +2 columns -- and calib_epix10ka_any applies them in
        that order, each pass seeing the previous pass's output. The default for this detector is
        mode 2, columns, which psana's own note calls the one that works best for epix10ka; mode 3
        (rows and columns) is the Jungfrau default.

        Groups are bounded by the bank, never by the panel: a column group is the 176 rows of one
        ASIC row, a row group is the 48 columns of one bank. Columns therefore need the reduce axis
        transposed to the end -- that copy is the price of the exact median.
        """
        cp = self.cp
        nseg, H, W = a.shape
        hr, nb, bw = H // 2, 8, W // 8

        if self.mode & 4:                                        # banks: (176, 48) -> one offset
            va = a.reshape(nseg, 2, hr, nb, bw).transpose(0, 1, 3, 2, 4).reshape(nseg, 2, nb, -1)
            ga = g.reshape(nseg, 2, hr, nb, bw).transpose(0, 1, 3, 2, 4).reshape(nseg, 2, nb, -1)
            cm = self._seg_offset(cp.ascontiguousarray(va), cp.ascontiguousarray(ga))
            a -= (cm[:, :, None, :, None] * g.reshape(nseg, 2, hr, nb, bw)).reshape(nseg, H, W)

        if self.mode & 1:                                        # rows within a bank: 48 pixels
            cm = self._seg_offset(a.reshape(nseg, H, nb, bw), g.reshape(nseg, H, nb, bw))
            a -= (cm[..., None] * g.reshape(nseg, H, nb, bw)).reshape(nseg, H, W)

        if self.mode & 2:                                        # columns within an ASIC row: 176
            va = cp.ascontiguousarray(a.reshape(nseg, 2, hr, W).transpose(0, 1, 3, 2))
            ga = cp.ascontiguousarray(g.reshape(nseg, 2, hr, W).transpose(0, 1, 3, 2))
            cm = self._seg_offset(va, ga)                        # (nseg, 2, W)
            a -= (cm[:, :, None, :] * g.reshape(nseg, 2, hr, W)).reshape(nseg, H, W)

    def __call__(self, raw):
        """raw: (nseg,352,384) uint16 -> calibrated float on the device, same shape."""
        cp = self.cp
        r = cp.asarray(raw).reshape(self.shape)
        if r.dtype != cp.uint16:
            r = r.astype(cp.uint16)
        hi = (r & self.B14) != 0                                  # the one per-event bit
        a = (r & self.DATA_MASK).astype(self.dt) - cp.where(hi, self._p1, self._p0)
        if self.mode:
            self._common_mode(a, cp.where(hi, self._h1, self._h0))
        a *= cp.where(hi, self._f1, self._f0)                     # gfac = 1/gain, see item 1
        if self._mask is not None:
            a *= self._mask
        return a
