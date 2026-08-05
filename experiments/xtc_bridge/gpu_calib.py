"""Pedestal + gain calibration on the GPU, from raw uint16 -- STATUS.md item 4.

`det.calib()` is 97.7% of a LUTE xtc event: 145.8 ms of 149.2 ms measured on mfxx49820 r0016
(Epix10ka2M, 2.16 Mpix), against 0.94 ms to advance the event stream and 0.86 ms for `det.raw`. It is
numpy on one core, and it is the whole cost of the offline route.

The DAQ already solves this. `psdaq/drpGpu/Reader.cu` on drp-srcf-gpu003 uploads the raw frame as
uint16 and applies, per pixel:

    rangeMask = (1 << rangeBits) - 1
    dataMask  = (1 << rangeOffset) - 1
    range     = (raw >> rangeOffset) & rangeMask
    data      =  raw & dataMask
    calib     = (float(data) - pedArray[range][i]) * gainArray[range][i]

This is that kernel in cupy, fed from psana's own `det.pedestals(run)` / `det.gain(run)`, so the
arrays are the same ones `det.calib` uses.

WHAT THIS DOES NOT DO, and it matters. `det.calib` also applies COMMON MODE -- a data-dependent
per-ASIC/row median subtraction that Reader.cu has no equivalent of, because the DAQ's detectors
either do not need it or handle it upstream. Dropping it is not free and not obviously harmful; it
changes the background under every peak, and the hit set is measurably sensitive to that. So this
module is deliberately paired with `validate_gpu_calib.py`, which measures the difference against
`det.calib` on real frames rather than asserting it away. Do not ship the fast path on a detector
without running that first.

The gain-range decode is the part you cannot skip. Epix10ka2M and Jungfrau encode the gain range in
the raw value's high bits, and the pedestal/gain arrays are indexed BY that range -- get it wrong and
the calibration is wrong exactly on the bright pixels peak-finding depends on.

STATE, 2026-08-05: THE DECODE IS SOLVED; COMMON MODE IS THE WHOLE REMAINDER. Three runs of
validate_gpu_calib.py, mfxx49820 r0016, 40 events, identical apart from how the gain mode is
resolved:

                          guessed (13,3)   DAQ (14,2)    psana decode
                          job 34275323     job 34275620  job 34275959
    det.calib (CPU)        142.84 ms        142.33 ms     144.12 ms
    this kernel              1.40 ms          1.37 ms       1.11 ms   (130x)
    per-pixel residual      4050 ADU         4035 ADU      1431 ADU
    peak-set Jaccard          35.7%            55.5%         58.6%
      shared/onlyCPU/onlyGPU  177/268/51      282/163/63    287/158/45

Following psana cut the residual 2.8x and made the kernel FASTER -- a `where` plus an FMA beats
the per-pixel gather the range-index version needed. The Jaccard moved 3 points.

That is the diagnosis. With the gain mode correct, what remains is a per-ASIC offset, which is
precisely what common mode removes, and it explains the 158 still-missing peaks mechanically:
an un-subtracted ASIC pedestal lifts the local background, PeakFinderV4's annulus SNR falls,
and weak peaks in those ASICs drop below threshold. Before the decode was fixed a wrong bit
field and a missing common mode were indistinguishable; they are now separated.

COMMON MODE IS THEREFORE NOT OPTIONAL on this detector, and implementing it is the remaining
work for item 4. psana applies it per ASIC/bank as a median with outlier rejection (see
Detector/UtilsCommonMode.py and the detector's `common_mode` parameters); a GPU version is a
segmented median, not a reduction that maps to `where`. Until it exists this module must not be
wired into the reader: 130x on 98% of the event while losing a third of the real peaks is not a
speedup anyone can use.
"""
from __future__ import annotations

import numpy as np


def gain_mode_planes(det, run):
    """Per-pixel (pedestal, gain) planes for data-bit-14 CLEAR and SET, using psana's own decode.

    Epix10ka's gain range is NOT the raw high bits alone. psana's Detector/UtilsEpix10ka.py
    `gain_maps_epix10ka_any` builds a per-pixel control word from the DETECTOR CONFIGURATION
    (test/mask/gain/ga bits and trbit, all run-constant) and then ORs in ONE bit from the data --
    bit 14, shifted down to bit 5 -- before classifying into the 7 modes:

        FH_H 0   FM_M 1   FL_L 2   AHL_H 3   AML_M 4   AHL_L 5   AML_L 6

    That is why a 2-bit raw field cannot address 7 planes, and why the earlier `raw range k ->
    psana plane k` clamp lost a third of the peaks: for an auto-gain pixel the mode depends on the
    configuration, not on the raw value.

    Since the config half is constant for a run and the data half is a single bit, the whole decode
    reduces to TWO precomputed plane sets. This calls psana's function twice -- once with synthetic
    all-zero data, once with every pixel's bit 14 set -- so the classification is psana's, not a
    reimplementation of it, and then folds pedestal and gain into those two cases. Per event the GPU
    then does one select and one fused multiply-add.
    """
    from Detector.UtilsEpix10ka import gain_maps_epix10ka_any, B14

    peds = np.asarray(det.pedestals(run), dtype=np.float32)
    if peds is None or peds.ndim != 4:
        raise NotImplementedError(
            "gain-mode decode is implemented for the Epix10ka family (4-D pedestals, "
            f"got shape {None if peds is None else peds.shape})")
    nmodes = peds.shape[0]
    gains = det.gain(run)
    gains = (np.ones_like(peds) if gains is None else np.asarray(gains, dtype=np.float32))
    if gains.shape != peds.shape:
        gains = np.broadcast_to(gains.reshape((1,) + gains.shape[-3:]), peds.shape)

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
        out.append((np.take_along_axis(peds, idx[None].astype(np.intp), 0)[0],
                    np.take_along_axis(gains, idx[None].astype(np.intp), 0)[0]))
    return out[0], out[1], shape


class GpuCalibrator:
    """Pedestal + gain on the device: one select and one FMA per pixel per event.

    Setup resolves the gain mode with psana's own decode (see `gain_mode_planes`) into two
    per-pixel plane pairs -- bit 14 clear and bit 14 set -- and uploads them ONCE. Four float
    planes for Epix10ka2M is 34 MB; re-deriving them per event would cost more than the
    calibration saves.

    STILL NOT COMMON MODE. det.calib also subtracts a data-dependent per-ASIC/row median, which
    this does not do and Reader.cu has no equivalent of. validate_gpu_calib.py is what says
    whether that matters on a given detector; the number to read there is the peak-set Jaccard.
    """

    DATA_MASK = np.uint16((1 << 14) - 1)      # 14 data bits; bit 14 is the gain bit (B14)
    B14 = np.uint16(1 << 14)

    def __init__(self, det, run, dtype=None):
        import cupy as cp
        self.cp = cp
        self.dt = dtype or cp.float32
        (p0, g0), (p1, g1), shape = gain_mode_planes(det, run)
        self.shape = shape
        self._p0 = cp.asarray(p0.reshape(-1), self.dt)
        self._g0 = cp.asarray(g0.reshape(-1), self.dt)
        self._p1 = cp.asarray(p1.reshape(-1), self.dt)
        self._g1 = cp.asarray(g1.reshape(-1), self.dt)

    def __call__(self, raw):
        """raw: (nseg,H,W) uint16 -> calibrated float on the device, same shape."""
        cp = self.cp
        r = cp.asarray(raw).reshape(-1)
        if r.dtype != cp.uint16:
            r = r.astype(cp.uint16)
        hi = (r & self.B14) != 0                                  # the one per-event bit
        data = (r & self.DATA_MASK).astype(self.dt)
        ped = cp.where(hi, self._p1, self._p0)
        gn = cp.where(hi, self._g1, self._g0)
        return ((data - ped) * gn).reshape(self.shape)
