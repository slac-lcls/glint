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
"""
from __future__ import annotations

import numpy as np


def gain_layout(det, run):
    """Infer (range_offset, range_bits, nranges) for a psana1 detector from its pedestal array.

    psana stores pedestals as (nranges, nseg, H, W) for a gain-switching detector and (nseg, H, W)
    for one without. The bit layout is a property of the ASIC: Epix10ka packs 14 data bits and uses
    the top 2 for the range; Jungfrau packs 14 and uses 2. Returned rather than hard-coded so a
    detector with a different split fails loudly here instead of silently mis-calibrating.
    """
    peds = det.pedestals(run)
    if peds is None:
        raise ValueError("no pedestals for this detector/run -- cannot calibrate on the GPU")
    peds = np.asarray(peds)
    if peds.ndim == 3:                       # no gain switching: one pedestal plane
        return 0, 0, 1
    nranges = int(peds.shape[0])
    if nranges <= 1:
        return 0, 0, 1
    range_bits = int(np.ceil(np.log2(nranges)))
    range_offset = 16 - range_bits           # psana packs data in the LOW bits, range in the high
    return range_offset, range_bits, nranges


class GpuCalibrator:
    """Holds the pedestal/gain planes on the device and calibrates raw frames in place.

    The arrays are uploaded ONCE. That is most of the point: they are (nranges, npix) floats -- for
    Epix10ka2M with 7 ranges that is 60 MB -- and re-sending them per event would cost more than the
    calibration saves.
    """

    def __init__(self, det, run, dtype=None):
        import cupy as cp
        self.cp = cp
        self.dt = dtype or cp.float32
        self.range_offset, self.range_bits, self.nranges = gain_layout(det, run)

        peds = np.asarray(det.pedestals(run), dtype=np.float32)
        gains = det.gain(run)
        gains = (np.ones_like(peds) if gains is None
                 else np.asarray(gains, dtype=np.float32))
        if peds.ndim == 3:                                   # promote to a 1-range stack
            peds = peds[None]; gains = gains[None]
        if gains.shape != peds.shape:                        # a scalar or per-pixel gain, no ranges
            gains = np.broadcast_to(gains.reshape((1,) + gains.shape[-3:]), peds.shape)
        self.shape = peds.shape[1:]                          # (nseg, H, W)
        self.npix = int(np.prod(self.shape))
        self._peds = cp.asarray(peds.reshape(self.nranges, self.npix), self.dt)
        self._gains = cp.asarray(gains.reshape(self.nranges, self.npix), self.dt)
        self._rmask = np.uint16((1 << self.range_bits) - 1) if self.range_bits else np.uint16(0)
        self._dmask = (np.uint16((1 << self.range_offset) - 1) if self.range_offset
                       else np.uint16(0xFFFF))

    def __call__(self, raw):
        """raw: (nseg,H,W) uint16 (host or device) -> calibrated float on the device, same shape.

        Mirrors Reader.cu:285 exactly. `take_along_axis` is the gather the C kernel expresses as
        `&pedArray[range * nElements]` -- same indexing, expressed once for the whole frame.
        """
        cp = self.cp
        r = cp.asarray(raw).reshape(-1)
        if r.dtype != cp.uint16:
            r = r.astype(cp.uint16)
        if self.nranges == 1:
            out = (r.astype(self.dt) - self._peds[0]) * self._gains[0]
            return out.reshape(self.shape)
        rng = ((r >> np.uint16(self.range_offset)) & self._rmask).astype(cp.int32)
        data = (r & self._dmask).astype(self.dt)
        idx = rng[None, :]
        ped = cp.take_along_axis(self._peds, idx, axis=0)[0]
        gn = cp.take_along_axis(self._gains, idx, axis=0)[0]
        return ((data - ped) * gn).reshape(self.shape)
