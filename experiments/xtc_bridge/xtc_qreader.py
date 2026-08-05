"""Reader FUNCTION for psana2 xtc, run in the psana2 env by envbridge (conda2: psana2 + cupy, numpy 2.3).

xtc -> per-event calibrated frame -> GPU peak-find -> q from psana per-pixel lab coords. The peak-find
+ q math is shared with the psana1 in-process reader (xtc_core); only the psana2 data-access is here.
envbridge calls this in conda2 and ships the returned q-arrays back to the GLINT/torch driver, so there
is no transport code here. Torch-free by construction.

Geometry, the thing to trust least (see the README gate):
  * psana per-pixel Z is NOMINAL -> zdist REQUIRED, replaces it; only transverse X,Y are per-pixel.
  * get_pixel_coords() is the PSANA frame (cframe=0): self-consistent + correctly handed for GLINT's
    blind indexing, NOT the CrystFEL lab frame -- do not hand these orientations to indexamajig unfixed.
  * calib() is DAQ panel order, coords are geometry-file order; the size assert catches a mismatch,
    not a permutation -- verify segment order against det.raw.image() on first real data.
"""
from __future__ import annotations

import numpy as np

import xtc_core


def run_to_qframes(exp, run, det="jungfrau", zdist=0.0, wavelength=0.0, energy_det="ebeamh",
                   min_peaks=6, max_events=0, rank=0, nranks=1, peakfinder="v4",
                   min_pix=xtc_core.PF_MIN_PIX, son_min=xtc_core.PF_SON_MIN,
                   thr_high=xtc_core.PF_THR_HIGH, thr_low=xtc_core.PF_THR_LOW):
    """Peak-find a whole psana2 run in conda2 and return its q-frames (see xtc_qreader_psana1 for the
    identical psana1 return contract): dict of qframes / events / counts, all envbridge-sendable.

    rank/nranks shard events round-robin for the MPI wrapper (each rank's conda1 process calls this over
    its own bridge worker with its shard, so there is no MPI inside conda2). Default rank=0/nranks=1 owns
    everything -- single-call behaviour unchanged; `events` are global enumerate indices, so shards are
    disjoint and the partial streams merge by concatenation."""
    if zdist <= 0:
        raise ValueError("zdist (sample-detector distance, m) is REQUIRED: psana per-pixel Z is nominal")

    from psana import DataSource
    from psana.pscalib.geometry.GeometryAccess import GeometryAccess

    ds = DataSource(exp=exp, run=int(run))
    prun = next(ds.runs())
    detector = prun.Detector(det)
    edet = None
    if not wavelength:
        try:
            edet = prun.Detector(energy_det)
        except Exception:
            raise ValueError(f"no wavelength and energy detector {energy_det!r} unavailable")

    geo = GeometryAccess()
    geo.load_pars_from_str(detector.calibconst["geometry"][0])
    Xf, Yf, Zf = geo.get_pixel_coords()

    def good_mask(shape):
        nseg, H, W = shape
        try:
            status = detector.calibconst["pixel_status"][0].reshape(-1, nseg, H, W)
            return ~(status != 0).any(axis=0)
        except Exception:
            return None

    def wl(evt):
        if wavelength:
            return wavelength
        try:
            eV = edet.raw.ebeamPhotonEnergy(evt)
            # isfinite is NOT optional: `eV > 1000.0` is True for +inf (seen on 100% of events in
            # some runs), which would yield HC/inf = 0.0 and a divide-by-zero in frame_q.
            if eV is not None and np.isfinite(eV) and float(eV) > 1000.0:
                return xtc_core.HC_EV_A / float(eV)
        except Exception:
            pass
        return None

    X = Y = Zc = kin = None
    finders = []
    qframes, events = [], []
    n_events = n_skipped_wl = 0
    for i, evt in enumerate(prun.events()):
        if max_events and i >= max_events:
            break
        if not xtc_core.event_in_shard(i, rank, nranks):
            continue                         # not this rank's event -- skip before the expensive calib
        frame = detector.raw.calib(evt)
        if frame is None:
            continue
        # wavelength FIRST: peakfinder8's per-pixel q map is q(lambda), so it must exist before the
        # geometry is built. Mirrors the psana1 reader exactly.
        lam = wl(evt)
        if not lam:                          # None, or 0.0 from a non-finite photon energy
            n_skipped_wl += 1
            continue
        if X is None:
            X, Y, Zc, kin, finders = xtc_core.prep_geometry(
                Xf, Yf, Zf, frame.shape, good_mask(frame.shape), zdist,
                min_pix=min_pix, son_min=son_min, thr_high=thr_high, thr_low=thr_low,
                peakfinder=peakfinder, lam=lam)
        n_events += 1
        q = xtc_core.frame_q(frame, finders, X, Y, Zc, kin, lam, min_peaks)
        if len(q):
            qframes.append(np.ascontiguousarray(q))
            events.append(i)

    return {"qframes": qframes, "events": events,
            "n_events": n_events, "n_sent": len(qframes), "n_skipped_wl": n_skipped_wl}
