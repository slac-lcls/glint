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
                   min_peaks=6, max_events=0):
    """Peak-find a whole psana2 run in conda2 and return its q-frames (see xtc_qreader_psana1 for the
    identical psana1 return contract): dict of qframes / events / counts, all envbridge-sendable."""
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
            if eV and float(eV) > 1000.0:
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
        frame = detector.raw.calib(evt)
        if frame is None:
            continue
        if X is None:
            X, Y, Zc, kin, finders = xtc_core.prep_geometry(Xf, Yf, Zf, frame.shape, good_mask(frame.shape), zdist)
        lam = wl(evt)
        if lam is None:
            n_skipped_wl += 1
            continue
        n_events += 1
        q = xtc_core.frame_q(frame, finders, X, Y, Zc, kin, lam, min_peaks)
        if len(q):
            qframes.append(np.ascontiguousarray(q))
            events.append(i)

    return {"qframes": qframes, "events": events,
            "n_events": n_events, "n_sent": len(qframes), "n_skipped_wl": n_skipped_wl}
