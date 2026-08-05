"""Reader FUNCTION for psana1 xtc (LCLS-I), run IN-PROCESS in the GLINT env (conda1: psana1 + torch +
cupy). No bridge: psana1 and torch coexist here, so GLINT reads xtc1 in one interpreter and hands q
straight to hybrid_index -- the simpler half of "both xtc1 and xtc2".

Same peak-find + q core (xtc_core) as the psana2 reader; only the psana1 data-access differs:
  psana1  DataSource("exp=X:run=N") . Detector("name") . det.calib(evt) . det.coords_x/y/z(par) . EBeam
  psana2  DataSource(exp=X,run=N)  . run.Detector      . det.raw.calib  . GeometryAccess              .

The geometry caveats are IDENTICAL to the psana2 reader (psana1 coords default to cframe=CFRAME_PSANA,
the same PSANA frame; per-pixel Z is nominal so zdist REPLACES it) -- so the same README real-data gate
applies. Returns the same dict contract as xtc_qreader.run_to_qframes.
"""
from __future__ import annotations

import numpy as np

import xtc_core


def _wavelength_A(ebeam_det, evt):
    """psana1 EBeam idiom: Detector('EBeam').get(evt).ebeamPhotonEnergy(); None on any failure so one
    bad event is skipped, not fatal.

    np.isfinite is NOT optional: ebeamPhotonEnergy() returns +/-inf on some runs (100% of events on
    mfxx49820 r0016), and `eV > 1000.0` is True for +inf, which would return HC/inf = 0.0 -- a
    wavelength of zero that the caller's `if not lam` must then reject, or frame_q divides by it."""
    if ebeam_det is None:
        return None
    try:
        eb = ebeam_det.get(evt)
        if eb is None:
            return None
        eV = eb.ebeamPhotonEnergy()
        if eV is not None and np.isfinite(eV) and float(eV) > 1000.0:
            return xtc_core.HC_EV_A / float(eV)
    except Exception:
        pass
    return None


def run_to_qframes_psana1(exp, run, det="jungfrau", zdist=0.0, wavelength=0.0,
                          min_peaks=6, max_events=0, rank=0, nranks=1, calib_dir=None, geom=None,
                          peakfinder="v4",
                          min_pix=xtc_core.PF_MIN_PIX, son_min=xtc_core.PF_SON_MIN,
                          thr_high=xtc_core.PF_THR_HIGH, thr_low=xtc_core.PF_THR_LOW,
                          pf8_min_snr=xtc_core.PF8_MIN_SNR):
    """Peak-find a whole psana1 (LCLS-I) run in-process and return its q-frames -- same dict contract
    as the psana2 reader: {qframes, events, n_events, n_sent, n_skipped_wl}.

    rank/nranks shard events round-robin (rank r owns event i iff i % nranks == r) for the MPI wrapper;
    the default rank=0/nranks=1 owns everything, so single-process behaviour is unchanged. Events are
    numbered by the global enumerate index, so shards are disjoint and the returned `events` are global."""
    if zdist <= 0:
        raise ValueError("zdist (sample-detector distance, m) is REQUIRED: psana per-pixel Z is nominal")

    import psana

    if calib_dir:
        # MUST precede DataSource. psana resolves the highest run-range calib file it finds, which is
        # not necessarily the one a downstream refinement (CrystFEL/BayFAI) was built on -- e.g.
        # mfxx49820 deploys both 0-end.data and a later tilted 8-end.data, and the btx geometry is
        # built on 0-end. --zdist only replaces Z; X and Y come from psana verbatim, so picking the
        # wrong file silently shifts the transverse origin by millimetres.
        psana.setOption("psana.calib-dir", str(calib_dir))

    ds = psana.DataSource(f"exp={exp}:run={int(run)}")
    try:
        detector = psana.Detector(det)
    except KeyError as e:
        # psana raises "Source string not found in configStore" for TWO different causes, and its
        # message only describes one of them. The second is that this psana release cannot PARSE the
        # run's detector config object, in which case it drops that config silently: DetNames() still
        # lists the detector and evt.keys() still shows its data, but the configStore has no ConfigV*
        # and Detector() cannot be built. Measured on cxi/cxilu8823 r0226 (Jungfrau4M): absent under
        # ana-4.0.58-py3-minipytorch, present under ana-4.0.59-py3-minipytorch -- same stack, one
        # release up, same torch and cupy. Reading the error at face value sends you hunting for a
        # typo in a name that is perfectly correct.
        cs = ds.env().configStore()
        have = sorted({str(k.src()) for k in cs.keys() if "DetInfo" in str(k.src())})
        listed = [r for r in psana.DetNames("detectors") if det in str(r)]
        hint = (f"\n  '{det}' IS listed by DetNames but has NO config in the configStore -- this psana "
                f"release cannot parse its ConfigV. Try a NEWER ana release (4.0.58 cannot read "
                f"Jungfrau.ConfigV4; 4.0.59 can, with the same torch+cupy)."
                if listed else
                f"\n  '{det}' is not listed by DetNames either -- check the name.")
        raise KeyError(f"psana.Detector({det!r}) failed for {exp} run {run}.{hint}\n"
                       f"  configStore detectors: {have}\n  original: {e}") from e
    ebeam = None
    if not wavelength:
        try:
            ebeam = psana.Detector("EBeam")
        except Exception:
            raise ValueError("no wavelength and EBeam detector unavailable")

    X = Y = Zc = kin = None
    finders = []
    qframes, events = [], []
    n_events = n_skipped_wl = 0
    for i, evt in enumerate(ds.events()):
        if max_events and i >= max_events:
            break
        if not xtc_core.event_in_shard(i, rank, nranks):
            continue                         # not this rank's event -- skip before the expensive calib
        frame = detector.calib(evt)          # psana1: pedestal/gain/common-mode applied -> (nseg,H,W)
        if frame is None:
            continue
        # WAVELENGTH FIRST, then geometry. peakfinder8 bins the background in RADIAL shells, so its
        # per-pixel q map is q(lambda) and must exist before the first find() -- the old order built
        # the geometry before lambda was known. Reordering also means a frame skipped for a missing
        # wavelength no longer triggers the one-time geometry build.
        lam = wavelength or _wavelength_A(ebeam, evt)
        if not lam:                          # None, or 0.0 from a non-finite photon energy
            n_skipped_wl += 1
            continue
        if X is None:
            if geom:
                # psana's deployed calibration is often the UNREFINED starting geometry while the
                # trusted refinement lives only in a .geom -- on mfxx49820 r0016 that is a per-
                # QUADRANT |q| error of +-3-5% (median 3.16%) that no --zdist can absorb.
                import geom_coords
                Xf, Yf, Zf = geom_coords.coords_from_geom(geom, frame.shape, zdist)
            else:
                # psana1 per-pixel coords (um), default cframe=CFRAME_PSANA -- same frame as psana2
                Xf = detector.coords_x(evt); Yf = detector.coords_y(evt); Zf = detector.coords_z(evt)
            try:
                m = detector.mask(evt, status=True, calib=True, edges=False, central=False)
                good = m.astype(bool) if m is not None else None   # psana1 mask: 1=good, 0=bad
            except Exception:
                good = None
            X, Y, Zc, kin, finders = xtc_core.prep_geometry(
                Xf, Yf, Zf, frame.shape, good, zdist,
                min_pix=min_pix, son_min=son_min, thr_high=thr_high, thr_low=thr_low,
                pf8_min_snr=pf8_min_snr,
                peakfinder=peakfinder, lam=lam)
        n_events += 1
        q = xtc_core.frame_q(frame, finders, X, Y, Zc, kin, lam, min_peaks)
        if len(q):
            qframes.append(np.ascontiguousarray(q))
            events.append(i)

    return {"qframes": qframes, "events": events,
            "n_events": n_events, "n_sent": len(qframes), "n_skipped_wl": n_skipped_wl}
