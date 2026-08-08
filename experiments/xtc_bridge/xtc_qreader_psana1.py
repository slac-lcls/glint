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


def make_gpu_calibrator(detector, run):
    """`det.calib` on the device, or a clear refusal. Returns a callable raw(uint16) -> device frame.

    det.calib is 97.7% of an event on this route and 141 ms of it; gpu_calib reproduces it exactly on
    Epix10ka (0.000 ADU residual, identical 445-peak hit set over 40 events of mfxx49820 r0016) at
    0.73 ms with this run's common-mode constants, 3.82 ms with a set that actually corrects.

    This RAISES rather than falling back to det.calib when the GPU path is unavailable. A silent
    fallback would leave the caller believing it got a 190x speedup it did not get, and this
    codebase has already been bitten once by a check that quietly did nothing.
    """
    import gpu_calib
    return gpu_calib.GpuCalibrator(detector, int(run))


def frames_for_events(exp, run, det, wanted, calib_dir=None, max_events=0, gpu_calib=False):
    """PASS 2 of the two-pass offline route: re-read a run and yield the CALIBRATED frames of the
    events that pass 1 managed to index, as `(event_index, frame_2d)`.

    Indexing needs only q; integration needs PIXELS. Holding every calibrated frame from pass 1 is
    not an option -- 8.65 MB each on Epix10ka2M, 16.8 on Jungfrau4M, so a 6000-event run is 50-100 GB
    -- and the cell is not known until the cross-frame consensus has seen every frame, so the pixels
    cannot be consumed on the way past either. Re-reading is the way out.

    It is cheaper than it sounds because `det.calib` is the whole cost of an event and it is called
    ONLY for wanted events: measured on mfxx49820 r0016, calib is 145.8 ms of a 149.2 ms event while
    advancing the event stream is 0.94 ms. So pass 2 costs about
        n_events * 0.94 ms  +  n_indexed * 145.8 ms
    rather than a second full pass. (psana1 `idx` mode would allow true random access and skip the
    walk entirely; sequential-with-skip is simpler and the walk is not the expensive part.)

    Yields the frame reshaped to the CrystFEL (nseg*H, W) slab, which is the layout a .geom's
    min/max_fs/ss address and what `integrate_spots` expects.

    `gpu_calib` calibrates on the device instead. Unlike pass 1 this must come BACK to the host:
    `integrate_spots` is numpy and its `np.asarray` refuses a cupy array outright. The copy is ~1 ms
    against the ~145 ms of det.calib it replaces, so the trade is still overwhelmingly worth it.
    """
    import psana

    if calib_dir:
        psana.setOption("psana.calib-dir", str(calib_dir))
    ds = psana.DataSource(f"exp={exp}:run={int(run)}")
    detector = psana.Detector(det)
    gcal = make_gpu_calibrator(detector, run) if gpu_calib else None
    wanted = set(int(e) for e in wanted)
    for i, evt in enumerate(ds.events()):
        if max_events and i >= max_events:
            break
        if i not in wanted:
            continue                         # skip BEFORE calib -- that is the entire saving
        if gcal is not None:
            import cupy as cp
            raw = detector.raw(evt)
            frame = None if raw is None else cp.asnumpy(gcal(raw))
        else:
            frame = detector.calib(evt)
        if frame is None:
            continue
        f = np.asarray(frame)
        yield i, (f.reshape(-1, f.shape[-1]) if f.ndim == 3 else f)


def run_to_qframes_psana1(exp, run, det="jungfrau", zdist=0.0, wavelength=0.0,
                          min_peaks=6, max_events=0, rank=0, nranks=1, calib_dir=None, geom=None,
                          peakfinder="v4", peaks_out=None,
                          min_pix=xtc_core.PF_MIN_PIX, son_min=xtc_core.PF_SON_MIN,
                          thr_high=xtc_core.PF_THR_HIGH, thr_low=xtc_core.PF_THR_LOW,
                          pf8_min_snr=xtc_core.PF8_MIN_SNR, gpu_calib=False):
    """Peak-find a whole psana1 (LCLS-I) run in-process and return its q-frames -- same dict contract
    as the psana2 reader: {qframes, events, n_events, n_sent, n_skipped_wl}.

    rank/nranks shard events round-robin (rank r owns event i iff i % nranks == r) for the MPI wrapper;
    the default rank=0/nranks=1 owns everything, so single-process behaviour is unchanged. Events are
    numbered by the global enumerate index, so shards are disjoint and the returned `events` are global.

    `gpu_calib` replaces `det.calib` with the device path (Epix10ka only; see make_gpu_calibrator).
    The frame then STAYS ON THE DEVICE all the way into the peak-finders, which take cupy arrays --
    so this removes the host upload of every frame as well as the calibration itself. Only the peak
    COORDINATES come back, a few hundred floats instead of 8.65 MB. Nothing else in the loop needs to
    know: `frame.shape` and `frame[p]` mean the same thing on either array type, and the geometry is
    built from psana coords, not from the frame.

    `peaks_out`, if given a list, also collects each sent frame's peaks as (n,4) float32
    [fs, ss, intensity, 0] in CrystFEL SLAB coordinates -- one row per q row, same order. Consumers
    that need detector pixels rather than q (a .stream peak list, the streaming monitor's geometry
    check) get them without a second pass over the run; q-only callers pay nothing."""
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
        # ana-4.0.58-py3-minipytorch, present under ana-4.0.59-py3-minipytorch. Reading the error at
        # face value sends you hunting for a typo in a name that is perfectly correct.
        #
        # The two envs do NOT carry the same torch -- 4.0.58 has 2.1.0, 4.0.59 has 1.11.0 -- and this
        # message used to claim they did, which was the same wrong assertion that made 5d6c9e0 mark
        # STATUS.md item 5 done. Switching to 4.0.59 is now safe because glint_index.py runs on
        # torch 1.11 (see _first_index_per_group), but it is a torch DOWNGRADE and the message should
        # say so rather than imply the envs are interchangeable.
        cs = ds.env().configStore()
        have = sorted({str(k.src()) for k in cs.keys() if "DetInfo" in str(k.src())})
        listed = [r for r in psana.DetNames("detectors") if det in str(r)]
        hint = (f"\n  '{det}' IS listed by DetNames but has NO config in the configStore -- this psana "
                f"release cannot parse its ConfigV. Use a NEWER ana release: 4.0.58 cannot read "
                f"Jungfrau.ConfigV4, 4.0.59 can. Set GLINT_ANA_ENV=ana-4.0.59-py3-minipytorch. "
                f"Note 4.0.59 is a torch DOWNGRADE (2.1.0 -> 1.11.0); GLINT supports 1.11, but "
                f"anything else you import in that env has to as well."
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

    gcal = make_gpu_calibrator(detector, run) if gpu_calib else None

    X = Y = Zc = kin = None
    finders = []
    qframes, events = [], []
    n_events = n_skipped_wl = 0
    for i, evt in enumerate(ds.events()):
        if max_events and i >= max_events:
            break
        if not xtc_core.event_in_shard(i, rank, nranks):
            continue                         # not this rank's event -- skip before the expensive calib
        if gcal is not None:                 # same arithmetic, on the device; frame is a cupy array
            raw = detector.raw(evt)
            frame = None if raw is None else gcal(raw)
        else:
            frame = detector.calib(evt)      # psana1: pedestal/gain/common-mode applied -> (nseg,H,W)
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
            # psana's own coords are read either way -- they are what GLINT uses when --geom is
            # absent, and the thing --geom is checked AGAINST when it is present. Two calls, once.
            Xp = detector.coords_x(evt); Yp = detector.coords_y(evt); Zp = detector.coords_z(evt)
            if geom:
                # psana's deployed calibration is often the UNREFINED starting geometry while the
                # trusted refinement lives only in a .geom -- on mfxx49820 r0016 that is a per-
                # QUADRANT |q| error of +-2-4% that no --zdist can absorb (it explains 0.2% of it).
                import geom_coords
                Xf, Yf, Zf = geom_coords.coords_from_geom(geom, frame.shape, zdist)
            else:
                # psana1 per-pixel coords (um), default cframe=CFRAME_PSANA -- same frame as psana2
                Xf, Yf, Zf = Xp, Yp, Zp
            # STATUS.md item 7: say which geometry this is and whether anything corroborates it.
            # Warns, never raises -- see geom_provenance.report.
            try:
                import geom_provenance
                geom_provenance.report(det, run, calib_dir=calib_dir, geom=geom,
                                       coords_psana=(Xp, Yp, Zp),
                                       coords_geom=((Xf, Yf, Zf) if geom else None),
                                       shape=tuple(frame.shape), zdist=zdist)
            except Exception as e:                    # provenance must never break a run
                print(f"geometry provenance check skipped ({type(e).__name__}: {e})", flush=True)
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
        if peaks_out is None:
            q = xtc_core.frame_q(frame, finders, X, Y, Zc, kin, lam, min_peaks)
        else:
            # CrystFEL slab coordinates for consumers that need detector pixels (the streaming
            # monitor's geometry check, a .stream peak list). n_cols == W is enforced by the geom
            # loader, so the slab index is just seg-major: fs_slab = fs, ss_slab = seg*H + ss.
            q, px, inten = xtc_core.frame_q(frame, finders, X, Y, Zc, kin, lam, min_peaks,
                                            return_px=True)
            if len(q):
                Hs = frame.shape[1]
                peaks_out.append(np.stack([px[:, 2], px[:, 0] * Hs + px[:, 1], inten,
                                           np.zeros(len(px))], axis=1).astype(np.float32))
        if len(q):
            qframes.append(np.ascontiguousarray(q))
            events.append(i)

    return {"qframes": qframes, "events": events,
            "n_events": n_events, "n_sent": len(qframes), "n_skipped_wl": n_skipped_wl}
