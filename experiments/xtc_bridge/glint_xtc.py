"""GLINT from xtc, single command -- reads LCLS-I (xtc1/psana1) OR LCLS-II (xtc2/psana2).

    source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
    conda activate ana-4.0.58-py3-minipytorch
    python glint_xtc.py --exp <exp> --run <run> --det jungfrau --zdist 0.246 -o out.stream   # xtc1
    python glint_xtc.py --exp <exp> --run <run> --zdist 0.246 --psana 2 -o out.stream         # xtc2

The two data eras are NOT symmetric, and the driver hides the difference:

  --psana 1 (LCLS-I, default): psana1 lives in the GLINT env alongside torch, so the reader runs
    IN-PROCESS -- one interpreter, no bridge. This is the bulk of existing data (MFX/CXI).
  --psana 2 (LCLS-II): psana2 needs numpy 2.3 and cannot co-import with torch, so the reader runs in
    conda2 via envbridge and only the q-vectors cross the pickle-free wire.

Both paths share the same peak-find + q core (xtc_core) and end at the same GLINT blind hybrid_index
-> CrystFEL .stream. The psana2 path needs envbridge in THIS env:
  pip install 'git+https://github.com/slac-lcls/drp-benchmarks.git#subdirectory=envbridge'

The read and index halves are factored into read_qframes() / index_and_write() so glint_xtc_mpi.py can
run this same body sharded across MPI ranks; main() is just the single-process (rank 0 of 1) case.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # xtc_core sits beside this file
# ...and the repo root, so `import glint` works however this file is invoked. `python
# experiments/xtc_bridge/glint_xtc.py` puts THIS file's directory on sys.path, not the working
# directory, so glint_launch.sh's `cd` to the repo root does NOT make the package importable:
# the read and peak-find stages run fine (they only need xtc_core, which sits here) and the run
# then dies at the first `import glint` inside index_and_write, after all the expensive work.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import xtc_core


def build_parser():
    ap = argparse.ArgumentParser(description="GLINT from xtc (LCLS-I in-process / LCLS-II over envbridge)")
    ap.add_argument("--exp", default=os.environ.get("GLINT_EXP"), help="experiment id (default $GLINT_EXP)")
    ap.add_argument("--run", type=int, default=int(os.environ.get("GLINT_RUN", "0")))
    ap.add_argument("--det", default="jungfrau", help="psana detector name")
    ap.add_argument("--zdist", type=float, required=True,
                    help="sample-detector distance in m (REQUIRED: psana per-pixel Z is nominal)")
    ap.add_argument("--wavelength", type=float, default=0.0, help="A; else per-event photon energy")
    ap.add_argument("--energy-det", default="ebeamh")
    ap.add_argument("--min-peaks", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=0, help="0 = all (global cap across ranks)")
    ap.add_argument("--psana", choices=("1", "2"), default="1",
                    help="1 = LCLS-I xtc1, read in-process (psana1 is in this env); "
                         "2 = LCLS-II xtc2, read in conda2 over envbridge. Default 1.")
    ap.add_argument("--reader-env", default="xpp_drp_gpu_311",
                    help="--psana 2 only: conda2 env with psana2 + cupy that does the peak-finding")
    ap.add_argument("--gpu-calib", action="store_true",
                    help="calibrate on the GPU instead of det.calib (psana1 + Epix10ka only). "
                         "det.calib is 97.7%% of an event on this route; the device path reproduces "
                         "it exactly (0.000 ADU residual, identical hit set on mfxx49820 r0016) at "
                         "0.73-3.8 ms against 141 ms, and in pass 1 the frame never leaves the GPU. "
                         "Raises rather than falling back if the detector is not supported, so a "
                         "silent return to the slow path cannot be mistaken for a speedup.")
    ap.add_argument("--calib-dir", default=None,
                    help="psana calib-dir override (psana1 only). Use when psana would resolve a "
                         "different geometry than the one a trusted refinement was built on; --zdist "
                         "only replaces Z, so X/Y still come from whatever psana picks.")
    ap.add_argument("--geom", default=None,
                    help="CrystFEL .geom to take per-pixel X/Y from INSTEAD of psana (psana1 only). "
                         "Use when the trusted geometry is a refinement that was never written back "
                         "into psana: on mfxx49820 r0016 psana's deployed 0-end.data is the "
                         "unrefined 2021 start, and against btx's refined r0016.geom it is off by a "
                         "median 3.16% in |q|, signed per detector quadrant -- which no --zdist can "
                         "absorb, and which is enough to stop blind indexing finding the true cell.")
    pf = ap.add_argument_group(
        "peak finder (PeakFinderV4)",
        "Defaults are calibrated on Epix10ka2M at MFX to match CrystFEL peakfinder8's hit rate; "
        "tune per detector. Too loose and blank frames become 'hits' -- PeakFinderV4's own library "
        "defaults (min-pix 1, no noise floor) pass 100% of frames.")
    pf.add_argument("--peakfinder", choices=("v4", "pf8", "pf8-panel"), default="v4",
                    help="v4 = local-annulus finder (default; needs no geometry). pf8 = the vendored "
                         "peakfinder8 over the WHOLE detector, seams masked -- radial-shell background, "
                         "the closer match to what PeakFinderSFX/CrystFEL run upstream, and the one to "
                         "use OFF-LINE when the xtc hit set must line up with a peakfinder8 reference. "
                         "pf8-panel = the same finder per panel: ~1/nseg the statistics per radial "
                         "shell, but panels are independent, which is what a latency-bound STREAMING "
                         "consumer wants. Both need a wavelength before the first frame (per-pixel q "
                         "is q(lambda)), so pass --wavelength when per-event photon energy is "
                         "unreliable.")
    pf.add_argument("--min-pix", type=int, default=xtc_core.PF_MIN_PIX,
                    help=f"min connected pixels per peak (default {xtc_core.PF_MIN_PIX})")
    pf.add_argument("--son-min", type=float, default=xtc_core.PF_SON_MIN,
                    help=f"min integrated peak SNR, V4 ONLY -- pf8 has no integrated-SNR cut "
                         f"(default {xtc_core.PF_SON_MIN})")
    pf.add_argument("--thr-high", type=float, default=xtc_core.PF_THR_HIGH,
                    help=f"seed SNR, V4 ONLY -- pf8's brightness cut is --pf8-min-snr "
                         f"(default {xtc_core.PF_THR_HIGH})")
    pf.add_argument("--thr-low", type=float, default=xtc_core.PF_THR_LOW,
                    help=f"component-extent SNR: v4 grow threshold, pf8 thr_snr "
                         f"(default {xtc_core.PF_THR_LOW})")
    pf.add_argument("--pf8-min-snr", type=float, default=xtc_core.PF8_MIN_SNR,
                    help=f"pf8 ONLY: min per-peak max-pixel SNR. Not tied to --thr-high; it stands "
                         f"in for the integrated-SNR cut pf8 lacks, and is sharp -- 10 turns this "
                         f"into a pass-through (default {xtc_core.PF8_MIN_SNR})")
    ig = ap.add_argument_group(
        "integration (xtc route, psana1)",
        "Off by default, in which case the stream is ORIENTATION-ONLY: every reflection row carries "
        "I=0/sigma=0 and a merge of it produces zeros. --integrate adds a SECOND PASS that re-reads "
        "the indexed events, predicts their reflections from the recovered orientation and "
        "box-integrates, giving a stream partialator can merge. Needs --geom for the panel model. "
        "Costs roughly n_events*0.9ms + n_indexed*146ms, since only wanted events are calibrated.")
    ig.add_argument("--integrate", action="store_true",
                    help="second pass: predict + box-integrate, writing real I/sigma")
    ig.add_argument("--int-dmin", type=float, default=2.0,
                    help="resolution limit for prediction, A (default 2.0)")
    ig.add_argument("--int-tol", type=float, default=0.006,
                    help="Ewald excitation-error half-width, 1/A (default 0.006)")
    ap.add_argument("--cell", nargs="+", metavar="V",
                    help='known cell "a b c al be ga" (skip consensus); omit for fully-blind')
    ap.add_argument("--nbest", type=int, default=3)
    ap.add_argument("-o", "--out", default="glint_xtc.stream")
    return ap


def read_qframes(args, rank=0, nranks=1, verbose=True):
    """Read + peak-find one event shard (rank of nranks; default the whole run) -> the reader's dict
    {qframes, events, n_events, n_sent, n_skipped_wl}. Dispatches xtc1 in-process vs xtc2 over the
    bridge; identical q-core either way."""
    pf_kw = dict(min_pix=args.min_pix, son_min=args.son_min, pf8_min_snr=args.pf8_min_snr,
                 thr_high=args.thr_high, thr_low=args.thr_low, peakfinder=args.peakfinder)
    if args.psana == "1":
        # LCLS-I: psana1 is in THIS env with torch -- read in-process, no bridge.
        import xtc_qreader_psana1
        if verbose:
            print(f"peak-finding {args.exp} run {args.run} in-process (psana1)"
                  f"{' with GPU calibration' if args.gpu_calib else ''} ...", flush=True)
        return xtc_qreader_psana1.run_to_qframes_psana1(
            args.exp, args.run, args.det, args.zdist, args.wavelength,
            args.min_peaks, args.max_events, rank, nranks, args.calib_dir, args.geom,
            gpu_calib=args.gpu_calib, **pf_kw)
    # LCLS-II: psana2 cannot co-import with torch -- read in conda2 over the bridge.
    if args.gpu_calib:
        # Checked BEFORE the envbridge import, so an unsupported combination reports itself rather
        # than sending the user off to install a package that would not have helped.
        # The psana2 reader runs in a conda2 env over envbridge, and only the peak COORDINATES cross
        # back. Device calibration there would have to live on the far side of the bridge, in that
        # env's cupy, which is a different piece of work than this flag.
        sys.exit("--gpu-calib is psana1-only so far (the psana2 reader runs over envbridge in conda2)")
    try:
        import envbridge
    except ImportError:
        sys.exit("--psana 2 needs envbridge in this env. Install it:\n"
                 "  pip install 'git+https://github.com/slac-lcls/drp-benchmarks.git#subdirectory=envbridge'\n"
                 "  (or put a checkout on PYTHONPATH)")
    # the conda2 worker must import xtc_qreader (this dir); it loads peakfinder_v4 by file path,
    # so the glint package need not be importable there.
    bridge_dir = str(Path(__file__).resolve().parent)
    reader_env = envbridge.Env.conda(args.reader_env, stack="conda2", pythonpath=[bridge_dir])
    if verbose:
        print(f"peak-finding {args.exp} run {args.run} in conda2 env {args.reader_env} (psana2) ...", flush=True)
    if args.calib_dir or args.geom:
        sys.exit("--calib-dir/--geom are psana1-only (psana2 takes geometry from calibconst)")
    return envbridge.call(
        reader_env, "xtc_qreader:run_to_qframes",
        args.exp, args.run, args.det, args.zdist, args.wavelength, args.energy_det,
        args.min_peaks, args.max_events, rank, nranks, args.peakfinder,
        args.min_pix, args.son_min, args.thr_high, args.thr_low, args.pf8_min_snr)


def index_and_write(out, args, out_path, report=True):
    """Index the reader's q-frames (blind, or --cell known) in conda1/torch and write a CrystFEL
    .stream. Always writes a valid (possibly header-only) stream so an empty shard still produces a
    mergeable part. Returns (results, stats, n_indexed)."""
    from glint.stream import write_stream
    frames = [np.asarray(q, float) for q in out["qframes"]]
    if not frames:
        n_idx = write_stream([], out_path)   # header-only, valid empty stream
        return None, None, n_idx
    from glint.hybrid_stream import hybrid_index, _report
    Mc_known = None
    if args.cell:
        from glint.lattice import cell_to_Ar
        Mc_known = cell_to_Ar(*[float(x) for x in " ".join(args.cell).split()])
    images = [{"image": f"xtc://{args.exp}_r{args.run}", "event": e} for e in out["events"]]
    results, stats = hybrid_index(frames, images, Mc_known=Mc_known, nbest=args.nbest)
    if getattr(args, "integrate", False):
        n_idx = integrate_and_write(results, args, out_path, report=report)
    else:
        n_idx = write_stream(results, out_path)
    if report:
        _report(stats, out_path)
    return results, stats, n_idx


def integrate_and_write(results, args, out_path, report=True):
    """PASS 2: re-read the indexed events, predict their reflections and box-integrate, then write a
    stream with REAL I/sigma that partialator can merge.

    Without this the stream is orientation-only: `glint/stream.py` writes every reflection row as
    `h k l 0.00 0.00 ...`, so only the Miller indices are real and a merge of it produces zeros.

    REQUIRES --geom. Prediction projects q onto named CrystFEL panels (corner, fs/ss basis, res,
    coffset), and psana's per-pixel coordinates do not carry that panel model -- they are positions,
    not a tiling. The frame is handed over reshaped to the (nseg*H, W) slab a .geom addresses.
    """
    from glint.lute_bridge import parse_geom
    from glint.predict import predict_spots, integrate_spots, write_stream_integrated
    import xtc_qreader_psana1 as rd

    if not args.geom:
        sys.exit("--integrate on the xtc route needs --geom: prediction projects onto CrystFEL "
                 "panels, which psana per-pixel coords do not define. Either pass the .geom the "
                 "reference geometry lives in, or drop --integrate for an orientation-only stream.")
    if args.psana != "1":
        sys.exit("--integrate is wired on the psana1 (xtc1) route only so far")

    panels, _glob = parse_geom(args.geom)
    by_event = {int(r["event"]): r for r in results if r.get("M") is not None}
    if not by_event:
        return write_stream_integrated([], out_path, geom_text=open(args.geom).read())

    if report:
        print(f"  integrating {len(by_event)} indexed frames (pass 2: re-read + predict + box-sum)",
              flush=True)
    lam = args.wavelength
    out = []
    for ev, frame in rd.frames_for_events(args.exp, args.run, args.det, by_event,
                                          calib_dir=args.calib_dir, max_events=args.max_events,
                                          gpu_calib=args.gpu_calib):
        r = by_event[ev]
        pred = predict_spots(r["M"], panels, args.zdist, lam,
                             dmin=args.int_dmin, tol=args.int_tol)
        if not len(pred):
            continue
        I, sig, peak, bg = integrate_spots(frame, pred)
        out.append({"image": r["image"], "event": ev, "M": r["M"],
                    "pred": pred, "I": I, "sigma": sig, "peak": peak, "bg": bg})

    n = write_stream_integrated(out, out_path, geom_text=open(args.geom).read(),
                                clen_m=args.zdist,
                                photon_eV=(12398.419843320026 / lam) if lam else 9392.7)
    if report:
        print(f"  wrote {n} integrated chunks -> {out_path}", flush=True)
    return n


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.exp:
        sys.exit("set --exp or GLINT_EXP (the beamtime id is not committed -- see README_beamtime.md)")

    out = read_qframes(args)
    print(f"  {out['n_sent']} frames with >= {args.min_peaks} peaks "
          f"({out['n_events']} events, {out['n_skipped_wl']} skipped no-wavelength)", flush=True)
    if not out["qframes"]:
        sys.exit("no frames with enough peaks -- nothing to index")
    index_and_write(out, args, args.out)


if __name__ == "__main__":
    main()
