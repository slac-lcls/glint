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
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description="GLINT from psana2 xtc via envbridge (single command)")
    ap.add_argument("--exp", default=os.environ.get("GLINT_EXP"), help="experiment id (default $GLINT_EXP)")
    ap.add_argument("--run", type=int, default=int(os.environ.get("GLINT_RUN", "0")))
    ap.add_argument("--det", default="jungfrau", help="psana detector name")
    ap.add_argument("--zdist", type=float, required=True,
                    help="sample-detector distance in m (REQUIRED: psana per-pixel Z is nominal)")
    ap.add_argument("--wavelength", type=float, default=0.0, help="A; else per-event photon energy")
    ap.add_argument("--energy-det", default="ebeamh")
    ap.add_argument("--min-peaks", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=0, help="0 = all")
    ap.add_argument("--psana", choices=("1", "2"), default="1",
                    help="1 = LCLS-I xtc1, read in-process (psana1 is in this env); "
                         "2 = LCLS-II xtc2, read in conda2 over envbridge. Default 1.")
    ap.add_argument("--reader-env", default="xpp_drp_gpu_311",
                    help="--psana 2 only: conda2 env with psana2 + cupy that does the peak-finding")
    ap.add_argument("--cell", nargs="+", metavar="V",
                    help='known cell "a b c al be ga" (skip consensus); omit for fully-blind')
    ap.add_argument("--nbest", type=int, default=3)
    ap.add_argument("-o", "--out", default="glint_xtc.stream")
    args = ap.parse_args(argv)
    if not args.exp:
        sys.exit("set --exp or GLINT_EXP (the beamtime id is not committed -- see README_beamtime.md)")

    if args.psana == "1":
        # LCLS-I: psana1 is in THIS env with torch -- read in-process, no bridge.
        import xtc_qreader_psana1
        print(f"peak-finding {args.exp} run {args.run} in-process (psana1) ...", flush=True)
        out = xtc_qreader_psana1.run_to_qframes_psana1(
            args.exp, args.run, args.det, args.zdist, args.wavelength,
            args.min_peaks, args.max_events)
    else:
        # LCLS-II: psana2 cannot co-import with torch -- read in conda2 over the bridge.
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
        print(f"peak-finding {args.exp} run {args.run} in conda2 env {args.reader_env} (psana2) ...", flush=True)
        out = envbridge.call(
            reader_env, "xtc_qreader:run_to_qframes",
            args.exp, args.run, args.det, args.zdist, args.wavelength, args.energy_det,
            args.min_peaks, args.max_events)
    frames = [np.asarray(q, float) for q in out["qframes"]]
    print(f"  {out['n_sent']} frames with >= {args.min_peaks} peaks "
          f"({out['n_events']} events, {out['n_skipped_wl']} skipped no-wavelength)", flush=True)
    if not frames:
        sys.exit("no frames with enough peaks -- nothing to index")

    # index locally in conda1 (torch), exactly as the --qframes path does
    from glint.hybrid_stream import hybrid_index, _report
    from glint.stream import write_stream
    Mc_known = None
    if args.cell:
        from glint.lattice import cell_to_Ar
        Mc_known = cell_to_Ar(*[float(x) for x in " ".join(args.cell).split()])
    images = [{"image": f"xtc://{args.exp}_r{args.run}", "event": e} for e in out["events"]]
    results, stats = hybrid_index(frames, images, Mc_known=Mc_known, nbest=args.nbest)
    write_stream(results, args.out)
    _report(stats, args.out)


if __name__ == "__main__":
    main()
