"""GLINT command-line indexer -- the productized LUTE/LCLS entry point.

Fully-blind GPU SFX indexing: peaks (+ geometry) -> consensus cell -> CrystFEL .stream, a
drop-in for the existing CrystFEL/partialator flow. Two input modes:

  # what LUTE/peakfinder8 emits: a CrystFEL peak-search stream + a .geom
  python glint_cli.py --peaks peaks.stream --geom detector.geom -o indexed.stream

  # pre-bridged reciprocal q-vectors (FRAME blocks, 3 cols, 1/A)
  python glint_cli.py --qframes frames.txt -o indexed.stream

Options: --cell "a b c al be ga" (skip consensus, rescue against this cell) ; -N limit ;
--min-peaks (default 6) ; --device cpu|auto (cpu forces CUDA off for the fallback path).
"""
import os, sys, argparse
import numpy as np


def _load_frames(args):
    """Return (frames [list of (N,3) q in 1/A], images [list of {image,event}])."""
    from glint.geom import parse_geom, read_crystfel_peaks, peaks_to_q
    if args.qframes:
        from glint.glint_fast import load
        frames = [np.asarray(q, float) for q in load(args.qframes)]
        images = [{"image": os.path.basename(args.qframes), "event": i} for i in range(len(frames))]
    elif args.images:
        from glint.lute_bridge import frames_from_cxi
        frames, images = frames_from_cxi(args.images, args.geom, wavelength_A=args.wavelength,
                                         n=args.N, min_peaks=args.min_peaks,
                                         peakfinder=args.peakfinder, top_n=args.top_peaks)
    else:
        geom = parse_geom(args.geom)
        if geom["wavelength_A"] is None and args.wavelength is None:
            sys.exit("error: no wavelength in .geom; pass --wavelength <A>")
        chunks = read_crystfel_peaks(args.peaks)
        frames, images = [], []
        for ch in chunks:
            frames.append(peaks_to_q(ch["peaks"], geom, wavelength_A=args.wavelength))
            images.append({"image": ch["image"] or "glint.cxi", "event": ch["event"]})
    keep = [(q, im) for q, im in zip(frames, images) if len(q) >= args.min_peaks]
    if args.N:
        keep = keep[:args.N]
    return [q for q, _ in keep], [im for _, im in keep]


def main():
    ap = argparse.ArgumentParser(prog="glint", description="Blind GPU SFX indexer -> CrystFEL .stream")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--peaks", help="CrystFEL peak-search stream (needs --geom)")
    src.add_argument("--images", help="raw detector .cxi; GLINT peak-finds it (see --peakfinder) then indexes "
                                      "-- self-contained GPU front end (needs --geom)")
    src.add_argument("--qframes", help="pre-bridged q-vector FRAME blocks (1/A)")
    ap.add_argument("--geom", help="CrystFEL .geom (with --peaks or --images)")
    ap.add_argument("--wavelength", type=float, help="wavelength in A (overrides .geom)")
    ap.add_argument("--cell", help='known cell "a b c al be ga" (skip consensus)')
    ap.add_argument("-N", type=int, default=0, help="limit to first N frames")
    ap.add_argument("--min-peaks", type=int, default=6, help="skip frames with fewer peaks")
    ap.add_argument("--peakfinder", choices=("v4", "pf9", "pf8", "stored"), default="v4",
                    help="with --images: 'v4'/'pf9' self peak-find the images; 'stored' reuses the .cxi's own "
                         "peakfinder8/Cheetah peaks (/entry_1/result_1, no redundant peak-find); 'pf8' TBD")
    ap.add_argument("--top-peaks", type=int, default=0,
                    help="with --images: keep only the N strongest peaks/frame (0=all; guards over-finding)")
    ap.add_argument("--device", choices=("auto", "cpu"), default="auto")
    ap.add_argument("--nbest", type=int, default=3,
                    help="keep N-best cell hypotheses/frame for consensus (1 = top-1 only)")
    ap.add_argument("--mode", choices=("auto", "sparse", "dense"), default="auto",
                    help="front end: sparse=nbest+consensus (SFX stills); dense=local-cluster FFT "
                         "(rotation clouds, self-indexing); auto=pick by median rlp count (default)")
    ap.add_argument("--cascade", metavar="DRIVER",
                    help="optional external cell-given indexer binary (e.g. ffbidx/xgandalf driver) to "
                         "fall back on for frames left unindexed; must use the FRAME-in / basis-out protocol")
    ap.add_argument("--integrate", action="store_true",
                    help="native predict+integrate: emit a stream with REAL I/sigma (fast QC path; needs "
                         "--geom and the frame images via --image-dir). For the best merge use --fromfile instead")
    ap.add_argument("--image-dir", default=".", help="base directory for the frame image files (with --integrate)")
    ap.add_argument("--fromfile", metavar="SOL",
                    help="also emit a CrystFEL --indexing=file solution file (the refined-merge handoff): "
                         "run 'indexamajig --indexing=file --fromfile-input-file=SOL --tolerance=10,10,10,3' "
                         "so CrystFEL refines+integrates the GLINT orientations (best merge)")
    ap.add_argument("--lattice", default="aP",
                    help="Bravais lattice code for --fromfile (e.g. tPc tetragonal, aP triclinic); default aP")
    ap.add_argument("-o", "--out", default="glint.stream")
    args = ap.parse_args()
    if (args.peaks or args.images) and not args.geom:
        ap.error("--peaks/--images requires --geom")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    frames, images = _load_frames(args)
    if not frames:
        sys.exit("error: no frames with >= %d peaks" % args.min_peaks)

    Mc_known = None
    if args.cell:
        from glint.lattice import cell_to_Ar
        Mc_known = cell_to_Ar(*[float(x) for x in args.cell.split()])

    from glint.hybrid_stream import hybrid_index, dense_index, _report   # torch import deferred to here
    from glint.stream import write_stream
    from glint.glint_fast import CLUSTER_MIN
    mode = args.mode
    if mode == "auto":
        med = int(np.median([len(q) for q in frames]))
        mode = "dense" if med >= CLUSTER_MIN else "sparse"
    if mode == "dense":
        results, stats = dense_index(frames, images)
    else:
        casc = None
        if args.cascade:
            from glint.cascade import external_cascade
            casc = external_cascade(args.cascade)
        results, stats = hybrid_index(frames, images, Mc_known=Mc_known, nbest=args.nbest, cascade=casc)
    if args.integrate:
        if not args.geom:
            ap.error("--integrate requires --geom (and --image-dir for the frame images)")
        from glint.predict import integrate_frames, write_stream_integrated
        from glint.geom import parse_geom
        geomd = parse_geom(args.geom); gg = geomd.get("global", {})
        nint, tot = integrate_frames(results, geomd, image_dir=args.image_dir)
        write_stream_integrated(results, args.out, geom_text=open(args.geom).read(),
                                photon_eV=float(gg.get("photon_energy", 9392.7)), clen_m=float(gg.get("clen", 0.15)))
    else:
        write_stream(results, args.out)
    _report(stats, args.out)
    if args.integrate:
        print(f"  integrated         : {nint} frames / {tot} reflections (real I/sigma) -> {args.out}")
    if args.fromfile:
        from glint.predict import write_fromfile
        nsol = write_fromfile(results, args.fromfile, args.lattice)
        print(f"  fromfile solutions : {nsol} ({args.lattice}) -> {args.fromfile}"
              f"  [indexamajig --indexing=file --fromfile-input-file={args.fromfile} --tolerance=10,10,10,3]")


if __name__ == "__main__":
    main()
