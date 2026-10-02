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


def _lattice_type_from_lattice_code(lattice_code):
    """Bravais code -> lattice_type, or None when the code asserts no symmetry.

    ``aP`` is CrystFEL's placeholder for "unconstrained", which is also what --tofile writes when
    the flag is absent -- so it must NOT be read back as a claim that the crystal is triclinic.
    Doing so set laue="-1" on every integrated frame, and "-1" is one of the classes
    ``standardize_axes`` leaves exactly as handed in: the per-frame setting the merge depends on
    was silently switched off, and it was off by default (glint#185). No code and ``aP`` both mean
    "no class", which takes the length rule instead."""
    c = str(lattice_code or "").strip().lower()
    if c in ("", "ap"):
        return None
    if c.startswith("hr"):
        return "rhombohedral"
    return {"a": "triclinic", "m": "monoclinic", "o": "orthorhombic",
            "t": "tetragonal", "h": "hexagonal", "c": "cubic"}.get(c[:1])


def _load_frames(args):
    """Return (frames [list of (N,3) q in 1/A], images [list of {image,event}])."""
    from glint.geom import clean_frames, parse_geom, read_crystfel_peaks, peaks_to_q
    if args.qframes:
        from glint.glint_fast import load
        frames = [np.asarray(q, float) for q in load(args.qframes)]
        images = [{"image": os.path.basename(args.qframes), "event": i} for i in range(len(frames))]
    elif args.images:
        from glint.lute_bridge import frames_from_cxi
        rf = None
        if getattr(args, "ring_focus", False):
            cv = " ".join(args.cell).split() if args.cell else []
            if len(cv) < 6:
                sys.exit("error: --ring-focus needs --cell \"a b c al be ga\"")
            rf = ([float(v) for v in cv[:6]], args.ring_qlow)
        frames, images = frames_from_cxi(args.images, args.geom, wavelength_A=args.wavelength,
                                         n=args.N, min_peaks=args.min_peaks, data_key=args.data_path,
                                         peakfinder=args.peakfinder, top_n=args.top_peaks, ring_focus=rf)
    else:
        geom = parse_geom(args.geom)
        if geom["wavelength_A"] is None and args.wavelength is None:
            sys.exit("error: no wavelength in .geom; pass --wavelength <A>")
        chunks = read_crystfel_peaks(args.peaks)
        frames, images = [], []
        for ch in chunks:
            frames.append(peaks_to_q(ch["peaks"], geom, wavelength_A=args.wavelength))
            images.append({"image": ch["image"] or "glint.cxi", "event": ch["event"]})
    # Every route, --qframes included: drop NaN/inf and zero-length rows before --min-peaks counts them
    # (glint.geom.q_rows_ok). --images and --peaks already did; a clean frame is the same object.
    frames = clean_frames(frames)
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
    ap.add_argument("--cell", nargs="+", metavar="V",
                    help='known cell (skip consensus): one quoted string "a b c al be ga" OR six '
                         'space-separated values a b c al be ga (both accepted, e.g. for LUTE which '
                         'splits the field into separate argv tokens)')
    ap.add_argument("-N", type=int, default=0, help="limit to first N frames")
    ap.add_argument("--min-peaks", type=int, default=6, help="skip frames with fewer peaks")
    ap.add_argument("--peakfinder", choices=("v4", "pf9", "pf8", "stored"), default="v4",
                    help="with --images: 'v4'/'pf9' self peak-find the images; 'stored' reuses the .cxi's own "
                         "peakfinder8/Cheetah peaks (/entry_1/result_1, no redundant peak-find); 'pf8' TBD")
    ap.add_argument("--top-peaks", type=int, default=0,
                    help="with --images: keep only the N strongest peaks/frame (0=all; guards over-finding)")
    ap.add_argument("--ring-focus", action="store_true",
                    help="with --images + --cell: search only the cell's powder-ring annuli (low-order shells, "
                         "|q|<=--ring-qlow) -- a known-cell scan / blank-veto throughput lever")
    ap.add_argument("--ring-qlow", type=float, default=0.15,
                    help="--ring-focus low-order shell cutoff in 1/A (default 0.15 ~ d>6.7 A)")
    ap.add_argument("--device", choices=("auto", "cpu"), default="auto")
    ap.add_argument("--nbest", type=int, default=3,
                    help="keep N-best cell hypotheses/frame for consensus (1 = top-1 only)")
    ap.add_argument("--mode", choices=("auto", "sparse", "dense"), default="auto",
                    help="front end: sparse=nbest+consensus (SFX stills); dense=local-cluster FFT "
                         "(rotation clouds, self-indexing); auto=pick by median rlp count (default)")
    ap.add_argument("--cascade", metavar="DRIVER",
                    help="optional external cell-given indexer binary (e.g. ffbidx/xgandalf driver) to "
                         "fall back on for frames left unindexed; must use the FRAME-in / basis-out protocol")
    ap.add_argument("--escalate", action="store_true",
                    help="sparse mode: give frames that still fail the observable gate a deeper known-cell search, "
                         "accepted only if the fit beats all 32 of its own azimuth-scrambled copies (sequential "
                         "null, ~8 searches per missed frame). Off by default")
    ap.add_argument("--select", choices=("first", "matched"), default="first",
                    help="sparse mode: which consensus-consistent candidate a frame keeps. first = the first N-best "
                         "cell, the known-cell search only if there is none (default); matched = the known-cell search "
                         "on every frame, keep whichever candidate matches the most peaks")
    ap.add_argument("--gate", choices=("none", "strict"), default="none",
                    help="what a frame must satisfy to be WRITTEN as a crystal. none (default): every registration, "
                         "as before -- with --cell that is nearly every frame, since a known-cell search always "
                         "returns the asked-for cell. strict: >= 10 peaks and >= 25%% of the frame's peaks matched "
                         "(the paper's scoring bar); a failing frame is written as unindexed and is skipped by "
                         "--integrate and --tofile. Not null-calibrated: ~5%% of dense lattice-free frames pass")
    ap.add_argument("--integrate", action="store_true",
                    help="native predict+integrate -> a stream with REAL I/sigma, self-contained (no CrystFEL). "
                         "With --images the frames are read straight from the stacked .cxi by event; with --peaks "
                         "supply the per-frame image files via --image-dir. For the best (refined) merge use --tofile")
    ap.add_argument("--image-dir", default=".", help="base directory for per-file frame images (--integrate with --peaks)")
    ap.add_argument("--data-path", help="HDF5 dataset path of the frame images (overrides the .geom "
                    "'data =' key; default: the .geom key, else /data/data with --peaks, "
                    "/entry_1/data_1/data with --images)")
    ap.add_argument("--int-dmin", type=float, default=2.0, help="--integrate resolution limit in A (default 2.0)")
    ap.add_argument("--int-tol", type=float, default=0.006,
                    help="--integrate Ewald excitation-error gate in 1/A (stills partiality window; default 0.006)")
    # Both --integrate routes face the layout question: --peaks through integrate_frames/_load_image,
    # and --images through integrate_cxi, whose (event, ss, fs) reading is checked against the file
    # by the same decision (an un-assembled panel stack is refused by name, glint#148).
    ap.add_argument("--event-axis", choices=("auto", "event", "panel"), default="auto",
                    help="--integrate: what the leading axis of a 3-D image dataset means. "
                         "auto (default) asks the file's per-event metadata and refuses to guess "
                         "when a multi-panel geometry makes it ambiguous; event|panel say so "
                         "outright (glint#136; applies to both the --peaks and the --images route)")
    # The escape hatch for glint#131. Without it "median" is reachable only from Python, which makes
    # every intensity GLINT produced before that change irreproducible through the shipped routes.
    ap.add_argument("--bg-mode", choices=("clipmean", "median", "mean"), default="clipmean",
                    help="--integrate annulus background estimator: clipmean (default, MAD-clipped "
                         "mean), median (what shipped before glint#131 -- use it to reproduce "
                         "pre-#131 intensities), mean (unbiased but not robust; diagnostic)")
    ap.add_argument("--tofile", metavar="SOL",
                    help="WRITE a CrystFEL --indexing=file solution file (the refined-merge handoff): "
                         "run 'indexamajig --indexing=file --fromfile-input-file=SOL --tolerance=10,10,10,3' "
                         "so CrystFEL refines+integrates the GLINT orientations (best merge)")
    # Was --fromfile, which named the flag after CrystFEL's READER (--fromfile-input-file) even though
    # GLINT is the WRITER -- so it read backwards from this side. Kept working, hidden from --help.
    ap.add_argument("--fromfile", metavar="SOL", help=argparse.SUPPRESS)
    ap.add_argument("--lattice",
                    help="Bravais lattice code (e.g. tPc tetragonal, oP orthorhombic). Labels the "
                         "--tofile solution file, and with --images --integrate also names the Laue "
                         "class the per-frame axis setting is standardized under. Default and 'aP' "
                         "both mean unconstrained: the file is labelled aP. The solution file's axis "
                         "setting: tetragonal/orthorhombic/hexagonal codes get the standard setting "
                         "(unique axis c); every other code (aP, monoclinic, hR) gets the setting of "
                         "--cell when it is given, else the one decided by axis lengths alone")
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
        vals = [float(x) for x in " ".join(args.cell).split()]   # accepts one quoted string OR 6 tokens
        Mc_known = cell_to_Ar(*vals)

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
        results, stats = hybrid_index(frames, images, Mc_known=Mc_known, nbest=args.nbest, cascade=casc,
                                      escalate=args.escalate or None, select=args.select)
    if args.gate != "none":                                      # before --integrate / --tofile / the stream
        from glint.hybrid_stream import gate_results
        stats["n_gated"] = gate_results(results, frames, args.gate)
        stats["gate"] = args.gate
        stats["n_idx"] -= stats["n_gated"]
    if args.integrate:
        if not args.geom:
            ap.error("--integrate requires --geom (and --image-dir for the frame images)")
        from glint.predict import write_stream_integrated
        _ev_axis = {"auto": None, "event": True, "panel": False}[args.event_axis]
        if args.images:                                          # stacked .cxi: read data[event] directly (self-contained)
            from glint.predict import integrate_cxi
            from glint.lute_bridge import parse_geom as _pg
            code = str(args.lattice or "").strip()
            lt = _lattice_type_from_lattice_code(code)
            if lt is not None:
                centering = code[1:2].upper()
                for r in results:
                    r.setdefault("lattice_type", lt)
                    if centering:
                        r.setdefault("centering", centering)
            nint, tot = integrate_cxi(results, args.geom, wavelength_A=args.wavelength,
                                      dmin=args.int_dmin, tol=args.int_tol, bg_mode=args.bg_mode,
                                      data_key=args.data_path, event_axis=_ev_axis)
            _panels, _g = _pg(args.geom)
            _pnames = [p["name"] for p in _panels]
            def _f(v, d):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return d
            write_stream_integrated(results, args.out, geom_text=open(args.geom).read(),
                                    photon_eV=_f(_g.get("photon_energy"), 9392.7), clen_m=_f(_g.get("clen"), 0.15),
                                    panel_names=_pnames)
        else:                                                    # per-file images (legacy detectors)
            from glint.predict import integrate_frames
            from glint.geom import parse_geom
            geomd = parse_geom(args.geom); gg = geomd.get("global", {})
            nint, tot = integrate_frames(results, geomd, image_dir=args.image_dir,
                                         data_path=args.data_path,          # None -> the .geom 'data =' key (glint#143)
                                         dmin=args.int_dmin, tol=args.int_tol, bg_mode=args.bg_mode,
                                         event_axis=_ev_axis)
            _pnames = list(geomd.get("panels", {}).keys())
            write_stream_integrated(results, args.out, geom_text=open(args.geom).read(),
                                    photon_eV=float(gg.get("photon_energy", 9392.7)), clen_m=float(gg.get("clen", 0.15)),
                                    panel_names=_pnames or None)
    else:
        # the .geom is what makes the stream readable at all -- see stream.write_stream
        write_stream(results, args.out,
                     geom_text=open(args.geom).read() if args.geom else None)
    _report(stats, args.out)
    if args.integrate:
        print(f"  integrated         : {nint} frames / {tot} reflections (real I/sigma) -> {args.out}")
    sol_path = args.tofile or args.fromfile
    if sol_path:
        if args.fromfile and not args.tofile:
            print("  note: --fromfile is deprecated, use --tofile (GLINT WRITES this file; "
                  "'fromfile' was named for CrystFEL, which reads it)", file=sys.stderr)
        from glint.predict import write_solution_file
        lattice_code = args.lattice or "aP"
        nsol = write_solution_file(results, sol_path, lattice_code, ref_cell=Mc_known)
        print(f"  solution file      : {nsol} ({lattice_code}) -> {sol_path}"
              f"  [indexamajig --indexing=file --fromfile-input-file={sol_path} --tolerance=10,10,10,3]")


if __name__ == "__main__":
    main()
