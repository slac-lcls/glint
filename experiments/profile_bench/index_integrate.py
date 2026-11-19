"""Index one run's peaks with GLINT, then integrate the SAME orientations with each integrator.

The comparison is meant to isolate integration, so indexing runs once and its results are deep-copied
into every arm. Each arm then predicts from the identical orientation, at the identical dmin and
excitation-error gate, and writes a CrystFEL .stream. Peaks come from find_peaks.py. Indexing is
hybrid_index (known cell) on the GPU, and the paper's strict gate withdraws weak registrations
before anything is integrated.

    python index_integrate.py --peaks RUN.peaks.npz --images RUN.h5 --geom A.geom \\
        --cell 79.2 79.2 37.9 90 90 90 --lattice tP --out-dir streams --name RUN --adu-per-photon 10

--tol defaults to 0.002 1/A, not GLINT's 0.006: on CXIDB 33 the found spots' excitation errors are
0.00084 at the 95th percentile and 0.0014 at the 99th. 0.006 predicts ~1770 reflections a frame,
mostly empty, which also shrinks the profile fit's automatic radius (it is set by prediction
spacing).

Writes OUT_DIR/<integrator>/<name>.stream and OUT_DIR/<name>.json: counts and per-stage timings.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--cell", nargs=6, type=float, required=True)
    ap.add_argument("--lattice", default="tP")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--integrators", default="box,profile")
    ap.add_argument("--adu-per-photon", type=float, default=None)
    ap.add_argument("--dmin", type=float, default=1.9)
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--gate", default="strict")
    ap.add_argument("--profile-batch", type=int, default=8)
    a = ap.parse_args()

    from glint.lute_bridge import parse_geom
    from glint.lattice import cell_to_Ar
    from glint.glint_cli import _lattice_type_from_lattice_code, _stream_photon_eV, _num
    from glint.hybrid_stream import hybrid_index, gate_results
    from glint.predict import integrate_cxi, write_stream_integrated

    log = {"images": os.path.abspath(a.images), "args": vars(a)}
    pk = np.load(a.peaks)
    frames = [pk[f"q_{i}"] for i in range(int(pk["n"]))]
    images = json.loads(str(pk["images"]))
    for im in images:                    # the stack's path as THIS run sees it
        im["image"] = a.images
    log["t_peakfind_s"] = float(pk["t_peakfind_s"])
    log["n_hits_with_peaks"] = len(frames)
    log["peaks_per_frame_median"] = float(np.median([len(q) for q in frames])) if frames else 0.0

    t0 = time.time()
    results, stats = hybrid_index(frames, images, Mc_known=cell_to_Ar(*a.cell), nbest=3)
    n_gated = gate_results(results, frames, a.gate)
    log["t_index_s"] = time.time() - t0
    log["n_indexed"] = int(sum(r.get("M") is not None for r in results))
    log["n_gated"] = int(n_gated)

    lt = _lattice_type_from_lattice_code(a.lattice)
    for r in results:
        if lt is not None:
            r.setdefault("lattice_type", lt)
            r.setdefault("centering", a.lattice[1:2].upper())

    panels, glob = parse_geom(a.geom)
    geom_text = open(a.geom).read()
    for integ in a.integrators.split(","):
        rr = copy.deepcopy(results)
        t0 = time.time()
        n, tot = integrate_cxi(rr, a.geom, dmin=a.dmin, tol=a.tol, integrator=integ,
                               adu_per_photon=a.adu_per_photon,
                               profile_kw=dict(batch=a.profile_batch))
        dt = time.time() - t0
        os.makedirs(os.path.join(a.out_dir, integ), exist_ok=True)
        out = os.path.join(a.out_dir, integ, a.name + ".stream")
        write_stream_integrated(rr, out, geom_text=geom_text,
                                photon_eV=_stream_photon_eV(None, glob.get("photon_energy")),
                                clen_m=_num(glob.get("clen"), 0.15),
                                panel_names=[p["name"] for p in panels])
        log[integ] = {"t_integrate_s": dt, "n_frames": n, "n_refl": tot, "stream": out}
        print(f"{a.name} {integ}: {n} frames, {tot} reflections, {dt:.1f} s", flush=True)
    with open(os.path.join(a.out_dir, a.name + ".json"), "w") as f:
        json.dump(log, f, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "args"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
