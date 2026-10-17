#!/usr/bin/env python
"""Step 1: per-frame q lists from raw xtc, with GLINT's own psana1 reader at 95d51ce.

    python j4_qlists.py TAG OUTDIR [--max-events N]      TAG in r0016 r0278 r0278z060 r0058

Calls experiments/xtc_bridge/xtc_qreader_psana1.run_to_qframes_psana1 of the pinned tree with its
default peak finder (v4) and default thresholds. The per-run arguments are the table RUNS below;
they repeat the arguments of the August runs whose frame counts are on record, so the number of
frames with >= 6 peaks can be checked against them (geometry does not enter peak finding).
Writes OUTDIR/TAG_qlists.npz (q_<i>, n, events, lams) and OUTDIR/TAG_qlists.json (provenance).
--max-events is for a dry run only; the measurement uses the table value.
"""
import argparse
import json
import os
import platform
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j4_common as C  # noqa: E402

BTX = "/sdf/data/lcls/ds/mfx/mfxx49820/results/btx/geom/r0016.geom"
DET = "MfxEndstation.0:Epix10ka2M.0"
RUNS = {
    # refined geometry: btx r0016.geom on its base calibration 0-end.data (README of xtc_bridge,
    # job8 arm B). Same arguments as the cached r0016_q6000.npz (2228 frames), plus --geom.
    "r0016": dict(exp="mfxx49820", run=16, zdist=0.102973, wavelength=1.290757, max_events=6000,
                  calib_dir=os.path.join(os.environ["VERIFY_INPUTS"], "glint_gt_mfxx49820", "calib"), geom=BTX,
                  expect_frames=2228),
    # no refined geometry exists for mfxl1038923: btx r0016 panel layout (same detector model) and
    # the distance of the consensus-support scan (max support at 0.058 m, job 34400191; ~8 mm
    # beyond r0058). The August dump that gave 1940 frames used 0.060 m (job 34492864).
    "r0278": dict(exp="mfxl1038923", run=278, zdist=0.058, wavelength=1.26296, max_events=3000,
                  calib_dir=None, geom=BTX, expect_frames=1940),
    # the same run at 0.060 m, the distance behind the FACTS key indexed_r0278 (1785) (August dump,
    # job 34492864: same reader arguments, 1940 frames). Added 16 Oct so that both
    # distances are measured in one submission; distance does not enter v4 peak finding.
    "r0278z060": dict(exp="mfxl1038923", run=278, zdist=0.060, wavelength=1.26296, max_events=3000,
                      calib_dir=None, geom=BTX, expect_frames=1940),
    # 0.050 m: consensus-support scan, corroborated by a 3.669 A ice ring at 49.4 mm (LUTE
    # sfx_mfxl1038923_r58 test geometry). wavelength 0 = per-event EBeam.
    "r0058": dict(exp="mfxl1038923", run=58, zdist=0.050, wavelength=0.0, max_events=3000,
                  calib_dir=None, geom=BTX, expect_frames=2370),
}
MIN_PEAKS = 6          # the reader's and the CLI's default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=sorted(RUNS))
    ap.add_argument("outdir")
    ap.add_argument("--max-events", type=int, default=None, help="dry run only")
    ap.add_argument("--cpu-calib", action="store_true", help="dry run on a node without a GPU")
    a = ap.parse_args()
    commit = C.pin()
    knobs = C.default_settings_guard()
    r = dict(RUNS[a.tag])
    expect = r.pop("expect_frames")
    if a.max_events is not None:
        r["max_events"] = a.max_events
    import xtc_qreader_psana1
    kw = dict(det=DET, min_peaks=MIN_PEAKS, gpu_calib=not a.cpu_calib, **r)
    print(f"qlists {a.tag}: run_to_qframes_psana1({kw})", flush=True)
    t0 = time.time()
    out = xtc_qreader_psana1.run_to_qframes_psana1(**kw)
    dt = time.time() - t0
    q = [np.asarray(x, float) for x in out["qframes"]]
    n = len(q)
    full = a.max_events is None
    print(f"{n} frames with >= {MIN_PEAKS} peaks from {out.get('n_events')} events in {dt:.0f} s"
          + (f"; on record {expect}: {'MATCH' if n == expect else 'DIFFERS'}" if full else " (dry run)"),
          flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.join(a.outdir, f"{a.tag}_qlists" + ("" if full else "_dryrun"))
    np.savez_compressed(stem + ".npz", n=n, events=np.asarray(out.get("events", []), int),
                        lams=np.asarray(out.get("lams", []), float),
                        **{f"q_{i}": x for i, x in enumerate(q)})
    prov = dict(job="per_frame_accuracy_vs_peaks", step="qlists", tag=a.tag, commit=commit, argv=sys.argv,
                reader="experiments/xtc_bridge/xtc_qreader_psana1.run_to_qframes_psana1",
                reader_kwargs={k: v for k, v in kw.items()}, peakfinder="v4 (reader default)",
                n_frames=n, n_events=out.get("n_events"), n_skipped_wl=out.get("n_skipped_wl"),
                expect_frames=expect if full else None,
                frame_count_check=(("MATCH" if n == expect else "DIFFERS") if full else "dry run"),
                seconds=round(dt, 1),
                host=socket.gethostname(), python=platform.python_version(),
                conda_env=os.environ.get("CONDA_DEFAULT_ENV"), crystfel="none (no CrystFEL step)",
                knobs_checked_unset=knobs,
                geom_manifest=os.environ.get("GLINT_GEOM_MANIFEST", "on (default)"),
                slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    json.dump(prov, open(stem + ".json", "w"), indent=1, default=str)
    print(f"wrote {stem}.npz and .json", flush=True)


if __name__ == "__main__":
    main()
