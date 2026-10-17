#!/usr/bin/env python
"""Rebuild the btx peak-finding .cxi of mfxx49820 r0016 for the first N psana events.

The .cxi that btx's own CrystFEL indexing of r0016 read (2023, /cds/data/drpsrcf/.../scratch/btx/index/r0016/)
was purged with the experiment scratch; only its stream survives (results/btx/index/r0016_sample2.stream). Both
arms need pixels (CrystFEL integrates), so the .cxi is rebuilt here with btx's own PeakFinder
(/sdf/group/lcls/ds/tools/btx/stable, commit 0f1af1d) and the r0016 settings of results/btx/mfxx49820_16.yaml
(find_peaks block; mask = the file btx's fetch_latest picks for run 16, r0008.npy; psana_mask False; clen from
btx's default PV for epix10k2M). The only change from btx's find_peaks task is the event cap: events
0..N-1 of the run in time order (the first-6000-event window of GLINT's xtc_bridge/README.md r0016 numbers),
split in contiguous blocks over the MPI ranks, so the virtual dataset keeps time order.
report() (the elog post) is never called.

    srun -n 32 python r0016_cxi.py --outdir DIR [--max-events 6000]
Writes DIR/mfxx49820_r0016_<rank>.cxi, the virtual DIR/mfxx49820_r0016.cxi, DIR/r0016.lst,
DIR/peakfinding.summary and DIR/r0016_cxi.json (settings + per-rank counts).
"""
import argparse
import json
import os
import sys

E = "/sdf/data/lcls/ds/mfx/mfxx49820/results/btx"
BTX_FIND_PEAKS = dict(min_peaks=10, max_peaks=2048, npix_min=2, npix_max=30, amax_thr=40.0, atot_thr=180.0,
                      son_min=10.0, peak_rank=3, r0=3.0, dr=2.0, nsigm=10.0)      # mfxx49820_16.yaml find_peaks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-events", type=int, default=6000)
    ap.add_argument("--mask", default=f"{E}/mask/r0008.npy")
    ap.add_argument("--clen-mm", type=float, default=None,
                    help="override the clen btx reads from MFX:ROB:CONT:POS:Z (btx's own r0016 stream: 240.573)")
    a = ap.parse_args()

    from btx.processing.peak_finder import PeakFinder
    os.makedirs(a.outdir, exist_ok=True)
    pf = PeakFinder(exp="mfxx49820", run=16, det_type="epix10k2M", outdir=a.outdir,
                    event_receiver=None, event_code=None, event_logic=None, tag="",
                    mask=a.mask, psana_mask=False, pv_camera_length=None, calibdir=None, **BTX_FIND_PEAKS)
    n_run = len(pf.psi.times)
    n = min(a.max_events, n_run) if a.max_events > 0 else n_run
    pf.psi.distribute_events(pf.rank, pf.size, max_events=n)     # contiguous block of events 0..n-1 per rank
    pf.n_events = pf.psi.max_events
    first = int(pf.psi.counter)
    if a.clen_mm is not None:
        pf.clen = a.clen_mm
    pf.find_peaks()
    pf.curate_cxi()
    pf.summarize()
    counts = pf.comm.gather((pf.rank, first, int(pf.psi.max_events), int(pf.n_hits)), root=0)
    if pf.rank == 0:
        rec = {"exp": "mfxx49820", "run": 16, "det_type": "epix10k2M", "events_in_run": n_run, "events_used": n,
               "find_peaks": BTX_FIND_PEAKS, "mask": a.mask, "psana_mask": False, "clen_mm": float(pf.clen),
               "clen_source": "override" if a.clen_mm is not None else "btx default PV MFX:ROB:CONT:POS:Z",
               "btx": os.path.dirname(os.path.dirname(sys.modules["btx"].__file__)),
               "ranks": [{"rank": r, "first_event": f, "end_event": e, "hits": h} for r, f, e, h in counts],
               "hits_total": int(sum(c[3] for c in counts))}
        json.dump(rec, open(os.path.join(a.outdir, "r0016_cxi.json"), "w"), indent=1)
        print(json.dumps({k: rec[k] for k in ("events_used", "hits_total", "clen_mm", "clen_source")}))


if __name__ == "__main__":
    main()
