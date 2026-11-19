"""Peak-find one run with GLINT's v4 finder (host CPU) and save the q-frames for indexing.

This is the CPU half of the benchmark, split out so a workflow can run many runs in parallel on
the host and keep the GPU steps (index_integrate.py) to a few at a time.

    python find_peaks.py --images RUN.h5 --geom A.geom -o RUN.peaks.npz [--sig-floor 10 --abs-thr 100]

``sig_floor`` and ``abs_thr`` are in detector units. CXIDB 33 is 10 ADU per photon and clipped at
zero, so without a floor the local sigma of an empty patch is 0, every lit pixel is a high-SNR
"peak", and the finder returns 500-3000 per frame where ~30-300 are real.
"""
import argparse
import json
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--min-peaks", type=int, default=10)
    ap.add_argument("--sig-floor", type=float, default=10.0)
    ap.add_argument("--abs-thr", type=float, default=100.0)
    ap.add_argument("-N", type=int, default=0)
    a = ap.parse_args()
    from glint.lute_bridge import frames_from_cxi
    from glint.geom import clean_frames
    t0 = time.time()
    frames, images = frames_from_cxi(a.images, a.geom, n=a.N, min_peaks=a.min_peaks,
                                     sig_floor=a.sig_floor, abs_thr=a.abs_thr)
    frames = clean_frames(frames)
    keep = [(q, im) for q, im in zip(frames, images) if len(q) >= a.min_peaks]
    dt = time.time() - t0
    np.savez(a.out, n=len(keep), images=json.dumps([im for _, im in keep]),
             t_peakfind_s=dt, **{f"q_{i}": q for i, (q, _) in enumerate(keep)})
    print(f"{a.images}: {len(keep)} frames with >= {a.min_peaks} peaks, {dt:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
