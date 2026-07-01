"""Optional external-indexer cascade stage for the hybrid pipeline. After blind -> consensus ->
GPU rescue, hand the frames still left UNINDEXED to an external cell-given indexer (e.g. the ffbidx
fast-feedback indexer, or the xgandalf driver), keeping only solutions that match the consensus cell.

It is FREE INSURANCE for the marginal regime (few frames / no strong consensus) and largely redundant
when consensus is strong -- on the 120 cxidb frames the full consensus hybrid already reaches 117/120,
so a fallback catches only a handful; its value is single-frame / low-frame-count data. Default OFF.

The external binary must read a FRAME file (arg 1) -- lines "FRAME id npk" then npk reciprocal spots
"qx qy qz" (1/A) -- and print one line per frame "id ax ay az bx by bz cx cy cz [score] ms" (the
output cell's three real-space axis vectors, rows) or "id NONE ms". Both experiments/xgandalf/
ffbidx_driver and xg_driver already use exactly this format.
"""
import os
import subprocess
import tempfile
import numpy as np
from fftindex.multishot import same_lattice


def write_frames_file(frames, path):
    """Write the shared driver input format: 'FRAME i npk' + npk reciprocal spots 'qx qy qz' (1/A)."""
    with open(path, "w") as f:
        for i, q in enumerate(frames):
            q = np.asarray(q, float)
            f.write("FRAME %d %d\n" % (i, len(q)))
            for x, y, z in q:
                f.write("%.6f %.6f %.6f\n" % (x, y, z))


def parse_basis_out(path):
    """Parse driver stdout -> {frame_id: M} with M's COLUMNS = real axes a,b,c (so q @ M = hkl);
    None for 'id NONE ms'. The optional trailing score before ms is ignored (only cols 1..9 used)."""
    out = {}
    for ln in open(path):
        t = ln.split()
        if len(t) < 2:
            continue
        fid = int(t[0])
        if t[1] == "NONE":
            out[fid] = None
        else:
            out[fid] = np.array(list(map(float, t[1:10]))).reshape(3, 3).T   # rows=axes -> cols=axes
    return out


def external_cascade(driver, extra_args=()):
    """Build a cascade callable `run(frames, cell) -> {local_index: M}` for hybrid_index(cascade=...).

    `driver` is the path to an indexer binary using the FRAME-file / basis-out protocol above. The
    consensus `cell` is passed through for cell-aware drivers / logging (the reference ffbidx_driver
    fixes its own prior cell, so it only makes sense when the consensus cell matches that prior)."""
    driver = os.path.expanduser(driver)

    def run(frames, cell):
        if not frames:
            return {}
        with tempfile.TemporaryDirectory() as d:
            fin = os.path.join(d, "resid.frames")
            fout = os.path.join(d, "resid.out")
            write_frames_file(frames, fin)
            with open(fout, "w") as fo:
                subprocess.run([driver, fin, *map(str, extra_args)], stdout=fo, check=True)
            return parse_basis_out(fout)

    return run
