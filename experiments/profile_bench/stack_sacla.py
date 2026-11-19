"""SACLA Cheetah multi-event HDF5 (one /tag-N group per hit) -> a stacked (event, ss, fs) copy that
GLINT's --images route reads.

The copy is int16, one chunk per event, lzf-compressed: ~6.8 GB a run, ~265 GB for CXIDB 33
Dataset A. The pixels stay in ADU; CXIDB 33 is 10 ADU per photon, and
``glint --integrate --adu-per-photon 10`` converts them before integration.

    python stack_sacla.py RAW.h5 OUT.h5

Writes /entry_1/data_1/data (n, ss, fs) int16, /LCLS/photon_energy_eV (n,), /LCLS/tag (n,).

Why not a zero-copy virtual dataset (the first version of this script): read through h5py's
default chunk cache, HDF5 keeps every source frame it has touched open with its decompressed
chunk, ~14 MB per frame, ~17 GB for a 1217-frame run. Eight peak-finding jobs reached 82 GB.
"""
import argparse
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    a = ap.parse_args()
    src = os.path.abspath(a.raw)
    with h5py.File(src, "r") as f:
        tags = sorted((k for k in f if k.startswith("tag-")), key=lambda k: int(k[4:]))
        d0 = f[tags[0]]["data"]
        shape, dtype = d0.shape, d0.dtype
        eV = np.array([f[t]["photon_energy_ev"][()] for t in tags])
        g = h5py.File(a.out + ".part", "w")
        d = g.create_dataset("/entry_1/data_1/data", shape=(len(tags),) + shape, dtype=dtype,
                             chunks=(1,) + shape, compression="lzf")
        for i, t in enumerate(tags):
            d[i] = f[t]["data"][()]
        g["/LCLS/photon_energy_eV"] = eV
        g["/LCLS/tag"] = np.array([int(t[4:]) for t in tags])
        g.attrs["source"] = src
        g.close()
    os.replace(a.out + ".part", a.out)
    print(f"{a.out}: {len(tags)} events from {src}, {eV.mean():.1f} eV mean")


if __name__ == "__main__":
    main()
