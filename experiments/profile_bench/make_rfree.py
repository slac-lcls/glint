"""The R-free flags both arms share: one FreeR_flag per reciprocal-ASU index (Friedel mates share
one), CCP4 convention, integers 0..N-1 with 0 the test set. Covers the ASU to --dmin, past the data,
so every merge finds its flag. The same flags as ``rs.rfree -d 1.0 -r 0.1`` up to the random draw.

    python make_rfree.py -o r-free-flags.mtz [--cell 79.2 79.2 37.9 90 90 90 --sg "P 43 21 2"]
"""
import argparse

import gemmi
import numpy as np
import reciprocalspaceship as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--cell", nargs=6, type=float, default=[79.2, 79.2, 37.9, 90, 90, 90])
    ap.add_argument("--sg", default="P 43 21 2")
    ap.add_argument("--dmin", type=float, default=1.0)
    ap.add_argument("--nsets", type=int, default=10, help="test fraction is 1/nsets")
    ap.add_argument("--seed", type=int, default=2015)
    a = ap.parse_args()
    cell, sg = gemmi.UnitCell(*a.cell), gemmi.SpaceGroup(a.sg)
    hkl = rs.utils.generate_reciprocal_asu(cell, sg, a.dmin, anomalous=False)
    ds = rs.DataSet({"H": hkl[:, 0], "K": hkl[:, 1], "L": hkl[:, 2],
                     "FreeR_flag": np.random.default_rng(a.seed).integers(0, a.nsets, len(hkl))},
                    cell=cell, spacegroup=sg).infer_mtz_dtypes().set_index(["H", "K", "L"])
    ds.write_mtz(a.out)
    print(f"{a.out}: {len(ds)} reflections to {a.dmin} A, {np.mean(ds.FreeR_flag == 0):.3f} free")


if __name__ == "__main__":
    main()
