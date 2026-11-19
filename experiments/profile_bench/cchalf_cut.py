"""Resolution limit from partialator's half-datasets: where CC1/2 first falls below --min.

Shells hold --shell reflections each (Bijvoet mates counted apart, as they were merged), walked out
from low resolution; an arm's limit is the outer edge of its last shell before the first failure.
The arms are compared on the same reflections, so `common` is the lowest-resolution of their limits,
and it is the one the workflow cuts at.

    python cchalf_cut.py WORK/merge --integrators box,profile --cell 79.2 79.2 37.9 90 90 90 --sg "P 43 21 2" -o cut.json
"""
import argparse
import json

import gemmi
import numpy as np
import reciprocalspaceship as rs

from hkl_to_mtz import read_hkl, reindex


def load(path, cell, sg, op="h,k,l"):
    x = read_hkl(path)
    H = reindex(x[:, :3], op)
    ds = rs.DataSet({"H": H[:, 0], "K": H[:, 1], "L": H[:, 2],
                     "I": x[:, 3], "SIGI": x[:, 4]}, cell=cell, spacegroup=sg).infer_mtz_dtypes().set_index(["H", "K", "L"])
    ds = ds[ds["SIGI"] > 0].hkl_to_asu(anomalous=True).remove_absences()
    ds.merged = True
    return ds[["I"]]


def shell_cchalf(a, b, n):
    """(outer d of each shell, its CC1/2), low resolution first, on the indices in both halves."""
    j = a.join(b, rsuffix="2", how="inner").dropna().compute_dHKL()
    j = j.sort_values("dHKL", ascending=False)
    d, i1, i2 = j["dHKL"].to_numpy(float), j["I"].to_numpy(float), j["I2"].to_numpy(float)
    k = max(len(d) // n, 1)
    return [(float(d[s[-1]]), float(np.corrcoef(i1[s], i2[s])[0, 1])) for s in np.array_split(np.arange(len(d)), k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merge_dir")
    ap.add_argument("--integrators", default="box,profile")
    ap.add_argument("--cell", nargs=6, type=float, required=True)
    ap.add_argument("--sg", required=True)
    ap.add_argument("--min", type=float, default=0.3)
    ap.add_argument("--reindex", default="h,k,l", help="merge indices -> the --cell/--sg setting")
    ap.add_argument("--shell", type=int, default=300)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    cell, sg = gemmi.UnitCell(*a.cell), gemmi.SpaceGroup(a.sg)
    out = {"min_cchalf": a.min, "shell": a.shell, "arms": {}, "shells": {}}
    for arm in a.integrators.split(","):
        sh = shell_cchalf(load(f"{a.merge_dir}/{arm}/merged.hkl1", cell, sg, a.reindex),
                          load(f"{a.merge_dir}/{arm}/merged.hkl2", cell, sg, a.reindex), a.shell)
        n_ok = next((i for i, (_, cc) in enumerate(sh) if not cc >= a.min), len(sh))
        if n_ok == 0:
            raise SystemExit(f"{arm}: CC1/2 is below {a.min} in the lowest-resolution shell")
        out["arms"][arm] = round(sh[n_ok - 1][0], 3)
        out["shells"][arm] = [[round(d, 3), round(cc, 3)] for d, cc in sh]
    out["common"] = max(out["arms"].values())
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(out["arms"], "common", out["common"])


if __name__ == "__main__":
    main()
