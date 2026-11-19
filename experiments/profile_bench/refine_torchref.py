"""Refine the reference model against one arm's abismal merge with torchref.

The protocol of the cxidb_81 abismal benchmarks, which used phenix.refine: B reset to 20, one
rigid-body pass, then 5 macro cycles of scale + individual isotropic ADP. Coordinates stay fixed
past the rigid body, so the two arms differ only in how well their data fit one model.
Anomalous scattering is on (--wavelength), and F(+)/F(-) are read as Bijvoet pairs, so the
refined MTZ carries ANOM/PANOM, the anomalous difference map.

    python refine_torchref.py --mtz asu_0_epoch_30.mtz --flags r-free-flags.mtz --pdb 4YOP.pdb \\
        --wavelength 1.7684 --dmin 1.9 -o OUTDIR

Writes OUTDIR/data.mtz (the merge with the shared FreeR_flag), start.pdb, refined.pdb,
refined.mtz and stats.json (R-work/R-free at the start, after rigid body, and at the end).
"""
import argparse
import json
import os

import gemmi
import reciprocalspaceship as rs
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtz", required=True, help="abismal output: F(+),SIGF(+),F(-),SIGF(-)")
    ap.add_argument("--flags", required=True, help="FreeR_flag MTZ from make_rfree.py")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--wavelength", type=float, required=True)
    ap.add_argument("--dmin", type=float, default=None)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--b-start", type=float, default=20.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("-o", "--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = lambda name: os.path.join(a.out_dir, name)

    ds = rs.read_mtz(a.mtz)
    keep = [c for c in ("F(+)", "SIGF(+)", "F(-)", "SIGF(-)") if c in ds.columns]
    flags = rs.read_mtz(a.flags)[["FreeR_flag"]]
    ds = ds[keep].join(flags, how="left")
    if ds["FreeR_flag"].isna().any():
        raise SystemExit(f"{ds['FreeR_flag'].isna().sum()} reflections have no flag in {a.flags}")
    ds["FreeR_flag"] = ds["FreeR_flag"].astype("I")
    ds.write_mtz(out("data.mtz"))

    st = gemmi.read_structure(a.pdb)
    st.remove_hydrogens()
    for m in st:
        for ch in m:
            for res in ch:
                for at in res:
                    at.b_iso = a.b_start
    st.write_pdb(out("start.pdb"))

    from torchref.refinement.lbfgs_refinement import LBFGSRefinement
    from torchref.cli._common import build_column_names, parse_device_str

    ref = LBFGSRefinement(data_file=out("data.mtz"), pdb=out("start.pdb"), max_res=a.dmin,
                          device=parse_device_str(a.device), wavelength=a.wavelength,
                          column_names=build_column_names(out("data.mtz"), None, None))
    stats = {"n_work": int(ref.reflection_data.work.n), "n_free": int(ref.reflection_data.free.n)}

    def r(tag):
        with torch.no_grad():
            rw, rf = ref.get_rfactor()
        stats[tag] = {"rwork": float(rw), "rfree": float(rf)}
        print(f"{tag}: Rwork {float(rw):.4f}  Rfree {float(rf):.4f}", flush=True)

    ref.get_scales()
    r("start")
    ref.refine_rigid_body(commit=True)
    r("rigid_body")
    for c in range(a.cycles):
        ref._refresh_scales()
        ref.refine_adp()
        r(f"cycle_{c + 1}")
    ref.get_scales()
    r("final")
    ref.model.write_pdb(out("refined.pdb"))
    ref.write_out_mtz(out("refined.mtz"), anomalous=True)
    with open(out("stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
