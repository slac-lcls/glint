"""Anomalous difference peak heights at the anomalous scatterers of a refined model.

Reads the refined MTZ's anomalous map coefficients (torchref's ANOM/PANOM; phenix's are
ANOM/PHANOM), FFTs them to a map in units of its own RMS, and reports the maximum within --radius A of each S (Cys SG, Met SD), Se (selenomethionine SE), Cl and Na. As the null it
also reports the highest peak more than 3 A from every one of those atoms. A site's height means
nothing without the height of the best peak that is not a site.

    python anom_peaks.py refined.mtz refined.pdb -o peaks.json
"""
import argparse
import json

import gemmi
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mtz")
    ap.add_argument("pdb")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--f", default="ANOM")
    ap.add_argument("--phi", default="PANOM")
    a = ap.parse_args()

    mtz = gemmi.read_mtz_file(a.mtz)
    grid = mtz.transform_f_phi_to_map(a.f, a.phi, sample_rate=4)
    arr = np.array(grid, copy=False)
    rms = float(np.sqrt(np.mean(arr.astype(float) ** 2)))
    grid.normalize()                                   # zero mean, unit RMS: values are in sigma

    st = gemmi.read_structure(a.pdb)
    st.setup_entities()
    st.setup_cell_images()                             # symmetry mates for the distance test below
    sites = []
    for ch in st[0]:
        for res in ch:
            for at in res:
                if (res.name == "CYS" and at.name == "SG") or (res.name == "MET" and at.name == "SD") \
                        or (res.name == "MSE" and at.name == "SE") \
                        or at.element.name in ("Cl", "Na"):
                    sites.append((f"{ch.name}/{res.name}{res.seqid.num}/{at.name}", at.pos))
    out = {"map_rms": rms, "sites": {}}
    for name, pos in sites:
        best = -np.inf
        for p in _sphere(pos, a.radius, grid):
            best = max(best, grid.interpolate_value(p))
        out["sites"][name] = round(float(best), 2)

    # null: the highest grid point further than 3 A from every site, over the asymmetric unit
    frac = np.argwhere(np.array(grid) > 3.0)
    null = -np.inf
    site_pos = [p for _, p in sites]
    for idx in frac:
        p = grid.point_to_position(grid.get_point(*map(int, idx)))
        if all(st.cell.find_nearest_image(s, p, gemmi.Asu.Any).dist() > 3.0 for s in site_pos):
            null = max(null, float(grid.get_value(*map(int, idx))))
    out["max_non_site_peak"] = round(null, 2) if np.isfinite(null) else None
    s_heights = [v for k, v in out["sites"].items() if k.endswith(("/SG", "/SD"))]
    out["mean_S_height"] = round(float(np.mean(s_heights)), 2) if s_heights else None
    se_heights = [v for k, v in out["sites"].items() if k.endswith("/SE")]
    out["mean_Se_height"] = round(float(np.mean(se_heights)), 2) if se_heights else None
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def _sphere(center, r, grid, step=0.25):
    """Points on a cubic lattice of spacing ``step`` inside a sphere of radius r about center."""
    g = np.arange(-r, r + 1e-9, step)
    for x in g:
        for y in g:
            for z in g:
                if x * x + y * y + z * z <= r * r:
                    yield gemmi.Position(center.x + x, center.y + y, center.z + z)


if __name__ == "__main__":
    main()
