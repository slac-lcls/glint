"""Collect the integrator benchmark into one summary.json.

  * throughput and counts per arm: frames, reflections, peak-find / index / integrate seconds;
  * merge-free accuracy: per resolution shell, the Pearson CC between each arm's single-shot
    (partial) intensities and |Fcalc|^2 from the reference model. Partiality spreads both arms
    the same way, so the difference between arms is the integrator's;
  * the arms against each other on the observations both kept: CC of I, median sigma ratio;
  * I/sigma per shell;
  * torchref final Rwork/Rfree (refine_torchref.py) and the anomalous peak heights (anom_peaks.py).

    python report.py --work WORK --integrators box,profile --runs r1 r2 ... --pdb 4YOP.pdb -o summary.json
"""
import argparse
import glob
import itertools
import json
import os

import gemmi
import numpy as np
import reciprocalspaceship as rs

from hkl_to_mtz import reindex


def read_stream(path, max_chunks=None):
    """GLINT .stream -> (chunk, h, k, l, I, sigma) float32 array. `chunk` counts indexed lattices from 0
    in file order, not stream chunks: GLINT writes a chunk for every frame, CrystFEL here only for indexed
    ones, and the arms are paired on this number.

    Reflection lines are collected as text and parsed in one np.loadtxt per file: a Python tuple per
    reflection needs ~20x the memory, and the two arms of Dataset A hold 83 M of them.
    """
    lines, chunks = [], []
    chunk = -1
    inref = False
    with open(path) as f:
        for ln in f:
            if ln.startswith("Reflections measured after indexing"):
                chunk += 1
                if max_chunks and chunk >= max_chunks:
                    break
                inref = True
                next(f)                                  # the column header
            elif ln.startswith("End of reflections"):
                inref = False
            elif inref:
                lines.append(ln)
                chunks.append(chunk)
    if not lines:
        return np.zeros((0, 6), np.float32)
    x = np.loadtxt(lines, usecols=(0, 1, 2, 3, 4), dtype=np.float32, ndmin=2)
    return np.column_stack([np.asarray(chunks, np.float32), x])


def _hkl_key(H):
    return (H[:, 0].astype(np.int64) + 512) * 1048576 + (H[:, 1] + 512) * 1024 + (H[:, 2] + 512)


def fcalc_lookup(pdb, cell, sg, dmin):
    """|Fcalc|^2 on the reciprocal ASU to dmin, as (sorted ASU keys, |Fcalc|^2) for _lookup."""
    st = gemmi.read_structure(pdb)
    st.setup_entities()
    calc = gemmi.StructureFactorCalculatorX(st.cell)
    hkl = rs.utils.generate_reciprocal_asu(cell, sg, dmin, anomalous=False)
    f2 = np.array([abs(calc.calculate_sf_from_model(st[0], [int(x) for x in h])) ** 2 for h in hkl])
    k = _hkl_key(hkl)
    o = np.argsort(k)
    return k[o], f2[o]


def _lookup(table, H):
    keys, f2 = table
    k = _hkl_key(H)
    i = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
    return np.where(keys[i] == k, f2[i], np.nan)


def shells(d, n=8):
    edges = np.quantile(1 / d ** 2, np.linspace(0, 1, n + 1))
    return np.clip(np.searchsorted(edges, 1 / d ** 2, side="right") - 1, 0, n - 1), 1 / np.sqrt(edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--integrators", default="box,profile")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--refine-dir", default=None, help="default: WORK/refine")
    ap.add_argument("--cell", nargs=6, type=float, default=[79.2, 79.2, 37.9, 90, 90, 90])
    ap.add_argument("--sg", default="P 43 21 2")
    ap.add_argument("--dmin", type=float, default=1.9)
    ap.add_argument("--reindex", default="h,k,l", help="stream indices -> the --cell/--sg (model) setting")
    ap.add_argument("--max-chunks", type=int, default=0, help="per run, for the per-observation stats")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    integs = a.integrators.split(",")
    cell, sg = gemmi.UnitCell(*a.cell), gemmi.SpaceGroup(a.sg)
    out = {"runs": a.runs, "integrators": integs}

    # --- throughput -------------------------------------------------------------------------
    tot = {"t_peakfind_s": 0.0, "t_index_s": 0.0, "n_hits_with_peaks": 0, "n_indexed": 0}
    for i in integs:
        tot[i] = {"t_integrate_s": 0.0, "n_frames": 0, "n_refl": 0}
    for r in a.runs:
        j = json.load(open(f"{a.work}/streams/{r}.json"))
        for k in ("t_peakfind_s", "t_index_s", "n_hits_with_peaks", "n_indexed"):
            tot[k] += j[k]
        for i in integs:
            ji = j[i] if i in j else json.load(open(f"{a.work}/streams/{i}/{r}.json"))
            for k in tot[i]:
                tot[i][k] += ji[k]
    for i in integs:
        tot[i]["ms_per_frame"] = 1e3 * tot[i]["t_integrate_s"] / max(tot[i]["n_frames"], 1)
        tot[i]["refl_per_frame"] = tot[i]["n_refl"] / max(tot[i]["n_frames"], 1)
    out["throughput"] = tot

    # --- per-observation accuracy against the model --------------------------------------------
    F2 = fcalc_lookup(a.pdb, cell, sg, a.dmin)
    obs = {}
    for i in integs:
        cols = {k: [] for k in ("run", "chunk", "H", "hkl", "I", "sig", "d", "f2")}
        for ri, r in enumerate(a.runs):        # per run: hkl_to_asu expands every symop in int64
            x = read_stream(f"{a.work}/streams/{i}/{r}.stream", a.max_chunks or None)
            if not len(x):
                continue
            hkl = reindex(x[:, 1:4], a.reindex).astype(np.int16)
            H = rs.utils.hkl_to_asu(hkl.astype(int), sg)[0]
            cols["run"].append(np.full(len(x), ri, np.int16))
            cols["chunk"].append(x[:, 0].astype(np.int32))
            cols["hkl"].append(hkl)
            cols["H"].append(H.astype(np.int16))
            cols["I"].append(x[:, 4])
            cols["sig"].append(x[:, 5])
            cols["d"].append(cell.calculate_d_array(H).astype(np.float32))
            cols["f2"].append(_lookup(F2, H).astype(np.float32))
        obs[i] = {k: np.concatenate(v) for k, v in cols.items()}
    ref_d = np.concatenate([obs[i]["d"] for i in integs])
    _, edges = shells(ref_d[ref_d >= a.dmin])
    def shell_of(d):
        e = 1 / edges ** 2
        return np.clip(np.searchsorted(e, 1 / d ** 2, side="right") - 1, 0, len(edges) - 2)
    acc = {"d_edges": [round(float(e), 3) for e in edges]}
    for i in integs:
        o = obs[i]
        ok = np.isfinite(o["f2"]) & (o["d"] >= a.dmin) & (o["sig"] > 0)
        sh = shell_of(o["d"][ok])
        I, s, f2 = o["I"][ok], o["sig"][ok], o["f2"][ok]
        acc[i] = {
            "cc_vs_fcalc2": [float(np.corrcoef(I[sh == b], f2[sh == b])[0, 1]) for b in range(len(edges) - 1)],
            "cc_vs_fcalc2_overall": float(np.corrcoef(I, f2)[0, 1]),
            "median_isigi": [float(np.median(I[sh == b] / s[sh == b])) for b in range(len(edges) - 1)],
            "frac_isigi_gt3": [float(np.mean(I[sh == b] / s[sh == b] > 3)) for b in range(len(edges) - 1)],
            "n_obs": int(ok.sum()),
        }
    out["accuracy"] = acc

    # --- the arms against each other, on the observations both kept ----------------------------
    key = lambda o: ((o["run"].astype(np.int64) * 1000000 + o["chunk"]) * 1000000
                     + (o["hkl"].astype(np.int64) + 50) @ np.array([10000, 100, 1]))
    out["paired"] = {}
    for x, y in itertools.combinations(integs, 2):
        A, B = obs[x], obs[y]
        ka, kb = key(A), key(B)
        common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
        out["paired"][f"{x}_{y}"] = {
            "n_common": int(len(common)), f"n_only_{x}": int(len(ka) - len(common)),
            f"n_only_{y}": int(len(kb) - len(common)),
            "cc_I": float(np.corrcoef(A["I"][ia], B["I"][ib])[0, 1]),
            f"mean_I_ratio_{y}_over_{x}": float(np.mean(B["I"][ib]) / np.mean(A["I"][ia])),
            f"median_sigma_ratio_{y}_over_{x}": float(np.median(B["sig"][ib] / A["sig"][ia])),
        }

    # --- refinement and anomalous peaks -------------------------------------------------------
    ref = {}
    rdir = a.refine_dir or f"{a.work}/refine"
    for i in integs:
        sp = f"{rdir}/{i}/stats.json"
        st = json.load(open(sp)) if os.path.exists(sp) else {}
        ref[i] = {"rwork": st.get("final", {}).get("rwork"), "rfree": st.get("final", {}).get("rfree"),
                  "start_rwork": st.get("start", {}).get("rwork"),
                  "start_rfree": st.get("start", {}).get("rfree")}
        pk = f"{rdir}/{i}/anom_peaks.json"
        if os.path.exists(pk):
            ref[i]["anom"] = json.load(open(pk))
    out["refinement"] = ref
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
