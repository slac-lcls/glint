#!/usr/bin/env python3
"""Analysis: the most common cell CrystFEL finds blind on r0278 and r0058, against GLINT's all-frame cell
(the vote of per_frame_accuracy_vs_peaks).

For each set, every crystal in the stream is a cell (load_stream: a copy of
experiments/score_crystfel.py's parse_stream at 95d51ce). The most common cell is found two ways:
  (1) glint.multishot.consensus_cell over the pooled cells, with the shipped batch gate
      (CONSENSUS_MIN_FRAC, CONSENSUS_MIN_LEAD) and also ungated -- a mode-finder over CrystFEL's cells;
  (2) a plain histogram of Buerger-reduced parameters rounded to 1 A and 1 deg, independent of GLINT's grouping.
Each is compared with GLINT's vote cell by glint.multishot.same_lattice (rtol 0.05, ctol 0.06, vtol 0.10) and
by the volume ratio. The scrambled sets are the null: their cells should form no dominant group.

    python cells.py OUTDIR    (reads OUTDIR/cf/<set>.stream and OUTDIR/inputs/<set>.txt; writes OUTDIR/crystfel_cell.json, crystfel_cell.md)
"""
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ["GLINT"])
from glint.glint_fast import load                                                      # noqa: E402
from glint.lattice import buerger_reduce, cell_params, cell_to_Ar, reduced_params       # noqa: E402
from glint.multishot import CONSENSUS_MIN_FRAC, CONSENSUS_MIN_LEAD, consensus_cell, same_lattice  # noqa: E402

# GLINT's all-frame cells, per_frame_accuracy_vs_peaks (95d51ce): reference.top1_vote of r0278_summary.json / r0058_summary.json
GLINT_CELL = {"r0278": (44.41, 70.60, 90.72, 88.40, 90.50, 88.95),
              "r0058": (43.87, 67.68, 89.02, 90.87, 88.15, 90.05)}


def load_stream(path, n):
    out, chunks, ev, a, b = {}, set(), None, None, None
    for ln in open(path):
        if ln.startswith("Event:"):
            mm = re.search(r"//(\d+)", ln); ev = int(mm.group(1)) if mm else None
            if ev is not None:
                chunks.add(ev)
        elif ln.startswith("astar ="):
            a = [float(x) for x in ln.split("=")[1].split()[:3]]
        elif ln.startswith("bstar ="):
            b = [float(x) for x in ln.split("=")[1].split()[:3]]
        elif ln.startswith("cstar ="):
            c = [float(x) for x in ln.split("=")[1].split()[:3]]
            if ev is not None and a and b:
                B = np.array([a, b, c]).T / 10.0
                out.setdefault(ev, []).append(np.linalg.inv(B).T)
            a = b = None
    cells = [out.get(i, []) for i in range(n)]
    return cells, len(chunks)


def params(M):
    """a, b, c, alpha, beta, gamma of the Buerger-reduced cell, and the volume."""
    M = np.asarray(M, float)
    return [round(float(x), 2) for x in cell_params(buerger_reduce(M))] + [round(float(abs(np.linalg.det(M))), 1)]


def fingerprint(M):
    """Sorted reduced edges (A) and the angles of the sorted |cos| (deg), rounded to 1 A and 1 deg."""
    lens, cs = reduced_params(np.asarray(M, float))
    return (tuple(int(round(float(x))) for x in lens)
            + tuple(int(round(float(np.degrees(np.arccos(min(float(c), 1.0)))))) for c in cs))


def analyse(outdir, s):
    run = s.split("_")[0]
    n = len(load(os.path.join(outdir, "inputs", s + ".txt")))
    cells, nproc = load_stream(os.path.join(outdir, "cf", s + ".stream"), n)
    pool = [M for cs in cells for M in cs]
    ref = cell_to_Ar(*GLINT_CELL[run])
    r = dict(set=s, frames=n, frames_processed=nproc, frames_indexed=sum(1 for cs in cells if cs),
             cells_pooled=len(pool), glint_cell=list(GLINT_CELL[run]))
    for tag, kw in (("gated", dict(min_frac=CONSENSUS_MIN_FRAC, min_lead=CONSENSUS_MIN_LEAD)), ("ungated", {})):
        Mb, sup = consensus_cell(pool, **kw) if pool else (None, 0)
        r[tag] = dict(support=int(sup or 0), support_frac=round((sup or 0) / max(len(pool), 1), 4),
                      cell=(params(Mb) if Mb is not None else None),
                      same_as_glint=(bool(same_lattice(Mb, ref)) if Mb is not None else None),
                      vol_ratio=(round(abs(np.linalg.det(Mb)) / abs(np.linalg.det(ref)), 4) if Mb is not None else None))
    fps = [fingerprint(M) for M in pool]
    hist = collections.Counter(fps)
    if hist:
        top, cnt = hist.most_common(1)[0]
        rep = pool[fps.index(top)]
        near = [M for M in pool if same_lattice(M, rep)]
        med = np.median(np.array([fingerprint(M) for M in near], float), axis=0)
        r["histogram"] = dict(mode_edges_A_and_angles_deg=list(top), mode_count=int(cnt),
                              same_lattice_as_mode=len(near), median_of_those=[round(float(x), 2) for x in med],
                              mode_cell=params(rep), mode_same_as_glint=bool(same_lattice(rep, ref)))
    r["glint_cell_count"] = sum(1 for M in pool if same_lattice(M, ref))
    return r


def main():
    outdir = sys.argv[1]
    res = [analyse(outdir, s) for s in os.environ["SETS"].split()]
    json.dump(dict(job="r0278_r0058_crystfel_cell", commit=open(os.path.join(os.environ["GLINT"], "COMMIT")).read().strip(),
                   crystfel=os.environ["CF"], results=res), open(os.path.join(outdir, "crystfel_cell.json"), "w"), indent=1)
    with open(os.path.join(outdir, "crystfel_cell.md"), "w") as f:
        f.write("| set | frames | indexed | cells | gated: support, cell (reduced a b c al be ga, V) | = GLINT's? | vol ratio |"
                " ungated: support, = GLINT's? | histogram mode (count) | cells = GLINT's |\n|" + "---|" * 10 + "\n")
        for r in res:
            g, u, h = r["gated"], r["ungated"], r.get("histogram", {})
            f.write(f"| {r['set']} | {r['frames']} | {r['frames_indexed']} | {r['cells_pooled']} | "
                    f"{g['support']} ({100 * g['support_frac']:.1f}%), {g['cell']} | {g['same_as_glint']} | {g['vol_ratio']} | "
                    f"{u['support']}, {u['same_as_glint']} | {h.get('mode_edges_A_and_angles_deg')} ({h.get('mode_count')}; {h.get('same_lattice_as_mode')} same lattice; = GLINT's: {h.get('mode_same_as_glint')}) | "
                    f"{r['glint_cell_count']} |\n")
    print(open(os.path.join(outdir, "crystfel_cell.md")).read())


if __name__ == "__main__":
    main()
