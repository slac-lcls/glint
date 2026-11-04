"""Score one arm, frame by frame, at the paper's gate; write <out>/<arm>_frames.csv and
<out>/<arm>_summary.json.

The recorded scorer of the asdf and MOSFLM checks (md5 d4182cf73c292b4e4c22d907e85d260e), logic unchanged;
the pinned GLINT tree is read from VERIFY_GLINT instead of a fixed path.

The gate is the one score_xg_gate.py and score_crystfel.py apply at 95d51ce: correct lattice
(glint.multishot.same_lattice against the textbook lysozyme cell) and, for the strict bar, at least
GATE_FRAC of the frame's peaks within GATE_TOL of integer hkl and at least GATE_MIN of them
(constants from glint.glint_fast). Three bars are reported per frame:
  lattice  correct lattice, no coverage requirement   (the paper's "lattice" column)
  strict   correct lattice AND >= 25% AND >= 10        (the paper's "indexed" column)
  ge10     correct lattice AND >= 10 indexed peaks     (reported only; at chance on dense frames)

  --kind xg      result = XGANDALF driver stdout ("id b00..b22 ms" | "id NONE ms"); parsing as in
                 experiments/xgandalf/score_xg_gate.py:parse_xg. ms = the indexer.index() call.
  --kind stream  result = indexamajig stream; parsing as in experiments/score_crystfel.py
                 (astar/bstar/cstar in nm^-1 -> real basis in A); --profile = indexamajig stdout
                 under --profile with -j 1, one line per image in event-list order; --events = the
                 event list, so line i of the profile is event i.
"""
import argparse
import json
import os
import re
import statistics as st
import sys

import numpy as np

PIN = os.environ["VERIFY_GLINT"]   # the pinned GLINT tree (../README.md); no default
sys.path.insert(0, PIN)
from glint.glint_fast import GATE_TOL, GATE_FRAC, GATE_MIN, load  # noqa: E402
from glint.lattice import cell_to_Ar                                # noqa: E402
from glint.multishot import same_lattice                            # noqa: E402

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)


def gate(M, q):
    """(lattice, strict, ge10, n_matched): score_xg_gate.gate with the matched count exposed."""
    if M is None or not same_lattice(M, LYSO):
        return 0, 0, 0, 0
    H = q @ M
    m = int((np.abs(H - np.rint(H)).max(1) < GATE_TOL).sum())
    return 1, int(m / len(q) >= GATE_FRAC and m >= GATE_MIN), int(m >= GATE_MIN), m


def parse_xg(path):
    res = {}
    for line in open(path):
        p = line.split()
        if len(p) < 3 or line.startswith("#"):
            continue
        if p[1] == "NONE":
            res[int(p[0])] = (None, float(p[2]))
        else:
            B = np.array(list(map(float, p[1:10]))).reshape(3, 3)
            M = np.linalg.inv(B).T if np.median(np.linalg.norm(B, axis=0)) < 1 else B
            res[int(p[0])] = (M, float(p[10]))
    return res


def parse_stream(path):
    out, by = {}, {}
    ev = a = b = c = None
    for ln in open(path):
        if ln.startswith("Event:"):
            mm = re.search(r"//(\d+)", ln)
            ev = int(mm.group(1)) if mm else None
        elif ln.startswith("indexed_by =") and ev is not None:
            by[ev] = ln.split("=", 1)[1].strip()
        elif ln.startswith("astar ="):
            a = [float(x) for x in ln.split("=")[1].split()[:3]]
        elif ln.startswith("bstar ="):
            b = [float(x) for x in ln.split("=")[1].split()[:3]]
        elif ln.startswith("cstar ="):
            c = [float(x) for x in ln.split("=")[1].split()[:3]]
            if ev is not None and a and b and ev not in out:      # first crystal only (no --multi)
                B = np.array([a, b, c]).T / 10.0
                out[ev] = np.linalg.inv(B).T
            a = b = c = None
    return out, by


TOK = re.compile(r"\(([A-Za-z0-9_-]+) ([0-9.]+)")


def parse_profile(path):
    rows = []
    for ln in open(path):
        if "(root" not in ln:
            continue
        agg, ncall = {}, 0
        for name, t in TOK.findall(ln):
            agg[name] = agg.get(name, 0.0) + float(t)
            ncall += name == "xgandalf"
        rows.append({"process_image_ms": 1e3 * agg.get("process-image", float("nan")),
                     "index_ms": 1e3 * agg.get("index", float("nan")),
                     "xgandalf_ms": 1e3 * agg.get("xgandalf", 0.0),
                     "xgandalf_calls": ncall,
                     "cellcheck_ms": 1e3 * (agg.get("prerefine-cell-check", 0.0) + agg.get("postrefine-cell-check", 0.0)),
                     "refine_ms": 1e3 * agg.get("refine", 0.0)})
    return rows


def pct(x, p):
    return float(np.percentile(np.asarray(x, float), p)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--kind", choices=["xg", "stream"], required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--frames", required=True, help="FRAME-block q list the gate scores against")
    ap.add_argument("--profile")
    ap.add_argument("--events")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="{}", help="JSON merged into the summary (argv, host, ...)")
    a = ap.parse_args()

    frames = [np.asarray(q, float) for q in load(a.frames)]
    rows = []
    if a.kind == "xg":
        res = parse_xg(a.result)
        for fid in sorted(res):
            M, ms = res[fid]
            lat, strict, g10, m = gate(M, frames[fid])
            rows.append(dict(frame=fid, npk=len(frames[fid]), returned=int(M is not None),
                             lattice=lat, strict=strict, ge10=g10, matched=m, ms=ms))
    else:
        S, by = parse_stream(a.result)
        ev = [int(re.search(r"//(\d+)", ln).group(1)) for ln in open(a.events) if "//" in ln]
        prof = parse_profile(a.profile) if a.profile else []
        if prof and len(prof) != len(ev):
            raise SystemExit(f"{a.arm}: {len(prof)} profile lines for {len(ev)} events; "
                             "with -j 1 they must match one to one")
        for i, fid in enumerate(ev):
            M = S.get(fid)
            lat, strict, g10, m = gate(M, frames[fid])
            r = dict(frame=fid, npk=len(frames[fid]), returned=int(M is not None),
                     lattice=lat, strict=strict, ge10=g10, matched=m, indexed_by=by.get(fid, ""))
            if prof:
                r.update(prof[i])
                r["ms"] = r["xgandalf_ms"]            # the library calls, retries included
            rows.append(r)

    os.makedirs(a.out, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with open(os.path.join(a.out, f"{a.arm}_frames.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    n = len(rows)
    summ = {"arm": a.arm, "kind": a.kind, "result": os.path.abspath(a.result),
            "frames": os.path.abspath(a.frames), "n": n,
            "gate": {"tol": GATE_TOL, "frac": GATE_FRAC, "min": GATE_MIN, "cell": "79.02 79.02 37.98 90 90 90"},
            "returned": sum(r["returned"] for r in rows), "lattice": sum(r["lattice"] for r in rows),
            "strict": sum(r["strict"] for r in rows), "ge10": sum(r["ge10"] for r in rows)}
    for col in ("ms", "index_ms", "process_image_ms", "cellcheck_ms", "refine_ms"):
        v = [r[col] for r in rows if col in r]
        if v:
            summ[col] = {"mean": st.mean(v), "median": st.median(v), "p10": pct(v, 10), "p90": pct(v, 90),
                         "sum_s": sum(v) / 1e3}
    if a.kind == "stream":
        summ["xgandalf_calls"] = sum(r.get("xgandalf_calls", 0) for r in rows)
        summ["indexed_by"] = sorted({r["indexed_by"] for r in rows if r["indexed_by"]})
    summ.update(json.loads(a.meta))
    with open(os.path.join(a.out, f"{a.arm}_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    ms = summ.get("ms", {})
    print(f"{a.arm:22s} n={n} returned={summ['returned']} lattice={summ['lattice']}/{n} "
          f"strict={summ['strict']}/{n} ge10={summ['ge10']}/{n} "
          f"ms mean={ms.get('mean', float('nan')):.0f} median={ms.get('median', float('nan')):.0f}")


if __name__ == "__main__":
    main()
