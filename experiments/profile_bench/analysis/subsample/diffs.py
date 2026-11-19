"""Paired differences (arm - CrystFEL) per shell and merge size, from per-draw rows."""
import json, math, sys
import numpy as np
base = "/pscratch/sd/k/kmdalton/cxidb_45/work/probe/subsample"
dd = json.load(open(f"{base}/deck_data.json"))
out = {}
for ds in ["cxidb45", "cxidb33"]:
    rows = json.load(open(f"{base}/{ds}/all.json"))["rows"]
    sizes = dd[ds]["sizes"]; ns = len(dd[ds]["edges"]) - 1
    idx = {(r["arm"], r["size"], r["rep"]): r for r in rows}
    out[ds] = {"sizes": sizes, "edges": dd[ds]["edges"]}
    for met in ["cchalf", "ccano_w", "ccano_rank"]:
        for arm in ["box", "profile"]:
            M, T = [], []
            for n in sizes:
                reps = sorted(k[2] for k in idx if k[0] == arm and k[1] == n)
                col_m, col_t = [], []
                for s in range(ns):
                    d = []
                    for r in reps:
                        a, c = idx[(arm, n, r)][met][s], idx[("crystfel", n, r)][met][s]
                        if a is not None and c is not None and not (math.isnan(a) or math.isnan(c)):
                            d.append(a - c)
                    if len(d) == 0:
                        col_m.append(None); col_t.append(None); continue
                    m = float(np.mean(d))
                    t = None
                    if len(d) >= 3:
                        sd = float(np.std(d, ddof=1))
                        t = m / (sd / math.sqrt(len(d))) if sd > 0 else None
                    col_m.append(round(m, 3)); col_t.append(None if t is None else round(t, 2))
                M.append(col_m); T.append(col_t)
            out[ds][f"{met}_{arm}"] = {"m": M, "t": T}
json.dump(out, open(f"{base}/diff/diffs.json", "w"))
for ds in out:
    for k, v in out[ds].items():
        if k in ("sizes", "edges"): continue
        print(ds, k)
        for c, (m, t) in enumerate(zip(v["m"], v["t"])):
            print("  ", out[ds]["sizes"][c], " ".join(f"{x:+.3f}{'*' if (tt is not None and abs(tt) > 2.776) else ' '}" if x is not None else "  --   " for x, tt in zip(m, t)))
