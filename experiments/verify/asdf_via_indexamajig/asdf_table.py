#!/usr/bin/env python3
"""One table from the <arm>_summary.json files ../common/score_arm.py wrote. asdf has no 'xgandalf' profile token, so the
per-frame time is CrystFEL's own 'index' time (index_ms), one core, -j 1; wall_s is the whole arm."""
import glob, json, os, sys
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "*_summary.json"))):
    s = json.load(open(f))
    im = s.get("index_ms", {})
    rows.append((s["arm"], s["n"], s["returned"], s["lattice"], s["strict"], s["ge10"],
                 im.get("median"), im.get("mean"), s.get("time", "")))
with open(os.path.join(out, "asdf.md"), "w") as fo:
    fo.write("| arm | frames | returned | lattice | strict | ge10 | index ms median | index ms mean | wall |\n|" + "---|" * 9 + "\n")
    for r in rows:
        med = f"{r[6]:.0f}" if isinstance(r[6], (int, float)) else "-"
        mean = f"{r[7]:.0f}" if isinstance(r[7], (int, float)) else "-"
        fo.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {med} | {mean} | {r[8]} |\n")
print(open(os.path.join(out, "asdf.md")).read())
