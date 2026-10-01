#!/usr/bin/env python3
"""Derive every number in lute/STATUS.md's Jungfrau common-mode section from the run log.

common_mode_ab.py is the script that made the measurement, committed verbatim (md5 baa3c89e...).
Its output, common_mode_ab_run200.log, is the record: one JSON line per event plus a SUMMARY
block. This reads the log and nothing else -- no psana, no data -- so the numbers in STATUS.md and
in common_mode_ab_result.json can be regenerated and checked on any machine:

    python common_mode_ab_summary.py                      # print the summary JSON
    python common_mode_ab_summary.py -o common_mode_ab_result.json
    python common_mode_ab_summary.py --check              # exit 1 if the committed JSON is stale

To re-run the measurement itself (psana2 env, S3DF; the script imports peakfinder_v4 flat, and
glint/peakfinder_v4.py is the file it ran with, byte for byte):

    PYTHONPATH=../../glint python common_mode_ab.py 200 > common_mode_ab_run200.log
"""
import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOG = HERE / "common_mode_ab_run200.log"
SCRIPT = HERE / "common_mode_ab.py"
RESULT = HERE / "common_mode_ab_result.json"
ARM = "doc_default_7_3_200_10"
HIT = 6          # common_mode_ab.py's hit test: len(peaks) >= 6
DENSE = 20       # "events with >= 20 peaks" (arm A count)


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _q(a, p):
    return round(float(np.percentile(a, p)), 4)


def summarize(log_path):
    text = log_path.read_text().splitlines()
    rows, head, snr = [], {}, {}
    m = re.match(r"host (\S+) psana (\S+) numpy (\S+)", text[0])
    if m:
        head = {"host": m.group(1), "psana": m.group(2), "numpy": m.group(3)}
    for line in text:
        if line.startswith('{"ev"'):
            rows.append(json.loads(line))
        elif line.startswith("calibconst keys:"):
            head["calibconst_keys"] = ast.literal_eval(line.split(":", 1)[1].strip())
        elif line.startswith("shape"):
            head["good_frac"] = float(line.split("good frac")[1])
        elif line.startswith(f"cache {ARM}"):
            head["cmps"] = [int(x) for x in re.search(r"cmps = \(([^)]*)\)", line).group(1).split(",")]
        s = re.match(r"\s+(matched|onlyA|onlyC)\s+n=\s*(\d+) snr median\s+([\d.]+) p10\s+([\d.]+)"
                     r" frac snr<30 ([\d.]+)\s+frac within 4px of bank edge ([\d.]+)", line)
        if s:
            snr[s.group(1)] = {"n": int(s.group(2)), "snr_median": float(s.group(3)),
                               "snr_p10": float(s.group(4)), "frac_snr_below_30": float(s.group(5)),
                               "frac_within_4px_of_bank_edge": float(s.group(6))}
    if not rows:
        raise SystemExit(f"no per-event records in {log_path}")
    C = [r[ARM] for r in rows]
    frac = np.array([c["frac_px_changed"] for c in C])
    dmed = np.array([c["absdiff_median_changed"] for c in C])
    dmax = np.array([c["absdiff_max"] for c in C])
    nA = np.array([r["nA"] for r in rows]); nC = np.array([c["nC"] for c in C])
    mt = np.array([c["matched"] for c in C])
    hitA, hitC = nA >= HIT, nC >= HIT
    assert all(bool(c["hitA"]) == bool(h) for c, h in zip(C, hitA)), "log hit flags disagree with >= 6"

    def jac(sel):
        u = nA[sel] + nC[sel] - mt[sel]
        return np.where(u > 0, mt[sel] / np.maximum(u, 1), 1.0)

    dense, hits = nA >= DENSE, hitA | hitC
    tot = int(nA.sum()), int(nC.sum()), int(mt.sum())
    return {
        "experiment": "mfx101555026", "run": 13, "detector": "jungfrau",
        "events": len(rows), "cmpars_arm_C": head.get("cmps"),
        "arm_A": "det.raw.calib(evt)  (no cmpars: what xtc_qreader.py calls)",
        "arm_C": "det.raw.calib(evt, cmpars=(7, 3, 200, 10))",
        "finder": {"code": "glint/peakfinder_v4.py @ b91ce63, numpy path, per panel, float32",
                   "min_pix": 3, "son_min": 15.0, "thr_high": 10.0, "thr_low": 5.0,
                   "mask": "pixel_status == 0 in every gain stage", "good_frac": head.get("good_frac")},
        "match_radius_px": 1.5,
        "pixels": {  # per-event statistics over live pixels (either arm nonzero)
            "frac_changed_median": _q(frac, 50), "frac_changed_p10": _q(frac, 10),
            "frac_changed_p90": _q(frac, 90),
            "abs_delta_kev_per_event_median_median": _q(dmed, 50),
            "abs_delta_kev_per_event_median_p90": _q(dmed, 90),
            "abs_delta_kev_per_event_max_median": _q(dmax, 50),
            "abs_delta_kev_max_over_run": round(float(dmax.max()), 3),
        },
        "peaks": {"off": tot[0], "on": tot[1], "matched": tot[2],
                  "jaccard": round(tot[2] / (tot[0] + tot[1] - tot[2]), 4),
                  f"events_with_{DENSE}_plus_peaks": int(dense.sum()),
                  f"jaccard_{DENSE}_plus_median": _q(jac(dense), 50),
                  f"jaccard_{DENSE}_plus_min": round(float(jac(dense).min()), 4),
                  "events_hit_in_either_arm": int(hits.sum()),
                  "jaccard_hits_median": _q(jac(hits), 50), "jaccard_hits_p10": _q(jac(hits), 10),
                  "jaccard_hits_min": round(float(jac(hits).min()), 4)},
        "hits": {"off": int(hitA.sum()), "on": int(hitC.sum()),
                 "lost": sorted(f"{a}->{c}" for a, c, ha, hc in zip(nA, nC, hitA, hitC) if ha and not hc),
                 "gained": sorted(f"{a}->{c}" for a, c, ha, hc in zip(nA, nC, hitA, hitC) if hc and not ha)},
        "snr_and_bank_edge": snr,  # copied from the log's SUMMARY block (per-peak values are not logged)
        "calib_seconds_per_event_median": {
            "off": _q([r["t_calibA"] for r in rows], 50), "on": _q([c["t_calib"] for c in C], 50)},
        "provenance": {**{k: head.get(k) for k in ("host", "psana", "numpy", "calibconst_keys")},
                       "measured": "2026-09-30, interactive on sdfiana027",
                       "script": SCRIPT.name, "script_md5": _md5(SCRIPT),
                       "log": log_path.name, "log_md5": _md5(log_path),
                       "derived_by": Path(__file__).name},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", nargs="?", type=Path, default=LOG)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--check", action="store_true", help=f"compare against {RESULT.name}")
    a = ap.parse_args()
    out = json.dumps(summarize(a.log), indent=2) + "\n"
    if a.check:
        ok = RESULT.exists() and RESULT.read_text() == out
        print(f"{RESULT.name}: {'up to date' if ok else 'STALE -- regenerate with -o'}")
        sys.exit(0 if ok else 1)
    if a.output:
        a.output.write_text(out)
        print(f"wrote {a.output}")
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
