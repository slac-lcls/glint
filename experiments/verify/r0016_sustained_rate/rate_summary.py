#!/usr/bin/env python3
"""Summary: one table from the rate_<tag>.json files (pixel arms) and rate_<tag>_null.json files (the
record_stream_replay null) in OUT_DIR. Reads only; writes rate_summary.json next to them.

    python rate_summary.py OUT_DIR [--q6000 $VERIFY_INPUTS/glint_gt_mfxx49820/q6000_geom.npz]

The null: the defaults arm's own per-frame q, replayed index-only through the shipped recorder at the
driver's defaults (B 64, live gate 6 inliers and 15%), first as recorded ("real", the driver locks on
these), then azimuth-scrambled ("scr": every peak rotated by its own random azimuth about the beam,
|q| and q_z kept, so peak counts and the radial distribution are unchanged and the lattice is gone).
The scrambled frames' strict count is what the strict bar returns from frames with the same peaks and
no lattice. The real half doubles as a check that the pixel pass's decisions reproduce from q alone.
"""
import glob
import json
import os
import sys

import numpy as np


def med(v):
    return float(np.median(v)) if len(v) else float("nan")


def main(argv):
    out = os.path.expanduser(argv[0])
    q6000 = os.path.expanduser(argv[argv.index("--q6000") + 1]) if "--q6000" in argv else \
        os.path.join(os.environ["VERIFY_INPUTS"], "glint_gt_mfxx49820", "q6000_geom.npz")   # no default without VERIFY_INPUTS
    runs = sorted(p for p in glob.glob(os.path.join(out, "rate_*.json"))
                  if not p.endswith("_null.json") and not p.endswith("rate_summary.json"))
    if not runs:
        raise SystemExit(f"no rate_<tag>.json in {out}")
    table = {}
    for p in runs:
        d = json.load(open(p))
        tag = os.path.basename(p)[3:-5]
        prov = d["provenance"]
        print(f"\n=== {tag}: job {prov.get('slurm_job')}.{prov.get('slurm_array_task')} on {prov['host']} "
              f"({prov.get('gpu')}), commit {prov['commit_file']}, env {prov.get('conda_env')}, torch {prov.get('torch')}, "
              f"cupy {prov.get('cupy')}")
        r = d["run"]
        print(f"    {r['exp']} r{r['run']:04d}: {r['events_pushed']} events pushed of {r['events_read']} read "
              f"({r['events_without_frame']} without a frame); calib {r['calib']}, gpu-vs-det.calib max |diff| "
              f"{max(r['calib_check_max_abs_adu']) if r['calib_check_max_abs_adu'] else 'n/a'} ADU; read+calib "
              f"{r['read_and_calib_s']} s (outside the timed sections); cold start {d['cold_start']}")
        g = d["geometry"]
        print(f"    geometry {g['geom']} md5 {g['md5'][:8]}, clen {g['clen_m']:.6f} m + coffset {g['coffset_m']:.6f} m "
              f"= {g['zdist_m']} m, lambda {g['wavelength_A']} A, mask good {g['mask_good_fraction']:.4f}; pf_kw {d['pf_kw']}")
        null = None
        pn = p[:-5] + "_null.json"
        if os.path.exists(pn):
            nj = json.load(open(pn))
            h = nj["header"]
            bs = h["by_species"]
            gmin = int(h.get("gate", {}).get("min_refl", 10))
            sc10 = [r for r in nj.get("records", []) if r.get("truth") == "scr" and (r.get("npk") or 0) >= gmin]
            real, scr = bs.get("real", {}), bs.get("scr", {})
            hr, hs = real.get("n", 0) - real.get("blank", 0), scr.get("n", 0) - scr.get("blank", 0)
            null = dict(real_strict=real.get("ok"), real_hits=hr, scr_strict=scr.get("ok"), scr_hits=hs,
                        scr_live=scr.get("indexed"),
                        strict_chance=(scr.get("ok", 0) / hs if hs else None),
                        live_chance=(scr.get("indexed", 0) / hs if hs else None),
                        scr_hits_ge_gate_min=len(sc10), scr_strict_ge_gate_min=sum(int(r.get("ok", 0)) for r in sc10),
                        strict_chance_ge_gate_min=(sum(int(r.get("ok", 0)) for r in sc10) / len(sc10) if sc10 else None),
                        driver_kw=h.get("driver_kw"), argv=h["provenance"].get("argv"))
        rows = {}
        print(f"    {'arm':12s} {'steady f/s':>10s} {'hits/s':>8s} {'whole f/s':>9s} {'win f/s min/med/max':>22s} "
              f"{'lock@':>6s} {'strict/hits':>14s} {'live':>6s} {'wresc':>5s}  cell")
        for name, a in d["arms"].items():
            st = a.get("steady", {})
            y = a.get("yield", {})
            w = [x["fps"] for x in a.get("windows", [])]
            sy = (f"{y['strict']}/{y['hits']}={100 * y['strict_of_hits']:.1f}%" if y.get("hits") else "-")
            print(f"    {name:12s} {st.get('fps', float('nan')):10.1f} {st.get('hits_per_s', float('nan')):8.1f} "
                  f"{a['whole_run_fps']:9.1f} {min(w) if w else float('nan'):7.1f}/{med(w):6.1f}/{max(w) if w else float('nan'):6.1f} "
                  f"{str(a['lock_i']):>6s} {sy:>14s} {str(y.get('live_accepted', '-')):>6s} {str(y.get('warmup_rescued', '-')):>5s}  "
                  f"{a['cell']} same_as_ref={a['cell_same_as_ref']}")
            rows[name] = dict(steady_fps=st.get("fps"), steady_hits_per_s=st.get("hits_per_s"),
                              whole_run_fps=a["whole_run_fps"], windows_fps=[min(w), med(w), max(w)] if w else None,
                              lock_i=a["lock_i"], strict=y.get("strict"), hits=y.get("hits"),
                              strict_of_hits=y.get("strict_of_hits"), steady_strict_of_hits=st.get("strict_of_hits"),
                              live_accepted=y.get("live_accepted"), warmup_rescued=y.get("warmup_rescued"),
                              integrated=y.get("integrated"), cell=a["cell"], cell_same_as_ref=a["cell_same_as_ref"],
                              hits_ge_gate_min=y.get("hits_ge_gate_min"),
                              strict_of_hits_ge_gate_min=y.get("strict_of_hits_ge_gate_min"),
                              counters=a.get("counters"), scoring_error=a.get("scoring_error"))
            if a.get("scoring_error"):
                print(f"    !! {name}: scoring failed: {a['scoring_error']} (events in {a.get('events_dump')})")
        if "defaults" in rows and "events_off" in rows:
            dd, de = rows["defaults"], rows["events_off"]
            same = all(dd[k] == de[k] for k in ("lock_i", "cell", "counters"))
            print(f"    defaults (event log on) and events_off (shipped default, log off) took the same decisions: {same} "
                  f"(lock frame, cell, driver counters)" + ("" if same else f"  defaults {dd['counters']} vs events_off {de['counters']}"))
            rows["events_off"]["same_decisions_as_defaults"] = same
        for name, rw in rows.items():
            if rw.get("strict_of_hits_ge_gate_min") is not None:
                print(f"    {name:12s} strict over hits with >= 10 peaks: {rw['strict']}/{rw['hits_ge_gate_min']} = "
                      f"{100 * rw['strict_of_hits_ge_gate_min']:.1f}%" + (
                          f"   (steady window, over all hits: {100 * rw['steady_strict_of_hits']:.1f}%)"
                          if rw.get("steady_strict_of_hits") is not None else ""))
        if "defaults" in rows and "events_off" in rows and rows["defaults"]["steady_fps"] and rows["events_off"]["steady_fps"]:
            print(f"    event log cost: defaults {rows['defaults']['steady_fps']:.1f} vs events_off "
                  f"{rows['events_off']['steady_fps']:.1f} f/s ({100 * (rows['events_off']['steady_fps'] / rows['defaults']['steady_fps'] - 1):+.1f}%)")
        if null:
            dflt = rows.get("defaults", {})
            print(f"    null (scrambled peaks, same counts): strict {null['scr_strict']}/{null['scr_hits']} = "
                  f"{100 * (null['strict_chance'] or 0):.2f}%; live gate {null['scr_live']}/{null['scr_hits']} = "
                  f"{100 * (null['live_chance'] or 0):.1f}%")
            if dflt.get("strict_of_hits") is not None and null["strict_chance"] is not None:
                print(f"    defaults strict yield {100 * dflt['strict_of_hits']:.2f}% minus null {100 * null['strict_chance']:.2f}% "
                      f"= {100 * (dflt['strict_of_hits'] - null['strict_chance']):.2f} points")
            if null["strict_chance_ge_gate_min"] is not None:
                print(f"    null over scrambled hits with >= 10 peaks: strict {null['scr_strict_ge_gate_min']}/"
                      f"{null['scr_hits_ge_gate_min']} = {100 * null['strict_chance_ge_gate_min']:.2f}%")
            print(f"    q-only replay of the same frames: strict {null['real_strict']}/{null['real_hits']} "
                  f"(pixel pass: {dflt.get('strict')}/{dflt.get('hits')})")
        # finder check against the validated reader (first 6000 events, btx geometry, per-panel finders)
        fz = p[:-5] + "_frames.npz"
        if os.path.exists(fz) and os.path.exists(q6000):
            z = np.load(fz)
            ev = z["psana_event"]
            if "defaults_npk" in z.files and (ev < 6000).sum() >= 6000:
                sel = ev < 6000
                o = z["defaults_outcome"][sel]
                npk = z["defaults_npk"][sel]
                hit = o != 0                             # TERMINAL[0] == "blank"
                qz = np.load(q6000)
                n_ref = [len(qz[k]) for k in qz.files if k.lstrip("q_").isdigit()]
                print(f"    finder check, first 6000 events: driver {int(hit.sum())} hits, median {med(npk[hit]):.0f} peaks; "
                      f"validated reader (q6000_geom.npz) {len(n_ref)} hits, median {med(n_ref):.0f} peaks")
        table[tag] = dict(arms=rows, null=null, host=prov["host"], gpu=prov.get("gpu"), job=prov.get("slurm_job"),
                          array_task=prov.get("slurm_array_task"), commit=prov.get("commit_file"),
                          conda_env=prov.get("conda_env"), crystfel=prov.get("crystfel"), argv=prov.get("argv"),
                          bar=d.get("bar"), steady_state="from one full batch (B frames) after the lock frame to the end "
                          "of the run; timed = push() and close() only, device-synchronized; read + calibration excluded")
    with open(os.path.join(out, "rate_summary.json"), "w") as f:
        json.dump(table, f, indent=1)
    print(f"\nwrote {os.path.join(out, 'rate_summary.json')}")


if __name__ == "__main__":
    main(sys.argv[1:])
