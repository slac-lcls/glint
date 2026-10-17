#!/usr/bin/env python3
"""Analysis: throughput and yield of fast XGANDALF on 32 cores, set against GLINT's blind_ms.

    j6_analyze.py RUN_DIR --glint GLINT_TREE --f120 F --f480 F --sub24 F --scr120 F [--controls-only]

RUN_DIR/raw holds the driver outputs and timestamp files written by j6_xg32.sbatch. Yields come from
the SHIPPED scorer (GLINT_TREE/experiments/xgandalf/score_xg_gate.py, run unmodified in a temporary
directory of symlinks), FACTS values from GLINT_TREE/experiments/check_numbers.py (parsed with ast,
not imported). Writes RUN_DIR/summary.json and RUN_DIR/summary.md.
"""
import argparse
import ast
import glob
import json
import os
import re
import statistics as st
import subprocess
import tempfile

INPUTS = os.environ["VERIFY_INPUTS"]   # xg_fast/{A_default,B_fast}.txt, xgd480_fix.txt, q480_fix.txt (README); no default
CONTROLS = {  # published driver outputs, with the strict / lattice counts FACTS records for them;
    # always scored against the canonical lists (c120 = the pinned frames_cxidb_clean.txt, c480 = $VERIFY_INPUTS/q480_fix.txt)
    "xg_blind120 (defaults, the 11542 ms run)": ("experiments/xgandalf/xg_blind120.txt", "c120",
                                                 "xgandalf_blind_strict_of120", "xgandalf_lattice_of120"),
    "B_fast (job 37187585, fast)": (f"{INPUTS}/xg_fast/B_fast.txt", "c120",
                                    "xgandalf_fast_strict_of120", "xgandalf_fast_lattice_of120"),
    "xgd480_fix (defaults, n=480)": (f"{INPUTS}/xgd480_fix.txt", "c480", "xgandalf_blind_strict_of480", None),
}


# ----------------------------------------------------------------------------------------- inputs
def facts(glint):
    src = open(os.path.join(glint, "experiments", "check_numbers.py")).read()
    for node in ast.walk(ast.parse(src)):
        tgt = getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(tgt, ast.Name) and tgt.id == "FACTS" and isinstance(node.value, ast.Dict):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                try:
                    out[ast.literal_eval(k)] = ast.literal_eval(v)
                except ValueError:
                    pass
            return out
    raise SystemExit("FACTS not found in check_numbers.py")


def parse_out(path):
    """driver output -> {fid: (basis string or None, ms)}"""
    res = {}
    for line in open(path):
        p = line.split()
        if len(p) < 3:
            continue
        if p[1] == "NONE":
            res[int(p[0])] = (None, float(p[2]))
        else:
            res[int(p[0])] = (" ".join(p[1:10]), float(p[10]))
    return res


def read_times(paths):
    rows = []
    for path in paths:
        for line in open(path):
            p = line.split()
            if len(p) >= 7:
                rows.append({"tag": p[0], "a": int(p[1]), "b": int(p[2]), "cpu": int(p[3]),
                             "t0": float(p[4]), "t1": float(p[5]), "rc": int(p[6])})
    return rows


def score(glint, frames, out):
    """Run the shipped scorer unmodified: it reads frames_cxidb_clean.txt and xg_blind120.txt in cwd."""
    with tempfile.TemporaryDirectory() as d:
        os.symlink(os.path.abspath(frames), os.path.join(d, "frames_cxidb_clean.txt"))
        os.symlink(os.path.abspath(out), os.path.join(d, "xg_blind120.txt"))
        env = dict(os.environ, PYTHONPATH=glint, CUDA_VISIBLE_DEVICES="")
        r = subprocess.run(["python3", os.path.join(glint, "experiments", "xgandalf", "score_xg_gate.py")],
                           cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
    m = re.search(r"BLIND\s+N=(\d+): correct-lattice (\d+)/\d+=\d+%\s+>=25%\(Table1\) (\d+)/\d+=\d+%\s+"
                  r">=10refl (\d+)/\d+", r.stdout)
    if not m:
        return {"error": (r.stdout + r.stderr)[-800:]}
    n, lat, strict, g10 = map(int, m.groups())
    return {"n": n, "strict": strict, "lattice": lat, "ge10_given_lattice": g10, "scorer_line": m.group(0)}


def _ratio(a, b, nd=3):
    """a / b rounded, or None when either is missing or b is 0 (an empty arm must not
    stop the analysis of the others)."""
    return round(a / b, nd) if a is not None and b else None


def _med(xs, nd=3):
    return round(st.median(xs), nd) if xs else None


def stats(ms):
    if not ms:
        return {}
    s = sorted(ms)
    q = lambda f: s[min(len(s) - 1, int(f * len(s)))]
    return {"n": len(s), "median_ms": round(st.median(s), 1), "mean_ms": round(st.mean(s), 1),
            "p10_ms": round(q(0.10), 1), "p90_ms": round(q(0.90), 1), "max_ms": round(s[-1], 1),
            "sum_s": round(sum(s) / 1e3, 2)}


# ------------------------------------------------------------------------------------------- arms
def arm_single(raw, tag):
    out = os.path.join(raw, f"{tag}.out")
    if not os.path.exists(out):
        return None
    res = parse_out(out)
    t = read_times([os.path.join(raw, f"{tag}.times")])
    wall = t[0]["t1"] - t[0]["t0"] if t else None
    s = stats([v[1] for v in res.values()])
    s.update({"kind": "single", "wall_s": round(wall, 2) if wall else None,
              "frames_per_s_wall": round(len(res) / wall, 3) if wall else None,
              "frames_per_s_from_mean": round(1e3 / s["mean_ms"], 3) if s else None,
              "rc": [r["rc"] for r in t], "cpu": [r["cpu"] for r in t]})
    return s, {tag: res}


def arm_parallel(raw, tag):
    outs = sorted(glob.glob(os.path.join(raw, f"{tag}.i*.p*.out")))
    if not outs:
        return None
    times = read_times(glob.glob(os.path.join(raw, f"{tag}.i*.times")))
    by = {(r["a"], r["b"]): r for r in times}
    runs, comps, allms = {}, [], []
    for path in outs:
        i, p = map(int, re.search(r"\.i(\d+)\.p(\d+)\.out$", path).groups())
        res = parse_out(path)
        runs[f"{tag}.i{i}.p{p}"] = res
        r = by.get((i, p))
        ms = [v[1] for v in res.values()]  # dict keeps file order = processing order
        allms += ms
        if r:
            startup = max(0.0, (r["t1"] - r["t0"]) - sum(ms) / 1e3)
            c = r["t0"] + startup
            for x in ms:
                c += x / 1e3
                comps.append(c)
    ninst = len({r["a"] for r in times})
    start = {}
    end = {}
    for r in times:
        start[r["a"]] = min(start.get(r["a"], r["t0"]), r["t0"])
        end[r["a"]] = max(end.get(r["a"], r["t1"]), r["t1"])
    w0, w1 = max(start.values()), min(end.values())
    t_first, t_last = min(start.values()), max(end.values())
    eps = 1e-3  # completions are rebuilt from per-frame ms; allow for rounding at the window edges
    in_win = sum(1 for c in comps if w0 - eps <= c <= w1 + eps)
    s = stats(allms)
    total = len(allms)
    fps_win = in_win / (w1 - w0) if w1 > w0 else None
    per_inst = {}
    for k, res in runs.items():
        i = int(re.search(r"\.i(\d+)\.p", k).group(1))
        per_inst[i] = per_inst.get(i, 0) + len(res)
    fps_sum = sum(per_inst[i] / (end[i] - start[i]) for i in per_inst if i in end and end[i] > start[i])
    s.update({"kind": "parallel", "instances": ninst, "solves": total,
              "window_s": round(w1 - w0, 2), "solves_in_window": in_win,
              "frames_per_s_window": round(fps_win, 3) if fps_win else None,
              "makespan_s": round(t_last - t_first, 2),
              "frames_per_s_makespan": round(total / (t_last - t_first), 3),
              "frames_per_s_sum_of_instances": round(fps_sum, 3),
              "effective_ms_per_frame_window": round(1e3 / fps_win, 2) if fps_win else None,
              "effective_ms_per_frame_makespan": round(1e3 * (t_last - t_first) / total, 2) if total else None,
              "start_spread_s": round(w0 - t_first, 3), "end_spread_s": round(t_last - w1, 2),
              "rc_nonzero": sum(1 for r in times if r["rc"] != 0)})
    return s, runs


def arm_queue(raw, tag, n_expected):
    outs = sorted(glob.glob(os.path.join(raw, f"{tag}.s*.out")))
    if not outs:
        return None
    times = read_times(glob.glob(os.path.join(raw, f"{tag}.s*.times")))
    merged, startup = {}, []
    for path in outs:
        slot = int(re.search(r"\.s(\d+)\.out$", path).group(1))
        res = parse_out(path)
        merged.update(res)
        mine = sorted((r for r in times if r["a"] == slot), key=lambda r: r["t0"])
        for r, (fid, (b, ms)) in zip(mine, res.items()):
            startup.append((r["t1"] - r["t0"]) * 1e3 - ms)
    t_first = min(r["t0"] for r in times)
    t_last = max(r["t1"] for r in times)
    last_pop = max(r["t0"] for r in times)
    busy = sum(r["t1"] - r["t0"] for r in times)
    nw = len({r["a"] for r in times})
    span = t_last - t_first
    s = stats([v[1] for v in merged.values()])
    s.update({"kind": "queue", "workers": nw, "frames": len(merged), "frames_expected": n_expected,
              "makespan_s": round(span, 3), "frames_per_s": round(len(merged) / span, 3) if span > 0 else None,
              "effective_ms_per_frame": round(1e3 * span / len(merged), 2) if merged else None,
              "queue_empty_at_s": round(last_pop - t_first, 3), "drain_tail_s": round(t_last - last_pop, 3),
              "busy_fraction": round(busy / (nw * span), 3),
              "process_overhead_ms_median": round(st.median(startup), 1) if startup else None,
              "rc_nonzero": sum(1 for r in times if r["rc"] != 0)})
    return s, {tag: merged}


def identical(ref, runs):
    """frames whose basis string differs from the reference run (None = no lattice)"""
    bad = 0
    for res in runs.values():
        for fid, (b, _) in res.items():
            if fid in ref and ref[fid][0] != b:
                bad += 1
    return bad


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--glint", required=True)
    ap.add_argument("--f120", required=True)
    ap.add_argument("--f480", required=True)
    ap.add_argument("--sub24", default=None)
    ap.add_argument("--scr120", default=None)
    ap.add_argument("--controls-only", action="store_true")
    a = ap.parse_args()
    F = facts(a.glint)
    frames = {"c120": os.path.join(a.glint, "experiments", "frames_cxidb_clean.txt"),
              "c480": f"{INPUTS}/q480_fix.txt"}
    S = {"facts_at_commit": open(os.path.join(a.glint, "COMMIT")).read().strip(),
         "facts": {k: F.get(k) for k in (
             "blind_ms", "blind_fps", "glint1_strict_of120", "glint1_lattice_of120", "glint1_strict_of480",
             "xgandalf_fast_strict_of120", "xgandalf_fast_lattice_of120", "xgandalf_fast_blind_ms",
             "xgandalf_fast_ratio", "xgandalf_fast_speedup", "xgandalf_blind_ms",
             "xgandalf_blind_strict_of120", "xgandalf_lattice_of120", "xgandalf_blind_strict_of480")},
         "bar": "strict correct-lattice bar: same_lattice(M, lysozyme 79.02/79.02/37.98) "
                "AND >= GATE_FRAC (0.25) of the frame's peaks index within GATE_TOL (0.15) (the scorer's "
                ">=25% column; every frame of the 120 and 480 lists has >= 38 peaks, so this also implies "
                ">= GATE_MIN (10) indexed peaks); 'lattice' = correct reduced cell only. Scored by the shipped "
                "experiments/xgandalf/score_xg_gate.py.",
         "controls": {}, "arms": {}, "checks": {}}

    for name, (path, fk, kstrict, klat) in CONTROLS.items():
        p = path if os.path.isabs(path) else os.path.join(a.glint, path)
        r = score(a.glint, frames[fk], p)
        r["expect_strict"], r["expect_lattice"] = F.get(kstrict), F.get(klat) if klat else None
        r["pass"] = (r.get("strict") == r["expect_strict"] and
                     (klat is None or r.get("lattice") == r["expect_lattice"]))
        S["controls"][name] = r
    if a.controls_only:
        json.dump(S, open(os.path.join(a.run, "controls.json"), "w"), indent=1)
        for k, v in S["controls"].items():
            print(f"{k}: {v.get('scorer_line', v.get('error'))}  expect strict {v['expect_strict']} "
                  f"lattice {v['expect_lattice']}  {'PASS' if v['pass'] else 'FAIL'}")
        return

    raw = os.path.join(a.run, "raw")
    allruns = {}
    plan = [("S1_120", "single", "f120"), ("D1_24", "single", "sub24"), ("S1_480", "single", "f480"),
            ("P32_120", "parallel", "f120"), ("Q32_120_r1", "queue", "f120"), ("Q32_120_r2", "queue", "f120"),
            ("Q32_120_r3", "queue", "f120"), ("N32_120", "queue", "scr120"), ("P32_480", "parallel", "f480"),
            ("Q32_480", "queue", "f480"), ("S1_120b", "single", "f120"), ("P64_120", "parallel", "f120")]
    fmap = {"f120": a.f120, "f480": a.f480, "sub24": a.sub24, "scr120": a.scr120}
    nexp = {"f120": 120, "f480": 480, "sub24": 24, "scr120": 120}
    for tag, kind, fk in plan:
        if kind == "single":
            got = arm_single(raw, tag)
        elif kind == "parallel":
            got = arm_parallel(raw, tag)
        else:
            got = arm_queue(raw, tag, nexp[fk])
        if got is None:
            continue
        s, runs = got
        allruns[tag] = runs
        # yield: score the first output of the arm with the shipped scorer
        first = sorted(glob.glob(os.path.join(raw, f"{tag}.out")) +
                       glob.glob(os.path.join(raw, f"{tag}.i0.p1.out")))
        if kind == "queue":  # merge the slots into one file in frame order, then score
            merged = os.path.join(a.run, "score", f"{tag}.merged.out")
            os.makedirs(os.path.dirname(merged), exist_ok=True)
            res = runs[tag]
            with open(merged, "w") as f:
                for fid in sorted(res):
                    b, ms = res[fid]
                    f.write(f"{fid} {'NONE' if b is None else b} {ms}\n")
            first = [merged]
        if first and fmap[fk]:
            s["yield"] = score(a.glint, fmap[fk], first[0])
        s["frames_file"] = fmap[fk]
        S["arms"][tag] = s

    # ---- checks
    C = S["checks"]
    if "S1_120" in allruns:
        ref = allruns["S1_120"]["S1_120"]
        bf = f"{INPUTS}/xg_fast/B_fast.txt"
        if os.path.exists(bf):
            C["S1_120_bases_vs_job37187585_mismatch"] = identical(parse_out(bf), {"x": ref})
        for tag in ("P32_120", "Q32_120_r1", "Q32_120_r2", "Q32_120_r3", "S1_120b", "P64_120"):
            if tag in allruns:
                C[f"{tag}_bases_vs_S1_120_mismatch"] = identical(ref, allruns[tag])
        if "S1_120b" in S["arms"]:
            C["drift_S1_120b_over_S1_120_mean"] = _ratio(S["arms"]["S1_120b"].get("mean_ms"),
                                                         S["arms"]["S1_120"].get("mean_ms"))
        for tag in ("P32_120", "P64_120"):
            if tag in allruns:
                ratios = []
                for res in allruns[tag].values():
                    for fid, (_, ms) in res.items():
                        if fid in ref and ref[fid][1] > 0:
                            ratios.append(ms / ref[fid][1])
                C[f"contention_{tag}_per_frame_ms_ratio_median"] = _med(ratios)
                C[f"contention_{tag}_mean_ratio"] = _ratio(S["arms"][tag].get("mean_ms"),
                                                          S["arms"].get("S1_120", {}).get("mean_ms"))
    if "S1_480" in allruns:
        ref = allruns["S1_480"]["S1_480"]
        for tag in ("P32_480", "Q32_480"):
            if tag in allruns:
                C[f"{tag}_bases_vs_S1_480_mismatch"] = identical(ref, allruns[tag])
        if "P32_480" in allruns:
            ratios = [ms / ref[fid][1] for res in allruns["P32_480"].values()
                      for fid, (_, ms) in res.items() if fid in ref and ref[fid][1] > 0]
            C["contention_P32_480_per_frame_ms_ratio_median"] = _med(ratios)
    if "D1_24" in allruns:
        d1 = allruns["D1_24"]["D1_24"]
        for name, path in (("xg_blind120 (published defaults run)",
                            os.path.join(a.glint, "experiments/xgandalf/xg_blind120.txt")),
                           ("A_default (EPYC 7702, job 37187585)", f"{INPUTS}/xg_fast/A_default.txt")):
            if os.path.exists(path):
                ref = parse_out(path)
                r = [d1[f][1] / ref[f][1] for f in d1 if f in ref and ref[f][1] > 0]
                C[f"D1_24_ms_over_{name}_median"] = _med(r)
                C[f"D1_24_bases_vs_{name}_mismatch"] = identical(ref, {"x": d1})

    # ---- MHz trace
    mhz = os.path.join(raw, "mhz.tsv")
    if os.path.exists(mhz):
        acc = {}
        for line in open(mhz):
            p = line.split()
            if len(p) == 3:
                acc.setdefault(p[1], []).append(float(p[2]))
        C["mean_MHz_of_pinned_cpus_by_arm"] = {k: round(st.mean(v)) for k, v in acc.items()}

    # ---- the comparison with GLINT's blind throughput
    cmp = {}
    bm, bf = F.get("blind_ms"), F.get("blind_fps")
    p = S["arms"].get("P32_120")
    s1 = S["arms"].get("S1_120")
    if p and bm:
        cmp["P32_120_frames_per_s_window"] = p.get("frames_per_s_window")
        cmp["P32_120_effective_ms_window"] = p.get("effective_ms_per_frame_window")
        cmp["glint_blind_fps_over_P32_120_fps"] = _ratio(bf, p.get("frames_per_s_window"), 2)
        cmp["P32_120_effective_ms_over_blind_ms"] = _ratio(p.get("effective_ms_per_frame_window"), bm, 2)
    if s1 and bm and s1.get("mean_ms") is not None:
        cmp["S1_120_median_ms_over_blind_ms (as xgandalf_fast_speedup is built)"] = round(s1["median_ms"] / bm, 1)
        cmp["S1_120_mean_ms_over_blind_ms"] = round(s1["mean_ms"] / bm, 1)
        cmp["linear_estimate_from_S1_mean_ms_over_32"] = round(s1["mean_ms"] / 32, 1)
        cmp["linear_estimate_from_FACTS_median_ms_over_32"] = round(F["xgandalf_fast_blind_ms"] / 32, 1)
    q = [S["arms"][t]["effective_ms_per_frame"] for t in ("Q32_120_r1", "Q32_120_r2", "Q32_120_r3")
         if t in S["arms"] and S["arms"][t].get("effective_ms_per_frame") is not None]
    if q and bm:
        cmp["Q32_120_effective_ms_reps"] = q
        cmp["Q32_120_effective_ms_median_over_blind_ms"] = round(st.median(q) / bm, 2)
    S["comparison"] = cmp

    json.dump(S, open(os.path.join(a.run, "summary.json"), "w"), indent=1)
    write_md(S, os.path.join(a.run, "summary.md"))
    print(open(os.path.join(a.run, "summary.md")).read())


def write_md(S, path):
    F = S["facts"]
    L = ["# Summary: fast XGANDALF on 32 cores against GLINT's blind_ms", "",
         f"GLINT tree {S['facts_at_commit']}; XGANDALF = libxgandalf of CrystFEL 0.12.0 via xg_driver; "
         "fast = sampling pitch 2, gradient-descent iterations 3 (CrystFEL's --xgandalf-fast-execution).", "",
         "Bar: " + S["bar"], "",
         "| arm | per-frame ms median / mean | frames/s | effective ms/frame | strict | lattice |",
         "|---|---|---|---|---|---|",
         f"| GLINT-1 blind, one A100 (FACTS blind_ms, a mean over the 120) | - / {F['blind_ms']} | "
         f"{F['blind_fps']} | {F['blind_ms']} | {F['glint1_strict_of120']}/120 (+/-1, A100) | "
         f"{F['glint1_lattice_of120']}/120 |"]
    for tag, s in S["arms"].items():
        y = s.get("yield", {})
        ystr = (f"{y.get('strict')}/{y.get('n')}", f"{y.get('lattice')}/{y.get('n')}") if "n" in y else ("?", "?")
        if s["kind"] == "single":
            fps, eff = s.get("frames_per_s_from_mean"), s.get("mean_ms")
        elif s["kind"] == "parallel":
            fps, eff = s.get("frames_per_s_window"), s.get("effective_ms_per_frame_window")
        else:
            fps, eff = s.get("frames_per_s"), s.get("effective_ms_per_frame")
        L.append(f"| {tag} ({s['kind']}) | {s.get('median_ms')} / {s.get('mean_ms')} | {fps} | {eff} | "
                 f"{ystr[0]} | {ystr[1]} |")
    L += ["", "## Comparison", ""] + [f"- {k}: {v}" for k, v in S["comparison"].items()]
    L += ["", "## Controls", ""]
    for k, v in S["controls"].items():
        L.append(f"- {k}: strict {v.get('strict')} (expect {v['expect_strict']}), lattice {v.get('lattice')} "
                 f"(expect {v['expect_lattice']}) -> {'PASS' if v['pass'] else 'FAIL'}")
    L += ["", "## Checks", ""] + [f"- {k}: {v}" for k, v in S["checks"].items()]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
