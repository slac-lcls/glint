#!/usr/bin/env python3
"""Analysis (CPU): replays of the shipped stopping rule on the cached per-frame cells, and the tables.

    python stop_replay.py [--procs 32] [--orders 400]

Reads $STOP_OUT/prep/{votes,live}_*.json|npz and $STOP_OUT/scramble/task*/; writes $STOP_OUT/replay/:
    stop_results.json   every number below, with provenance
    stop_tables.md      the tables

Parts:
  A  arrival order (the live StreamDriver decision from stop_prep.py), with its replay parity
  B  arrival order from later starting points (every 100 frames; overlapping cases, not a rate)
  C  400 random orders per dataset, shipped rule and support-and-lead only, seed 1; shipped also seeds 0, 2-7
  D  the 480-frame controlled set: part C on cxidb480, against seqstop_* (median 6, p90 12, 0/400)
  E  scrambled groups (arm a): false decisions of the live shipped rule, the support-and-lead-only rule and
     the batch requirements on the same groups; counts by frames seen; one-sided 95% upper bounds
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stop_common as C                                                       # noqa: E402

P = os.path.join(C.OUT, "prep")
R = os.path.join(C.OUT, "replay")
VARIANTS = ("shipped", "gaponly")
SEEDS_EXTRA = (0, 2, 3, 4, 5, 6, 7)          # shipped rule only: is the summary stable across seeds?
G_MARKS = (6, 12, 16, 24, 32, 50, 75, 100)    # frames seen, for the cumulative false-decision counts
FACTS = dict(seqstop_median_lock=6, seqstop_p90_lock=12, seqstop_false_locks=0, seqstop_trials=400,
             poolgate_clean_diff_72=0, poolgate_bad_locks_gaponly=342, poolgate_bad_wrong_gaponly=194,
             poolgate_bad_locks_gated=3, poolgate_bad_wrong_gated=2)   # check_numbers.py at 95d51ce

_DS = {}
_FR = {}


def load_ds(name):
    if name in _DS:
        return _DS[name]
    z = np.load(os.path.join(P, f"votes_{name}.npz"), allow_pickle=False)
    import stop_prep
    _DS[name] = (C.unpack_cells(z["flat"], z["nper"]), z["voted"], stop_prep.references(name, z), z["events"])
    return _DS[name]


def one_order(args):
    """Part B: arrival order of dataset `name`, starting at frame `start`, to the end of the run."""
    name, variant, _, _, start = args
    per_frame, voted, refs, _ = load_ds(name)
    r = C.replay(per_frame, voted, np.arange(start, len(per_frame)), variant)
    return dict(name=name, variant=variant, seed=None, i=None, start=start, decided=r["decided"],
                pushed=r["pushed"], voting=r["voting"], npool=r["npool"], support=r["support"],
                runner=r["runner"], edges=C.edges(r["M"]),
                vs={k: C.verdict_vs(r["M"], ref) for k, ref in refs.items()})


def _perms(n, seed, lo, hi):
    """Orders lo..hi-1 of an order scheme.
    seed = int : order i is the i-th permutation drawn from default_rng(seed) -- the scheme of
                 experiments/test_seqstop_replay.py (seed 1 reproduces poolgate_clean_*; stop_check120.py);
    seed = "k" : order i is default_rng(i).permutation(n) -- the scheme of the August script wrong_sweep.py
                 (19 Aug), behind poolgate_bad_*, used only for the unrefined-r0016 bridge."""
    if seed == "k":
        return [np.random.default_rng(i).permutation(n) for i in range(lo, hi)]
    rng = np.random.default_rng(seed)
    return [rng.permutation(n) for _ in range(hi)][lo:hi]


def _orders_job(args):
    """Part C: orders lo..hi-1 of order scheme `seed` (see _perms)."""
    name, variant, seed, lo, hi = args
    per_frame, voted, refs, _ = load_ds(name)
    out = []
    for i, order in zip(range(lo, hi), _perms(len(per_frame), seed, lo, hi)):
        r = C.replay(per_frame, voted, order, variant)
        out.append(dict(name=name, variant=variant, seed=seed, i=i, start=None, decided=r["decided"],
                        pushed=r["pushed"], voting=r["voting"], npool=r["npool"], support=r["support"],
                        runner=r["runner"], edges=C.edges(r["M"]),
                        vs={k: C.verdict_vs(r["M"], ref) for k, ref in refs.items()}))
    return out


DEV_N = int(os.environ.get("STOP_DEV_N", "64"))   # frames per real dataset re-indexed on CPU (Part F)


def _device_parity(args):
    """Part F, one frame: index_blind_nbest on this CPU node against the cells stop_prep.py cached on the
    A100. Frame k of the evenly spaced DEV_N. Tells whether the scramble arm could run on CPU (milano)
    without changing what the rule sees, and how often GPU and CPU disagree at all."""
    import torch
    torch.set_num_threads(1)
    from glint.glint_fast import index_blind_nbest
    import stop_prep
    name, k = args
    per_frame, voted, _, _ = load_ds(name)
    i = int(np.linspace(0, len(per_frame) - 1, DEV_N).astype(int)[k])
    if not voted[i]:
        return dict(name=name, i=i, voted=False, same=False, same_first=False, ms=0.0)
    if name not in _FR:
        _FR[name] = stop_prep.load_frames(name)[0]
    q = C.clean([_FR[name][i]], C.shipped_defaults()["min_peaks"])[0][0]
    t = time.time()
    cpu = [np.asarray(c, float) for c, _ in index_blind_nbest(q, 3) if c is not None]
    ms = 1e3 * (time.time() - t)
    gpu = per_frame[i]
    same = len(cpu) == len(gpu) and all(C.same_lattice(a, b) for a, b in zip(cpu, gpu))
    first = bool(cpu and gpu and C.same_lattice(cpu[0], gpu[0])) or (not cpu and not gpu)
    return dict(name=name, i=i, voted=True, same=bool(same), same_first=first, ms=ms)


def _rank90(x):
    """90th percentile by nearest rank (the smallest value with >= 90% of draws at or below it)."""
    s = np.sort(np.asarray(x))
    return int(s[int(np.ceil(0.9 * len(s))) - 1])


def summarize(rows, primary_ref):
    """Counts and frames to decision. `pushed` = hit frames pushed (the stream position, which is what a
    run spends); `voting` = frames with >= min_peaks usable peaks (RunningConsensus.nframes, the count
    test_seqstop_replay.py reports). p90 is given both ways: linear interpolation (np.percentile) and
    nearest rank; the FACTS seqstop_p90_lock does not say which it used."""
    dec = [r for r in rows if r["decided"]]
    ok = [r for r in dec if r["vs"].get(primary_ref, {}).get("same")]
    bad = [r for r in dec if r["vs"].get(primary_ref, {}).get("same") is False]
    fr = np.array([r["pushed"] for r in ok]) if ok else np.array([])
    fv = np.array([r["voting"] for r in ok]) if ok else np.array([])
    vr = [r["vs"][primary_ref]["vol_ratio"] for r in bad]
    return dict(n=len(rows), decided=len(dec), correct=len(ok), false=len(bad), none=len(rows) - len(dec),
                unjudged=len(dec) - len(ok) - len(bad),            # decided, but no reference cell to judge by
                frames_correct_median=float(np.median(fr)) if fr.size else None,
                frames_correct_p90=float(np.percentile(fr, 90)) if fr.size else None,
                frames_correct_p90_rank=_rank90(fr) if fr.size else None,
                frames_correct_max=int(fr.max()) if fr.size else None,
                frames_correct_mean=float(fr.mean()) if fr.size else None,
                voting_correct_median=float(np.median(fv)) if fv.size else None,
                voting_correct_p90_rank=_rank90(fv) if fv.size else None,
                false_frames=sorted(int(r["pushed"]) for r in bad),
                false_pools=sorted(int(r["npool"]) for r in bad),
                false_vol_ratio_median=float(np.median(vr)) if vr else None,
                max_pool_at_decision=max((r["npool"] for r in dec), default=None),
                vs_all_refs={k: dict(same=sum(1 for r in dec if r["vs"][k]["same"]),
                                     different=sum(1 for r in dec if r["vs"][k]["same"] is False))
                             for k in (rows[0]["vs"] if rows else {})})


def primary_ref(name):
    return {"cxidb480": "lyso_textbook"}.get(name, "btx_sample2" if name.startswith("r0016") else "batch_whole_run")


def upper95(k, n):
    """One-sided 95% upper bound on a rate after k events in n trials (k = 0: 1 - 0.05**(1/n) ~ 3/n)."""
    if n == 0:
        return None
    if k == 0:
        return 1.0 - 0.05 ** (1.0 / n)
    try:
        from scipy.stats import beta
        return float(beta.ppf(0.95, k + 1, n - k))
    except Exception:
        return None


def scrambled_part(procs):
    rows, cells = [], []
    for f in sorted(glob.glob(os.path.join(C.OUT, "scramble", "task*", "*_r*_*.npz"))):
        z = np.load(f, allow_pickle=False)
        rr = json.loads(str(z["rows"]))
        groups = C.unpack_cells(z["flat"].astype(np.float64), z["nper"])
        off = 0
        for r_ in rr:
            rows.append(r_)
            cells.append(groups[off:off + r_["G"]])
            off += r_["G"]
    if not rows:
        return dict(note="no scramble outputs found")
    with Pool(procs) as pool:
        extra = pool.map(_scr_replay, cells, chunksize=20)
    out = {}
    for kind in ("main", "long"):
        idx = [i for i, r_ in enumerate(rows) if r_["kind"] == kind]
        if not idx:
            continue
        per_run = {}
        for run in sorted({rows[i]["run"] for i in idx}) + ["all"]:
            sel = [i for i in idx if run == "all" or rows[i]["run"] == run]
            n = len(sel)
            arms = {}
            for arm in ("shipped_live", "gaponly", "batch_whole_group"):
                if arm == "shipped_live":
                    first = [rows[i]["pushed"] if rows[i]["decided"] else None for i in sel]
                elif arm == "gaponly":
                    first = [extra[i]["gaponly_first"] for i in sel]
                else:
                    first = [rows[i]["G"] if extra[i]["batch_accept"] else None for i in sel]
                k = sum(1 for x in first if x is not None)
                by_g = ({g: sum(1 for x in first if x is not None and x <= g) for g in G_MARKS}
                        if kind == "main" and arm != "batch_whole_group" else None)
                arms[arm] = dict(groups=n, false_decisions=k, per_1000=1000.0 * k / n if n else None,
                                 upper95_per_1000=(1000.0 * upper95(k, n)) if n else None,
                                 by_frames_seen=by_g)
            dec = [rows[i] for i in sel if rows[i]["decided"]]
            per_run[run] = dict(groups=n, frames=sum(rows[i]["G"] for i in sel), arms=arms,
                                shipped_false_cells=[dict(run=r_["run"], group=r_["group"], frame=r_["pushed"],
                                                          support=r_["support"], runner=r_["runner"],
                                                          pool=r_["npool"], edges=r_["edges"], vs=r_["vs"])
                                                     for r_ in dec],
                                ms_per_frame=1e3 * sum(rows[i]["index_s"] for i in sel) /
                                max(1, sum(min(rows[i]["pushed"], rows[i]["G"]) for i in sel)))
        out[kind] = per_run
    return out


def _scr_replay(group_cells):
    voted = np.ones(len(group_cells), bool)
    g = C.replay(group_cells, voted, range(len(group_cells)), "gaponly")
    pool = [m for c in group_cells for m in c]
    Mb, _ = C.consensus_cell(pool, min_frac=C.CONSENSUS_MIN_FRAC, min_lead=C.CONSENSUS_MIN_LEAD) if pool else (None, 0)
    return dict(gaponly_first=g["pushed"] if g["decided"] else None, batch_accept=Mb is not None)


def md_tables(res):
    L = ["# Stopping-rule tables (generated by stop_replay.py; numbers only, no interpretation)", ""]
    L += ["## A. Arrival order, live StreamDriver (shipped rule)", "",
          "| dataset | frames | decided | at pushed frame (event) | support / runner-up / pool | cell (sorted edges, Å) | correct vs reference | replay parity |",
          "|---|---|---|---|---|---|---|---|"]
    for name, a in res["A"].items():
        lv = a["live"]
        L.append(f"| {name} | {a['n_frames']} | {lv['decided']} | {lv['pushed']} ({lv.get('event')}) | "
                 f"{lv['support']} / {lv['runner']} / {lv['npool']} | {lv['edges']} | "
                 f"{ {k: v['same'] for k, v in lv['vs'].items()} } | {a['parity']} |")
    L += ["", "## C/D. Random orders (400 per dataset, seed 1)", "",
          "| dataset | rule | reference | decided | correct | false (median vol ratio) | no decision | frames to a correct decision: median / p90 / max | max pool at decision |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, by_v in res["C"].items():
        for v, s in by_v.items():
            if not isinstance(s, dict) or "ref" not in s:
                continue
            L.append(f"| {name} | {v} | {s['ref']} | {s['decided']} | {s['correct']} | {s['false']} "
                     f"({s['false_vol_ratio_median']}) | {s['none']} | {s['frames_correct_median']} / "
                     f"{s['frames_correct_p90']} / {s['frames_correct_max']} | {s['max_pool_at_decision']} |")
    L += ["", "Orders in which the shipped rule and the support-and-lead-only rule differ (decision, frame, cell, "
          "support or runner-up), seed 1: " + ", ".join(f"{n} {by_v['orders_differing_shipped_vs_gaponly']}"
                                                       for n, by_v in res["C"].items())]
    L += ["", "Seed stability, shipped rule (decided / correct / false / median frames to a correct decision): "]
    for n, by_v in res["C"].items():
        L.append(f"- {n}: " + "; ".join(f"seed {sd}: {x['decided']}/{x['correct']}/{x['false']}/{x['frames_correct_median']}"
                                       for sd, x in by_v["seeds_shipped"].items()))
    if "C2_r0016unref_wrong_sweep_orders" in res:
        c2 = res["C2_r0016unref_wrong_sweep_orders"]
        L += ["", "## C2. Unrefined r0016, wrong_sweep.py's orders (bridge to poolgate_bad_*)", "",
              f"FACTS: {c2['facts']}", ""]
        for v in VARIANTS:
            x = c2[v]
            L.append(f"- {v}: decided {x['decided']}, correct {x['correct']}, false {x['false']} "
                     f"(median vol ratio {x['false_vol_ratio_median']}, pools {x['false_pools']}), none {x['none']}")
    if res.get("F_device_parity"):
        L += ["", "## F. Device parity (cached A100 cells vs the same call on a CPU node)", ""]
        for n, x in res["F_device_parity"].items():
            L.append(f"- {n}: {x['identical']}/{x['frames']} frames identical, {x['same_first_cell']} same first cell, "
                     f"{x['ms_per_frame_cpu']} ms/frame on CPU (1 thread)")
    if res.get("check120"):
        L += ["", f"Harness check against experiments/test_seqstop_replay.py at n = 120: {res['check120']}"]
    L += ["", "## E. Scrambled groups (false decisions; one-sided 95% upper bound per 1,000 groups)", ""]
    for kind, per_run in res.get("E", {}).items():
        if not isinstance(per_run, dict):
            continue
        L += [f"### {kind} groups", "", "| run | groups | frames | rule | false decisions | per 1,000 | upper 95% per 1,000 | by frames seen |",
              "|---|---|---|---|---|---|---|---|"]
        for run, s in per_run.items():
            for arm, x in s["arms"].items():
                L.append(f"| {run} | {s['groups']} | {s['frames']} | {arm} | {x['false_decisions']} | "
                         f"{x['per_1000']:.3f} | {x['upper95_per_1000']:.3f} | {x['by_frames_seen']} |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--orders", type=int, default=400)
    ap.add_argument("--skip-scramble", action="store_true")
    a = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    t0 = time.time()
    names = [n for n in C.DATASETS if os.path.exists(os.path.join(P, f"votes_{n}.npz"))]
    res = dict(provenance=C.provenance(), facts_at_95d51ce=FACTS, datasets=names, A={}, B={}, C={})
    import stop_check120                                     # the replay harness against the repo's own test
    res["check120"] = stop_check120.check()

    # A: the live arrival-order decision and its parity (computed in stop_prep.py)
    for n in names:
        f = os.path.join(P, f"live_{n}.json")
        if os.path.exists(f):
            j = json.load(open(f))
            res["A"][n] = dict(n_frames=j["n_frames"], live=j["live"], replay=j["replay"], parity=j["parity"],
                               batch=j["batch"])

    # the whole-run cells: against btx for r0016; for mfxl1038923 no independent cell exists, so the
    # whole-run cell is GLINT's own and is compared with the edges the 8 Aug dump log printed
    # (r278 46.6/72.7/92.9, r58 43.6/67.8/89.1 A; a log line). r0278 against
    # r0058 is recorded only: the Aug cells differ by ~6% in every edge (zdist 0.060 vs 0.050), so
    # the replay does not assume the two runs share a cell.
    bc = {n: load_ds(n)[2]["batch_whole_run"] for n in names}
    res["whole_run_cells"] = {n: C.edges(m) for n, m in bc.items()}
    res["whole_run_cells_aug_log"] = {"r0278": [46.6, 72.7, 92.9], "r0058": [43.6, 67.8, 89.1]}
    res["cross_checks"] = {
        "r0278_vs_r0058_whole_run_record_only": C.verdict_vs(bc.get("r0278"), bc.get("r0058")),
        **{f"{n}_whole_run_vs_btx": C.verdict_vs(bc.get(n), load_ds(n)[2].get("btx_sample2"))
           for n in names if n.startswith("r0016")}}

    jobs_b = [(n, v, None, 0, s) for n in names if n != "cxidb480" for v in VARIANTS
              for s in range(100, len(load_ds(n)[0]), 100)]
    chunk = max(1, a.orders // 25)
    jobs_c = [(n, v, 1, lo, min(lo + chunk, a.orders)) for n in names for v in VARIANTS
              for lo in range(0, a.orders, chunk)]
    jobs_c += [(n, "shipped", s, lo, min(lo + chunk, a.orders)) for n in names for s in SEEDS_EXTRA
               for lo in range(0, a.orders, chunk)]
    # C2: unrefined r0016 under wrong_sweep.py's own order scheme, the bridge to poolgate_bad_*
    jobs_c += [(n, v, "k", lo, min(lo + chunk, a.orders)) for n in names if n == "r0016unref"
               for v in VARIANTS for lo in range(0, a.orders, chunk)]
    with Pool(a.procs) as pool:
        rb = pool.map(one_order, jobs_b, chunksize=1)
        rc = [r for part in pool.imap_unordered(_orders_job, jobs_c, chunksize=1) for r in part]
        rf = pool.map(_device_parity, [(n, k) for n in names if n != "cxidb480" for k in range(DEV_N)],
                      chunksize=4)
    print(f"[replay] B {len(rb)} cases, C {len(rc)} orders, F {len(rf)} frames in {time.time() - t0:.0f} s",
          flush=True)

    for n in names:
        ref = primary_ref(n)
        res["B"][n] = {v: [dict(start=r["start"], decided=r["decided"], pushed=r["pushed"], support=r["support"],
                                runner=r["runner"], npool=r["npool"], edges=r["edges"],
                                same=r["vs"].get(ref, {}).get("same"))
                           for r in sorted(rb, key=lambda x: x["start"]) if r["name"] == n and r["variant"] == v]
                       for v in VARIANTS}
        res["C"][n] = {}
        for v in VARIANTS:
            rows = sorted([r for r in rc if r["name"] == n and r["variant"] == v and r["seed"] == 1],
                          key=lambda x: x["i"])
            s = summarize(rows, ref)
            s["ref"] = ref
            res["C"][n][v] = s
        # orders in which the two rules differ (decision, frame, cell, support or runner-up)
        a1 = {r["i"]: r for r in rc if r["name"] == n and r["variant"] == "shipped" and r["seed"] == 1}
        a2 = {r["i"]: r for r in rc if r["name"] == n and r["variant"] == "gaponly" and r["seed"] == 1}
        res["C"][n]["orders_differing_shipped_vs_gaponly"] = sum(
            1 for i in a1 if (a1[i]["decided"], a1[i]["pushed"], a1[i]["edges"], a1[i]["support"], a1[i]["runner"])
            != (a2[i]["decided"], a2[i]["pushed"], a2[i]["edges"], a2[i]["support"], a2[i]["runner"]))
        res["C"][n]["seeds_shipped"] = {s: summarize(sorted([r for r in rc if r["name"] == n and r["variant"] == "shipped"
                                                             and r["seed"] == s], key=lambda x: x["i"]), ref)
                                        for s in (1,) + SEEDS_EXTRA}
    if "r0016unref" in names:
        ref = primary_ref("r0016unref")
        res["C2_r0016unref_wrong_sweep_orders"] = dict(
            scheme="order i = default_rng(i).permutation(n), i = 0..399 (wrong_sweep.py)",
            facts=dict((k, FACTS[k]) for k in FACTS if k.startswith("poolgate_bad")),
            note="cells regenerated at 95d51ce and grouped by RunningConsensus (merge on); wrong_sweep.py "
                 "re-implemented the grouping (first match, no merge, gap rate over all frames), so a "
                 "difference from FACTS is a finding to report, not a fault",
            **{v: summarize(sorted([r for r in rc if r["name"] == "r0016unref" and r["variant"] == v
                                    and r["seed"] == "k"], key=lambda x: x["i"]), ref) for v in VARIANTS})
    # F: device parity -- the cached (GPU, stop_prep.py) cells against index_blind_nbest on this CPU node
    res["F_device_parity"] = {}
    for n in sorted({x["name"] for x in rf}):
        xs = [x for x in rf if x["name"] == n and x["voted"]]
        res["F_device_parity"][n] = dict(frames=len(xs), identical=sum(x["same"] for x in xs),
                                         same_first_cell=sum(x["same_first"] for x in xs),
                                         ms_per_frame_cpu=float(np.mean([x["ms"] for x in xs])) if xs else None)
    if "cxidb480" in names:
        res["D_g89"] = dict(facts=dict((k, FACTS[k]) for k in FACTS if k.startswith("seqstop")),
                            shipped=res["C"]["cxidb480"]["shipped"], gaponly=res["C"]["cxidb480"]["gaponly"],
                            orders_differing=res["C"]["cxidb480"]["orders_differing_shipped_vs_gaponly"])
    if not a.skip_scramble:
        res["E"] = scrambled_part(a.procs)
    res["secs"] = time.time() - t0
    C.dump_json(os.path.join(R, "stop_results.json"), res)
    open(os.path.join(R, "stop_tables.md"), "w").write(md_tables(res))
    print(f"[replay] wrote {R}/stop_results.json and stop_tables.md in {res['secs']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
