"""Routes scoring: the strict scorer of stills_score.py (sha256 f11a88e0... as run) applied to the route arms.

Copied from stills_score.py, with these changes only:
  - load_dials: the `ok` mask is built with dtype=bool (stills_score.py:85 built it without, so an arm whose timed
    frames are all failures, or a 1-frame dry run, gave a float array and `~ok` raised); the timing unit is per route.
  - the arm lists are the routes' nine; B0, K2, K3, G_pf and G_hyb are read back, read-only, from the stills_process outputs
    for the paired tests (they are not re-run).
  - regression checks for the two earlier rows (561 and 548 of 907 for 1D-FFT blind, length and full gate; 453 of
    907 for grid with the cell), and S1 against B0.
  - 16 Oct, before any full-run frame: arm S3n (null for S3) added; `complete` covers the four headline arms
    (V1, V2, V1n, V2n) and `complete_secondary` the S arms; a paired test is computed only when both arms cover all
    frames; pairs S3 vs V1, S3 vs G_pf and S3 vs B0 added; a bvs self-check (the driver's own peak count of the kept
    candidate against matched_strict on the stored M, per frame).
score(), wilson(), mcnemar() and the bar are unchanged:
strict bar (glint_fast.gpass with the ProK reference in place of LYSO):
    same_lattice(M, ProK)  AND  matched_strict(M, q) / N >= GATE_FRAC (0.25)  AND  matched_strict >= GATE_MIN (10)
also reported, never as the headline: correct (same_lattice alone), loose10, lengate (the earlier rows' gate).
Run in the GLINT env with PYTHONPATH=<pinned tree>."""
import argparse
import glob
import json
import math
import os

import numpy as np

from glint.glint_fast import GATE_FRAC, GATE_MIN, GATE_TOL, matched_strict
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice

PROK = cell_to_Ar(68.7, 68.7, 108.6, 90.0, 90.0, 90.0)
AXES = sorted([68.7, 68.7, 108.6])
ROUTE_ARMS = [("V1_bvs_fft1d_blind", "real", "bvs"), ("V2_bvs_rsgs_kc", "real", "bvs"),
             ("V1n_bvs_fft1d_blind_null", "null", "bvs"), ("V2n_bvs_rsgs_kc_null", "null", "bvs"),
             ("S1_idx_fft1d_rac", "real", "indexer"), ("S2_idx_fft1d_norac", "real", "indexer"),
             ("S3_idx_fft1d_rac_mc200", "real", "indexer"), ("S4_idx_rsgs_kc_norac", "real", "indexer"),
             ("S3n_idx_fft1d_rac_mc200_null", "null", "indexer")]
HEADLINE = ("V1_bvs_fft1d_blind", "V2_bvs_rsgs_kc", "V1n_bvs_fft1d_blind_null", "V2n_bvs_rsgs_kc_null")
STILLS_DIALS_REF = [("B0_blind_fft1d", "real", "stills_process"), ("K2_kc_xfel", "real", "stills_process"),
                 ("K3_kc_rsgs", "real", "stills_process")]
STILLS_GLINT_REF = [("G_pf", "real"), ("G_hyb", "real")]
PUBLISHED = {"V1_bvs_fft1d_blind": dict(orig_len_gate=561, orig_full_gate=548),     # index_dials.out (61.9%, 60.4%)
             "V2_bvs_rsgs_kc": dict(orig_len_gate=453)}                              # index_kc.out (49.9%)
UNIT = {"bvs": "s per frame, one EPYC core, bare basis-vector search + 25-candidate assembly + pick (no refinement)",
        "indexer": "s per frame, one EPYC core, Indexer.from_parameters + index() (stills indexer, refinement included)",
        "stills_process": "s per frame, one EPYC core, stills_process dispatch (hit finder + index + refinement)"}


def len_gate(M, ltol=0.05):
    L = sorted(np.linalg.norm(np.asarray(M, float), axis=0))
    return all(abs(l - p) <= ltol * p for l, p in zip(L, AXES))


def score(M, q):
    if M is None or not np.all(np.isfinite(M)):
        return dict(found=False, correct=False, m=0, strict=False, loose10=False, lengate=False)
    corr = bool(same_lattice(M, PROK))
    m = int(matched_strict(M, q))
    return dict(found=True, correct=corr, m=m, strict=bool(corr and m / len(q) >= GATE_FRAC and m >= GATE_MIN),
                loose10=bool(corr and m >= GATE_MIN), lengate=bool(len_gate(M)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h)))


def mcnemar(a, b):
    """a, b: boolean arrays over the same frames. Discordant counts and the exact two-sided p."""
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, p


def load_dials(root, arm, n, route):
    recs = {}
    first = set()
    for f in sorted(glob.glob(os.path.join(root, arm, "*.jsonl"))):
        frames_here = []
        for line in open(f):
            r = json.loads(line)
            recs[r["frame"]] = r
            frames_here.append(r["frame"])
        if frames_here:
            first.add(min(frames_here))                  # each task's first frame carries import warm-up
    Ms = [None] * n
    for i, r in recs.items():
        if r.get("ok") and r.get("M_glint"):
            Ms[i] = np.array(r["M_glint"], float).reshape(3, 3)
    t = np.array([r["t_proc"] for i, r in sorted(recs.items()) if r.get("t_proc") is not None and i not in first])
    ok = np.array([bool(recs[i].get("ok")) for i in sorted(recs) if recs[i].get("t_proc") is not None and i not in first],
                  dtype=bool)
    timing = dict(unit=UNIT[route] + ", spot finding excluded",
                  n=int(len(t)), median=float(np.median(t)) if len(t) else None, mean=float(t.mean()) if len(t) else None,
                  p90=float(np.percentile(t, 90)) if len(t) else None,
                  median_ok=float(np.median(t[ok])) if ok.any() else None,
                  median_fail=float(np.median(t[~ok])) if (~ok).any() else None,
                  total_cpu_h=float(t.sum() / 3600) if len(t) else None)
    own = sum(bool(r.get("ok")) for r in recs.values())
    build_fail = sum(1 for r in recs.values() if str(r.get("err") or "").startswith("build:"))
    extra = dict(build_fail=build_fail,
                 max_rt_err=max((r.get("rt_err") or 0) for r in recs.values()) if recs else None,
                 errors=sum(1 for r in recs.values() if r.get("err") and not str(r["err"]).startswith("build:")))
    if route == "bvs":
        extra["orig_len_gate"] = sum(bool(r.get("orig_len_gate")) for r in recs.values())
        extra["orig_full_gate"] = sum(bool(r.get("orig_full_gate")) for r in recs.values())
        extra["n_vectors_median"] = float(np.median([r["n_vectors"] for r in recs.values() if "n_vectors" in r])) \
            if any("n_vectors" in r for r in recs.values()) else None
        extra["_m_best"] = {i: r["m_best"] for i, r in recs.items() if r.get("m_best") is not None}
    if route == "indexer":
        mcs = [r["max_cell_used"] for r in recs.values() if isinstance(r.get("max_cell_used"), (int, float))]
        extra["max_cell_used_median"] = float(np.median(mcs)) if mcs else None
        extra["max_cell_used_p10_p90"] = [float(np.percentile(mcs, 10)), float(np.percentile(mcs, 90))] if mcs else None
    return Ms, len(recs), own, timing, extra


def load_glint(root, arm, n):
    p = os.path.join(root, arm + ".npz")
    if not os.path.exists(p):
        return [None] * n, 0, 0, {}, {}
    Z = np.load(p)
    M = Z["M"]
    Ms = [None if not np.all(np.isfinite(M[i])) else M[i] for i in range(len(M))]
    t = Z["t"]
    if arm.startswith("G_pf"):
        timing = dict(unit="s per frame, one A100, index_blind_fast after warm-up", n=int(len(t)),
                      median=float(np.median(t)), mean=float(t.mean()), p90=float(np.percentile(t, 90)))
    else:
        timing = dict(unit="s per frame = wall of one hybrid_index call over all frames / frames, one A100",
                      n=int(len(M)), wall_s=float(t.sum()), per_frame=float(t.sum() / len(M)))
    return Ms, len(M), sum(m is not None for m in Ms), timing, json.loads(str(Z["stats"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--npz-null", required=True)
    ap.add_argument("--dials", required=True, help="routes dials output root (<root>/<ARM>/*.jsonl)")
    ap.add_argument("--stills-dials", required=True, help="the stills_process dials output root, read-only")
    ap.add_argument("--stills-glint", required=True, help="the stills_process run's glint output root, read-only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D, DN = np.load(a.npz), np.load(a.npz_null)
    n = int(D["n"])
    Q = {"real": [np.asarray(D["q_%d" % i], float) for i in range(n)],
         "null": [np.asarray(DN["q_%d" % i], float) for i in range(n)]}
    assert all(len(q) == 170 for q in Q["real"]), "the scorer assumes 170 peaks per frame"
    rows, flags = {}, {}
    jobs = [("routes", arm, inp, (lambda arm, route=route: load_dials(a.dials, arm, n, route))) for arm, inp, route in ROUTE_ARMS]
    jobs += [("stills_process (read back)", arm, inp, (lambda arm, route=route: load_dials(a.stills_dials, arm, n, route)))
             for arm, inp, route in STILLS_DIALS_REF]
    jobs += [("stills_process (read back)", arm, inp, (lambda arm: load_glint(a.stills_glint, arm, n))) for arm, inp in STILLS_GLINT_REF]
    for src, arm, inp, loader in jobs:
        Ms, covered, own, timing, extra = loader(arm)
        S = [score(M, q) for M, q in zip(Ms, Q[inp])]
        mb = extra.pop("_m_best", None)
        if mb is not None:     # bvs self-check: the driver's count of the kept candidate vs matched_strict on stored M
            extra["m_best_vs_matched_strict_mismatch"] = sum(1 for i, m in mb.items() if m != S[i]["m"])
        f = {k: np.array([s[k] for s in S], dtype=bool) for k in ("found", "correct", "strict", "loose10", "lengate")}
        flags[arm] = f
        row = dict(source=src, input=inp, frames=n, covered=covered, own_indexed=own, timing=timing, extra=extra)
        for k in ("correct", "strict", "loose10", "lengate"):
            c = int(f[k].sum())
            row[k] = c
            row[k + "_pct"] = 100.0 * c / n
            row[k + "_ci95"] = wilson(c, n)
        row["m_median_strict"] = float(np.median([s["m"] for s in S if s["strict"]])) if f["strict"].any() else None
        rows[arm] = row
    pairs = [("V1_bvs_fft1d_blind", "G_pf"),            # the earlier per-frame comparison, at the strict bar
             ("V1_bvs_fft1d_blind", "V2_bvs_rsgs_kc"),  # does the cell help the bare route
             ("V1_bvs_fft1d_blind", "B0_blind_fft1d"),  # bare route vs stills_process, blind
             ("V2_bvs_rsgs_kc", "K3_kc_rsgs"),          # bare grid vs stills_process grid, cell supplied
             ("V2_bvs_rsgs_kc", "K2_kc_xfel"),          # bare grid vs DIALS's best known-cell arm
             ("G_hyb", "V1_bvs_fft1d_blind"),           # GLINT consensus vs the bare blind route
             ("S1_idx_fft1d_rac", "B0_blind_fft1d"),    # consistency: indexer alone vs the full dispatch
             ("S2_idx_fft1d_norac", "S1_idx_fft1d_rac"),       # refine_all_candidates
             ("S3_idx_fft1d_rac_mc200", "S1_idx_fft1d_rac"),   # max_cell 200 vs DIALS's estimate
             ("S4_idx_rsgs_kc_norac", "K3_kc_rsgs"),           # refine_all_candidates, grid with the cell
             ("S3_idx_fft1d_rac_mc200", "V1_bvs_fft1d_blind"),  # real indexer at the bare route's max_cell vs the bare route
             ("S3_idx_fft1d_rac_mc200", "G_pf"),               # DIALS blind indexer vs GLINT single-frame blind
             ("S3_idx_fft1d_rac_mc200", "B0_blind_fft1d")]     # max_cell 200 vs the blind stills_process arm
    paired = {}
    for x, y in pairs:
        k = "%s vs %s" % (x, y)
        if rows[x]["covered"] == n and rows[y]["covered"] == n:
            paired[k] = dict(zip(("only_first", "only_second", "p_exact"), mcnemar(flags[x]["strict"], flags[y]["strict"])))
        else:
            paired[k] = dict(skipped="arm incomplete (%d and %d of %d frames)" % (rows[x]["covered"], rows[y]["covered"], n))
    checks = {}
    for arm, pub in PUBLISHED.items():
        for k, v in pub.items():
            checks["%s %s vs published" % (arm, k)] = dict(now=rows[arm]["extra"].get(k), published=v)
        checks["%s lengate on stored M vs orig_len_gate on cell" % arm] = dict(
            on_M=rows[arm]["lengate"], on_cell=rows[arm]["extra"].get("orig_len_gate"))
    checks["S1 strict vs stills_process B0 strict"] = dict(S1=rows["S1_idx_fft1d_rac"]["strict"], B0=rows["B0_blind_fft1d"]["strict"])
    complete = all(rows[arm]["covered"] == n for arm in HEADLINE)
    complete_secondary = all(rows[arm]["covered"] == n for arm, _, _ in ROUTE_ARMS if arm not in HEADLINE)
    out = dict(frames=n, bar=dict(GATE_TOL=GATE_TOL, GATE_FRAC=GATE_FRAC, GATE_MIN=GATE_MIN, reference="68.7 68.7 108.6 90 90 90"),
               complete=complete, complete_secondary=complete_secondary, rows=rows, paired_strict=paired,
               regression_checks=checks)
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    L = ["# DIALS routes results (cxidb-45, 907 frames x 170 peaks; GLINT 95d51ce gate; DIALS 3.dev.1332)", "",
         "complete: %s (the four headline arms V1, V2, V1n, V2n each cover all %d frames)" % (complete, n),
         "complete_secondary: %s (the S arms)" % complete_secondary, "",
         "| arm | source | input | covered | own 'indexed' | correct lattice | **strict** (95% CI) | >=10 refl | length gate | time per frame |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, r in rows.items():
        t = r["timing"]
        ts = ("%.0f ms median" % (1e3 * t["median"])) if t.get("median") is not None else (
            "%.2f ms (wall/frames)" % (1e3 * t["per_frame"]) if t.get("per_frame") is not None else "-")
        L.append("| %s | %s | %s | %d | %d | %d | **%d/%d = %.1f%%** (%.1f-%.1f) | %d | %d | %s |" % (
            arm, r["source"], r["input"], r["covered"], r["own_indexed"], r["correct"], r["strict"], n, r["strict_pct"],
            r["strict_ci95"][0], r["strict_ci95"][1], r["loose10"], r["lengate"], ts))
    L += ["", "Paired on the strict bar (frames passing only the first / only the second arm, exact McNemar p):", ""]
    L += [("- %s: %d / %d, p = %.3g" % (k, v["only_first"], v["only_second"], v["p_exact"])) if "skipped" not in v
          else ("- %s: not computed, %s" % (k, v["skipped"])) for k, v in paired.items()]
    L += ["", "Regression checks:", ""]
    L += ["- %s: %s" % (k, json.dumps(v)) for k, v in checks.items()]
    L += ["", "Per-arm extras:", ""]
    L += ["- %s: %s" % (arm, json.dumps(r["extra"])) for arm, r in rows.items() if r["source"] == "routes"]
    open(os.path.join(a.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("ROUTES_SCORE_DONE complete=%s complete_secondary=%s" % (complete, complete_secondary))


if __name__ == "__main__":
    main()
