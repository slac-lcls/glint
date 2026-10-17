"""Scoring: every arm, DIALS and GLINT, through GLINT's own gate code at 95d51ce.

strict bar (glint_fast.gpass with the ProK reference in place of LYSO):
    same_lattice(M, ProK)  AND  matched_strict(M, q) / N >= GATE_FRAC (0.25)  AND  matched_strict >= GATE_MIN (10)
    matched_strict = peaks whose three indices q @ M are each within GATE_TOL (0.15, "<") of an integer.
    Every frame here has N = 170 peaks, so the 25% clause (>= 43 peaks) implies the >= 10 clause: the three
    wordings of the strict bar coincide on this set.
also reported, never as the headline:
    correct  = same_lattice(M, ProK) alone (correct-lattice, no peak requirement)
    loose10  = correct AND >= 10 peaks (at chance on dense frames)
    lengate  = sorted column lengths of M within 5% of 68.7/68.7/108.6, the gate behind the earlier
               32.0 / 61.9 / 49.9 / 97.8 (prok_rerun_s3df.py:len_match), kept only for continuity.
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
DIALS_ARMS = [("B0_blind_fft1d", "real"), ("K1_kc_fft1d", "real"), ("K2_kc_xfel", "real"),
              ("K3_kc_rsgs", "real"), ("N2_kc_xfel_scrambled", "null")]
GLINT_ARMS = [("G_pf", "real"), ("G_hyb", "real"), ("G_kc", "real"),
              ("G_pf_null", "null"), ("G_hyb_null", "null"), ("G_kc_null", "null")]
PUBLISHED = {"G_pf": 290, "G_hyb": 887}     # the earlier 32.0% and 97.8% (887/907), length gate


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


def load_dials(root, arm, n):
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
    ok = np.array([bool(recs[i].get("ok")) for i in sorted(recs) if recs[i].get("t_proc") is not None and i not in first])
    timing = dict(unit="s per frame, one EPYC core, stills_process dispatch (hit finder + index + refinement), "
                       "spot finding excluded",
                  n=int(len(t)), median=float(np.median(t)) if len(t) else None, mean=float(t.mean()) if len(t) else None,
                  p90=float(np.percentile(t, 90)) if len(t) else None,
                  median_ok=float(np.median(t[ok])) if ok.any() else None,
                  median_fail=float(np.median(t[~ok])) if (~ok).any() else None,
                  total_cpu_h=float(t.sum() / 3600) if len(t) else None)
    own = sum(bool(r.get("ok")) for r in recs.values())
    build_fail = sum(1 for r in recs.values() if str(r.get("err") or "").startswith("build:"))
    return Ms, len(recs), own, timing, dict(build_fail=build_fail,
                                             max_rt_err=max((r.get("rt_err") or 0) for r in recs.values()) if recs else None)


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
    ap.add_argument("--dials", required=True)
    ap.add_argument("--glint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D, DN = np.load(a.npz), np.load(a.npz_null)
    n = int(D["n"])
    Q = {"real": [np.asarray(D["q_%d" % i], float) for i in range(n)],
         "null": [np.asarray(DN["q_%d" % i], float) for i in range(n)]}
    assert all(len(q) == 170 for q in Q["real"]), "the scorer assumes 170 peaks per frame"
    rows, flags = {}, {}
    for kind, arms, loader, root in (("DIALS", DIALS_ARMS, load_dials, a.dials), ("GLINT", GLINT_ARMS, load_glint, a.glint)):
        for arm, inp in arms:
            Ms, covered, own, timing, extra = loader(root, arm, n)
            S = [score(M, q) for M, q in zip(Ms, Q[inp])]
            f = {k: np.array([s[k] for s in S]) for k in ("found", "correct", "strict", "loose10", "lengate")}
            flags[arm] = f
            row = dict(program=kind, input=inp, frames=n, covered=covered, own_indexed=own, timing=timing, extra=extra)
            for k in ("correct", "strict", "loose10", "lengate"):
                c = int(f[k].sum())
                row[k] = c
                row[k + "_pct"] = 100.0 * c / n
                row[k + "_ci95"] = wilson(c, n)
            row["m_median_strict"] = float(np.median([s["m"] for s in S if s["strict"]])) if f["strict"].any() else None
            rows[arm] = row
    pairs = [("K2_kc_xfel", "G_hyb"), ("K2_kc_xfel", "G_kc"), ("K1_kc_fft1d", "B0_blind_fft1d"),
             ("K2_kc_xfel", "K1_kc_fft1d"), ("K3_kc_rsgs", "K1_kc_fft1d"), ("B0_blind_fft1d", "G_pf")]
    paired = {"%s vs %s" % (x, y): dict(zip(("only_first", "only_second", "p_exact"), mcnemar(flags[x]["strict"], flags[y]["strict"])))
              for x, y in pairs}
    checks = {}
    pub = D["glint_prok"] if "glint_prok" in D.files else None
    for arm, v in PUBLISHED.items():
        checks[arm + " lengate vs published"] = dict(now=rows[arm]["lengate"], published=v)
    if pub is not None:
        checks["G_pf lengate vs prok_q.npz glint_prok (July)"] = dict(now=rows["G_pf"]["lengate"], stored=int(pub.sum()))
    complete = all(rows[arm]["covered"] == n for arm, _ in DIALS_ARMS + GLINT_ARMS)
    out = dict(frames=n, bar=dict(GATE_TOL=GATE_TOL, GATE_FRAC=GATE_FRAC, GATE_MIN=GATE_MIN, reference="68.7 68.7 108.6 90 90 90"),
               complete=complete, rows=rows, paired_strict=paired, regression_checks=checks)
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    L = ["# stills_process head-to-head results (cxidb-45, 907 frames x 170 peaks; GLINT 95d51ce gate; DIALS 3.dev.1332)", "",
         "complete: %s (every arm covers all %d frames)" % (complete, n), "",
         "| arm | input | covered | own 'indexed' | correct lattice | **strict** (95% CI) | >=10 refl | length gate | time per frame |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm, r in rows.items():
        t = r["timing"]
        ts = ("%.0f ms median" % (1e3 * t["median"])) if t.get("median") is not None else (
            "%.2f ms (wall/frames)" % (1e3 * t["per_frame"]) if t.get("per_frame") is not None else "-")
        L.append("| %s | %s | %d | %d | %d | **%d/%d = %.1f%%** (%.1f-%.1f) | %d | %d | %s |" % (
            arm, r["input"], r["covered"], r["own_indexed"], r["correct"], r["strict"], n, r["strict_pct"],
            r["strict_ci95"][0], r["strict_ci95"][1], r["loose10"], r["lengate"], ts))
    L += ["", "Paired on the strict bar (frames indexed by only the first / only the second arm, exact McNemar p):", ""]
    L += ["- %s: %d / %d, p = %.3g" % (k, v["only_first"], v["only_second"], v["p_exact"]) for k, v in paired.items()]
    L += ["", "Regression checks (length gate, for continuity only):", ""]
    L += ["- %s: %s" % (k, json.dumps(v)) for k, v in checks.items()]
    open(os.path.join(a.out, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("J12_SCORE_DONE complete=%s" % complete)


if __name__ == "__main__":
    main()
