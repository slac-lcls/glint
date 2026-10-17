#!/usr/bin/env python
"""Step 3: bin the per-frame records by peaks per frame and write the tables.

    python j4_bin.py OUTDIR

Reads OUTDIR/{r0016,r0278,r0278z060,r0058,cxidb45}_frames.csv (and cxidb45_thin_frames.csv), whatever exists,
and writes OUTDIR/J4_bins.csv (every run x arm x bar x bin, with Wilson 95% intervals),
OUTDIR/J4_table.md (the readable table) and OUTDIR/J4_accuracy_vs_peaks.png when matplotlib imports.
Bars, from (correct, m, N_f) per frame, GATE_* read from the pinned glint_fast:
  lattice   the cell is the reference lattice (same_lattice), no coverage requirement
  s25       lattice AND m >= 0.25 N_f                       (gpass' first flag; FACTS sf_strict_of120)
  strict    lattice AND m >= 0.25 N_f AND m >= 10           (the strict bar as FACTS words it)
  cover     m >= 0.25 N_f AND m >= 10, cell identity ignored (how often the coverage part passes alone)
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j4_common as C  # noqa: E402

EDGES = [6, 10, 15, 20, 30, 40, 60, 100, 150, 250, 10 ** 9]
RUNS = ("r0016", "r0278", "r0278z060", "r0058", "cxidb45")
ARMS = (("blind", "real"), ("blind", "scr"), ("known", "real"), ("known", "scr"))


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def flags(r, kind, frac, gmin):
    Nf = int(r["Nf"])
    cor = int(r["correct"] if kind == "blind" else r["known_ok"])
    m = int(r["m"] if kind == "blind" else r["mk"])
    cov = m >= frac * Nf and m >= gmin
    return dict(lattice=cor, s25=int(cor and m >= frac * Nf), strict=int(cor and cov), cover=int(cov))


def label(lo, hi):
    return f"{lo}-{hi - 1}" if hi < 10 ** 9 else f">={lo}"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else C.SCRATCH
    C.pin()
    from glint.glint_fast import GATE_FRAC, GATE_MIN
    rows_out, md = [], []
    curves = {}
    for run in RUNS:
        p = os.path.join(out, f"{run}_frames.csv")
        if not os.path.exists(p):
            md.append(f"\n## {run}: no per-frame file (task failed or reference refused; see {run}_summary.json)\n")
            continue
        recs = load(p)
        summ = json.load(open(os.path.join(out, f"{run}_summary.json")))
        ref = summ["reference"]
        qp = summ.get("qlists") or {}
        chk = []
        if qp:
            chk.append(f"frame count {qp.get('n_frames')} against {qp.get('expect_frames')} on record: "
                       f"{qp.get('frame_count_check')}")
        chk.append(f"vote over best cells {ref.get('top1_vote', {}).get('cell')} "
                   f"(support {ref.get('top1_vote', {}).get('support')}), is the reference: {ref.get('top1_vote_is_ref')}")
        if "record_check" in ref:
            rc = ref["record_check"]
            chk.append(f"vote edges {rc['vote_edges']} against {rc['record_edges']} on record "
                       f"({rc['record_source']}): within 5% {rc['edges_within_5pct']}")
        md.append(f"\n## {run}: {summ['n_frames']} frames; reference {ref['ref_cell']} ({ref['source']})\n")
        md.append("Checks: " + "; ".join(chk) + ".\n")
        md.append("| peaks per frame | n | blind strict | blind lattice | null strict (scrambled) | "
                  "known-cell strict | blind strict among known-cell strict | cover alone, real / scrambled |")
        md.append("|---|---|---|---|---|---|---|---|")
        # the bins, then one unbinned row over every frame with >= 6 peaks (the run's total)
        for lo, hi, lab in ([(l_, h_, label(l_, h_)) for l_, h_ in zip(EDGES[:-1], EDGES[1:])]
                            + [(EDGES[0], EDGES[-1], f"all >={EDGES[0]}")]):
            sel = {arm: [r for r in recs if r["arm"] == arm and lo <= int(r["Nf"]) < hi] for arm in ("real", "scr")}
            n = len(sel["real"])
            if n == 0:
                continue
            res = {}
            for kind, arm in ARMS:
                fl = [flags(r, kind, GATE_FRAC, GATE_MIN) for r in sel[arm]]
                for bar in ("lattice", "s25", "strict", "cover"):
                    k = sum(f[bar] for f in fl)
                    ci = C.wilson(k, len(fl))
                    res[(kind, arm, bar)] = (k, len(fl))
                    rows_out.append(dict(run=run, bin=lab, lo=lo, hi=hi, kind=kind, arm=arm, bar=bar,
                                         n=len(fl), k=k, rate=round(k / len(fl), 4) if fl else None,
                                         ci_lo=round(ci[0], 4), ci_hi=round(ci[1], 4)))
            # conditional: blind strict among the frames the reference cell indexes at the strict bar
            kn = [(flags(r, "blind", GATE_FRAC, GATE_MIN)["strict"], flags(r, "known", GATE_FRAC, GATE_MIN)["strict"])
                  for r in sel["real"]]
            den = sum(b for _, b in kn)
            num = sum(a and b for a, b in kn)
            f = lambda kn_: f"{kn_[0]}/{kn_[1]} ({100 * kn_[0] / kn_[1]:.0f}%)" if kn_[1] else "-"
            md.append(f"| {lab} | {n} | {f(res[('blind', 'real', 'strict')])} | {f(res[('blind', 'real', 'lattice')])} | "
                      f"{f(res[('blind', 'scr', 'strict')])} | {f(res[('known', 'real', 'strict')])} | {f((num, den))} | "
                      f"{f(res[('blind', 'real', 'cover')])} / {f(res[('blind', 'scr', 'cover')])} |"
                      + ("  (n < 20)" if n < 20 else ""))
            if lab.startswith("all"):
                continue
            curves.setdefault(run, []).append((lo, hi, res[("blind", "real", "strict")], res[("blind", "scr", "strict")],
                                               res[("known", "real", "strict")]))
    # thinning (cxidb45), its own table: never pooled with the natural bins
    tp = os.path.join(out, "cxidb45_thin_frames.csv")
    if os.path.exists(tp):
        t = load(tp)
        md.append("\n## cxidb45, random thinning of the 170-peak lists (not natural frames; see README)\n")
        md.append("| peaks kept k | n | blind strict | blind lattice | null strict (scrambled) | known-cell strict |")
        md.append("|---|---|---|---|---|---|")
        for k in sorted({int(r["k"]) for r in t}):
            res = {}
            for kind, arm in ARMS:
                fl = [flags(r, kind, GATE_FRAC, GATE_MIN) for r in t if int(r["k"]) == k and r["arm"] == arm]
                for bar in ("lattice", "s25", "strict", "cover"):
                    kk = sum(x[bar] for x in fl)
                    ci = C.wilson(kk, len(fl))
                    res[(kind, arm, bar)] = (kk, len(fl))
                    rows_out.append(dict(run="cxidb45_thin", bin=str(k), lo=k, hi=k + 1, kind=kind, arm=arm, bar=bar,
                                         n=len(fl), k=kk, rate=round(kk / len(fl), 4) if fl else None,
                                         ci_lo=round(ci[0], 4), ci_hi=round(ci[1], 4)))
            f = lambda kn_: f"{kn_[0]}/{kn_[1]} ({100 * kn_[0] / kn_[1]:.0f}%)" if kn_[1] else "-"
            md.append(f"| {k} | {res[('blind', 'real', 'strict')][1]} | {f(res[('blind', 'real', 'strict')])} | "
                      f"{f(res[('blind', 'real', 'lattice')])} | {f(res[('blind', 'scr', 'strict')])} | "
                      f"{f(res[('known', 'real', 'strict')])} |")
    if rows_out:
        with open(os.path.join(out, "J4_bins.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    head = ("# Per-frame accuracy against peaks per frame (GLINT 95d51ce, index_blind_fast at defaults)\n\n"
            "Strict bar: the frame's best blind cell is the reference lattice AND indexes at least 25% of the "
            "frame's peaks AND at least 10 (|hkl - round| < 0.15). Null: the same frames azimuth-scrambled. "
            "Known-cell: the reference cell supplied (index_known_fast). Counts are frames; percentages rounded. "
            "No CrystFEL step. Bins with n < 20 are marked.\n")
    open(os.path.join(out, "J4_table.md"), "w").write(head + "\n".join(md) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for run, cv in curves.items():
            x = [np.sqrt(lo * min(hi, 400)) for lo, hi, *_ in cv]
            y = [b[0] / b[1] for _, _, b, _, _ in cv]
            lo_ = [y_ - C.wilson(*b)[0] for y_, (_, _, b, _, _) in zip(y, cv)]
            hi_ = [C.wilson(*b)[1] - y_ for y_, (_, _, b, _, _) in zip(y, cv)]
            ax.errorbar(x, y, yerr=[lo_, hi_], marker="o", capsize=2, label=run)
            ax.plot(x, [s[0] / s[1] for _, _, _, s, _ in cv], ls=":", color=ax.lines[-1].get_color(), lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("peaks per frame (bin center)")
        ax.set_ylabel("fraction of frames, best blind cell correct (strict bar)")
        ax.set_ylim(0, 1)
        ax.legend(title="dotted: scrambled", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "J4_accuracy_vs_peaks.png"), dpi=150)
    except Exception as e:  # the table is the result; the figure is a convenience
        print(f"figure skipped: {type(e).__name__}: {e}")
    print(open(os.path.join(out, "J4_table.md")).read())


if __name__ == "__main__":
    main()
