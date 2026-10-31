#!/usr/bin/env python
"""Step 2: score each frame's best blind cell against the reference cell, with its controls.

    python accuracy_score.py TAG OUTDIR [--max-frames N] [--dry] [--seed 20260928]
    TAG: r0016 r0278 r0278z060 r0058 (q lists from accuracy_qlists.py), cxidb45 (prok_q.npz of the pinned tree),
         ctrl17 (driver check on the 120 cxidb-17 frames; not part of the default array)

For every frame, four arms, all at the pinned tree's default settings:
  blind/real   glint.glint_fast.index_blind_fast(q): the shipped per-frame indexer, top-1 cell
  blind/scr    the same on the frame azimuth-scrambled by experiments/scramble_qlist.scramble_frames
               (|q| and q_z kept, lattice destroyed): the arm where the indexer can find nothing
  known/real   glint.hybrid_stream.index_known_fast(frames, REF): the reference cell supplied
  known/scr    the same on the scrambled frames: the chance level of the known-cell arm
Per frame it records the peak count N_f, whether the cell is the reference lattice
(multishot.same_lattice, the shipped tolerances) and the number m of peaks it indexes
(glint_fast.matched_strict, |hkl - round| < GATE_TOL on all three indices). The bars are applied
in accuracy_bin.py from (correct, m, N_f). cxidb45 adds a thinning arm: k peaks drawn at random from each
frame's 170, k in K_THIN, because every cxidb-45 frame on S3DF has exactly 170 peaks.
Writes OUTDIR/TAG_frames.csv, OUTDIR/TAG_summary.json (and TAG_thin_frames.csv for cxidb45).
"""
import argparse
import csv
import json
import os
import platform
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accuracy_common as C  # noqa: E402

K_THIN = [8, 12, 16, 20, 30, 40, 60, 80, 120, 170]
TAGS = ("r0016", "r0278", "r0278z060", "r0058", "cxidb45", "ctrl17")
LCLS = ("r0016", "r0278", "r0278z060", "r0058")
# cells on record for the internal-reference runs (reduced edges, A), compared with the vote below
# as a check only; none of them is used as the reference.
RECORD = {"r0278": ([44.5, 70.4, 90.4], "GLINT consensus at 0.058 m, z-scan job 34400191 (Aug)"),
          "r0278z060": ([46.6, 72.7, 92.9], "GLINT all-frame vote at 0.060 m, dump job 34492864 (Aug)"),
          "r0058": ([43.6, 67.8, 89.1], "GLINT consensus at 0.050 m, LUTE sfx_mfxl1038923_r58_crystfel README")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=TAGS)
    ap.add_argument("outdir")
    ap.add_argument("--max-frames", type=int, default=None, help="dry run only")
    ap.add_argument("--dry", action="store_true", help="read TAG_qlists_dryrun.npz")
    ap.add_argument("--seed", type=int, default=20260928, help="scramble_qlist.py's default seed")
    a = ap.parse_args()
    commit = C.pin()
    knobs = C.default_settings_guard()

    import torch
    from glint.geom import clean_q
    from glint.glint_fast import (GATE_FRAC, GATE_MIN, GATE_TOL, LYSO, gpass, index_blind_fast,
                                  load, matched_strict)
    from glint.hybrid_stream import index_known_fast
    from glint.lattice import cell_to_Ar
    from glint.multishot import CONSENSUS_MIN_FRAC, CONSENSUS_MIN_LEAD, consensus_cell, same_lattice
    from scramble_qlist import scramble_frames

    # ---------------------------------------------------------------- frames
    qprov = None
    if a.tag in LCLS:
        src = os.path.join(a.outdir, f"{a.tag}_qlists" + ("_dryrun" if a.dry else "") + ".npz")
        z = np.load(src)
        frames = [np.asarray(z[f"q_{i}"], float) for i in range(int(z["n"]))]
        events = list(np.asarray(z["events"], int)) if "events" in z.files else [-1] * len(frames)
        try:
            qprov = json.load(open(src[:-4] + ".json"))
            print(f"q lists: {qprov.get('n_frames')} frames, on record {qprov.get('expect_frames')}: "
                  f"{qprov.get('frame_count_check')}", flush=True)
        except (OSError, ValueError) as e:
            qprov = {"error": f"{type(e).__name__}: {e}"}
    elif a.tag == "cxidb45":
        src = os.path.join(C.GL, "experiments", "prok_q.npz")
        z = np.load(src, allow_pickle=True)
        frames = [np.asarray(z[f"q_{i}"], float) for i in range(int(z["n"]))]
        events = list(range(len(frames)))
    else:  # ctrl17, oracle_blind.py's selection
        src = os.path.join(C.GL, "experiments", "frames_cxidb_clean.txt")
        frames = [q for q in load(src) if len(q) >= 6][:120]
        events = list(range(len(frames)))
    if a.max_frames:
        frames, events = frames[:a.max_frames], events[:a.max_frames]
    frames = [clean_q(np.asarray(q, float)) for q in frames]
    n = len(frames)
    print(f"score {a.tag}: {n} frames from {src}", flush=True)

    # ---------------------------------------------------------------- blind arm (real, scrambled)
    t0 = time.time()
    index_blind_fast(frames[0])                         # warm-up, as glint_fast's own __main__ does
    best = [index_blind_fast(q) for q in frames]
    t_blind = time.time() - t0
    scr = scramble_frames(frames, a.seed)
    t0 = time.time()
    best_s = [index_blind_fast(q) for q in scr]
    t_blind_s = time.time() - t0
    print(f"blind arm: {t_blind:.0f} s real, {t_blind_s:.0f} s scrambled", flush=True)

    # ---------------------------------------------------------------- reference cell
    ref_note = {}
    top1 = [np.asarray(M, float) for M in best if M is not None]
    Mc, sup = consensus_cell(top1, min_frac=CONSENSUS_MIN_FRAC, min_lead=CONSENSUS_MIN_LEAD) if top1 else (None, 0)
    ref_note["top1_vote"] = dict(n_cells=len(top1), support=int(sup or 0),
                                 cell=None if Mc is None else [round(x, 2) for x in C.cell_params(Mc)],
                                 min_frac=CONSENSUS_MIN_FRAC, min_lead=CONSENSUS_MIN_LEAD)
    if a.tag == "r0016":
        REF = cell_to_Ar(79.327, 79.461, 38.406, 90.111, 89.981, 90.083)
        ref_note["source"] = "external: /sdf/data/lcls/ds/mfx/mfxx49820/results/btx/cell/sample2.cell"
    elif a.tag == "cxidb45":
        REF = np.diag([68.7, 68.7, 108.6])
        ref_note["source"] = "external: P4(3)2(1)2 68.7/68.7/108.6 A, experiments/phaseB_prok.py PROK at 95d51ce"
    elif a.tag == "ctrl17":
        REF = np.asarray(LYSO, float)
        ref_note["source"] = "external: glint_fast.LYSO 79.02/79.02/37.98 A (glint_fast's built-in reference)"
    else:
        REF = Mc
        ref_note["source"] = ("internal: vote over every frame's best blind cell (consensus_cell, shipped "
                              "min_frac/min_lead); no unit cell was ever recorded for this sample")
    if REF is None:
        json.dump(dict(job="per_frame_accuracy_vs_peaks", tag=a.tag, commit=commit, argv=sys.argv, n_frames=n, reference=ref_note,
                       verdict="NO REFERENCE: the vote over best cells refused; nothing scored"),
                  open(os.path.join(a.outdir, f"{a.tag}_summary.json"), "w"), indent=1, default=str)
        print("NO REFERENCE CELL: the vote refused; nothing scored", flush=True)
        return
    REF = np.asarray(REF, float)
    ref_note["ref_cell"] = [round(x, 3) for x in C.cell_params(REF)]
    ref_note["top1_vote_is_ref"] = bool(Mc is not None and same_lattice(Mc, REF))
    if a.tag in RECORD and Mc is not None:
        rec_e, rec_src = RECORD[a.tag]
        ve = sorted(C.cell_params(Mc)[:3])
        ref_note["record_check"] = dict(record_edges=rec_e, record_source=rec_src,
                                        vote_edges=[round(x, 2) for x in ve],
                                        edges_within_5pct=bool(all(abs(x - y) <= 0.05 * y for x, y in zip(ve, rec_e))))
    print(f"reference {ref_note['ref_cell']}  ({ref_note['source']}); top-1 vote {ref_note['top1_vote']}", flush=True)

    # ---------------------------------------------------------------- known-cell arm (real, scrambled)
    t0 = time.time()
    kres, _ = index_known_fast(frames, REF)
    kres_s, _ = index_known_fast(scr, REF)
    t_known = time.time() - t0

    # ---------------------------------------------------------------- the strict bar's code path
    # gpass() is the published gate with LYSO hard-wired; check that our (correct, m) bars reproduce
    # it exactly when the reference is LYSO, on whatever cells these frames gave.
    ident = 0
    for M, q in zip(best[:50], frames[:50]):
        okL = M is not None and same_lattice(M, LYSO)
        m = matched_strict(M, q) if M is not None else 0
        mine = (int(m / len(q) >= GATE_FRAC), int(m >= GATE_MIN)) if okL else (0, 0)
        ident += int(tuple(gpass(M, q)) == mine)
    if ident != min(50, n):
        raise SystemExit(f"bar code path disagrees with glint_fast.gpass on {min(50, n) - ident} frames")

    vref = abs(np.linalg.det(REF))

    def row(arm, i, q, M, kr):
        Nf = len(q)
        rec = dict(tag=a.tag, arm=arm, i=i, event=int(events[i]), Nf=Nf, none=int(M is None))
        if M is None:
            rec.update(correct=0, m=0, a=None, b=None, c=None, al=None, be=None, ga=None, vol_ratio=None)
        else:
            p = C.cell_params(M)
            rec.update(correct=int(same_lattice(M, REF)), m=matched_strict(M, q),
                       a=round(p[0], 2), b=round(p[1], 2), c=round(p[2], 2), al=round(p[3], 2),
                       be=round(p[4], 2), ga=round(p[5], 2), vol_ratio=round(p[6] / vref, 3))
        Mk = kr["M"]
        rec.update(known_ok=int(Mk is not None), mk=matched_strict(Mk, q) if Mk is not None else 0)
        return rec

    rows = [row("real", i, q, M, k) for i, (q, M, k) in enumerate(zip(frames, best, kres))]
    rows += [row("scr", i, q, M, k) for i, (q, M, k) in enumerate(zip(scr, best_s, kres_s))]
    cols = list(rows[0].keys())
    out_csv = os.path.join(a.outdir, f"{a.tag}_frames.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    def bars(rs, cor="correct", mm="m"):
        L = sum(r[cor] for r in rs)
        S25 = sum(r[cor] and r[mm] >= GATE_FRAC * r["Nf"] for r in rs)
        S = sum(r[cor] and r[mm] >= GATE_FRAC * r["Nf"] and r[mm] >= GATE_MIN for r in rs)
        return dict(n=len(rs), lattice=int(L), correct_and_25pct=int(S25), strict=int(S))

    tot = {}
    for arm in ("real", "scr"):
        rs = [r for r in rows if r["arm"] == arm]
        tot[f"blind_{arm}"] = bars(rs)
        tot[f"known_{arm}"] = bars(rs, "known_ok", "mk")
    print("totals:", json.dumps(tot), flush=True)
    if a.tag == "ctrl17":
        print(f"driver check against FACTS (95d51ce): lattice {tot['blind_real']['lattice']} vs sf_lattice_of120 85, "
              f">=25% {tot['blind_real']['correct_and_25pct']} vs sf_strict_of120 79 (CPU record; +/-1 is device)", flush=True)

    # ---------------------------------------------------------------- thinning arm (cxidb45 only)
    thin = None
    if a.tag == "cxidb45":
        t0 = time.time()
        trows = []
        for k in K_THIN:
            fk = [q if k >= len(q) else q[np.sort(np.random.default_rng([a.seed, i, k]).choice(len(q), k, replace=False))]
                  for i, q in enumerate(frames)]
            sk = scramble_frames(fk, a.seed)
            bk = [index_blind_fast(q) for q in fk]
            bks = [index_blind_fast(q) for q in sk]
            kk, _ = index_known_fast(fk, REF)
            kks, _ = index_known_fast(sk, REF)
            for i in range(n):
                trows.append(dict(row("real", i, fk[i], bk[i], kk[i]), k=k))
                trows.append(dict(row("scr", i, sk[i], bks[i], kks[i]), k=k))
            print(f"  thinning k={k}: blind/real lattice {sum(r['correct'] for r in trows if r['k'] == k and r['arm'] == 'real')}/{n}", flush=True)
        with open(os.path.join(a.outdir, f"{a.tag}_thin_frames.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
            w.writeheader()
            w.writerows(trows)
        thin = dict(K=K_THIN, seconds=round(time.time() - t0, 1),
                    draw="k rows without replacement, numpy default_rng([seed, frame, k]); k=170 keeps every row")

    import glint.glint_fast as gf
    summ = dict(job="per_frame_accuracy_vs_peaks", tag=a.tag, commit=commit, argv=sys.argv, input=src, n_frames=n, qlists=qprov,
                reference=ref_note, totals=tot, thinning=thin,
                bars=dict(lattice="same_lattice(M, REF): rtol 0.05, ctol 0.06, vtol 0.10",
                          correct_and_25pct=f"lattice AND m >= {GATE_FRAC} N_f (gpass first flag)",
                          strict=f"lattice AND m >= {GATE_FRAC} N_f AND m >= {GATE_MIN} (FACTS wording)",
                          m=f"matched_strict: peaks with all |hkl - round(hkl)| < {GATE_TOL}"),
                indexer="glint.glint_fast.index_blind_fast (top-1), default settings",
                defaults={k: getattr(gf, k) for k in ("NTOP", "KEEP", "STEPS", "QPOW", "TOL", "SCORER",
                                                       "ANNEAL_ITERS", "ANNEAL_FP32", "REFINER", "QDIST", "DETREJ")},
                knobs_checked_unset=knobs, scramble=dict(fn="experiments/scramble_qlist.scramble_frames", seed=a.seed),
                known_cell="glint.hybrid_stream.index_known_fast(frames, REF), batch 32",
                gpass_identity_frames=int(ident), seconds=dict(blind=round(t_blind, 1), blind_scr=round(t_blind_s, 1),
                                                               known=round(t_known, 1)),
                host=socket.gethostname(), device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                torch=torch.__version__, python=platform.python_version(),
                conda_env=os.environ.get("CONDA_DEFAULT_ENV"), crystfel="none (no CrystFEL step)",
                slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
    json.dump(summ, open(os.path.join(a.outdir, f"{a.tag}_summary.json"), "w"), indent=1, default=str)
    print(f"wrote {out_csv} and {a.tag}_summary.json", flush=True)


if __name__ == "__main__":
    main()
