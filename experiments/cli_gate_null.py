#!/usr/bin/env python3
"""Calibrate glint_cli's written-crystal gate (--gate) on a lattice-free null, for the CLI's OWN search.

With --cell, glint_cli runs hybrid_index(frames, Mc_known=cell): every frame is blind-indexed N-best
(index_blind_nbest, nbest=3); the first hypothesis with same_lattice(c, cell) is the registration (path
"nbest"); otherwise the per-frame known-cell rescue index_known_gpu_cell(q, cell) is, whenever it returns
the cell (path "rescue"), which a known-cell search does by construction. --gate strict then asks
matched_strict(M, q) >= GATE_MIN and >= GATE_FRAC * n. The gate therefore sees the best of two searches, a
blind one over all cells and a per-frame known-cell one, and its chance rate is the rate at which that
route's registration of a frame with NO lattice clears the bar. This is not StreamDriver's null
(live_gate_null.py, glint#214: the batched index_fused engine at the driver's depth and count), so its floor
constants do not carry over; this script measures this route's own.

`measure` runs hybrid_index itself (the CLI's call: Mc_known, nbest=3, no cascade, no escalation) on
  * every real frame,
  * a held-out azimuth-scrambled copy of every frame, rng default_rng([null_seed, i]) (scramble_qlist.py's
    stream, glint#213),
  * k_fit further scrambled copies per frame, rng default_rng([fit_seed, i, k]), used ONLY to fit,
and saves per frame the peak count, matched_strict of the registration (0 when there is none) and the path
it came from (0 none, 1 nbest, 2 rescue), for the real frame, the held-out copy and each fit copy, plus the
real registration M. The path is read off the run: a frame is "rescue" iff hybrid_index called the rescue on
it. With a known cell every frame is indexed independently, so the frames are sharded over processes, one
real frame and its copies per task; `--check-whole` re-runs the real frames as ONE hybrid_index call and
requires identical counts. Scrambling rotates each peak by its own azimuth about the beam: |q| and q_z (so
each peak's excitation error) are kept, the lattice is removed.

`fit` fits n_matched >= a*n + b (+ c*sqrt(n)) at quantile 1 - alpha of the fit copies (a linear quantile
regression, live_gate_null.py's fit_floor, glint#214), applies it ON TOP of the strict bar (the gate `--gate floor`
implements), and reports: null on the held-out copies and on the fit copies, per peak-count band and per path;
a frame-split cross-validation (fit on half the frames, score the other half's copies); and the real frames
kept against the strict count. `confirm` asks whether the strict frames the floor refuses are crystals, using
the published xgandalf arm on the same q (xgd480_fix.txt) finding the same lattice within 2 deg.

Committed results (RESULTS_cli_gate_null.md): cli_gate_null_480.npz (S3DF ~/q480_fix.txt, md5 d6d86c1b...) and
cli_gate_null_120.npz (the committed frames_cxidb_clean.txt), both --cell LYSO, 32 fit copies per frame, CPU torch.
fit and confirm read the npz (confirm also the xgandalf file) and need torch only for the shipped floor_value:

  python experiments/cli_gate_null.py measure --input ~/q480_fix.txt --k-fit 32 --procs 11 --check-whole \\
      --out experiments/cli_gate_null_480.npz                      # CPU torch: ~42,000 CPU-s; resumes if killed
  python experiments/cli_gate_null.py fit experiments/cli_gate_null_480.npz
  python experiments/cli_gate_null.py fit experiments/cli_gate_null_120.npz --floor 0.0246,5.39,1.188
  python experiments/cli_gate_null.py confirm experiments/cli_gate_null_480.npz --floor 0.0246,5.39,1.188 \\
      --xgandalf ~/xgd480_fix.txt
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)                          # live_gate_null (the fit helpers)

LYSO_CELL = "79.02,79.02,37.98,90,90,90"          # glint_fast.LYSO; glint_cli --cell "79.02 79.02 37.98 90 90 90"
PATHS = ("none", "nbest", "rescue")


def parse_cell(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 6:
        raise SystemExit(f"--cell needs 6 numbers, got {s!r}")
    return v


def load_frames(path):
    from glint.glint_fast import load
    return [np.asarray(q, float) for q in load(os.path.expanduser(path)) if len(q) >= 6]   # glint_cli --min-peaks 6


# ------------------------------------------------------------------------------------------ measure ----
def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)


def route(frames, Mc, nbest=3):
    """glint_cli's --cell route on `frames`: hybrid_index as the CLI calls it. Returns (matched, path, Ms):
    matched_strict of each registration (0 when none), its path code (0 none, 1 nbest, 2 rescue) and M.
    The rescue is wrapped only to record which frames reached it; its arguments and result pass through."""
    import glint.hybrid_stream as hs
    from glint.glint_fast import matched_strict
    reached = set()
    orig = hs.index_known_gpu_cell

    def recording(q, Mcell, *a, **kw):
        reached.add(id(q))
        return orig(q, Mcell, *a, **kw)
    hs.index_known_gpu_cell = recording
    try:
        res, _ = hs.hybrid_index(frames, Mc_known=Mc, nbest=nbest, cascade=None, escalate=None)
    finally:
        hs.index_known_gpu_cell = orig
    m = np.zeros(len(frames), int); p = np.zeros(len(frames), int); Ms = []
    for j, (q, r) in enumerate(zip(frames, res)):
        M = r["M"]
        Ms.append(None if M is None else np.asarray(M, float))
        if M is None:
            continue
        m[j] = matched_strict(np.asarray(M, float), q)
        p[j] = 2 if id(q) in reached else 1
    return m, p, Ms


def _task(args):
    from glint.multilattice import scramble_azimuth
    i, q, cell, k_fit, fit_seed, null_seed = args
    from glint.lattice import cell_to_Ar
    Mc = cell_to_Ar(*cell)
    copies = [q, scramble_azimuth(q, np.random.default_rng([null_seed, i]))]
    copies += [scramble_azimuth(q, np.random.default_rng([fit_seed, i, k])) for k in range(k_fit)]
    t0 = time.time()
    m, p, Ms = route(copies, Mc)
    return i, m, p, Ms[0], time.time() - t0


def measure(a):
    import multiprocessing as mp
    real = load_frames(a.input)
    if a.limit:
        real = real[:a.limit]
    F = len(real)
    n = np.array([len(q) for q in real], int)
    cell = parse_cell(a.cell)
    K = a.k_fit
    m_real = np.zeros(F, int); p_real = np.zeros(F, int)
    m_null = np.zeros(F, int); p_null = np.zeros(F, int)
    m_fit = np.zeros((F, K), int); p_fit = np.zeros((F, K), int)
    M_real = np.full((F, 3, 3), np.nan)
    secs = np.zeros(F)
    done_mask = np.zeros(F, bool)
    in_md5 = hashlib.md5(open(os.path.expanduser(a.input), "rb").read()).hexdigest()
    key = json.dumps(dict(md5=in_md5, cell=cell, k=K, fit_seed=a.fit_seed, null_seed=a.null_seed, F=F))
    part = a.out + ".partial.npz"                            # checkpoint: a killed run resumes where it stopped
    if os.path.exists(part):
        c = np.load(part)
        if str(c["key"]) == key:
            done_mask = c["done"]
            m_real, p_real, m_null, p_null = c["m_real"], c["p_real"], c["m_null"], c["p_null"]
            m_fit, p_fit, M_real, secs = c["m_fit"], c["p_fit"], c["M_real"], c["secs"]
            print(f"resuming from {part}: {int(done_mask.sum())}/{F} frames done", flush=True)

    def checkpoint():
        np.savez_compressed(part + ".tmp.npz", key=key, done=done_mask, m_real=m_real, p_real=p_real, m_null=m_null,
                            p_null=p_null, m_fit=m_fit, p_fit=p_fit, M_real=M_real, secs=secs)
        os.replace(part + ".tmp.npz", part)
    t0 = time.time()
    tasks = [(i, q, cell, K, a.fit_seed, a.null_seed) for i, q in enumerate(real) if not done_mask[i]]
    with mp.get_context("spawn").Pool(a.procs, initializer=_init_worker) as pool:
        for (i, m, p, M, dt) in pool.imap_unordered(_task, tasks):
            m_real[i], p_real[i] = m[0], p[0]
            m_null[i], p_null[i] = m[1], p[1]
            m_fit[i], p_fit[i] = m[2:], p[2:]
            if M is not None:
                M_real[i] = M
            secs[i] = dt
            done_mask[i] = True
            done = int(done_mask.sum())
            if done % 10 == 0 or done == F:
                checkpoint()
            if done % 20 == 0 or done == F:
                print(f"  {done}/{F} frames  {time.time() - t0:.0f} s", flush=True)
    whole = None
    if a.check_whole:                                         # sharding must not change a single count
        from glint.lattice import cell_to_Ar
        _init_worker()
        mw, pw, _ = route(real, cell_to_Ar(*cell))
        whole = dict(frames=F, matched_equal=int((mw == m_real).sum()), path_equal=int((pw == p_real).sum()))
        print(f"whole-list check: matched identical on {whole['matched_equal']}/{F}, path on {whole['path_equal']}/{F}")
        if whole["matched_equal"] != F or whole["path_equal"] != F:
            raise SystemExit("sharded and whole-list runs differ")
    import torch
    import glint.glint_fast as gf
    meta = dict(input=os.path.expanduser(a.input), n_frames=F,
                input_md5=hashlib.md5(open(os.path.expanduser(a.input), "rb").read()).hexdigest(),
                cell=cell, route="glint_cli --cell: hybrid_index(Mc_known, nbest=3), no cascade/escalation",
                null_seed=a.null_seed, fit_seed=a.fit_seed, k_fit=K, gate_tol=gf.GATE_TOL, gate_min=gf.GATE_MIN,
                gate_frac=gf.GATE_FRAC, torch=torch.__version__, device="cpu", steps=gf.STEPS,
                procs=a.procs, wall_seconds=time.time() - t0, cpu_seconds=float(secs.sum()), whole_check=whole)
    np.savez_compressed(a.out, meta=json.dumps(meta), n=n, m_real=m_real, p_real=p_real, m_null=m_null,
                        p_null=p_null, m_fit=m_fit, p_fit=p_fit, M_real=M_real)
    if os.path.exists(part):
        os.remove(part)
    print(f"wrote {a.out}  ({F} frames x {K + 2} route calls, {time.time() - t0:.0f} s wall)")


# ---------------------------------------------------------------------------------------------- fit ----
def strict_gate(m, n):
    """--gate strict as gate_results applies it: matched >= GATE_MIN and >= GATE_FRAC * n."""
    from glint.glint_fast import GATE_FRAC, GATE_MIN
    return (m >= GATE_MIN) & (m >= GATE_FRAC * n)


def floor_value(n, fl):
    """The shipped floor, a*n + b (+ c*sqrt(n)): hybrid_stream.floor_value, so the fit scores what the gate applies."""
    from glint.hybrid_stream import floor_value as fv
    return fv(n, fl)


def floor_gate(m, n, fl):
    """--gate floor as gate_results applies it: strict AND matched >= floor_value(n)."""
    return strict_gate(m, n) & (m >= floor_value(n, fl))


def _helpers():
    """The quantile fit, the peak-count bands and the xgandalf reader are live_gate_null.py's (glint#214), imported
    here rather than at module level: spawned measure workers re-import this module before _init_worker runs."""
    import live_gate_null as lgn
    return lgn


def _fmt(fl):
    return f"{fl[0]:.4f} n + {fl[1]:.2f}" + (f" + {fl[2]:.3f} sqrt(n)" if len(fl) > 2 else "")


def _band_rates(acc, n):
    out = []
    for lo, hi in _helpers().BANDS:
        sel = (n >= lo) & (n < hi)
        out.append(dict(band=[lo, hi], frames=int(sel.sum()),
                        rate=(float(acc[sel].mean()) if sel.any() else None), accepts=int(acc[sel].sum())))
    return out


def fit(a):
    d = np.load(a.npz)
    meta = json.loads(str(d["meta"]))
    n, mr, pr, mn, pn, mf, pf = (d[k] for k in ("n", "m_real", "p_real", "m_null", "p_null", "m_fit", "p_fit"))
    F, K = mf.shape
    nK = n[:, None]
    alpha = a.alpha
    strict_r = strict_gate(mr, n)
    fit_floor = _helpers().fit_floor
    floors = {sh: fit_floor(n, mf, alpha, sh) for sh in ("lin", "sqrt")}
    if a.floor:
        floors["given"] = tuple(float(v) for v in a.floor.split(","))
    rows = []

    def row(name, acc_r, acc_n, acc_f):
        rows.append(dict(gate=name, real=int(acc_r.sum()), strict_kept=int((acc_r & strict_r).sum()),
                         strict_lost=int((strict_r & ~acc_r).sum()), heldout_null=int(acc_n.sum()),
                         heldout_rate=float(acc_n.mean()), fit_rate=float(acc_f.mean()),
                         fit_accepts=int(acc_f.sum())))
    row("none (every registration)", mr > 0, mn > 0, mf > 0)
    row("strict (10, 0.25)", strict_r, strict_gate(mn, n), strict_gate(mf, nK))
    for sh, fl in floors.items():
        f = floor_value(n, fl)
        row(f"floor:{sh} strict & m >= {_fmt(fl)}", strict_r & (mr >= f), strict_gate(mn, n) & (mn >= f),
            strict_gate(mf, nK) & (mf >= f[:, None]))
    # the fraction alternative: smallest GATE_FRAC-like bar holding the fit copies at alpha
    frac = next((float(round(x, 3)) for x in np.arange(0.25, 0.80, 0.005)
                 if ((mf >= 10) & (mf >= x * nK)).mean() <= alpha), None)
    if frac is not None:
        row(f"fraction {frac:.3f} (m >= 10 and >= {frac:.3f} n)", (mr >= 10) & (mr >= frac * n),
            (mn >= 10) & (mn >= frac * n), (mf >= 10) & (mf >= frac * nK))
    bands = {"strict": _band_rates(strict_gate(mf, nK), nK.repeat(K, 1))}
    for sh, fl in floors.items():
        bands[sh] = _band_rates(strict_gate(mf, nK) & (mf >= floor_value(n, fl)[:, None]), nK.repeat(K, 1))
    # per path: where do the null copies' registrations come from, and which path's accepts are they
    paths = {}
    for code, name in enumerate(PATHS):
        sel = pf == code
        e = dict(fit_copies=int(sel.sum()), fit_share=float(sel.mean()),
                 real=int((pr == code).sum()), heldout=int((pn == code).sum()),
                 strict_accepts=int((sel & strict_gate(mf, nK)).sum()))
        for sh, fl in floors.items():
            e[f"{sh}_accepts"] = int((sel & strict_gate(mf, nK) & (mf >= floor_value(n, fl)[:, None])).sum())
        e["real_strict"] = int(((pr == code) & strict_r).sum())
        paths[name] = e
    # frame-split CV: fit on half the FRAMES, score the other half (a new run has new frames)
    rng = np.random.default_rng(a.cv_seed)
    cv = {sh: [] for sh in ("lin", "sqrt")}
    for _ in range(a.cv_reps):
        perm = rng.permutation(F)
        for tr, te in ((perm[:F // 2], perm[F // 2:]), (perm[F // 2:], perm[:F // 2])):
            for sh in cv:
                fl = fit_floor(n[tr], mf[tr], alpha, sh)
                f = floor_value(n[te], fl)
                sparse = n[te] < 60
                acc_f = strict_gate(mf[te], n[te][:, None]) & (mf[te] >= f[:, None])
                cv[sh].append((acc_f.mean(), (strict_gate(mn[te], n[te]) & (mn[te] >= f)).mean(),
                               acc_f[sparse].mean() if sparse.any() else np.nan,
                               int((strict_gate(mr[te], n[te]) & (mr[te] >= f)).sum()), int(strict_r[te].sum())))
    cvs = {}
    for sh, r in cv.items():
        r = np.array(r, float)
        cvs[sh] = dict(splits=len(r), fit_copies_median=float(np.median(r[:, 0])),
                       fit_copies_p90=float(np.quantile(r[:, 0], 0.9)), fit_copies_max=float(r[:, 0].max()),
                       heldout_median=float(np.median(r[:, 1])), heldout_max=float(r[:, 1].max()),
                       sparse_lt60_median=float(np.nanmedian(r[:, 2])), sparse_lt60_max=float(np.nanmax(r[:, 2])),
                       strict_kept_share_median=float(np.median(r[:, 3] / np.maximum(r[:, 4], 1))))
    # the null's quantiles by band, for the record
    q99 = []
    for lo, hi in _helpers().BANDS:
        sel = (n >= lo) & (n < hi)
        if sel.any():
            q99.append(dict(band=[lo, hi], q99=float(np.quantile(mf[sel], 0.99)), median=float(np.median(mf[sel])),
                            max=int(mf[sel].max())))
    res = dict(meta=meta, alpha=alpha, n_frames=F, k_fit=K, floors={k: list(v) for k, v in floors.items()},
               rows=rows, bands=bands, paths=paths, cv=cvs, null_quantiles=q99, strict_real=int(strict_r.sum()),
               n_range=[int(n.min()), int(np.median(n)), int(n.max())])
    w = max(len(r["gate"]) for r in rows)
    print(f"{meta['input']}: {F} frames (peaks {n.min()}-{n.max()}, median {int(np.median(n))}), {K} fit copies/frame, "
          f"target null {alpha:g}; real strict {int(strict_r.sum())}")
    print(f"{'gate':<{w}}  real (strict kept) strict_lost  held-out null  fit copies")
    for r in rows:
        print(f"{r['gate']:<{w}}  {r['real']:4d} ({r['strict_kept']:4d})  {r['strict_lost']:11d}  "
              f"{r['heldout_null']:4d}/{F} {100 * r['heldout_rate']:5.2f}%  {100 * r['fit_rate']:5.2f}% ({r['fit_accepts']})")
    print("null by peak-count band (fit copies):")
    for sh, b in bands.items():
        print(f"  {sh:7s} " + "  ".join(f"[{x['band'][0]},{x['band'][1]}) n={x['frames']} "
                                         + ("-" if x["rate"] is None else f"{100 * x['rate']:.2f}%") for x in b))
    print("paths:")
    for k, v in paths.items():
        print(f"  {k:7s} {v}")
    for sh, v in cvs.items():
        print(f"frame-split CV ({sh}): {v}")
    print("null matched quantiles by band:", q99)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")
    return res


# ------------------------------------------------------------------------------------------ confirm ----
def confirm(a):
    """Are the strict frames the floor refuses crystals? Independent evidence: the published xgandalf arm
    (blind, same q) finding the LYSO lattice within 2 deg of the CLI's registration. Scored on the refused
    frames and, as the control that shows the check can pass, on the strict frames the floor keeps at the
    same (sparse) peak counts, since xgandalf itself solves fewer sparse frames."""
    from glint.glint_fast import LYSO
    from glint.multilattice import misorientation_deg
    from glint.multishot import same_lattice
    d = np.load(a.npz)
    n, mr, pr = d["n"], d["m_real"], d["p_real"]
    F = len(n)
    fl = tuple(float(v) for v in a.floor.split(","))
    strict = strict_gate(mr, n)
    keep = floor_gate(mr, n, fl)
    M_real = [None if np.isnan(M).any() else M for M in d["M_real"]]
    xg = _helpers().load_xgandalf(a.xgandalf, F)

    def agree(mask):
        idx = np.where(mask)[0]; has = agr = 0; far = []
        for i in idx:
            X, M = xg[i], M_real[i]
            if X is None or M is None or abs(np.linalg.det(X)) < 1.0 or not same_lattice(X, LYSO):
                continue
            has += 1
            ang = misorientation_deg(M, X, laue="4/mmm")
            if ang < 2.0:
                agr += 1
            else:
                far.append(round(float(ang), 1))
        return dict(frames=int(len(idx)), xgandalf_lyso=int(has), agree_2deg=int(agr),
                    n_range=[int(n[idx].min()), int(n[idx].max())] if len(idx) else None,
                    paths={PATHS[c]: int((pr[idx] == c).sum()) for c in range(3)}, disagree_deg=sorted(far))
    sp = n < a.sparse
    out = {f"sparse (n<{a.sparse}) strict, floor keeps": agree(sp & strict & keep),
           f"sparse (n<{a.sparse}) strict, floor refuses": agree(sp & strict & ~keep),
           "all strict, floor refuses": agree(strict & ~keep),
           "all strict, floor keeps": agree(strict & keep),
           "registered, not strict": agree((mr > 0) & ~strict)}
    for k, v in out.items():
        print(f"{k:40s} {v}")
    if a.json:
        json.dump(dict(floor=list(fl), rows=out), open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--input", required=True, help="FRAME-block q list")
    m.add_argument("--cell", default=LYSO_CELL, help="the --cell glint_cli is given (a,b,c,al,be,ga)")
    m.add_argument("--null-seed", type=int, default=20260928, help="held-out copy rng [seed, i]")
    m.add_argument("--fit-seed", type=int, default=1, help="fit copies rng [seed, i, k]")
    m.add_argument("--k-fit", type=int, default=32)
    m.add_argument("--procs", type=int, default=10)
    m.add_argument("--limit", type=int, default=None)
    m.add_argument("--check-whole", action="store_true",
                   help="re-run the real frames as one hybrid_index call; exit non-zero unless identical")
    m.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("npz")
    f.add_argument("--alpha", type=float, default=0.01)
    f.add_argument("--floor", default=None, help="also score this a,b[,c] (e.g. one fitted on another list)")
    f.add_argument("--cv-reps", type=int, default=20)
    f.add_argument("--cv-seed", type=int, default=7)
    f.add_argument("--json", default=None)
    c = sub.add_parser("confirm")
    c.add_argument("npz")
    c.add_argument("--floor", required=True, help="a,b[,c]")
    c.add_argument("--xgandalf", required=True)
    c.add_argument("--sparse", type=int, default=62)
    c.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    dict(measure=measure, fit=fit, confirm=confirm)[a.cmd](a)


if __name__ == "__main__":
    main()
