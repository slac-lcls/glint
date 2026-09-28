#!/usr/bin/env python3
"""Calibrate StreamDriver's live accept gate (`_fits`) on a lattice-free null.

The live gate accepts a registration when it explains >= min_inliers peaks AND >= min_inlier_frac of them
(HKL_TOL = 0.15 per index). Its default 0.15 was chosen against a WRONG cell on 16 synthetic frames. The
case that matters on a dense stream is different: a frame with no lattice at all, searched with the RIGHT
cell. The known-cell search maximises the inlier count over ~10^4 orientations, so on a peak-rich frame the
best of those chance counts clears a count-and-fraction bar that was never calibrated against it.

`measure` runs the driver's own registration (glint.replica_gpu_batch.index_fused, the call flush() makes,
at the shipped depth) and the driver's own count (StreamDriver._inliers) on
  * every real frame,
  * a held-out azimuth-scrambled copy of every frame -- rng default_rng([null_seed, i]), the stream of
    scramble_qlist.py (glint#213; its q480_scrambled.txt, md5 bd132bd0..., gives the same 480 counts as the
    in-script copies), or the scrambled q list itself with --null-input,
  * k_fit further scrambled copies per frame, rng default_rng([fit_seed, i, k]), used ONLY to fit a floor,
and saves per frame (n_peaks, n_inl_real, n_inl_null, n_inl_fit[k], strict flags). Scrambling rotates each
peak independently about the beam, which keeps |q| and q_z (so each peak's excitation error) and removes
only the lattice. Scrambling a scrambled frame gives the same distribution, so a frame's fit copies are also
valid controls for its held-out null copy (the per-frame test, option c).

`fit` reads that file, fits the candidate gates on the FIT copies at a target null rate, and scores them on
the held-out null and on the real frames (live gate, strict gate, and the published xgandalf arm when its
solutions are given):
  (a) floor     n_inl >= a*n + b   (linear quantile regression of the fit copies at quantile 1 - alpha)
  (b) fraction  n_inl >= f*n       (smallest f whose fit-copy accept rate is <= alpha)
  (c) per-frame n_inl > every one of k scrambled copies' counts (escalate_batch's rule), k in --k-test

`confirm` asks whether the frames a floor refuses are crystals, using independent evidence: the published
xgandalf arm (xg_driver on the same q) finding the same lattice within 2 deg of the driver's registration.

Committed results (RESULTS_live_gate_null.md): live_gate_null_480.npz (q480_fix.txt, S3DF ~/q480_fix.txt, md5
d6d86c1b...; the 480 arm's lock cell; 32 fit copies; the relock cell and Proteinase K as transfer checks) and
live_gate_null_120.npz (the committed frames_cxidb_clean.txt, lysozyme cell, 16 copies). The fit and confirm
steps need no torch:

  python experiments/live_gate_null.py measure --input ~/q480_fix.txt --k-fit 32 \\
      --extra-cell cell1=87.5,87.6,109.5,69.3,72.7,100.7 --extra-cell prok=68.7,68.7,108.6,90,90,90 --k-extra 8 \\
      --out live_gate_null_480.npz                    # CPU torch: ~30 s per pass of 480, 52 passes
  python experiments/live_gate_null.py fit experiments/live_gate_null_480.npz --xgandalf ~/xgd480_fix.txt \\
      --input ~/q480_fix.txt
  python experiments/live_gate_null.py confirm experiments/live_gate_null_480.npz --floor 0.0224,5.32,1.211 \\
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
sys.path.insert(0, os.path.dirname(HERE))

from glint.glint_fast import GATE_FRAC, GATE_MIN, LYSO, load, matched_strict   # noqa: E402
from glint.lattice import cell_to_Ar                                          # noqa: E402
from glint.multilattice import scramble_azimuth                               # noqa: E402
from glint.multishot import same_lattice                                      # noqa: E402
import glint.stream_driver as sd                                              # noqa: E402

# The 480 published arm's primary lock (replay_480_off.json header "primary_cell", job 39344280)
CELL_480 = "78.706,78.792,37.813,89.811,90.083,90.189"


def parse_cell(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 6:
        raise SystemExit(f"--cell needs 6 numbers, got {s!r}")
    return v


def driver_counts(qs, Mc, B):
    """The driver's registration and count, exactly as _index_integrate makes them: index_fused, then a
    None or |det| < 1 result is a miss (0 inliers here), else StreamDriver._inliers."""
    import glint.replica_gpu_batch as rgb
    Ms = rgb.index_fused(qs, Mc, B=max(min(len(qs), B), 1))
    out, keep = [], []
    for q, M in zip(qs, Ms):
        M = np.asarray(M, float) if M is not None else None
        if M is None or abs(np.linalg.det(M)) < 1.0:
            out.append(0); keep.append(None)
        else:
            out.append(sd.StreamDriver._inliers(None, q, M)); keep.append(M)
    return np.array(out, int), keep


def measure(a):
    import glint.replica_gpu_batch as rgb
    real = [np.asarray(q, float) for q in load(os.path.expanduser(a.input))]
    if a.limit:
        real = real[:a.limit]
    F = len(real)
    n = np.array([len(q) for q in real], int)
    cell = parse_cell(a.cell)
    Mc = cell_to_Ar(*cell)
    if a.null_input:
        null = [np.asarray(q, float) for q in load(os.path.expanduser(a.null_input))][:F]
        if [len(q) for q in null] != list(n):
            raise SystemExit("--null-input does not have the same peak counts as --input")
        null_src = dict(path=os.path.expanduser(a.null_input),
                        md5=hashlib.md5(open(os.path.expanduser(a.null_input), "rb").read()).hexdigest())
    else:
        null = [scramble_azimuth(q, np.random.default_rng([a.null_seed, i])) for i, q in enumerate(real)]
        null_src = dict(seed=a.null_seed)
    t0 = time.time()
    inl_real, M_real = driver_counts(real, Mc, a.B)
    ref = cell_to_Ar(*parse_cell(a.ref)) if a.ref else LYSO
    strict = np.zeros(F, bool); sl = np.zeros(F, bool)
    for i, (q, M) in enumerate(zip(real, M_real)):
        if M is None:
            continue
        sl[i] = same_lattice(M, ref)
        m = matched_strict(M, q)
        strict[i] = sl[i] and m >= GATE_MIN and m / len(q) >= GATE_FRAC
    t1 = time.time()
    print(f"real: {F} frames in {t1 - t0:.1f} s", flush=True)
    inl_null, _ = driver_counts(null, Mc, a.B)
    print(f"null: {time.time() - t1:.1f} s", flush=True)
    inl_fit = np.zeros((F, a.k_fit), int)
    for k in range(a.k_fit):
        t2 = time.time()
        cps = [scramble_azimuth(q, np.random.default_rng([a.fit_seed, i, k])) for i, q in enumerate(real)]
        inl_fit[:, k], _ = driver_counts(cps, Mc, a.B)
        print(f"fit copy {k + 1}/{a.k_fit}: {time.time() - t2:.1f} s", flush=True)
    extra = {}
    for spec in a.extra_cell or []:
        name, s = spec.split("=", 1)
        Me = cell_to_Ar(*parse_cell(s))
        xr, _ = driver_counts(real, Me, a.B)
        xf = np.zeros((F, a.k_extra), int)
        for k in range(a.k_extra):
            cps = [scramble_azimuth(q, np.random.default_rng([a.fit_seed, i, k])) for i, q in enumerate(real)]
            xf[:, k], _ = driver_counts(cps, Me, a.B)
        extra[name] = dict(cell=parse_cell(s), inl_real=xr, inl_fit=xf)
        print(f"extra cell {name}: done", flush=True)
    meta = dict(input=os.path.expanduser(a.input), n_frames=F,
                input_md5=hashlib.md5(open(os.path.expanduser(a.input), "rb").read()).hexdigest(), cell=cell, ref=a.ref or "LYSO 79.02,79.02,37.98,90,90,90",
                B=a.B, null=null_src, fit_seed=a.fit_seed, k_fit=a.k_fit, hkl_tol=sd.HKL_TOL,
                device=rgb.DEV, fp=str(rgb.FP), torch_seconds=time.time() - t0,
                extra={k: v["cell"] for k, v in extra.items()}, k_extra=a.k_extra)
    M_arr = np.full((F, 3, 3), np.nan)
    for i, M in enumerate(M_real):
        if M is not None:
            M_arr[i] = M
    arrs = dict(n=n, inl_real=inl_real, inl_null=inl_null, inl_fit=inl_fit, strict=strict, same_lattice=sl,
                M_real=M_arr)
    for k, v in extra.items():
        arrs[f"x_{k}_inl_real"] = v["inl_real"]; arrs[f"x_{k}_inl_fit"] = v["inl_fit"]
    np.savez_compressed(a.out, meta=json.dumps(meta), **arrs)
    print(f"wrote {a.out}")


# ------------------------------------------------------------------------------------------- fit ----
BANDS = ((0, 60), (60, 90), (90, 130), (130, 200), (200, 400), (400, 100000))


def live_gate(inl, n, min_inliers=10, frac=0.15):
    return (inl >= min_inliers) & (inl >= frac * n)


def floor_value(n, fl):
    """The driver's floor: a*n + b (+ c*sqrt(n)) for fl = (a, b[, c]) -- StreamDriver(null_floor=fl)."""
    n = np.asarray(n, float)
    return fl[0] * n + fl[1] + (fl[2] * np.sqrt(n) if len(fl) > 2 else 0.0)


def quantile_fit(X, y, tau):
    """Linear quantile regression y ~ X @ beta at quantile tau (pinball loss, as a sparse linear program)."""
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, hstack, identity
    X = np.asarray(X, float); N, P = X.shape
    # variables: beta (P), u+ (N), u- (N);  X beta + u+ - u- = y;  min tau*sum u+ + (1-tau)*sum u-
    c = np.concatenate([np.zeros(P), np.full(N, tau), np.full(N, 1.0 - tau)])
    A = hstack([csr_matrix(X), identity(N, format="csr"), -identity(N, format="csr")], format="csr")
    bounds = [(None, None)] * P + [(0, None)] * (2 * N)
    r = linprog(c, A_eq=A, b_eq=np.asarray(y, float), bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    return [float(v) for v in r.x[:P]]


def fit_floor(n, counts, alpha, shape):
    """Quantile 1 - alpha of the scrambled copies' counts as a function of the peak count.
    shape "lin": (a, b) = a*n + b, the 16 Sep form.  shape "sqrt": (a, b, c) = a*n + b + c*sqrt(n) -- the best
    of K orientations of a Binomial(n, p) count sits near n p + sqrt(2 n p (1-p) ln K), so the null quantile
    bends; the straight line over-rejects sparse frames and under-rejects the middle (see the band table)."""
    counts = np.asarray(counts); K = counts.shape[1]
    x = np.repeat(np.asarray(n, float), K); y = counts.ravel().astype(float)
    if shape == "lin":
        a, b = quantile_fit(np.column_stack([x, np.ones(len(x))]), y, 1.0 - alpha)
        return (a, b)
    a, c, b = quantile_fit(np.column_stack([x, np.sqrt(x), np.ones(len(x))]), y, 1.0 - alpha)
    return (a, b, c)


def load_xgandalf(path, n_frames):
    """Published xgandalf arm (xg_driver, libxgandalf 0.12.0, on the same q): per line `i M(9) score`."""
    out = [None] * n_frames
    for line in open(os.path.expanduser(path)):
        t = line.split()
        if len(t) >= 10 and int(t[0]) < n_frames:
            out[int(t[0])] = np.array([float(v) for v in t[1:10]]).reshape(3, 3)
    return out


def _fmt(fl):
    return (f"{fl[0]:.4f} n + {fl[1]:.2f}" + (f" + {fl[2]:.3f} sqrt(n)" if len(fl) > 2 else ""))


def fit(a):
    d = np.load(a.npz)
    meta = json.loads(str(d["meta"]))
    n, ir, inull, ifit, strict = d["n"], d["inl_real"], d["inl_null"], d["inl_fit"], d["strict"]
    F, K = ifit.shape
    alpha = a.alpha
    live, live_null, live_fit = live_gate(ir, n), live_gate(inull, n), live_gate(ifit, n[:, None])
    rows = []

    def row(name, acc_real, acc_null, acc_fit=None, note=""):
        rows.append(dict(gate=name, real=int(acc_real.sum()), real_strict=int((acc_real & strict).sum()),
                         strict_lost=int((strict & ~acc_real).sum()), null=int(acc_null.sum()),
                         null_rate=float(acc_null.mean()),
                         fit_rate=(float(acc_fit.mean()) if acc_fit is not None else None), note=note))

    row("live (10, 0.15)", live, live_null, live_fit, "shipped default")
    row("strict (10, 0.25)", live_gate(ir, n, 10, 0.25), live_gate(inull, n, 10, 0.25),
        live_gate(ifit, n[:, None], 10, 0.25), "count/fraction part of the research bar")
    # (a) floors: the gate is live AND n_inl >= floor(n)
    floors = {sh: fit_floor(n, ifit, alpha, sh) for sh in ("lin", "sqrt")}
    for sh, fl in floors.items():
        f = floor_value(n, fl)
        row(f"(a:{sh}) live & n_inl >= {_fmt(fl)}", live & (ir >= f), live_null & (inull >= f),
            live_fit & (ifit >= f[:, None]), f"quantile {1 - alpha:g} of {K} fit copies")
    # (b) fraction: smallest f (0.005 steps) with fit-copy accept <= alpha
    frac = next((float(round(f, 3)) for f in np.arange(0.15, 0.60, 0.005)
                 if live_gate(ifit, n[:, None], 10, f).mean() <= alpha), None)
    if frac is not None:
        row(f"(b) min_inlier_frac {frac:.3f}", live_gate(ir, n, 10, frac), live_gate(inull, n, 10, frac),
            live_gate(ifit, n[:, None], 10, frac))
    # (c) per-frame: live AND the real count beats every one of k scrambled copies (escalate_batch's rule)
    for k in a.k_test:
        if k <= K:
            mx = ifit[:, :k].max(1)
            row(f"(c) live & n_inl > max of {k} copies", live & (ir > mx), live_null & (inull > mx), None,
                f"{k} extra searches per live-accepted frame; bound 1/(k+1) = {1 / (k + 1):.3f}")
    # (d) raising min_inliers alone: smallest count with fit-copy accept <= alpha
    mi = next((m for m in range(10, 60) if live_gate(ifit, n[:, None], m, 0.15).mean() <= alpha), None)
    if mi is not None:
        row(f"(d) min_inliers {mi}", live_gate(ir, n, mi), live_gate(inull, n, mi), live_gate(ifit, n[:, None], mi))

    # per-band null of the floors (fit copies): a floor can hold 1% on average and not per peak count
    bands = {}
    for sh, fl in floors.items():
        acc = live_fit & (ifit >= floor_value(n, fl)[:, None])
        bands[sh] = [dict(band=list(b), frames=int(((n >= b[0]) & (n < b[1])).sum()),
                          null=float(acc[(n >= b[0]) & (n < b[1])].mean()) if ((n >= b[0]) & (n < b[1])).any() else None)
                     for b in BANDS]
    bands["live"] = [dict(band=list(b), null=float(live_fit[(n >= b[0]) & (n < b[1])].mean())
                          if ((n >= b[0]) & (n < b[1])).any() else None) for b in BANDS]
    # frame-split cross-validation: fit on half the FRAMES, score the other half (a new run has new frames)
    rng = np.random.default_rng(a.cv_seed)
    cv = {sh: [] for sh in floors}
    for _ in range(a.cv_reps):
        perm = rng.permutation(F)
        for tr, te in ((perm[:F // 2], perm[F // 2:]), (perm[F // 2:], perm[:F // 2])):
            for sh in floors:
                fl = fit_floor(n[tr], ifit[tr], alpha, sh)
                f = floor_value(n[te], fl)
                cv[sh].append(((live_fit[te] & (ifit[te] >= f[:, None])).mean(),
                               (live_null[te] & (inull[te] >= f)).mean()))
    cv = {sh: dict(fit_copies_median=float(np.median([v[0] for v in r])), fit_copies_p90=float(np.quantile([v[0] for v in r], 0.9)),
                   fit_copies_max=float(max(v[0] for v in r)), heldout_null_median=float(np.median([v[1] for v in r])),
                   heldout_null_max=float(max(v[1] for v in r)), splits=len(r)) for sh, r in cv.items()}
    xg = None
    if a.xgandalf:
        qpath = os.path.expanduser(a.input) if a.input else meta["input"]
        real_q = [np.asarray(q, float) for q in load(qpath)][:F]
        Ms = load_xgandalf(a.xgandalf, F)
        xs = np.zeros(F, bool)
        for i, (q, M) in enumerate(zip(real_q, Ms)):
            if M is not None and abs(np.linalg.det(M)) >= 1.0 and same_lattice(M, LYSO):
                m = matched_strict(M, q)
                xs[i] = m >= GATE_MIN and m / len(q) >= GATE_FRAC
        xg = dict(strict=int(xs.sum()))
    xcells = {}
    for key in d.files:
        if key.startswith("x_") and key.endswith("_inl_fit"):
            name = key[2:-8]
            xf = d[key]; xr = d[f"x_{name}_inl_real"]
            e = dict(cell=meta["extra"][name], k=int(xf.shape[1]),
                     live_null=float(live_gate(xf, n[:, None]).mean()), real_live=int(live_gate(xr, n).sum()))
            for sh, fl in floors.items():
                f = floor_value(n, fl)
                e[f"{sh}_null"] = float((live_gate(xf, n[:, None]) & (xf >= f[:, None])).mean())
                e[f"{sh}_real"] = int((live_gate(xr, n) & (xr >= f)).sum())
            xcells[name] = e
    res = dict(meta=meta, alpha=alpha, floors={k: list(v) for k, v in floors.items()}, frac=frac, min_inliers=mi,
               rows=rows, bands=bands, cv=cv, xgandalf=xg, extra_cells=xcells, n_frames=F,
               strict_real=int(strict.sum()))
    w = max(len(r["gate"]) for r in rows)
    print(f"{F} frames, {K} fit copies/frame, target null {alpha:g}; strict-gate real {int(strict.sum())}")
    print(f"{'gate':<{w}}  real  (strict)  strict_lost  null/{F}  null_rate  fit_rate")
    for r in rows:
        fr = "" if r["fit_rate"] is None else f"{r['fit_rate']:.4f}"
        print(f"{r['gate']:<{w}}  {r['real']:4d}  ({r['real_strict']:4d})  {r['strict_lost']:11d}  "
              f"{r['null']:7d}  {r['null_rate']:.4f}     {fr}")
    print("null by peak-count band (fit copies):")
    for sh, b in bands.items():
        print(f"  {sh:5s} " + "  ".join(f"[{x['band'][0]},{x['band'][1]}) "
                                         + ("-" if x["null"] is None else f"{x['null']:.4f}") for x in b))
    for sh, v in cv.items():
        print(f"frame-split CV ({sh}): {v}")
    if xg:
        print("xgandalf (published arm, same q): strict", xg["strict"])
    for k, v in xcells.items():
        print(f"cell {k}: {v}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")
    return res


def confirm(a):
    """Is a frame the floor refuses a crystal? Independent evidence: the published xgandalf arm (blind, same
    q) finding the same lattice within 2 deg of the driver's registration. Scored on the frames the floor
    refuses AND, as the control that shows the check can pass, on the frames it keeps, per peak-count range
    (xgandalf itself finds fewer solutions on sparse frames, so an all-frames control would be confounded)."""
    from glint.multilattice import misorientation_deg
    d = np.load(a.npz)
    meta = json.loads(str(d["meta"]))
    n, ir, strict = d["n"], d["inl_real"], d["strict"]
    F = len(n)
    fl = tuple(float(v) for v in a.floor.split(","))
    ok = live_gate(ir, n) & (ir >= floor_value(n, fl))
    live = live_gate(ir, n)
    xg = load_xgandalf(a.xgandalf, F)
    if "M_real" in d.files:
        M_real = [None if np.isnan(M).any() else M for M in d["M_real"]]
    else:                                                            # older files: recompute (torch)
        qpath = os.path.expanduser(a.input) if a.input else meta["input"]
        real_q = [np.asarray(q, float) for q in load(qpath)][:F]
        cnt, M_real = driver_counts(real_q, cell_to_Ar(*meta["cell"]), meta["B"])
        if not np.array_equal(cnt, ir):
            print(f"!! recomputed counts differ on {int((cnt != ir).sum())} frames")

    def agree(mask):
        idx = np.where(mask)[0]; has = agr = 0
        for i in idx:
            X, M = xg[i], M_real[i]
            if X is None or M is None or abs(np.linalg.det(X)) < 1.0 or not same_lattice(X, LYSO):
                continue
            has += 1
            agr += misorientation_deg(M, X, laue="4/mmm") < 2.0
        return dict(frames=int(len(idx)), xgandalf_lyso=int(has), agree_2deg=int(agr),
                    n_median=float(np.median(n[idx])) if len(idx) else None)

    sp = n < a.sparse
    out = {f"sparse (n<{a.sparse}) strict, floor keeps": agree(sp & strict & ok),
           f"sparse (n<{a.sparse}) strict, floor refuses": agree(sp & strict & ~ok),
           f"sparse (n<{a.sparse}) live not strict": agree(sp & live & ~strict),
           "all: floor keeps, not strict": agree(ok & ~strict),
           "all: live accepts, floor refuses": agree(live & ~ok),
           "all: floor keeps": agree(ok)}
    for k, v in out.items():
        print(f"{k:42s} {v}")
    if a.json:
        json.dump(dict(floor=list(fl), rows=out), open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--input", required=True, help="FRAME-block q list")
    m.add_argument("--null-input", default=None, help="the scrambled q list (scramble_qlist.py output) to use as the held-out null")
    m.add_argument("--null-seed", type=int, default=20260928, help="held-out null rng [seed, i] when --null-input is not given")
    m.add_argument("--fit-seed", type=int, default=1, help="fit copies use rng [fit_seed, i, k]")
    m.add_argument("--k-fit", type=int, default=32)
    m.add_argument("--cell", default=CELL_480, help="the cell the driver is locked to (a,b,c,al,be,ga)")
    m.add_argument("--ref", default=None, help="strict-gate reference cell (default LYSO 79.02,79.02,37.98)")
    m.add_argument("--extra-cell", action="append", metavar="NAME=a,b,c,al,be,ga",
                   help="also measure the null under another cell (does a floor transfer?)")
    m.add_argument("--k-extra", type=int, default=4)
    m.add_argument("--B", type=int, default=20)
    m.add_argument("--limit", type=int, default=None)
    m.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("npz")
    f.add_argument("--alpha", type=float, default=0.01)
    f.add_argument("--k-test", type=int, nargs="*", default=[8, 16, 32])
    f.add_argument("--cv-reps", type=int, default=20)
    f.add_argument("--cv-seed", type=int, default=7)
    f.add_argument("--xgandalf", default=None, help="published xgandalf solutions on the same q (xgd480_fix.txt)")
    f.add_argument("--input", default=None, help="q list, when the path recorded in the npz is not on this machine")
    f.add_argument("--json", default=None)
    c = sub.add_parser("confirm")
    c.add_argument("npz")
    c.add_argument("--floor", required=True, help="a,b[,c] as passed to StreamDriver(null_floor=...)")
    c.add_argument("--xgandalf", required=True)
    c.add_argument("--sparse", type=int, default=62)
    c.add_argument("--input", default=None)
    c.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    dict(measure=measure, fit=fit, confirm=confirm)[a.cmd](a)


if __name__ == "__main__":
    main()
