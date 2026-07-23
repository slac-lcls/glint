"""Controlled synthetic SFX experiment for GLINT item 3 (partiality + per-crystal scale).

Merges a synthetic multi-crystal stills dataset with KNOWN truth I_full under three
conditions and scores each against the truth:

  (1) gated-only        current GLINT: PERTURBED geometry, 1/mean scale, inv-var weight
  (2) + Bravais refine  item 2 refine_sym.refine_bravais restores the tetragonal cell
  (3) + partiality      item 3: divide by G_c*p, weight p^2/sigma^2 (PartialityScaler)

Metrics per condition: CC1/2, Rsplit (odd/even FRAME half-sets, via MergeAccumulator),
and the DECISIVE R_vs_truth = sum|k*I_merged - I_full| / sum I_full against the KNOWN
full intensities (CC1/2 can be gamed by consistent-but-biased partials; R_vs_truth cannot).

Usage: python run_partiality.py [N] [--clean] [--oracle] [--fitB]
"""
from __future__ import annotations

import sys
import numpy as np

from glint.predict import recip_from_M
from glint.refine_sym import refine_bravais
from glint.stream_driver import MergeAccumulator, laue_ops_4mmm, _asu_key
from glint.partiality import PartialityScaler
import glint.synth_sfx as ss


def assign_hkl(q_obs, M):
    """Miller indices of observed reciprocal peaks under orientation M: h = round(q @ M)."""
    return np.rint(q_obs @ M).astype(int)


def score_vs_truth(merged, I_full):
    """R_vs_truth (single global scale k) and CC_vs_truth over common asu keys."""
    keys = [k for k in merged if k in I_full and I_full[k] > 0]
    if len(keys) < 10:
        return float("nan"), float("nan"), len(keys)
    Im = np.array([merged[k] for k in keys])
    If = np.array([I_full[k] for k in keys])
    denk = np.sum(Im * Im)
    k = np.sum(Im * If) / denk if denk > 0 else 1.0
    R = np.sum(np.abs(k * Im - If)) / np.sum(If)
    cc = float(np.corrcoef(Im, If)[0, 1])
    return float(R), cc, len(keys)


def merge_default(stills, use_refine, ops, sym_tol=0.02):
    """Conditions (1)/(2): assign hkl under perturbed (or refined) geometry, default merge.
    Returns (acc, arrays) where arrays=(key,v,w,crystal) mirror add_frame (for bootstrap)."""
    acc = MergeAccumulator(ops=ops)
    Kk, Vv, Ww, Cc = [], [], [], []
    for fi, st in enumerate(stills):
        M = st["M_pert"]
        if use_refine:
            q = st["q_obs"]
            q = q[np.isfinite(q).all(1)]
            if len(q) >= 6:
                try:
                    M, _, _, _ = refine_bravais(q, st["M_pert"], "tetragonal", sym_tol)
                except Exception:
                    M = st["M_pert"]
        hkl = assign_hkl(st["q_obs"], M)
        I = np.asarray(st["I_obs"], float); sigma = np.maximum(st["sigma"], 1e-3)
        acc.add_frame(hkl, I, sigma, fi)
        good = np.isfinite(I) & np.isfinite(sigma)
        scale = 1.0 / I[good].mean() if (good.sum() > 5 and I[good].mean() > 0) else 1.0
        keep = good & (I > 0)
        Kk.append(_asu_key(hkl[keep], ops)); Vv.append((I * scale)[keep])
        Ww.append((1.0 / sigma ** 2)[keep]); Cc.append(np.full(int(keep.sum()), fi))
    arrays = (np.concatenate(Kk), np.concatenate(Vv), np.concatenate(Ww), np.concatenate(Cc))
    return acc, arrays


def merge_partiality(stills, ops, eta_bar, dl_bar, wavelength_A, sym_tol=0.02,
                     n_iter=5, fit_B=False, oracle=False, use_refine=True):
    """Condition (3): (optionally refined) geometry + partiality/scale un-correction.

    use_refine=True  -> item-2 Bravais refine feeds the predicted excitation error (item 3).
    use_refine=False -> partiality on the PERTURBED geometry (diagnostic: isolates what the
                        refine contributes to the partiality model)."""
    # gather global measurement arrays (assigned under the chosen geometry)
    hkls = []
    Kk, Ii, Ss, Ee, Qq, Cc, Ff = [], [], [], [], [], [], []
    p_or, G_or = [], []
    for fi, st in enumerate(stills):
        M = st["M_pert"]
        q = st["q_obs"]; qf = q[np.isfinite(q).all(1)]
        if use_refine and len(qf) >= 6:
            try:
                M, _, _, _ = refine_bravais(qf, st["M_pert"], "tetragonal", sym_tol)
            except Exception:
                M = st["M_pert"]
        hkl = assign_hkl(st["q_obs"], M)
        R = recip_from_M(M)
        qpred = hkl @ R
        qn2 = np.einsum("ij,ij->i", qpred, qpred)
        exc = qpred[:, 2] + 0.5 * wavelength_A * qn2
        qmag = np.sqrt(qn2)
        keys = _asu_key(hkl, ops)
        n = len(keys)
        hkls.append(hkl)
        Kk.append(keys); Ii.append(st["I_obs"]); Ss.append(st["sigma"])
        Ee.append(exc); Qq.append(qmag)
        Cc.append(np.full(n, fi)); Ff.append(np.full(n, fi))
        if oracle:
            p_or.append(st["p_true"]); G_or.append(np.full(n, st["G_c"]))
    key = np.concatenate(Kk); I = np.concatenate(Ii); sigma = np.concatenate(Ss)
    exc = np.concatenate(Ee); qmag = np.concatenate(Qq)
    crystal = np.concatenate(Cc); frame = np.concatenate(Ff)

    sc = PartialityScaler(eta_bar, dl_bar, wavelength_A=wavelength_A, n_iter=n_iter, fit_B=fit_B)
    if oracle:
        # oracle self-consistency: use TRUE partiality and TRUE per-crystal scale
        p = np.clip(np.concatenate(p_or), 1e-3, 1.0); G = np.concatenate(G_or)
        values = I / (G * p)
        weights = p * p / np.maximum(sigma, 1e-3) ** 2
    else:
        sc.fit(key, I, sigma, exc, qmag, crystal, frame)
        values = sc.values
        weights = sc.weights

    # fold into the SAME accumulator machinery (values/weights path)
    acc = MergeAccumulator(ops=ops)
    off = 0
    Kk2, Vv2, Ww2, Cc2 = [], [], [], []
    for fi, st in enumerate(stills):
        n = len(st["I_obs"])
        sl = slice(off, off + n); off += n
        Ifr = np.asarray(st["I_obs"], float)
        acc.add_frame(hkls[fi], Ifr, st["sigma"], fi,
                      values=values[sl], weights=weights[sl])
        keep = np.isfinite(Ifr) & (Ifr > 0)                 # mirror add_frame's snr>thr[0] drop
        Kk2.append(_asu_key(hkls[fi][keep], ops)); Vv2.append(values[sl][keep])
        Ww2.append(weights[sl][keep]); Cc2.append(np.full(int(keep.sum()), fi))
    arrays = (np.concatenate(Kk2), np.concatenate(Vv2), np.concatenate(Ww2), np.concatenate(Cc2))
    return acc, sc, arrays


def _merge_from_arrays(key, v, w):
    order = np.argsort(key, kind="stable"); k = key[order]; v = v[order]; w = w[order]
    uk, idx = np.unique(k, return_index=True)
    sw = np.add.reduceat(w, idx); swv = np.add.reduceat(w * v, idx)
    ok = sw > 0
    return dict(zip(uk[ok].tolist(), (swv[ok] / sw[ok]).tolist()))


def bootstrap_R(arrays, I_full, nboot=200, seed=1):
    """Crystal-resampling bootstrap of R_vs_truth (G/p held at the point fit). 95% CI."""
    key, v, w, crystal = arrays
    uc = np.unique(crystal)
    # per-crystal measurement index lists
    order = np.argsort(crystal, kind="stable")
    cs = crystal[order]
    bounds = np.searchsorted(cs, uc, side="left")
    ends = np.r_[bounds[1:], len(cs)]
    slices = [order[bounds[i]:ends[i]] for i in range(len(uc))]
    rng = np.random.default_rng(seed)
    Rs = []
    for _ in range(nboot):
        pick = rng.integers(0, len(uc), len(uc))
        idx = np.concatenate([slices[p] for p in pick])
        merged = _merge_from_arrays(key[idx], v[idx], w[idx])
        R, _, _ = score_vs_truth(merged, I_full)
        if R == R:
            Rs.append(R)
    Rs = np.array(Rs)
    return float(np.mean(Rs)), float(np.percentile(Rs, 2.5)), float(np.percentile(Rs, 97.5))


def main():
    args = sys.argv[1:]
    N = 2000
    for a in args:
        if a.isdigit():
            N = int(a)
    clean = "--clean" in args
    oracle = "--oracle" in args
    fitB = "--fitB" in args
    noB = "--noB"  # unused

    cfg = ss.SynthConfig(N=N, use_B_c=fitB)
    if clean:
        cfg = ss.SynthConfig(N=N, noiseless=True, unit_partiality=True,
                             no_perturb=True, unit_scale=True)
    print(f"# synthetic SFX: N={N} clean={clean} oracle={oracle} fitB={fitB}")
    (ops, ukeys, I_full, qmag_by_key), stills = ss.generate(cfg)
    nmeas = sum(len(s["I_obs"]) for s in stills)
    print(f"# stills indexed={len(stills)}  measurements={nmeas}  unique_truth={len(ukeys)}")

    # global model kernel params (medians of the per-crystal distributions -- NOT peeking at truth per still)
    eta_bar = np.deg2rad(np.median([np.rad2deg(s["eta"]) for s in stills]))
    dl_bar = float(np.median([s["dloverl"] for s in stills]))

    boot = "--boot" in args

    def report(name, acc, arrays=None):
        s = acc.stats(thr=0.0)
        merged = acc.merged_by_key(thr=0.0)
        R, ccT, ncommon = score_vs_truth(merged, I_full)
        ci = ""
        if boot and arrays is not None:
            Rm, lo, hi = bootstrap_R(arrays, I_full)
            ci = f"  CI95=[{lo:.4f},{hi:.4f}]"
        print(f"{name:36s} cc1/2={s['cc_half']:.4f}  Rsplit={100*s['rsplit']:6.2f}%  "
              f"CCvT={ccT:.4f}  R_vs_truth={R:.4f}{ci}  (uniq={s['unique']}, red={s['redundancy']:.1f}, common={ncommon})")
        return R, s

    print()
    acc1, a1 = merge_default(stills, use_refine=False, ops=ops)
    R1, s1 = report("(1) gated-only", acc1, a1)
    acc2, a2 = merge_default(stills, use_refine=True, ops=ops)
    R2, s2 = report("(2) +Bravais refine", acc2, a2)
    acc3, sc, a3 = merge_partiality(stills, ops, eta_bar, dl_bar, cfg.wavelength_A,
                                    fit_B=fitB, oracle=oracle, use_refine=True)
    R3, s3 = report("(3) +refine+partiality", acc3, a3)
    acc3p, _, a3p = merge_partiality(stills, ops, eta_bar, dl_bar, cfg.wavelength_A,
                                     fit_B=fitB, oracle=oracle, use_refine=False)
    R3p, s3p = report("(3p) partiality, NO refine (diag)", acc3p, a3p)

    print()
    print(f"# R_vs_truth: (1) {R1:.4f}  (2) {R2:.4f}  (3) {R3:.4f}  (3p no-refine) {R3p:.4f}")
    if R1 == R1 and R2 == R2:
        print(f"# refine-only gain (1)->(2):   {100*(R1-R2)/R1:+.1f}% R")
    if R2 == R2 and R3 == R3:
        print(f"# partiality gain  (2)->(3):   {100*(R2-R3)/R2:+.1f}% R")
    if R3p == R3p and R3 == R3:
        print(f"# refine WITHIN partiality (3p)->(3): {100*(R3p-R3)/R3p:+.1f}% R")


if __name__ == "__main__":
    main()
