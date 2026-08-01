"""Multi-lattice residual peeling on ALL 42 streaming post-lock failures, with the acceptance
bar FIXED.

peel_hard.py used MIN_COV as a fraction of the CURRENT residual, so the bar fell every round --
the adversarial verifier showed that admits chance lattices (30/30 false accepts on
direction-randomized peaks once the residual is down to ~40). Here MIN_COV is a fraction of the
ORIGINAL peak count, which makes the rule monotonically harder as peeling proceeds.

Three groups, identical rule:
  1. every post-lock frame that FAILS the strict gate (the 42)
  2. every post-lock frame that PASSES it (dominance control)
  3. direction-randomized copies of group 1 (|q| kept, angles destroyed) -- re-calibrates the
     false-accept rate UNDER THE NEW RULE, which is the whole point of the fix

Reports the number that matters for offline rescue: how many failures reach >=0.25 as a SUMMED
LYSO FAMILY even though no single lattice does.

  python peel_all42.py [n_rand_seeds]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT)
sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import matched, index_blind_nbest
from glint.multishot import same_lattice
from glint.replica_gpu import index_known_gpu_cell
from what_are_the_failures import strict_gate, push_blind_q, LYSO, FRAMES_PATH
import glint.replica_gpu_batch as rgb
import glint.stream_driver as sd

TOL = 0.15
MIN_NEW = 10
MIN_COV = 0.20        # fraction of the ORIGINAL npk  <-- THE FIX
MAX_LAT = 5
GATE = 0.25           # the strict research gate


def inl_mask(M, q, tol=TOL):
    hf = np.asarray(q, float) @ np.asarray(M, float)
    return np.abs(hf - np.rint(hf)).max(1) < tol


def candidates(resid, nbest=3):
    out = []
    try:
        for k, (c, s) in enumerate(index_blind_nbest(resid, nbest)):
            if c is not None:
                out.append((f"blind{k}", np.asarray(c, float)))
    except Exception:
        pass
    try:
        Mr = index_known_gpu_cell(resid, LYSO)
        if Mr is not None:
            out.append(("known", np.asarray(Mr, float)))
    except Exception:
        pass
    return out


def peel(q, verbose=False):
    q = np.asarray(q, float)
    npk = len(q)
    taken = np.zeros(npk, bool)
    lats = []
    stop = "max_lat"
    for it in range(MAX_LAT):
        ridx = np.where(~taken)[0]
        if len(ridx) < 6:
            stop = f"residual<6 ({len(ridx)})"; break
        resid = q[ridx]
        cands = candidates(resid)
        if not cands:
            stop = f"no candidate (resid={len(resid)})"; break
        best = None
        for nm, M in cands:
            if not np.all(np.isfinite(M)) or abs(np.linalg.det(M)) < 1.0:
                continue
            n = int(inl_mask(M, resid).sum())
            if best is None or n > best[0]:
                best = (n, nm, M)
        if best is None:
            stop = f"all degenerate (resid={len(resid)})"; break
        nnew, via, M = best
        cov_orig = nnew / npk                      # <-- the bar is now against npk
        cov_resid = nnew / len(resid)
        if nnew < MIN_NEW or cov_orig < MIN_COV:
            stop = (f"rejected round {it+1}: best={via} new={nnew} "
                    f"cov_orig={cov_orig:.3f} cov_resid={cov_resid:.3f}")
            break
        taken[ridx[inl_mask(M, resid)]] = True
        lats.append(dict(via=via, n_new=nnew, cov_orig=float(cov_orig),
                         cov_resid=float(cov_resid), frac_orig_new=nnew / npk,
                         lyso=bool(same_lattice(M, LYSO)),
                         det=float(abs(np.linalg.det(M))),
                         cum=float(taken.sum() / npk)))
        if verbose:
            L = lats[-1]
            print(f"    L{it+1} via={via:7s} new={nnew:4d} cov_orig={cov_orig:.3f} "
                  f"cov_resid={cov_resid:.3f} LYSO={L['lyso']!s:5s} cum={L['cum']:.3f}")
    return lats, stop, float(taken.sum() / npk)


def randomize(q, rng):
    q = np.asarray(q, float)
    r = np.linalg.norm(q, axis=1)
    d = rng.normal(size=(len(q), 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * r[:, None]


def summarize(tag, rows):
    if not rows:
        print(f"{tag}: (empty)"); return
    nl = [len(r["lats"]) for r in rows]
    cums = [r["cum"] for r in rows]
    best1 = [max([L["frac_orig_new"] for L in r["lats"]], default=0.0) for r in rows]
    slyso = [r["summed_lyso"] for r in rows]
    print(f"\n{tag}: n={len(rows)}  lattices mean={np.mean(nl):.2f} median={np.median(nl):.1f} "
          f"max={max(nl)}  >=2: {sum(1 for x in nl if x>=2)}/{len(rows)}")
    print(f"    best SINGLE lattice frac_orig: mean={np.mean(best1):.3f} "
          f"range {min(best1):.3f}-{max(best1):.3f}")
    print(f"    summed-LYSO frac:              mean={np.mean(slyso):.3f} "
          f"range {min(slyso):.3f}-{max(slyso):.3f}   >={GATE}: "
          f"{sum(1 for x in slyso if x>=GATE)}/{len(rows)}")
    print(f"    cumulative explained:          mean={np.mean(cums):.3f}")


def main():
    nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.time()
    frames = list(gf.load(FRAMES_PATH))
    print(f"loaded {len(frames)} frames from {FRAMES_PATH}")
    print(f"RULE: accept if n_new >= {MIN_NEW} AND n_new/ORIGINAL_npk >= {MIN_COV} "
          f"(peel_hard.py used n_new/current_residual)\n")

    N = 400
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
                   res=1.0 / (0.1 / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
                   coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
    drv = sd.StreamDriver(None, panels, 0.1, 1.0, (N, N), dtype=np.uint16, B=20, dmin=2.0,
                          tol=0.002, warmup_nbest=3)
    warmup = []
    for i, qq in enumerate(frames):
        qq = np.asarray(qq, float)
        if drv._blind:
            warmup.append(i); push_blind_q(drv, qq)
        else:
            drv._q[drv._n] = qq; drv._n += 1; drv.n_pushed += 1
            if drv._n == drv.B:
                drv.flush()
    drv.flush()

    post = [i for i in range(len(frames)) if i not in set(warmup)]
    Ms = rgb.index_fused([np.asarray(frames[i], float) for i in post], drv.Mc, B=len(post))
    solved, failed = [], []
    for i, M in zip(post, Ms):
        (solved if strict_gate(np.asarray(M, float) if M is not None else None,
                               frames[i], drv.Mc) else failed).append(i)
    print(f"locked_after={drv.locked_after}  post-lock={len(post)}  "
          f"solved={len(solved)}  FAILED={len(failed)}")
    print(f"failed frames: {failed}\n")

    groups = {}
    for tag, idxs in (("FAILED", failed), ("solved", solved)):
        print("=" * 96); print(f"GROUP: {tag}  ({len(idxs)} frames)"); print("=" * 96)
        rows = []
        for i in idxs:
            q = frames[i]
            lats, stop, cum = peel(q)
            sl = sum(L["frac_orig_new"] for L in lats if L["lyso"])
            rows.append(dict(i=i, npk=len(q), lats=lats, stop=stop, cum=cum, summed_lyso=sl))
            fr = [f"{L['frac_orig_new']:.3f}" for L in lats] + ["-"] * 3
            ly = ",".join("Y" if L["lyso"] else "N" for L in lats) or "-"
            flag = "  <== FAMILY CLEARS GATE" if (sl >= GATE and
                   max([L["frac_orig_new"] for L in lats], default=0) < GATE) else ""
            print(f"  frame {i:3d} npk={len(q):4d} nlat={len(lats)} "
                  f"L1={fr[0]} L2={fr[1]} L3={fr[2]} cum={cum:.3f} "
                  f"sumLYSO={sl:.3f} [{ly}]{flag}")
        groups[tag] = rows

    print("\n" + "=" * 96)
    print(f"GROUP: randomized null over the {len(failed)} failures, {nseed} seed(s) each")
    print("=" * 96)
    rrows = []
    for i in failed:
        for s in range(nseed):
            rng = np.random.default_rng(1000 * i + s)
            lats, stop, cum = peel(randomize(frames[i], rng))
            rrows.append(dict(i=(i, s), npk=len(frames[i]), lats=lats, stop=stop, cum=cum,
                              summed_lyso=sum(L["frac_orig_new"] for L in lats if L["lyso"])))
    n_any = sum(1 for r in rrows if len(r["lats"]) >= 1)
    n_two = sum(1 for r in rrows if len(r["lats"]) >= 2)
    print(f"  false accepts under the FIXED rule: >=1 lattice {n_any}/{len(rrows)}, "
          f">=2 lattices {n_two}/{len(rrows)}")
    groups["random"] = rrows

    print("\n" + "=" * 96); print("SUMMARY"); print("=" * 96)
    for tag in ("FAILED", "solved", "random"):
        summarize(tag, groups[tag])

    f_rows, s_rows = groups["FAILED"], groups["solved"]
    fb = [max([L["frac_orig_new"] for L in r["lats"]], default=0.0) for r in f_rows]
    sb = [max([L["frac_orig_new"] for L in r["lats"]], default=0.0) for r in s_rows]
    print(f"\nDOMINANCE: best single lattice  FAILED {np.mean(fb):.3f}  vs  "
          f"solved {np.mean(sb):.3f}   (separation = {np.mean(sb)-np.mean(fb):+.3f})")
    rescuable = [r["i"] for r in f_rows if r["summed_lyso"] >= GATE and
                 max([L["frac_orig_new"] for L in r["lats"]], default=0) < GATE]
    print(f"RESCUE HEADROOM: {len(rescuable)}/{len(f_rows)} failures reach >={GATE} as a summed "
          f"LYSO family while no single lattice does: {rescuable}")
    print(f"\nDONE elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
