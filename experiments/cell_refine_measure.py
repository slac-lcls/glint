"""What would a RUNNING CELL REFINEMENT buy StreamDriver?

StreamDriver.Mc is locked once (_lock()) and never updated. This measures, on the real 120-frame
cxidb set, what a running refit of the cell from already-registered frames would change:
  (1) baseline: frozen driver.Mc -- gate count AND the full matched_frac distribution
  (2) refined arms: refit every K post-lock frames (K=10,25,50) x {naive, robust(frac>=0.25),
      symmetrized-tetragonal}, re-registering only the REMAINING frames against the update
  (3) ceiling: exact textbook LYSO
  (4) hkl-consistency hazard: fraction of reflections relabelled frozen->refined, vs resolution
  (5) drift: cell trajectory at every update step, distance to textbook LYSO

Only index_fused's known-cell registration is varied. index_fused consumes ONLY the cell geometry
of Mc (_axes_from_cell -> lengths+cosines, _adaptive_dirs, same_lattice guard), so a refit cell is
injected simply as cell_to_Ar(a,b,c,al,be,ga).
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT)
sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import matched
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar, cell_params
from glint.predict import _canonical_axes
import glint.replica_gpu_batch as rgb
import glint.stream_driver as sd

LYSO = gf.LYSO
GATE_FRAC, GATE_MIN, TOL = 0.25, 10, 0.15
FRAMES_PATH = WT + "/experiments/frames_cxidb_clean.txt"
LYSO_CELL = np.array(cell_params(_canonical_axes(LYSO)))


def strict_gate(M, q, truth=None):
    if M is None:
        return False
    truth = LYSO if truth is None else truth
    if not same_lattice(M, truth):
        return False
    m = matched(M, q, TOL)
    return (m / len(q) >= GATE_FRAC) and (m >= GATE_MIN)


def push_blind_q(driver, qq):                      # verbatim from what_are_the_failures.py
    driver.n_pushed += 1
    if len(qq) >= driver.min_peaks:
        nb = driver._blind_index(qq, driver.warmup_nbest)
        driver._rc.add_frame([c for c, _ in nb])
        driver.n_warmup += 1
        if getattr(driver, "_warmup_buf", None) is not None:
            driver._warmup_buf.append(qq)
    Mc, sup, _ = driver._rc.verdict()
    if Mc is not None:
        driver._lock(Mc, sup, standardize=True)


def frac_of(M, q):
    if M is None or abs(np.linalg.det(M)) < 1.0:
        return 0.0
    return matched(M, q, TOL) / len(q)


def dist_line(label, fr):
    fr = np.asarray(fr, float)
    dec = np.percentile(fr, np.arange(0, 101, 10))
    return (f"{label:34s} n={len(fr):3d} mean={fr.mean():.4f} med={np.median(fr):.4f} "
            f"| deciles " + " ".join(f"{d:.3f}" for d in dec))


def cell_of(M):
    """Per-frame cell parameters in the pipeline's own canonical (long,long,short) setting."""
    return np.array(cell_params(_canonical_axes(np.asarray(M, float))))


def refit(pool, rule):
    """pool = list of (cellparams6, matched_frac). Returns a 6-vector cell or None."""
    if not pool:
        return None
    if rule == "naive":
        sel = [c for c, f in pool]
    elif rule == "robust":
        sel = [c for c, f in pool if f >= GATE_FRAC]
    elif rule == "symm":                                   # robust + impose tetragonal (a=b, 90/90/90)
        sel = [c for c, f in pool if f >= GATE_FRAC]
    else:
        raise ValueError(rule)
    if len(sel) < 3:
        return None
    C = np.mean(np.stack(sel), 0)
    if rule == "symm":
        ab = 0.5 * (C[0] + C[1])
        C = np.array([ab, ab, C[2], 90.0, 90.0, 90.0])
    return C


def Mc_from_cell(C):
    return cell_to_Ar(*C)


def register(frames, idxs, Mc):
    qs = [np.asarray(frames[i], float) for i in idxs]
    Ms = rgb.index_fused(qs, np.asarray(Mc, float), B=max(len(qs), 1))
    out = {}
    for i, M in zip(idxs, Ms):
        out[i] = np.asarray(M, float) if M is not None else None
    return out


def run_arm(frames, idxs, Mc0, K, rule, refine=True):
    """Chunked streaming registration. refine=False -> frozen-cell control with identical chunking."""
    Mc = np.asarray(Mc0, float).copy()
    res, pool, traj = {}, [], []
    for s in range(0, len(idxs), K):
        chunk = idxs[s:s + K]
        r = register(frames, chunk, Mc)
        res.update(r)
        for i in chunk:
            M = r[i]
            if M is not None and abs(np.linalg.det(M)) >= 1.0:
                pool.append((cell_of(M), matched(M, frames[i], TOL) / len(frames[i])))
        if refine:
            C = refit(pool, rule)
            if C is not None:
                Mc = Mc_from_cell(C)
            traj.append((s + len(chunk), len(pool), np.array(cell_params(_canonical_axes(Mc)))))
    return res, traj


def score(frames, res, idxs, truth=None):
    fr = np.array([frac_of(res.get(i), frames[i]) for i in idxs])
    g = sum(1 for i in idxs if strict_gate(res.get(i), frames[i], truth))
    return fr, g


def dev(C):
    """max |Delta| on the three lengths vs textbook LYSO, and the angle spread."""
    return (np.max(np.abs(np.sort(C[:3]) - np.sort(LYSO_CELL[:3]))),
            np.max(np.abs(C[3:] - 90.0)))


# ------------------------------------------------------------------ hkl consistency -------------
def hkl_change(frames, res_f, res_r, idxs):
    """Fraction of reflections that receive a DIFFERENT integer hkl under frozen vs refined cell.

    Both matrices are put through the pipeline's own _canonical_axes first; the RESIDUAL
    sign/handedness/a<->b freedom that _canonical_axes does not fix is then absorbed by the best
    integer basis change S = round(pinv(Af) @ Ar) (checked |det S| == 1), so only genuine
    relabelling is counted. Raw (S = I) numbers are reported alongside.
    """
    tot = raw_diff = alg_diff = 0
    nframe_ok = nframe_S = 0
    bins = {}                                                    # |hkl|max bin -> [n, ndiff]
    for i in idxs:
        Mf, Mr = res_f.get(i), res_r.get(i)
        if Mf is None or Mr is None:
            continue
        q = np.asarray(frames[i], float)
        Af, Ar = _canonical_axes(Mf), _canonical_axes(Mr)
        hf, hr = q @ Af, q @ Ar
        okf = np.abs(hf - np.rint(hf)).max(1) < TOL
        okr = np.abs(hr - np.rint(hr)).max(1) < TOL
        sel = okf & okr
        if sel.sum() == 0:
            continue
        nframe_ok += 1
        Hf, Hr = np.rint(hf[sel]), np.rint(hr[sel])
        S = np.rint(np.linalg.pinv(Af) @ Ar)
        if abs(abs(np.linalg.det(S)) - 1.0) < 1e-6:
            HfS = Hf @ S
            nframe_S += 1
        else:
            HfS = Hf                                             # no valid integer alignment found
        d_raw = (Hf != Hr).any(1)
        d_alg = (HfS != Hr).any(1)
        tot += len(Hr); raw_diff += int(d_raw.sum()); alg_diff += int(d_alg.sum())
        r = np.abs(Hr).max(1)
        for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 1000)):
            m = (r >= lo) & (r < hi)
            if m.any():
                b = bins.setdefault((lo, hi), [0, 0])
                b[0] += int(m.sum()); b[1] += int(d_alg[m].sum())
    return tot, raw_diff, alg_diff, nframe_ok, nframe_S, bins


def main():
    t0 = time.time()
    frames = list(gf.load(FRAMES_PATH))
    n = len(frames)

    N = 400
    panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
                   res=1.0 / (0.1 / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
                   coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
    drv = sd.StreamDriver(None, panels, 100.0 / 1000.0, 1.0, (N, N), dtype=np.uint16,
                          B=20, dmin=2.0, tol=0.002, warmup_nbest=3)
    warmup_idx = []
    for i, qq in enumerate(frames):
        qq = np.asarray(qq, float)
        if drv._blind:
            warmup_idx.append(i)
            push_blind_q(drv, qq)
        else:
            drv._q[drv._n] = qq
            drv._n += 1; drv.n_pushed += 1
            if drv._n == drv.B:
                drv.flush()
    drv.flush()
    warmup_set = set(warmup_idx)
    post = [i for i in range(n) if i not in warmup_set]

    C0 = np.array(cell_params(_canonical_axes(drv.Mc)))
    print("=" * 100)
    print(f"DRIVER: locked_after={drv.locked_after}  n_warmup={len(warmup_idx)}  "
          f"post-lock frames={len(post)}  same_lattice(Mc,LYSO)={same_lattice(drv.Mc, LYSO)}")
    print(f"  driver.Mc canonical cell = {np.round(C0,4).tolist()}")
    print(f"  textbook LYSO   canonical = {np.round(LYSO_CELL,4).tolist()}")
    print(f"  max |dlength| = {dev(C0)[0]:.4f} A   max |dangle| = {dev(C0)[1]:.4f} deg")
    print(f"  [{time.time()-t0:.0f}s]")

    # ------------------------------------------------------ 1. baseline (frozen, one batch) ------
    res_base = register(frames, post, drv.Mc)
    fr_b, g_b = score(frames, res_base, post)
    _, g_b_own = score(frames, res_base, post, truth=drv.Mc)
    print("\n" + "=" * 100)
    print("1. BASELINE  frozen driver.Mc, all post-lock frames in one batch")
    print(f"   strict gate (same_lattice vs LYSO): {g_b}/{len(post)}    "
          f"(vs own Mc, as test_cell_precision did: {g_b_own}/{len(post)})")
    print("   " + dist_line("matched_frac", fr_b))
    print(f"   frames with frac < 0.25: {(fr_b < GATE_FRAC).sum()}   frac == 0 (no reg): "
          f"{(fr_b == 0).sum()}")

    # ------------------------------------------------------ 3. ceiling: exact LYSO ---------------
    res_ly = register(frames, post, LYSO)
    fr_l, g_l = score(frames, res_ly, post)
    print("\n" + "=" * 100)
    print("3. CEILING  exact textbook LYSO cell (bounds any refinement)")
    print(f"   strict gate: {g_l}/{len(post)}   (baseline {g_b})   delta = {g_l - g_b:+d}")
    print("   " + dist_line("matched_frac", fr_l))
    print(f"   frames with frac < 0.25: {(fr_l < GATE_FRAC).sum()}")
    d = fr_l - fr_b
    print(f"   per-frame delta vs baseline: mean {d.mean():+.5f}  median {np.median(d):+.5f}  "
          f"improved {(d > 1e-9).sum()}  worsened {(d < -1e-9).sum()}  identical {(np.abs(d) <= 1e-9).sum()}")

    # ------------------------------------------------------ 2. refined arms ----------------------
    print("\n" + "=" * 100)
    print("2. REFINED ARMS  (chunk size K; 'frozen-chunked' is the same chunking with NO refit,")
    print("   i.e. the batching-only control)")
    hdr = (f"{'arm':30s} {'K':>3s} {'gate':>8s} {'mean':>8s} {'med':>8s} {'<0.25':>6s} "
           f"{'a':>8s} {'b':>8s} {'c':>8s} {'dLmax':>7s} {'dAmax':>7s}")
    print(hdr); print("-" * len(hdr))
    store = {}
    for K in (10, 25, 50):
        res_c, _ = run_arm(frames, post, drv.Mc, K, "naive", refine=False)
        fr_c, g_c = score(frames, res_c, post)
        store[("frozen-chunked", K)] = (res_c, fr_c, g_c, C0, None)
        print(f"{'frozen-chunked (control)':30s} {K:3d} {g_c:4d}/{len(post):<3d} {fr_c.mean():8.4f} "
              f"{np.median(fr_c):8.4f} {(fr_c<GATE_FRAC).sum():6d} {C0[0]:8.3f} {C0[1]:8.3f} "
              f"{C0[2]:8.3f} {dev(C0)[0]:7.4f} {dev(C0)[1]:7.4f}")
        for rule in ("naive", "robust", "symm"):
            res_r, traj = run_arm(frames, post, drv.Mc, K, rule, refine=True)
            fr_r, g_r = score(frames, res_r, post)
            Cf = traj[-1][2]
            store[(rule, K)] = (res_r, fr_r, g_r, Cf, traj)
            print(f"{'refined: ' + rule:30s} {K:3d} {g_r:4d}/{len(post):<3d} {fr_r.mean():8.4f} "
                  f"{np.median(fr_r):8.4f} {(fr_r<GATE_FRAC).sum():6d} {Cf[0]:8.3f} {Cf[1]:8.3f} "
                  f"{Cf[2]:8.3f} {dev(Cf)[0]:7.4f} {dev(Cf)[1]:7.4f}")
    print(f"   [{time.time()-t0:.0f}s]")

    print("\n   full matched_frac deciles per arm:")
    print("   " + dist_line("BASELINE frozen (one batch)", fr_b))
    print("   " + dist_line("CEILING exact LYSO", fr_l))
    for K in (10, 25, 50):
        for arm in ("frozen-chunked", "naive", "robust", "symm"):
            print("   " + dist_line(f"{arm} K={K}", store[(arm, K)][1]))

    # ------------------------------------------------------ 5. drift ------------------------------
    print("\n" + "=" * 100)
    print("5. DRIFT  cell at EVERY update step, all rules x all K.")
    print(f"   start (locked driver.Mc): a={C0[0]:.4f} b={C0[1]:.4f} c={C0[2]:.4f} "
          f"al={C0[3]:.4f} be={C0[4]:.4f} ga={C0[5]:.4f}  dLmax={dev(C0)[0]:.4f}")
    print(f"   target (textbook LYSO)  : a={LYSO_CELL[0]:.4f} b={LYSO_CELL[1]:.4f} "
          f"c={LYSO_CELL[2]:.4f} al=90 be=90 ga=90")
    for K in (10, 25, 50):
        for rule in ("naive", "robust", "symm"):
            print(f"\n   --- rule={rule}  K={K} ---")
            print(f"   {'nseen':>6s} {'npool':>6s} {'a':>9s} {'b':>9s} {'c':>9s} "
                  f"{'alpha':>8s} {'beta':>8s} {'gamma':>8s} {'dLmax':>7s} {'d(a+b)/2':>9s} {'dc':>8s}")
            for nseen, npool, C in store[(rule, K)][4]:
                dab = 0.5 * (C[0] + C[1]) - LYSO_CELL[0]
                dc = C[2] - LYSO_CELL[2]
                print(f"   {nseen:6d} {npool:6d} {C[0]:9.4f} {C[1]:9.4f} {C[2]:9.4f} "
                      f"{C[3]:8.4f} {C[4]:8.4f} {C[5]:8.4f} {dev(C)[0]:7.4f} {dab:+9.4f} {dc:+8.4f}")

    # ------------------------------------------------------ 4. hkl consistency --------------------
    print("\n" + "=" * 100)
    print("4. hkl-CONSISTENCY HAZARD  frozen-chunked vs refined, same chunking (K=25)")
    for rule in ("naive", "robust", "symm"):
        res_f = store[("frozen-chunked", 25)][0]
        res_r = store[(rule, 25)][0]
        tot, raw, alg, nf, nS, bins = hkl_change(frames, res_f, res_r, post)
        if tot == 0:
            print(f"   {rule}: no comparable reflections"); continue
        print(f"   {rule:7s} reflections compared={tot:6d} over {nf} frames "
              f"({nS} with a valid integer basis alignment)")
        print(f"           relabelled RAW (no alignment)   : {raw:6d}  {100*raw/tot:6.2f}%")
        print(f"           relabelled after basis alignment: {alg:6d}  {100*alg/tot:6.2f}%")
        for (lo, hi), (nn, nd) in sorted(bins.items()):
            print(f"             |hkl|max in [{lo:2d},{hi if hi<1000 else 99:2d}): "
                  f"n={nn:6d}  changed={nd:5d}  {100*nd/max(nn,1):6.2f}%")
    # also frozen vs exact-LYSO registration (the ceiling's own relabelling cost)
    tot, raw, alg, nf, nS, bins = hkl_change(frames, res_base, res_ly, post)
    print(f"   LYSO    reflections compared={tot:6d} over {nf} frames ({nS} aligned)")
    print(f"           relabelled RAW: {raw} ({100*raw/max(tot,1):.2f}%)   "
          f"aligned: {alg} ({100*alg/max(tot,1):.2f}%)")
    for (lo, hi), (nn, nd) in sorted(bins.items()):
        print(f"             |hkl|max in [{lo:2d},{hi if hi<1000 else 99:2d}): "
              f"n={nn:6d}  changed={nd:5d}  {100*nd/max(nn,1):6.2f}%")

    print(f"\nDONE elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
