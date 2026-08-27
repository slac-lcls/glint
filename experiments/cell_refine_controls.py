"""Controls for cell_refine_measure.py's surprise: the NAIVE running refit scored 77/115 at the
strict gate -- ABOVE the exact-textbook-LYSO "ceiling" (75/115) -- while sitting FARTHER from
textbook LYSO (dLmax 0.39 vs 0.31 A). Three things must be checked before believing it:

  A. STATIC re-registration of each arm's FINAL cell over all 115 post-lock frames. The chunked arm
     registers early frames with the OLD cell, so its score mixes cells. If the final cell alone
     scores >= 77 static, the cell itself is genuinely better and textbook LYSO is simply not the
     optimum for this data+geometry. If it scores ~73, the chunked gain was an artifact.
  B. PAIRED frame-level flips vs baseline (gained/lost sets) + exact-binomial McNemar on the
     discordant pairs, so a +4 is not read as signal when it is 6 gained / 2 lost.
  C. ORDER SENSITIVITY: a running refit depends on frame order. Re-run naive/robust with shuffled
     post-lock order (3 seeds) and see whether the gain survives.
  D. Minimal shippable variant: ONE refit after the first K frames, then frozen (no running update).
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT)
sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import matched_strict
from glint.multishot import same_lattice
from glint.lattice import cell_to_Ar, cell_params
from glint.predict import _canonical_axes
import glint.replica_gpu_batch as rgb
import glint.stream_driver as sd

from cell_refine_measure import (strict_gate, push_blind_q, frac_of, dist_line, cell_of, refit,
                                 Mc_from_cell, register, run_arm, score, dev,
                                 LYSO, LYSO_CELL, GATE_FRAC, FRAMES_PATH)


def mcnemar(a_ok, b_ok):
    """exact two-sided binomial on discordant pairs; a_ok/b_ok are boolean arrays."""
    g = int((~a_ok & b_ok).sum())          # gained by b
    l = int((a_ok & ~b_ok).sum())          # lost by b
    nd = g + l
    if nd == 0:
        return g, l, 1.0
    from math import comb
    k = min(g, l)
    p = sum(comb(nd, i) for i in range(0, k + 1)) / 2.0 ** nd * 2.0
    return g, l, min(p, 1.0)


def gate_vec(frames, res, idxs):
    return np.array([strict_gate(res.get(i), frames[i]) for i in idxs])


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
            warmup_idx.append(i); push_blind_q(drv, qq)
        else:
            drv._q[drv._n] = qq; drv._n += 1; drv.n_pushed += 1
            if drv._n == drv.B:
                drv.flush()
    drv.flush()
    post = [i for i in range(n) if i not in set(warmup_idx)]
    C0 = np.array(cell_params(_canonical_axes(drv.Mc)))

    res_base = register(frames, post, drv.Mc)
    g_base = gate_vec(frames, res_base, post)
    fr_base = np.array([frac_of(res_base.get(i), frames[i]) for i in post])
    res_ly = register(frames, post, LYSO)
    g_ly = gate_vec(frames, res_ly, post)
    fr_ly = np.array([frac_of(res_ly.get(i), frames[i]) for i in post])
    print(f"baseline frozen Mc: {g_base.sum()}/{len(post)}   exact LYSO: {g_ly.sum()}/{len(post)}")
    gg, ll, p = mcnemar(g_base, g_ly)
    print(f"  LYSO vs baseline: gained {gg}, lost {ll}, McNemar exact p={p:.3f}")

    # ---------------- A + B: running arms, their final cells applied STATICALLY -----------------
    print("\n" + "=" * 108)
    print("A/B. running arm  vs  its FINAL cell applied statically to all 115 frames, with paired flips")
    hdr = (f"{'arm':22s} {'K':>3s} | {'run gate':>9s} {'gain':>4s} {'lost':>4s} {'p':>6s} "
           f"| {'static gate':>11s} {'gain':>4s} {'lost':>4s} {'p':>6s} | {'stat mean':>9s} {'<0.25':>6s}")
    print(hdr); print("-" * len(hdr))
    finals = {}
    for K in (10, 25, 50):
        for rule in ("naive", "robust", "symm"):
            res_r, traj = run_arm(frames, post, drv.Mc, K, rule, refine=True)
            g_r = gate_vec(frames, res_r, post)
            Cf = traj[-1][2]; finals[(rule, K)] = Cf
            Mcf = Mc_from_cell(Cf)
            res_s = register(frames, post, Mcf)
            g_s = gate_vec(frames, res_s, post)
            fr_s = np.array([frac_of(res_s.get(i), frames[i]) for i in post])
            a1, b1, p1 = mcnemar(g_base, g_r)
            a2, b2, p2 = mcnemar(g_base, g_s)
            print(f"{'refined ' + rule:22s} {K:3d} | {g_r.sum():4d}/{len(post):<4d} {a1:4d} {b1:4d} "
                  f"{p1:6.3f} | {g_s.sum():6d}/{len(post):<4d} {a2:4d} {b2:4d} {p2:6.3f} "
                  f"| {fr_s.mean():9.4f} {(fr_s<GATE_FRAC).sum():6d}")
    print(f"[{time.time()-t0:.0f}s]")

    # ---------------- C: order sensitivity -----------------------------------------------------
    print("\n" + "=" * 108)
    print("C. ORDER SENSITIVITY  (post-lock frames shuffled; gate counted over the SAME 115 frames)")
    print(f"   {'rule':8s} {'K':>3s} {'seed':>5s} {'gate':>9s} {'mean frac':>10s} {'a':>9s} {'b':>9s} "
          f"{'c':>9s} {'dLmax':>7s} {'d(a+b)/2':>9s}")
    for rule in ("naive", "robust", "symm"):
        for seed in (0, 1, 2, 3):
            rng = np.random.default_rng(seed)
            order = list(np.array(post)[rng.permutation(len(post))]) if seed else list(post)
            res_r, traj = run_arm(frames, order, drv.Mc, 25, rule, refine=True)
            g_r = gate_vec(frames, res_r, post)
            fr_r = np.array([frac_of(res_r.get(i), frames[i]) for i in post])
            C = traj[-1][2]
            tag = "native" if seed == 0 else str(seed)
            print(f"   {rule:8s} {25:3d} {tag:>5s} {g_r.sum():4d}/{len(post):<4d} {fr_r.mean():10.4f} "
                  f"{C[0]:9.4f} {C[1]:9.4f} {C[2]:9.4f} {dev(C)[0]:7.4f} "
                  f"{0.5*(C[0]+C[1])-LYSO_CELL[0]:+9.4f}")

    # ---------------- D: one-shot refit then frozen --------------------------------------------
    print("\n" + "=" * 108)
    print("D. ONE-SHOT refit after the first K frames, then FROZEN (minimal shippable variant).")
    print("   'rest' = gate over the remaining 115-K frames only, refit vs frozen, paired.")
    print(f"   {'rule':8s} {'K':>3s} {'all-115 gate':>13s} {'rest refit':>11s} {'rest frozen':>12s} "
          f"{'gain':>4s} {'lost':>4s} {'p':>6s} {'dLmax':>7s}")
    for K in (10, 25, 50):
        for rule in ("naive", "robust", "symm"):
            head, rest = post[:K], post[K:]
            r_head = register(frames, head, drv.Mc)
            pool = [(cell_of(r_head[i]), matched_strict(r_head[i], frames[i]) / len(frames[i]))
                    for i in head if r_head[i] is not None and abs(np.linalg.det(r_head[i])) >= 1.0]
            C = refit(pool, rule)
            if C is None:
                print(f"   {rule:8s} {K:3d}   refit failed (pool too small)"); continue
            r_rest = register(frames, rest, Mc_from_cell(C))
            res_all = dict(r_head); res_all.update(r_rest)
            g_all = gate_vec(frames, res_all, post)
            gr = gate_vec(frames, r_rest, rest)
            gf_ = gate_vec(frames, res_base, rest)
            a, b, p = mcnemar(gf_, gr)
            print(f"   {rule:8s} {K:3d} {g_all.sum():8d}/{len(post):<4d} {gr.sum():7d}/{len(rest):<4d} "
                  f"{gf_.sum():8d}/{len(rest):<4d} {a:4d} {b:4d} {p:6.3f} {dev(C)[0]:7.4f}")

    # ---------------- E: is textbook LYSO simply not optimal here? -----------------------------
    print("\n" + "=" * 108)
    print("E. 1-D scan of an ISOTROPIC scale on the textbook LYSO cell -- does the data prefer a cell")
    print("   different from textbook, independent of any refinement machinery?")
    print(f"   {'scale':>8s} {'a':>8s} {'c':>8s} {'gate':>9s} {'mean frac':>10s} {'<0.25':>6s}")
    for s in (0.994, 0.996, 0.998, 0.999, 1.000, 1.001, 1.002, 1.004, 1.006):
        C = LYSO_CELL.copy(); C[:3] *= s
        rr = register(frames, post, Mc_from_cell(C))
        g = gate_vec(frames, rr, post)
        fr = np.array([frac_of(rr.get(i), frames[i]) for i in post])
        print(f"   {s:8.4f} {C[0]:8.3f} {C[2]:8.3f} {g.sum():4d}/{len(post):<4d} {fr.mean():10.4f} "
              f"{(fr<GATE_FRAC).sum():6d}")
    print(f"   (driver.Mc for reference: a={C0[0]:.3f} b={C0[1]:.3f} c={C0[2]:.3f} -> "
          f"{g_base.sum()}/{len(post)}, mean {fr_base.mean():.4f})")

    print(f"\nDONE elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
