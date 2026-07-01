"""D2 diagnostic: WHY does peak broadening kill candidate generation (gen_miss)?
On simulated lysozyme frames (known true real-space axes = columns of M from sim_fat),
broaden peaks by sigma and separate two hypotheses:
  (1) DAMPING (fundamental): the objective value AT the true axis decays ~exp(-2pi^2 sig^2 |v|^2)
      (Debye-Waller-like; worst for the long 79A axis) -> true peak sinks below spurious.
  (2) FINDABILITY: the true axis is still a maximum but the ascent/selection misses it.
Also test one fix: cos (sharp=False) vs cos^2 (sharp=True) -- sharpening should HURT under
broadening. Report per sigma: objective(true)/objective(sigma=0) [damping], and generation
recall = frac of true axes within tol of a top-30 distinct maximum, for sharp on/off.

  python d2_generation.py [K_frames_per_sigma]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
from glint_index import objective, refine_vec, distinct_maxima, invq_weight, STARTS, DEV
from fat_ewald import sim_fat

SIGS = (0.0, 0.0003, 0.0006, 0.001, 0.0015, 0.002)
TOL = 0.8                                            # A: candidate-to-true-axis match (vector, allow sign)


def cands_and_obj(g, axes, sharp):
    Q = torch.as_tensor(np.asarray(g, float), dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    T = refine_vec(STARTS.clone(), Q, w, qmax)
    f, _ = objective(T, Q, w, sharp=sharp)
    C = distinct_maxima(T.cpu().numpy(), f.cpu().numpy())[:30]
    # objective AT the 3 true axes (on this, possibly broadened, g)
    A = torch.as_tensor(axes, dtype=torch.float32, device=DEV)
    fa, _ = objective(A, Q, w, sharp=sharp)
    fa = fa.cpu().numpy()
    # recall: each true axis within TOL of a candidate (allow +/- sign)
    rec = np.zeros(3, bool)
    if len(C):
        for i, a in enumerate(axes):
            d = min(np.linalg.norm(C - a, axis=1).min(), np.linalg.norm(C + a, axis=1).min())
            rec[i] = d < TOL
    return rec, fa


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = np.random.default_rng(0)
    # warmup
    g0, M0 = sim_fat(0.001, rng=rng); cands_and_obj(g0, M0.T, True)
    print(f"D2 generation diagnostic  dev={DEV}  K={K}/sigma  (true axes |a|,|b|,|c| ~ 79,79,38 A)")
    print(f"{'sigma':>8} {'damp(obj/obj0)':>15} {'recall sharp':>13} {'recall plain':>13} {'all3 sharp':>11} {'all3 plain':>11}")
    base_obj = None
    for sig in SIGS:
        rng = np.random.default_rng(1)
        damp = []; rs = []; rp = []; a3s = 0; a3p = 0; nf = 0
        for _ in range(K):
            g, M = sim_fat(0.001, rng=rng)            # 0.001 = intrinsic localisation jitter in sim
            if len(g) < 8:
                continue
            axes = M.T                                 # rows = real-space axes a,b,c
            gb = g + rng.normal(0, sig, g.shape) if sig > 0 else g
            rec_s, fa_s = cands_and_obj(gb, axes, True)
            rec_p, fa_p = cands_and_obj(gb, axes, False)
            damp.append(fa_s)                          # objective(true) under sharp
            rs.append(rec_s.mean()); rp.append(rec_p.mean())
            a3s += rec_s.all(); a3p += rec_p.all(); nf += 1
        damp = np.array(damp)                          # (nf,3)
        if base_obj is None:
            base_obj = np.median(damp, 0)              # per-axis obj at sigma=0
        dratio = np.median(damp, 0) / base_obj         # per-axis damping
        print(f"{sig:8.4f} {np.array2string(dratio, precision=2, floatmode='fixed'):>15} "
              f"{100*np.mean(rs):12.0f}% {100*np.mean(rp):12.0f}% {a3s:9d}/{nf} {a3p:9d}/{nf}")
