"""D2 fix test: the broadening cliff is an INLIER-TOLERANCE artifact, not generation death.
hkl residual under broadening = dq.a ~ sigma*|axis|, so along the 79A axis it is sigma*79,
crossing the fixed hkl TOL=0.15 at sigma~0.0019 (= the failmodes knee). The matched()
criterion |q@M - round| < 0.15 is ANISOTROPIC in hkl (penalises long axes). Fix: match in
RECIPROCAL DISTANCE |q - nearest_node| (1/A, isotropic, sigma-scaled). Here, on simulated
broadened lyso frames with the TRUE cell M, compare match-fraction under the two criteria
vs sigma, and the solved-rate (index_blind_fast cell) gated each way.

  python d2_tolerance.py [K]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
from glint_fast import index_blind_fast, LYSO
from glint.multishot import same_lattice
from fat_ewald import sim_fat

SIGS = (0.0, 0.0006, 0.001, 0.0015, 0.002, 0.003)
HKL_TOL = 0.15
QD_TOL = 0.004                                       # 1/A reciprocal-distance inlier tol


def fracs(M, q):
    """match fraction of spots to lattice M under hkl-residual vs reciprocal-distance."""
    H = q @ M
    r = H - np.rint(H)
    hkl_frac = (np.abs(r).max(1) < HKL_TOL).mean()
    qresid = r @ np.linalg.inv(M)                    # q - nearest node, in 1/A
    qd_frac = (np.linalg.norm(qresid, axis=1) < QD_TOL).mean()
    return hkl_frac, qd_frac


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.default_rng(0)
    g0, M0 = sim_fat(0.001, rng=rng); index_blind_fast(g0)        # warmup
    print(f"D2 tolerance test  K={K}/sigma  hkl_tol={HKL_TOL}  qdist_tol={QD_TOL} 1/A")
    print(f"{'sigma':>8} {'TRUEcell hkl-frac':>18} {'TRUEcell qd-frac':>17} "
          f"{'solved hkl':>11} {'solved qd':>10}")
    for sig in SIGS:
        rng = np.random.default_rng(1)
        hf = []; qf = []; sh = 0; sq = 0; nf = 0
        for _ in range(K):
            g, M = sim_fat(0.001, rng=rng)
            if len(g) < 8:
                continue
            gb = g + rng.normal(0, sig, g.shape) if sig > 0 else g
            h, q = fracs(M, gb)                       # TRUE cell match under both criteria
            hf.append(h); qf.append(q)
            Msel = index_blind_fast(gb)               # what GLINT actually returns
            if Msel is not None and same_lattice(Msel, LYSO):
                fh, fq = fracs(Msel, gb)
                sh += fh >= 0.25; sq += fq >= 0.25
            nf += 1
        print(f"{sig:8.4f} {100*np.median(hf):17.0f}% {100*np.median(qf):16.0f}% "
              f"{100*sh//max(nf,1):10d}% {100*sq//max(nf,1):9d}%")
