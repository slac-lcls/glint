"""Joint multi-shot cell recovery vs single-shot, at extreme sparsity.

Pools rotation-invariant |q|^2 over many shots -> powder-like spectrum -> fits the
shared metric tensor with NO per-shot indexing. Reaches n where single-shot (and the
bootstrap that needs single-shot successes) is 0%.

HONEST STATUS: the pooled-|q|^2 fit is the powder-indexing problem and inherits its
hard cases -- line overlap for anisotropic/low-symmetry cells, and ambiguity for large
cells (many lines). It works cleanly for smaller / higher-symmetry cells but the
figure of merit is sensitive to cell size and spot count (e.g. below it nails n=4-6 but
slips at n=8-12 for the same cell as the spectrum densens). Robustifying needs a proper
powder FOM (de Wolff M20 / DICVOL-class) and/or the within-shot ANGULAR info that powder
discards (pooled pairwise dot products q_i.q_j -> the off-diagonal metric terms).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import sys

import numpy as np

sys.path.insert(0, "..")
from fftindex import index_shot, score, simulate_shot
from fftindex.joint import cell_from_diag, fit_orthorhombic, pooled_q2, spectrum_peaks

CELL = (48.0, 48.0, 55.0, 90.0, 90.0, 90.0)   # tetragonal (favorable)
N_SHOTS = 1500


def single_rate(n, trials=20):
    sv = []
    for i in range(trials):
        s = simulate_shot(rng=np.random.default_rng(9000 + i), cell=CELL, n_target=n,
                          pos_sigma=0.0005, frac_spurious=0.1)
        sv.append(score(s, index_shot(s.g, s.meta["qmax"]))["solved"])
    return 100 * np.mean(sv)


def joint_fit(n):
    shots = [simulate_shot(rng=np.random.default_rng(9000 + s), cell=CELL, n_target=n,
                           pos_sigma=0.0005, frac_spurious=0.1) for s in range(N_SHOTS)]
    pk, _ = spectrum_peaks(pooled_q2(shots), shots[0].meta["qmax"], nbins=6000, smooth=1.5)
    ABC = fit_orthorhombic(pk)
    return cell_from_diag(ABC) if ABC is not None else None


if __name__ == "__main__":
    truth = np.sort(CELL[:3])[::-1]
    print(f"cell {CELL}, {N_SHOTS} pooled shots, jitter 0.0005, 10% spurious")
    print(f"{'n_spots':>7} {'single-shot':>12} {'joint cell':>20} {'correct':>8}")
    for n in [12, 8, 6, 4]:
        c = joint_fit(n)
        ok = c is not None and np.max(np.abs(np.sort(c)[::-1] - truth)) < 2.0
        print(f"{n:>7} {single_rate(n):>11.0f}% {str(np.round(c, 1)):>20} {'YES' if ok else 'no':>8}")
