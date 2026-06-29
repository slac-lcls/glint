"""The sparsity-floor lever: single-shot vs multi-shot consensus indexing.

Many shots of ONE fixed cell (random orientations) at each sparsity n:
  single-shot : index each independently (unknown cell)
  consensus   : the cell the indexed minority agree on (rotation-invariant signature)
  known-cell  : re-index ALL shots constrained to the consensus cell
If consensus emerges and known-cell indexing rescues the failures, the multi-shot
rate sits well below the single-shot sparsity floor.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "..")
from fftindex import index_shot, score, simulate_shot
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import (consensus_cell, index_known_pairangle,
                                reference_lattice, same_lattice)

TRUE_CELL = (62.0, 67.0, 71.0, 90.0, 90.0, 90.0)   # fixed primitive orthorhombic
M_TRUE = cell_to_Ar(*TRUE_CELL)                     # unrotated (signature is rot-invariant)
N_SHOTS = 50


def run(n):
    shots, singles = [], []
    for s in range(N_SHOTS):
        rng = np.random.default_rng(1000 + s)
        shot = simulate_shot(rng=rng, cell=TRUE_CELL, n_target=n,
                             pos_sigma=0.001, frac_spurious=0.1)
        res = index_shot(shot.g, shot.meta["qmax"])
        shots.append(shot)
        singles.append((res.M, score(shot, res)["solved"]))

    rate_single = 100 * np.mean([sv for _, sv in singles])
    M_cons, support = consensus_cell([M for M, _ in singles])
    cons_ok = same_lattice(M_cons, M_TRUE)

    if M_cons is None:
        return rate_single, support, cons_ok, float("nan")
    qmax = shots[0].meta["qmax"]
    Vref = reference_lattice(M_cons, qmax)          # cache: same consensus for all shots
    tree = cKDTree(Vref)
    rescued = []
    for shot in shots:
        res = index_known_pairangle(shot.g, qmax, M_cons, Vref=Vref, tree=tree)
        rescued.append(score(shot, res)["solved"])
    return rate_single, support, cons_ok, 100 * np.mean(rescued)


if __name__ == "__main__":
    print(f"cell {TRUE_CELL}, {N_SHOTS} shots/point, jitter 0.001, 10% spurious")
    print(f"{'n_spots':>7} {'single%':>8} {'consensus':>10} {'support':>8} {'known-cell%':>11}")
    for n in [18, 22, 28, 35]:
        rs, sup, ok, rk = run(n)
        print(f"{n:>7} {rs:>7.0f}% {'OK' if ok else 'no/wrong':>10} "
              f"{sup:>6}/{N_SHOTS} {rk:>10.0f}%")
