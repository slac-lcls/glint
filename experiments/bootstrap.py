"""Bootstrap test: is the sub-n=18 floor a shot-count knob, not a wall?

At extreme sparsity (n=12, 15) single-shot indexing is near-0, so a consensus needs
enough shots that >=3 happen to index. Index a pool once, then show the consensus
agreement (support) grow with shot count N; where it forms, the taketwo matcher
rescues the rest. If consensus forms only with more shots, the floor is shot count.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "..")
from glint import index_shot, score, simulate_shot
from glint.lattice import cell_to_Ar
from glint.multishot import (consensus_cell, index_known_pairangle,
                                reference_lattice, same_lattice)

TRUE_CELL = (62.0, 67.0, 71.0, 90.0, 90.0, 90.0)
M_TRUE = cell_to_Ar(*TRUE_CELL)
POOL = 200
PREFIXES = [50, 100, 150, 200]


def run(n):
    shots, Ms, singles = [], [], []
    for s in range(POOL):
        rng = np.random.default_rng(5000 + s)
        shot = simulate_shot(rng=rng, cell=TRUE_CELL, n_target=n,
                             pos_sigma=0.001, frac_spurious=0.1)
        res = index_shot(shot.g, shot.meta["qmax"])
        shots.append(shot); Ms.append(res.M); singles.append(score(shot, res)["solved"])

    print(f"\nn={n}: single-shot rate {100*np.mean(singles):.1f}%")
    print(f"  {'N_shots':>7} {'support':>8} {'consensus':>10}")
    M_cons = None
    for N in PREFIXES:
        mc, sup = consensus_cell(Ms[:N])
        ok = same_lattice(mc, M_TRUE)
        print(f"  {N:>7} {sup:>8} {'OK' if ok else '-':>10}")
        if ok:
            M_cons = mc
    if M_cons is None:
        print("  -> no consensus even at full pool (need more shots or a joint method)")
        return
    qmax = shots[0].meta["qmax"]
    Vref = reference_lattice(M_cons, qmax); tree = cKDTree(Vref)
    rk = [score(sh, index_known_pairangle(sh.g, qmax, M_cons, Vref=Vref, tree=tree))["solved"]
          for sh in shots[:60]]
    print(f"  -> known-cell rescue (60 shots): {100*np.mean(rk):.0f}%")


if __name__ == "__main__":
    print(f"cell {TRUE_CELL}, jitter 0.001, 10% spurious, pool {POOL}")
    for n in [12, 15]:
        run(n)
