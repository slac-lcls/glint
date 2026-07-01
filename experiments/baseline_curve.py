"""Honest characterization with the robust (difference-seeded, fail-safe) indexer.

Reports solved% / wrong-index% / no-index% under the lattice-match metric, so we
can see that failures are now honest no-index rather than confident wrong cells.
Cut (3) also ablates difference-vector seeding on/off to show it drives the gain.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"     # must precede numpy import to bind BLAS
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time

import numpy as np

sys.path.insert(0, "..")
from glint import index_shot, score, simulate_shot

N_TRIALS = 20
TOL = 0.02


def stats(n_target, pos_sigma=0.0, frac_spurious=0.0, seed_diff=True):
    sv, wr, ni = [], [], []
    for s in range(N_TRIALS):
        rng = np.random.default_rng(300 + s)
        shot = simulate_shot(rng=rng, n_target=n_target, pos_sigma=pos_sigma,
                             frac_spurious=frac_spurious)
        res = index_shot(shot.g, shot.meta["qmax"], n=128, localize=False,
                         tol_frac=TOL, seed_diff=seed_diff)
        sc = score(shot, res, tol_frac=TOL)
        sv.append(sc["solved"]); wr.append(sc["wrong_index"]); ni.append(sc["no_index"])
    return 100 * np.mean(sv), 100 * np.mean(wr), 100 * np.mean(ni)


def line(label, st):
    print(f"     {label:<16}: solved {st[0]:>3.0f}%  wrong {st[1]:>3.0f}%  no-index {st[2]:>3.0f}%")


if __name__ == "__main__":
    t0 = time.time()

    print("(1) vs spot count (clean)")
    for nt in [15, 20, 25, 30, 40, 60, 90]:
        line(f"n={nt}", stats(nt))

    print("\n(2) vs jitter sigma [1/A] (n=60)")
    for ps in [0.0, 0.0005, 0.001, 0.002, 0.004]:
        line(f"sigma={ps}", stats(60, pos_sigma=ps))

    print("\n(3) vs spurious fraction (n=60)  [seed_diff ON vs OFF]")
    for fs in [0.0, 0.1, 0.2, 0.4]:
        on = stats(60, frac_spurious=fs, seed_diff=True)
        off = stats(60, frac_spurious=fs, seed_diff=False)
        print(f"     spurious={fs:<4} ON : solved {on[0]:>3.0f}%  wrong {on[1]:>3.0f}%  no-index {on[2]:>3.0f}%")
        print(f"     spurious={fs:<4} OFF: solved {off[0]:>3.0f}%  wrong {off[1]:>3.0f}%  no-index {off[2]:>3.0f}%")

    print(f"\nwall time: {time.time() - t0:.1f}s")
