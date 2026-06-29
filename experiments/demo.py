"""Smoke test: index one clean shot, then sweep indexing rate vs spot count."""

import sys

import numpy as np

sys.path.insert(0, "..")
from fftindex import index_shot, score, simulate_shot


def one_shot(seed=0, **kw):
    rng = np.random.default_rng(seed)
    shot = simulate_shot(rng=rng, **kw)
    res = index_shot(shot.g, shot.meta["qmax"], n=128)
    return shot, res, score(shot, res)


if __name__ == "__main__":
    shot, res, sc = one_shot(seed=1, n_target=60)
    print("=== single clean shot (n_target=60) ===")
    print(f"observed spots: {len(shot.g)}   true: {sc['n_true']}")
    print(f"indexed: {sc['n_indexed']}/{sc['n_true']}  "
          f"(frac={sc['frac_indexed']:.2f})  lattice_match={sc['lattice_match']}  "
          f"solved={sc['solved']}")
    print(f"per-axis err: {sc['axis_err']:.3f} A")

    print("\n=== indexing rate vs spot count (20 shots each) ===")
    print("solved = correct lattice (integer unimodular T), not just frac indexed")
    print(f"{'n_target':>9} {'solved %':>9}")
    for nt in [15, 20, 25, 30, 40, 60, 90]:
        sv = []
        for s in range(20):
            _, _, sc = one_shot(seed=100 + s, n_target=nt,
                                pos_sigma=0.002, frac_spurious=0.1)
            sv.append(sc["solved"])
        print(f"{nt:>9} {100*np.mean(sv):>8.0f}%")
