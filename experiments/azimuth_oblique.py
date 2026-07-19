"""Rescue rate vs cell obliquity |c01|, half-turn vs full-turn azimuth sweep.

Companion to azimuth_coverage.py, which showed the effect on the two oblique cells of the standard
10-cell set (dense triclinic 67% -> 100%). Here: random triclinic cells binned by |c01| (the
unit-cosine between the two SHORTEST axes -- the pair the sweep actually constrains), so the loss
can be read as a function of obliquity rather than off two anecdotes. Stills carry position noise,
spurious peaks and partiality so the sparse regime is not sitting at its ceiling.

  python azimuth_oblique.py [ncell] [ntrial] [regime: both|dense|still]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glint.replica_gpu as rg
from glint.replica_gpu import index_known_gpu_cell
from glint.multishot import same_lattice
from experiments.azimuth_coverage import cell_to_A, rand_rot, rlps, still, gate, set_grid, ARMS

NCELL = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NTRIAL = int(sys.argv[2]) if len(sys.argv) > 2 else 60
REGIME = sys.argv[3] if len(sys.argv) > 3 else "both"
DENSE_CAP = 3000
POS_SIGMA = 5e-4          # rlp position noise (~detector/geometry error)
FRAC_SPUR = 0.10          # spurious (non-lattice) peaks
KEEP = 0.65               # partiality: fraction of shell-crossing rlps actually recorded

rg.NANG_BASE = rg.NANG


def rand_triclinic(rng):
    """Random triclinic cell; lengths 40-90 A, angles 75-105 deg (rejecting degenerate metrics)."""
    while True:
        abc = rng.uniform(40, 90, 3)
        ang = rng.uniform(75, 105, 3)
        try:
            A = cell_to_A(*abc, *ang)
        except (ValueError, FloatingPointError):
            continue
        if not np.all(np.isfinite(A)) or abs(np.linalg.det(A)) < 1e4:
            continue
        return A, tuple(abc) + tuple(ang)


def noisy_still(q, rng, target=45):
    s = still(q, rng, target=int(target / KEEP))
    if len(s) == 0:
        return s
    s = s[rng.random(len(s)) < KEEP]                       # partiality
    s = s + rng.normal(0, POS_SIGMA, s.shape)              # position noise
    nspur = int(round(FRAC_SPUR * max(len(s), 1)))
    if nspur:                                              # spurious peaks on the same shell
        r = np.linalg.norm(q, axis=1).max() if len(q) else 0.3
        d = rng.normal(size=(nspur, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        s = np.vstack([s, d * rng.uniform(0.05, r, (nspur, 1))])
    return s


def run(regime):
    print(f"\n=== {regime.upper()} ===  {NCELL} random triclinic cells x {NTRIAL} orientations"
          f"  (dev={rg.DEV}, NANG={rg.NANG_BASE})")
    rng = np.random.default_rng(7)
    rows = []
    for ci in range(NCELL):
        A0, cp = rand_triclinic(rng)
        _, c01, _, _, _ = rg._axes_from_cell(A0)
        frames, Mcs = [], []
        for t in range(NTRIAL):
            Ar = rand_rot(rng) @ A0
            q = rlps(Ar, rng)
            if regime == "dense":
                if len(q) > DENSE_CAP:
                    q = q[rng.choice(len(q), DENSE_CAP, replace=False)]
            else:
                q = noisy_still(q, rng)
            if len(q) >= 10:
                frames.append(q); Mcs.append(Ar)
        if not frames:
            continue
        ok = {}
        for aname, full, mult in ARMS:
            set_grid(full, mult)
            index_known_gpu_cell(frames[0], Mcs[0])
            res = [index_known_gpu_cell(q, Mc) for q, Mc in zip(frames, Mcs)]
            ok[aname] = sum(gate(M, q, Mc) for M, q, Mc in zip(res, frames, Mcs)) / len(frames)
        rows.append((abs(c01), ok, len(frames)))
        print(f"  |c01|={abs(c01):.4f}  n={len(frames):3d}  " +
              "  ".join(f"{a[0]}={100*ok[a[0]]:5.1f}%" for a in ARMS), flush=True)

    print(f"\n  {'|c01| bin':<14}{'cells':>6}{'frames':>8}" + "".join(f"{a[0]:>9}" for a in ARMS) +
          f"{'full-half':>11}")
    edges = [0.0, 0.02, 0.05, 0.10, 0.20, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[0] < hi]
        if not sel:
            continue
        nf = sum(r[2] for r in sel)
        m = {a[0]: sum(r[1][a[0]] * r[2] for r in sel) / nf for a in ARMS}
        print(f"  [{lo:.2f},{hi:.2f}){'':<4}{len(sel):>6}{nf:>8}" +
              "".join(f"{100*m[a[0]]:8.1f}%" for a in ARMS) +
              f"{100*(m['full']-m['half']):+10.1f}")


for r in (["dense", "still"] if REGIME == "both" else [REGIME]):
    run(r)
