"""Score a GLINT run at Table 1's exact gate -- the missing half of the reproduction path.

`experiments/xgandalf/score_xg_gate.py` scores the xgandalf arms of `tab:summary` from committed
solutions, so a reader can reproduce those rows in seconds. Nothing did the same for GLINT's own
rows: running `hybrid_index` prints its loose `n_idx`, which is NOT a table value, and telling a
reader to "apply the same gate" supplied neither a scorer nor a command (Copilot review of
glint#161). This is that scorer, applying the identical rule:

    correct lattice   same_lattice(M, LYSO)                 -- reduced-cell match
    AND >=25% spots   |q @ M - round(q @ M)|_max < TOL      -- matched fraction >= 0.25
    AND >=10 refl     ...and at least ten of them

TOL = 0.15 and the 0.25 fraction are read from score_xg_gate.py rather than restated, so the two
arms cannot drift apart -- comparing indexers under different gates is the exact mistake the
three-column output exists to make visible.

Blind by default; `--cell` scores the known-cell arm instead.

  PYTHONPATH=. python experiments/score_glint_gate.py [-N 120] [--cell]

Runs on CPU (~2 min for 120 frames on a laptop); set CUDA_VISIBLE_DEVICES= to force CPU on a
GPU host.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from glint.lattice import cell_to_Ar                                        # noqa: E402
from glint.multishot import same_lattice                                    # noqa: E402

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
TOL = 0.15            # same as experiments/xgandalf/score_xg_gate.py; see the note above
MIN_FRAC = 0.25
MIN_REFL = 10


def gate(M, q):
    """Table 1's gate, returning (correct_lattice, >=25% spots, >=10 reflections)."""
    if M is None or not same_lattice(np.asarray(M, float), LYSO):
        return (0, 0, 0)
    H = q @ np.asarray(M, float)
    m = int((np.abs(H - np.rint(H)).max(1) < TOL).sum())
    return (1, int(m / len(q) >= MIN_FRAC), int(m >= MIN_REFL))


def main(argv=None):
    ap = argparse.ArgumentParser(description="score a GLINT run at tab:summary's gate")
    ap.add_argument("-N", type=int, default=120, help="frames to score (default 120)")
    ap.add_argument("--cell", action="store_true",
                    help="known-cell arm: hand the lysozyme cell instead of deriving it")
    ap.add_argument("--frames", default=os.path.join(ROOT, "experiments/frames_cxidb_clean.txt"))
    a = ap.parse_args(argv)

    from glint.glint_fast import load
    from glint.hybrid_stream import hybrid_index

    frames = [np.asarray(q, float) for q in load(a.frames)][:a.N]
    frames = [q for q in frames if len(q) >= 6]
    results, stats = hybrid_index(frames, Mc_known=(LYSO if a.cell else None))

    n = lat = g25 = g10 = 0
    for q, r in zip(frames, results):
        n += 1
        c, f, t = gate(r.get("M") if isinstance(r, dict) else r, q)
        lat += c; g25 += f; g10 += t
    arm = "KNOWN-CELL" if a.cell else "BLIND"
    # ROUND, not floor -- the deliverables' convention; flooring is what printed the retired
    # 76/71 pair when the counts had already moved to 92/86.
    print(f"GLINT-(1) {arm:10s} N={n}: correct-lattice {lat}/{n}={round(100*lat/n)}%  "
          f">=25%(Table1) {g25}/{n}={round(100*g25/n)}%  >=10refl {g10}/{n}={round(100*g10/n)}%")
    if not a.cell:
        print(f"  consensus cell edges {np.round(stats['edges'], 1)}  support {stats['support']}"
              f"  (refused: {stats.get('consensus_refused')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
