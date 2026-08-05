"""Both sides of the consensus-barrier speedup, measured together on one machine.

glint.tex:915-918 claims the pooled-consensus barrier was "an ~0.9 s once-per-run barrier; caching
one reduction per hypothesis cuts it 5.3x (to 0.17 s)". Only the cached side can be re-measured from
the shipped code -- `_consensus_exact` IS the cache, so the 0.9 s baseline no longer exists to run.
That left the ratio unverifiable: `hist_timing.py` prints the barrier but not what it is a speedup
OVER, so a re-measure of the barrier alone (159.9 ms shipped, 142.0 ms pre-#87, vs a published
0.17 s) could correct the numerator and say nothing about the 5.3x.

This reconstructs the uncached path so both sides come from the same run.

  CACHED    the shipped path: one reduced_params per hypothesis into RP, then greedy grouping that
            compares cached (lengths, cosines, det) tuples. `_consensus_exact` in glint/multishot.py.
  UNCACHED  the identical greedy loop, but calling same_lattice(M_i, M_group_rep) per comparison, so
            each test re-runs the Buerger reduction on BOTH operands -- what the code did before the
            cache landed. Same grouping, same tolerances, same acceptance; only the reduction is
            repeated.

Input is experiments/nbest_120.npz, the recorded 120-frame x top-3 pool (360 hypotheses) that the
published barrier was measured on -- tracked in the repo, so this needs no GPU and no re-indexing.
Both sides are warmed before timing: cold-vs-warm on this codebase is ~3.9x and would swamp the
comparison.

  python experiments/consensus_barrier_ab.py [reps]
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.multishot import consensus_cell, same_lattice          # noqa: E402
from glint.lattice import reduced_params                          # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
RTOL, CTOL, VTOL = 0.05, 0.06, 0.10

here = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(here, "nbest_120.npz"))
cells = [np.asarray(M, float) for M in d["cells"]]
print(f"pool: {len(cells)} hypotheses ({int(d['nframes'])} frames x top-3)")


def uncached(valid):
    """The pre-cache path: greedy same_lattice grouping, one Buerger reduction per COMPARISON.

    Mirrors _grp_reduced's loop exactly -- first matching group wins, groups ranked by weight -- so
    the only difference from the shipped path is where the reduction happens. Returns (support,
    n_same_lattice_calls) so the cost model is visible, not just the wall time.
    """
    groups = []                                               # [rep_index, weight, members]
    ncalls = 0
    for idx, M in enumerate(valid):
        for grp in groups:
            ncalls += 1
            if same_lattice(M, valid[grp[0]], rtol=RTOL, ctol=CTOL, vtol=VTOL):
                grp[1] += 1
                grp[2].append(idx)
                break
        else:
            groups.append([idx, 1, [idx]])
    ranked = sorted(groups, key=lambda g: g[1], reverse=True)
    return ranked[0][1], ncalls


def bench(fn, reps):
    fn()                                                      # warm: ~3.9x cold-vs-warm on this code
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


_, sup_c = consensus_cell(cells)
sup_u, ncalls = uncached(cells)
print(f"support: cached {sup_c}   uncached {sup_u}   "
      f"{'AGREE' if sup_c == sup_u else 'DISAGREE -- the two paths are not equivalent'}")
print(f"uncached does {ncalls} same_lattice calls = {2*ncalls} Buerger reductions; "
      f"cached does {len(cells)}\n")

med_c, lo_c, hi_c = bench(lambda: consensus_cell(cells), REPS)
med_u, lo_u, hi_u = bench(lambda: uncached(cells), REPS)

print(f"{'path':10s} {'median':>10s} {'min':>10s} {'max':>10s}")
print("-" * 44)
print(f"{'cached':10s} {1e3*med_c:9.1f}ms {1e3*lo_c:9.1f}ms {1e3*hi_c:9.1f}ms")
print(f"{'uncached':10s} {1e3*med_u:9.1f}ms {1e3*lo_u:9.1f}ms {1e3*hi_u:9.1f}ms")
print("-" * 44)
print(f"speedup (uncached / cached): {med_u/med_c:.2f}x")
print(f"\npaper glint.tex:915-918 says ~0.9 s -> 0.17 s, a 5.3x")
print(f"measured here:              {med_u:.3f} s -> {med_c:.3f} s, a {med_u/med_c:.1f}x")
print("Both sides from one run, one machine, warmed. Cite the ratio only from a run that")
print("measured BOTH -- a barrier-only re-measure cannot check it.")
