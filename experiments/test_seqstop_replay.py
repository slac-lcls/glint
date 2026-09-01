"""Replay the sequential-stop consensus over the cached cxidb-120 N-best and pin the FACTS it backs.

WHY THIS EXISTS. check_numbers.py's FACTS carry four numbers about RunningConsensus on the clean
120-frame benchmark -- median frames to lock, false locks, the worst-order lock frame and the largest
pool at lock -- and until now nothing in CI re-derived any of them: they were measured once, by a
script that is not in the tree, and the guard only checks that the DOCUMENTS still quote them. A
change to RunningConsensus (the merge-on-multi-match repair, an adaptive-gap retune, a different
tie-break in leaders()) could move every one of them and leave CI green. This replays the
measurement itself, numpy-only, in ~2 s.

WHAT IT REPLAYS. experiments/nbest_120.npz is the top-3 blind hypotheses of each of the 120 cxidb
frames, frame-major (glint_fast.index_blind_nbest(q, 3) looped over frames_cxidb_clean.txt -- see
gen_nbest.py). Blind N-best is deterministic per frame, so the trials below differ ONLY in the order
frames arrive, which is what the claim is about. Each order is fed to
RunningConsensus(min_support=3, gap=2) -- adaptive gap and merge-on-multi-match ON, as shipped --
one frame at a time, verdict() after every frame, and the first non-None verdict is the lock. A lock
is CORRECT iff glint.multishot.same_lattice(locked cell, lyso) with the reference cell stored in the
file (79.02/79.02/37.98 tetragonal lysozyme), and it is also compared with the batch consensus_cell
over the whole pool under hybrid_stream's gate -- the two criteria the FACTS comment names.

WHAT IS PINNED, and how honestly:
  * deposition order (the order the file was written in) locks CORRECTLY, no later than the worst
    random order and no bigger a pool -- a weak consistency check, printed for the record;
  * 400 random orders at ONE fixed seed: every order locks; the MEDIAN lock frame and the FALSE-lock
    count equal FACTS seqstop_median_lock / seqstop_false_locks. These two are seed-STABLE: over
    seeds 0, 1, 2, 3, 7, 11, 42, 2024 the median was 6 and the false-lock count 0 every time;
  * the same 400 orders' WORST lock frame and LARGEST pool at lock equal FACTS
    poolgate_clean_lock_max / poolgate_clean_maxpool. These are the max of 400 draws and are NOT
    seed-stable: across those eight seeds they ranged 15-19 frames and 45-57 hypotheses. FACTS
    records 18 / 54 from an unrecorded seed; SEED = 1 below reproduces both exactly and is pinned
    for that reason -- so read those two checks as "the tail has not moved at this seed", not as an
    independent re-measurement. A regression of the stop rule moves the median or the false-lock
    count, which are the load-bearing claims, before it moves a tail value;
  * the structural identity behind poolgate_clean_maxpool: every frame in this file deposits
    exactly 3 hypotheses, so the pool at lock is 3 x the lock frame and 54 = 3 x 18; and the reason
    that FACTS key exists at all -- the clean benchmark never locks past the shipped
    lock_pool_switch (72), so the pool-keyed gate leaves it untouched -- which DOES hold at every
    seed tried (max 57 < 72) and is asserted as a bound.

If a FACTS value stops reproducing, do NOT edit FACTS to match: the measured value is printed next
to the recorded one, and the discrepancy is the finding.

  PYTHONPATH=. python experiments/test_seqstop_replay.py     # exit 0 = all pass, ~2 s
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from glint.running_consensus import RunningConsensus                        # noqa: E402
from glint.multishot import (consensus_cell, same_lattice,                  # noqa: E402
                             CONSENSUS_MIN_FRAC, CONSENSUS_MIN_LEAD)
from check_numbers import FACTS          # noqa: E402  -- import is defs only (~20 ms), no side effects

NPZ = os.path.join(HERE, "nbest_120.npz")
SEED = 1                                  # reproduces the recorded tail (18 / 54); see the docstring
TRIALS = int(FACTS["poolgate_clean_trials"])
# StreamDriver's shipped lock_pool_switch. Read from the signature rather than re-typed, so a retune
# there moves this bound with it; stream_driver imports cupy/torch only behind guards.
from inspect import signature                                               # noqa: E402
from glint.stream_driver import StreamDriver                                # noqa: E402
POOL_SWITCH = int(signature(StreamDriver.__init__).parameters["lock_pool_switch"].default)

fails = []


def check(name, cond, msg=""):
    print(f"  {name:72s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def replay(per_frame, order, lyso):
    """Feed frames in `order`; return the first lock as a dict, or None if no order locks."""
    rc = RunningConsensus(min_support=3, gap=2)              # adaptive=True, merge=True: as shipped
    for i in order:
        rc.add_frame(list(per_frame[i]))
        Mc, sup, lead = rc.verdict()
        if Mc is not None:
            return dict(frames=rc.nframes, npool=rc.npool, correct=same_lattice(Mc, lyso),
                        support=sup, runner=lead, M=Mc)
    return None


def main():
    d = np.load(NPZ)
    cells, fid, lyso, nframes = d["cells"], d["fid"], d["lyso"], int(d["nframes"])
    print(f"{os.path.basename(NPZ)}: {cells.shape[0]} hypotheses over {nframes} frames")
    check("file shape: cells (360,3,3), fid sorted (deposition order), nframes 120",
          cells.shape == (360, 3, 3) and bool(np.all(np.diff(fid) >= 0)) and nframes == 120,
          f"{cells.shape} sorted={bool(np.all(np.diff(fid) >= 0))} nframes={nframes}")
    per_frame = [cells[fid == i] for i in range(nframes)]
    per_count = np.array([len(p) for p in per_frame])
    check("every frame deposits exactly 3 hypotheses (so pool at lock == 3 x lock frame)",
          bool(np.all(per_count == 3)), f"min/max per frame {per_count.min()}/{per_count.max()}")

    # The batch reference: consensus_cell over the WHOLE pool under hybrid_stream's gate.
    Mb, sup_b = consensus_cell(list(cells), min_frac=CONSENSUS_MIN_FRAC, min_lead=CONSENSUS_MIN_LEAD)
    check("batch consensus_cell over the pool returns the lyso lattice",
          Mb is not None and same_lattice(Mb, lyso), f"support {sup_b}")

    # --- deposition order ----------------------------------------------------------------------
    dep = replay(per_frame, range(nframes), lyso)
    print(f"  deposition order: {dep and {k: v for k, v in dep.items() if k != 'M'}}")
    check("deposition order locks on the lyso lattice", dep is not None and dep["correct"])
    check("...and on the batch consensus cell", dep is not None and same_lattice(dep["M"], Mb))
    check(f"...no later than FACTS poolgate_clean_lock_max ({FACTS['poolgate_clean_lock_max']})",
          dep is not None and dep["frames"] <= FACTS["poolgate_clean_lock_max"], dep and dep["frames"])
    check(f"...with a pool <= FACTS poolgate_clean_maxpool ({FACTS['poolgate_clean_maxpool']})",
          dep is not None and dep["npool"] <= FACTS["poolgate_clean_maxpool"], dep and dep["npool"])

    # --- 400 random arrival orders at the pinned seed ---------------------------------------------
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    locks = [replay(per_frame, rng.permutation(nframes), lyso) for _ in range(TRIALS)]
    dt = time.time() - t0
    n_lock = sum(1 for r in locks if r is not None)
    fr = np.array([r["frames"] for r in locks if r is not None])
    pools = np.array([r["npool"] for r in locks if r is not None])
    false = sum(1 for r in locks if r is not None and not r["correct"])
    false_vs_batch = sum(1 for r in locks if r is not None and not same_lattice(r["M"], Mb))
    med = float(np.median(fr)) if fr.size else float("nan")
    print(f"  seed {SEED}, {TRIALS} orders in {dt:.1f} s: locked {n_lock}/{TRIALS}, lock frame "
          f"median {med:.0f} p90 {np.percentile(fr, 90):.0f} max {fr.max()}, pool at lock max "
          f"{pools.max()}, false locks {false} (vs batch cell {false_vs_batch})")
    check(f"all {TRIALS} orders lock", n_lock == TRIALS, f"{n_lock}/{TRIALS}")
    check(f"median lock frame == FACTS seqstop_median_lock ({FACTS['seqstop_median_lock']})",
          med == FACTS["seqstop_median_lock"], f"measured {med:.1f}")
    check(f"false locks vs lyso == FACTS seqstop_false_locks ({FACTS['seqstop_false_locks']})",
          false == FACTS["seqstop_false_locks"], f"measured {false}")
    check("false locks vs the batch consensus cell also == FACTS seqstop_false_locks",
          false_vs_batch == FACTS["seqstop_false_locks"], f"measured {false_vs_batch}")
    check(f"worst lock frame == FACTS poolgate_clean_lock_max ({FACTS['poolgate_clean_lock_max']})"
          f" [tail, seed-pinned]",
          fr.size and fr.max() == FACTS["poolgate_clean_lock_max"], f"measured {fr.max()}")
    check(f"largest pool at lock == FACTS poolgate_clean_maxpool ({FACTS['poolgate_clean_maxpool']})"
          f" [tail, seed-pinned]",
          pools.size and pools.max() == FACTS["poolgate_clean_maxpool"], f"measured {pools.max()}")
    check("pool at lock == 3 x lock frame in every order (structural)",
          bool(np.all(pools == 3 * fr)))
    check(f"largest pool at lock < StreamDriver lock_pool_switch ({POOL_SWITCH}) -- the pool-keyed "
          f"gate leaves the clean benchmark untouched",
          pools.size and pools.max() < POOL_SWITCH, f"measured {pools.max()}")
    return 0 if not fails else 1


if __name__ == "__main__":
    rc = main()
    print()
    if fails:
        print(f"FAIL: {len(fails)} check(s): {fails}")
    else:
        print("ALL PASS")
    sys.exit(rc)
