"""Does the FULL arsenal on the retry make the reorder a free win instead of a trade?

glint#76 measured registration-first at 88/120 vs the shipped 91, 4.4x faster. Two reasons that 88
may be an underestimate:

  1. A BUG in that script -- the retry loop broke unconditionally after the FIRST N-best candidate,
     so it was a blind-TOP1 retry, not N-best. Fixed here (break moved inside the test).
  2. It never tried the rest of offline's arsenal. Offline reaches 91 as 60 blind-top1 + 5 blind
     N-best + 26 known-cell rescue; the retry only ever ran blind.

This runs each arm separately on the frames the batched known-cell pass rejects, reports what each
one adds, and computes the ARSENAL CEILING -- for every frame, does ANY arm clear the gate -- which
bounds what any ordering can reach on this data.

The arms themselves now live in `glint.retry_cascade` and are IMPORTED here, because the streaming
driver's opt-in `retry_cascade` (glint#75) runs the same functions: a second implementation over
there would break the correspondence between what ships and the 94/120 measured here. This script
stays the reference for the number; that module is the single implementation of it.

  python offline_retry_arsenal.py [frames.txt]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = os.environ.get("GLINT_WT", "/sdf/home/s/smarches/glint_streamfix_wt")
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import glint.glint_fast as gf
from glint.glint_fast import index_blind_nbest, index_blind_fast
from glint.multishot import same_lattice
from glint.replica_gpu import index_known_gpu_cell
from glint.retry_cascade import arm_blind_top1, arm_blind_nbest, arm_known_perframe
from what_are_the_failures import strict_gate, LYSO
import glint.replica_gpu_batch as rgb

PATH = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
K = 5


def vote(cells):
    best, bn = None, 0
    for a in cells:
        n = sum(1 for b in cells if same_lattice(a, b))
        if n > bn:
            best, bn = a, n
    return best, bn


def main():
    frames = [np.asarray(f, float) for f in gf.load(PATH) if len(f) >= 6]
    N = len(frames)
    print(f"{PATH}: {N} frames, strict gate\n")

    # --- lock the cell from the first K, then one batched known-cell pass over all -------------
    cells = []
    for i in range(K):
        try:
            for c, _s in index_blind_nbest(frames[i], 3):
                if c is not None:
                    cells.append(np.asarray(c, float))
                break
        except Exception:
            pass
    Mc, sup = vote(cells)
    Ms = rgb.index_fused(frames, Mc, B=N)
    ok = [i for i, M in enumerate(Ms)
          if strict_gate(None if M is None else np.asarray(M, float), frames[i], Mc)]
    failed = [i for i in range(N) if i not in set(ok)]
    print(f"lock from K={K} (support {sup}/{K}); batched known-cell pass: "
          f"{len(ok)}/{N} pass, {len(failed)} fail\n")

    # --- each retry arm, run ONLY on the failures ---------------------------------------------
    # The arms are glint.retry_cascade's, bound here to the STRICT research gate (same_lattice +
    # >=25% matched + >=10 reflections). The live driver binds the same functions to its own accept
    # gate instead; that is the only difference between what is measured here and what ships.
    def gate_for(i, truth):
        return lambda M: strict_gate(M, frames[i], truth)

    arms = {}
    t0 = time.time()
    arms["blind top-1"] = {i for i in failed
                           if arm_blind_top1(frames[i], gate_for(i, Mc), index_blind_fast) is not None}
    t_top1 = time.time() - t0

    for nb in (3, 10):
        t0 = time.time()
        arms[f"blind N-best (k={nb})"] = {
            i for i in failed
            if arm_blind_nbest(frames[i], gate_for(i, Mc), index_blind_nbest, nb) is not None}
        if nb == 3:
            t_nb3 = time.time() - t0

    t0 = time.time()
    arms["known-cell, per-frame"] = {
        i for i in failed
        if arm_known_perframe(frames[i], Mc, gate_for(i, Mc), index_known_gpu_cell) is not None}
    t_kc = time.time() - t0

    # exact reference cell rather than the voted one
    arms["known-cell, exact LYSO"] = {
        i for i in failed
        if arm_known_perframe(frames[i], LYSO, gate_for(i, LYSO), index_known_gpu_cell) is not None}

    print(f"{'retry arm (on the ' + str(len(failed)) + ' failures)':34s} {'recovers':>9s} "
          f"{'total yield':>12s}")
    print("-" * 60)
    for name, got in arms.items():
        print(f"{name:34s} {len(got):9d} {len(ok) + len(got):8d}/{N:<3d}")

    legit = [k for k in arms if "exact LYSO" not in k]      # the exact cell is an ORACLE: a blind
    uni_l = set().union(*[arms[k] for k in legit])          # pipeline never has it. Exclude it.
    uni_b = set().union(*[arms[k] for k in legit if k.startswith("blind")])
    uni_a = set().union(*arms.values())
    print("-" * 60)
    print(f"{'UNION, blind arms only':34s} {len(uni_b):9d} {len(ok)+len(uni_b):8d}/{N:<3d}")
    print(f"{'UNION, all AVAILABLE arms':34s} {len(uni_l):9d} {len(ok)+len(uni_l):8d}/{N:<3d}  <== the honest number")
    print(f"{'UNION incl. exact-cell ORACLE':34s} {len(uni_a):9d} {len(ok)+len(uni_a):8d}/{N:<3d}  (not achievable blind)")
    print(f"\nshipped discovery-first offline reaches 91/{N}")
    print(f"blind solves: {K} warm-up + {len(failed)} retried = {K + len(failed)}, vs 120 shipped")
    print(f"retry cost: top-1 {t_top1:.1f}s, N-best(3) {t_nb3:.1f}s, known-cell {t_kc:.2f}s")


if __name__ == "__main__":
    main()
