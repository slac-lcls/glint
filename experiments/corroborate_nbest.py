"""Corroborate the N-best consensus result: is the +6% a REAL gate-passing recovery or a
metric-only coincidence? Sweep N and report, per frame:
  top1_gated     : top-1 cell is LYSO AND passes the gate (>=25% spots / >=10 refl)  [= current]
  topN_metric    : some top-N cell is LYSO by metric (same_lattice)                   [loose]
  topN_gated     : some top-N cell is LYSO AND gate-passes                            [honest ceiling]
  op_gated       : build consensus over the N-best POOL (production consensus_cell),
                   then per frame pick the consensus-consistent N-best cell that gate-passes [operational]
Also: is the pooled consensus cell LYSO, and its support. If op_gated > top1_gated, grows with N
then saturates, and consensus stays LYSO, the win is corroborated.

  python corroborate_nbest.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from glint_fast import index_blind_nbest, load, matched, LYSO
from fftindex.multishot import same_lattice, consensus_cell

GATE, MININL = 0.25, 10


def gated(M, q):
    m = matched(M, q)
    return m / len(q) >= GATE and m >= MININL


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    n = len(frames)
    index_blind_nbest(frames[0])                                 # warmup
    NB = [index_blind_nbest(q, 10) for q in frames]              # top-10 once, slice below
    print(f"N={n}  corroborating N-best (gate = frac>=.25 & >=10 refl)")
    print(f"{'topN':>5}{'top1_gated':>12}{'topN_metric':>13}{'topN_gated':>12}{'op_gated':>10}"
          f"{'consensus':>16}")
    for K in (1, 3, 5, 8, 10):
        nbK = [nb[:K] for nb in NB]
        t1 = tm = tg = 0
        pool = []
        for q, nb in zip(frames, nbK):
            if not nb:
                continue
            t1 += bool(same_lattice(nb[0][0], LYSO) and gated(nb[0][0], q))
            tm += any(same_lattice(c, LYSO) for c, _ in nb)
            tg += any(same_lattice(c, LYSO) and gated(c, q) for c, _ in nb)
            pool.extend(c for c, _ in nb)
        Mc, sup = consensus_cell(pool)
        mc_lyso = Mc is not None and same_lattice(Mc, LYSO)
        op = 0
        for q, nb in zip(frames, nbK):
            if Mc is None:
                continue
            if any(same_lattice(c, Mc) and gated(c, q) for c, _ in nb):
                op += 1
        print(f"{K:>5}{100*t1//n:>11d}%{100*tm//n:>12d}%{100*tg//n:>11d}%{100*op//n:>9d}%"
              f"   {'LYSO' if mc_lyso else 'WRONG':>5} sup{sup}", flush=True)
