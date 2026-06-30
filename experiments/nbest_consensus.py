"""N-best multi-hypothesis consensus. Sparse single-shot indexing is rank-deficient (orientation
ambiguous about the unobserved axis), so the true cell is often a reachable-but-not-top-1
hypothesis discarded as wrong_cell/sel_miss. Keep the top-N cells per frame (score-tagged) and
pool them for consensus: frame-specific aliases scatter (singletons), the true cell recurs
(clusters). Metrics: per-frame top-1-correct vs top-N-contains-LYSO; consensus (dominant 5A
metric cluster) over top-1-only pool vs the N-best pool. Repeated under peak SUBSAMPLING (few-peak
ambiguity -- the regime the idea targets).

  python nbest_consensus.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from collections import Counter
from glint_fast import index_blind_nbest, load, LYSO
from fftindex.multishot import same_lattice

NBEST = 5
LYSOKEY = tuple((np.round(np.sort(np.linalg.norm(LYSO, axis=0)) / 5) * 5).tolist())


def dom_cluster(cells):
    if not cells:
        return None, 0
    keys = [tuple((np.round(np.sort(np.linalg.norm(c, axis=0)) / 5) * 5).tolist()) for c in cells]
    k, s = Counter(keys).most_common(1)[0]
    return k, s


def subsample(q, k, rng):
    return q if len(q) <= k else q[rng.choice(len(q), k, replace=False)]


def run(frames, label):
    n = len(frames)
    top1 = topN = 0
    pool1, poolN = [], []
    for q in frames:
        nb = index_blind_nbest(q, NBEST)
        if not nb:
            continue
        top1 += bool(same_lattice(nb[0][0], LYSO))               # best hypothesis correct
        topN += any(same_lattice(c, LYSO) for c, _ in nb)        # LYSO anywhere in top-N
        pool1.append(nb[0][0])
        poolN.extend(c for c, _ in nb)
    k1, s1 = dom_cluster(pool1); kN, sN = dom_cluster(poolN)
    ok1 = "LYSO" if k1 == LYSOKEY else f"WRONG{k1}"
    okN = "LYSO" if kN == LYSOKEY else f"WRONG{kN}"
    print(f"  {label:16} top1={100*top1//n:3d}%  topN_has_LYSO={100*topN//n:3d}%  "
          f"| consensus: top1-pool={ok1}(sup{s1})  N-best-pool={okN}(sup{sN})", flush=True)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    index_blind_nbest(frames[0])                                 # warmup
    print(f"N={len(frames)}  N-best={NBEST}  (LYSO metric key={LYSOKEY})")
    run(frames, "full peaks")
    for k in (15, 10, 8):
        rng = np.random.default_rng(0)
        run([subsample(q, k, rng) for q in frames], f"subsample k={k}")
