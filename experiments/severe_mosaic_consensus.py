"""Severe-mosaic consensus recovery: does cross-frame consensus recover the true cell AFTER the
single-frame rate has collapsed? D2 says severe mosaic (sigma>=.0015) destroys the long-axis phase
(delta(q.v) ~ sigma|v| on the 79A axis) -> GENERATION fails, no single frame solves. The open
claim "only consensus helps there" is quantified here.

For each broadening sigma we report, on the SAME 120 cxidb pool:
  * per-frame top-1 correct rate  (the single-frame floor)
  * N-best REACH = % frames whose top-N candidate set contains LYSO at all (the leading indicator:
    consensus can only recover what at least a few frames independently surface)
  * consensus(K) over R random K-frame subsets, two readings that the multi-hypothesis section
    distinguishes:
      - PICK   : the dominant (largest) support>=3 cluster IS LYSO  (what production consensus_cell returns)
      - PRESENT: ANY support>=3 cluster is LYSO                     (recoverable, even if not dominant)
  plus the median support of the LYSO cluster. The gap PRESENT-PICK = how often the truth is there
  but out-voted by a spurious near-degenerate cell (where a cell prior / known-cell rescue would close it).

  python severe_mosaic_consensus.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from glint_fast import index_blind_nbest, load, LYSO
from glint.multishot import same_lattice

NB_N, R = 3, 40
KS = (4, 6, 8, 12, 20, 40, 80, 120)
SIGS = [(0.0008, "mild  s=.0008"), (0.0015, "severe s=.0015"),
        (0.0020, "severe s=.0020"), (0.0030, "severe s=.0030")]


def clusters(cells, min_support=3):
    """All same-lattice clusters, sorted by support desc. Mirrors consensus_cell's grouping
    but keeps every cluster so we can ask PRESENT (any) vs PICK (dominant)."""
    groups = []  # [rep, count]
    for M in (M for M in cells if M is not None):
        for g in groups:
            if same_lattice(M, g[0]):
                g[1] += 1
                break
        else:
            groups.append([M, 1])
    groups.sort(key=lambda g: g[1], reverse=True)
    return [(rep, c) for rep, c in groups if c >= min_support]


def lyso_pick_present_support(cells):
    cl = clusters(cells)
    if not cl:
        return False, False, 0
    pick = same_lattice(cl[0][0], LYSO)
    present = next((c for rep, c in cl if same_lattice(rep, LYSO)), 0)
    return pick, present > 0, present


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    index_blind_nbest(frames[0])  # warmup
    print(f"frames pool={len(frames)}  R={R} subsets/K  N-best={NB_N}  min_support=3")
    print("PICK = dominant cluster is LYSO ; PRESENT = any support>=3 cluster is LYSO\n")
    for sig, label in SIGS:
        jit = np.random.default_rng(1)
        bro = [q + jit.normal(0, sig, q.shape) for q in frames]
        NB = [index_blind_nbest(qq, NB_N) for qq in bro]
        top1 = [nb[0][0] if nb else None for nb in NB]
        per = sum(1 for t in top1 if t is not None and same_lattice(t, LYSO)) * 100 // len(frames)
        reach = sum(1 for nb in NB if any(same_lattice(c, LYSO) for c, _ in nb)) * 100 // len(frames)
        print(f"{label}:  per-frame top-1={per}%   N-best reach={reach}%")
        print(f"  {'K':>4}{'PICK':>8}{'PRESENT':>10}{'med-support':>13}")
        rng = np.random.default_rng(0)
        for K in KS:
            if K > len(frames):
                break
            npick = npres = 0
            sups = []
            for _ in range(R):
                idx = rng.choice(len(frames), K, replace=False)
                pool = [c for i in idx for c, _ in NB[i]]
                pk, pr, sp = lyso_pick_present_support(pool)
                npick += pk; npres += pr
                if sp:
                    sups.append(sp)
            ms = int(np.median(sups)) if sups else 0
            print(f"  {K:>4}{100*npick//R:>7d}%{100*npres//R:>9d}%{ms:>13d}", flush=True)
        print()
