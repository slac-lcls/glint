"""Where N-best consensus actually pays off: FEW frames (fragile consensus). Vary the frame count
K; over R random K-frame subsets, build consensus (production consensus_cell) over top-1 cells vs
the pooled N-best, and report how often the dominant cluster is the true LYSO cell. Run clean and
under a mild mosaic (lowers per-frame rate so consensus is genuinely marginal). N-best should
recover the cell with fewer frames.

  python nbest_fewframes.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from glint_fast import index_blind_nbest, load, LYSO
from fftindex.multishot import same_lattice, consensus_cell

NB_N, R = 3, 30
KS = (4, 6, 8, 12, 20, 40)
SIGS = [(0.0, "clean"), (0.0008, "mild mosaic s=.0008")]


def consensus_lyso(cells):
    Mc, sup = consensus_cell(cells)
    return bool(Mc is not None and same_lattice(Mc, LYSO))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    index_blind_nbest(frames[0])
    print(f"frames pool={len(frames)}  R={R} subsets/K  N-best={NB_N}  "
          f"(% of K-frame subsets whose consensus = LYSO)")
    for sig, label in SIGS:
        jit = np.random.default_rng(1)
        bro = [q + jit.normal(0, sig, q.shape) if sig else q for q in frames]
        NB = [index_blind_nbest(qq, NB_N) for qq in bro]
        top1 = [nb[0][0] if nb else None for nb in NB]
        per = sum(1 for t in top1 if t is not None and same_lattice(t, LYSO)) * 100 // len(frames)
        print(f"\n  {label}  (per-frame top-1 correct = {per}%)")
        print(f"  {'K':>4}{'top1-consensus':>16}{'N-best-consensus':>18}")
        rng = np.random.default_rng(0)
        for K in KS:
            if K > len(frames):
                break
            o1 = oN = 0
            for _ in range(R):
                idx = rng.choice(len(frames), K, replace=False)
                p1 = [top1[i] for i in idx if top1[i] is not None]
                pN = [c for i in idx for c, _ in NB[i]]
                o1 += consensus_lyso(p1); oN += consensus_lyso(pN)
            print(f"  {K:>4}{100*o1//R:>15d}%{100*oN//R:>17d}%", flush=True)
