"""Peak-tampering ensemble = phase-retrieval-style symmetry breaking on the INPUT. The thin
single-shot Ewald slice is rank-deficient, so the orientation is flip/twist ambiguous (the
indexing cousin of twin stagnation in HIO/ER). Instead of trusting one index of the full peak
set, PERTURB the input -- randomly drop a fraction of peaks (a Friedel sign-flip is a no-op since
the objective sum_i cos(2pi q.v) is even in q) -- re-index K times, and consensus over the K cells.
A spurious/degenerate cell is fragile to peak dropout; the true cell is robust -> the ensemble vote
breaks the ambiguity (a per-frame, input-side analog of cross-frame consensus). Compare single-shot
top-1 vs tamper-ensemble, clean and under mild mosaic (where the ambiguity is worst).

  python tamper_ensemble.py [frames.txt] [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np
from glint_fast import index_blind_fast, load, LYSO
from fftindex.multishot import same_lattice, consensus_cell

K = 8           # resamples per frame
DROP = 0.20     # fraction of peaks dropped per resample
CONDS = [(0.0, "clean"), (0.001, "mosaic s=.001")]


def tamper(q, rng):
    keep = rng.random(len(q)) > DROP
    return q[keep] if keep.sum() >= 6 else q


def correct(M):
    return M is not None and same_lattice(M, LYSO)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    n = len(frames)
    index_blind_fast(frames[0])
    print(f"N={n}  K={K} resamples, drop {int(100*DROP)}%/resample  (blind correct-cell)")
    print(f"{'cond':16}{'single top-1':>14}{'tamper-ens':>13}{'ens beats':>11}{'ens loses':>11}")
    for sig, label in CONDS:
        jit = np.random.default_rng(1)
        bro = [q + jit.normal(0, sig, q.shape) if sig else q for q in frames]
        rng = np.random.default_rng(0)
        s_ok = e_ok = beats = loses = 0
        for q in bro:
            single = correct(index_blind_fast(q))
            cells = [M for M in (index_blind_fast(tamper(q, rng)) for _ in range(K)) if M is not None]
            Mc, _ = consensus_cell(cells)
            ens = correct(Mc)
            s_ok += single; e_ok += ens
            beats += (ens and not single); loses += (single and not ens)
        print(f"  {label:14}{100*s_ok//n:>12d}%{100*e_ok//n:>12d}%{beats:>11d}{loses:>11d}", flush=True)
