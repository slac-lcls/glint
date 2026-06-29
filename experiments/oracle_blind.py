"""② diagnostic: split GLINT-blind failures into GENERATION vs SELECTION.
For each frame, anneal ALL valid triplets (as index_blind does) and ask:
  - REACHABLE: does ANY annealed triplet give a cell that is same_lattice(LYSO) AND
    indexes >=25% of spots?  (the true cell is in reach of the candidate set)
  - ORACLE-BEST: the reachable cell with the MOST matched reflections (what a perfect
    selector would pick).
  - SELECTED: what index_blind_fast's coverage-gated-defect scorer actually returns.
Tally: solved | SELECTION-miss (reachable but scorer missed it) | GENERATION-miss
(not reachable). The SELECTION bucket is exactly what ② (better assembly/reduction +
selection criterion) can recover; GENERATION needs richer candidates.

  python oracle_blind.py [frames.txt] [N]
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_index import (objective, refine_vec, distinct_maxima, invq_weight, buerger_reduce,
                         primitivize, STARTS, DEV)
from glint_fast import anneal_batch_t, index_blind_fast, load, matched
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice

LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)


def candidates(q):
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32, device=DEV)
    qmax = float(Q.norm(dim=1).max()); w = invq_weight(Q)
    T = refine_vec(STARTS.clone(), Q, w, qmax)
    f, _ = objective(T, Q, w, sharp=True)
    return distinct_maxima(T.cpu().numpy(), f.cpu().numpy())[:30], Q


def all_annealed(q):
    """anneal every valid triplet; return list of (M_reduced, matched_count, frac)."""
    cands, Q = candidates(q)
    if len(cands) < 3:
        return []
    nrm = np.linalg.norm(cands, axis=1)
    tris = np.array(list(itertools.combinations(range(len(cands)), 3)))
    M0 = np.transpose(cands[tris], (0, 2, 1))
    sc = nrm[tris].prod(1); det = np.abs(np.linalg.det(M0))
    M0 = M0[(sc > 0) & (det >= 0.1 * sc)]
    if len(M0) == 0:
        return []
    Qd = Q.double()
    Mt = anneal_batch_t(torch.as_tensor(M0, dtype=torch.float64, device=DEV), Qd).cpu().numpy()
    out = []
    for M in Mt:
        Mr = buerger_reduce(M)
        m = matched(Mr, q)
        out.append((Mr, m, m / len(q)))
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]; n = len(frames)
    index_blind_fast(frames[0])
    solved = sel_miss = gen_miss = 0
    oracle_reach = 0
    t0 = time.time()
    for q in frames:
        cells = all_annealed(q)
        reach = [(M, m) for (M, m, fr) in cells if fr >= 0.25 and same_lattice(M, LYSO)]
        is_reach = len(reach) > 0
        oracle_reach += is_reach
        M_sel = index_blind_fast(q)
        ok_sel = M_sel is not None and same_lattice(M_sel, LYSO) and matched(M_sel, q) / len(q) >= 0.25
        if ok_sel:
            solved += 1
        elif is_reach:
            sel_miss += 1
        else:
            gen_miss += 1
    dt = time.time() - t0
    print(f"=== ② oracle diagnostic  device={DEV}  N={n}  {1e3*dt/n:.0f} ms/frame ===")
    print(f"  SOLVED (selected ok)      : {solved}/{n} ({100*solved//n}%)")
    print(f"  SELECTION-miss (reachable): {sel_miss}/{n} ({100*sel_miss//n}%)   <- ② headroom")
    print(f"  GENERATION-miss (no reach): {gen_miss}/{n} ({100*gen_miss//n}%)   <- needs richer candidates")
    print(f"  reachable ceiling          : {oracle_reach}/{n} ({100*oracle_reach//n}%)  (solved+selection-miss)")
