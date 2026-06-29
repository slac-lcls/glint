"""D1: does cross-frame consensus+rescue already clean up the per-frame wrong_cell leak?
The idiosyncratic wrong cells are per-frame and inconsistent, so the consensus cell is the
true LYSO (huge support) and the wrong-cell frames should be RESCUED by re-indexing with the
consensus/known cell. Measure: per-frame blind solved/wrong, then final solved after rescue,
and specifically how many wrong_cell frames become solved.

  python d1_consensus.py [N]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_fast import index_blind_fast, load, matched, LYSO
from replica_gpu import index_known_gpu as rescue
from fftindex.multishot import same_lattice, consensus_cell


def solved(M, q):
    if M is None or not same_lattice(M, LYSO):
        return False
    m = matched(M, q)
    return m / len(q) >= 0.25 and m >= 10


def confident(M, q):                                    # indexed >=gate but maybe wrong lattice
    if M is None:
        return False
    m = matched(M, q)
    return m / len(q) >= 0.25 and m >= 10


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6][:N]
    index_blind_fast(frames[0]); rescue(frames[0])      # warmup
    blind = [index_blind_fast(q) for q in frames]
    Mc, sup = consensus_cell([M for M in blind if M is not None])
    b_solved = sum(solved(M, q) for M, q in zip(blind, frames))
    b_wrong = sum(confident(M, q) and not solved(M, q) for M, q in zip(blind, frames))
    # final: keep if already solved, else rescue with known cell
    fin_solved = 0; wrong_rescued = 0; wrong_total = 0
    for M, q in zip(blind, frames):
        was_wrong = confident(M, q) and not solved(M, q)
        wrong_total += was_wrong
        F = M if solved(M, q) else rescue(q)
        fs = solved(F, q)
        fin_solved += fs
        wrong_rescued += was_wrong and fs
    n = len(frames)
    print(f"D1 consensus+rescue on {n} cxidb frames  (consensus support {sup})")
    print(f"  per-frame blind : solved {b_solved}/{n} ({100*b_solved//n}%), wrong_cell {b_wrong}")
    print(f"  after rescue    : solved {fin_solved}/{n} ({100*fin_solved//n}%)")
    print(f"  wrong_cell frames rescued to correct: {wrong_rescued}/{wrong_total}")
