"""User case: MULTI-SHOT of TWO crystals of the SAME protein per shot (multi-lattice AND
multi-shot). Per shot: deflate-and-reindex recovers both LYSO orientations; across shots:
consensus over ALL recovered cells confirms the ONE cell with large support. Combines D3/D4
deflate + multi-shot consensus.

  python d34_multishot.py [N_shots]
"""
import sys
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
import numpy as np
from d34_deflate import two_crystal, deflate, recovered           # sets QDIST=1
from d5_multicell import consensus_cells, params
from glint_fast import LYSO
from fftindex.multishot import same_lattice

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    LREF = np.asarray(LYSO, float)
    rng = np.random.default_rng(1)
    deflate(two_crystal(rng)[0])                                   # warmup
    allM = []; both = 0; nlat = []
    for _ in range(N):
        q, lab = two_crystal(rng, frac2=1.0)                      # 2 LYSO crystals + spurious
        g1, g2 = q[lab == 1], q[lab == 2]
        cells = deflate(q)
        allM += list(cells)
        nlat.append(len(cells))
        both += (recovered(cells, g1) > 0.5 and recovered(cells, g2) > 0.5)
    clusters = consensus_cells(allM, min_support=3)
    print(f"multi-shot, 2 crystals of SAME protein per shot  N={N} shots")
    print(f"  per-shot BOTH lattices recovered: {100*both//N}%  (median {int(np.median(nlat))} lattices/shot)")
    print(f"  total lattices indexed across shots: {len(allM)}")
    print(f"  consensus clusters (support>=3): {len(clusters)}")
    for M, s in clusters:
        print(f"    cell {params(M)}  support {s:3d}  -> {'LYSO' if same_lattice(M, LREF) else '??'}")
