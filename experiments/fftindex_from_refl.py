"""DEFINITIVE real-data test: index DIALS rlp (q-vectors at Brewster's REFINED
geometry, indexed lysozyme spots) with fftindex. If it recovers 79/79/38, fftindex
works on real data given proper geometry."""
import sys

import numpy as np

sys.path.insert(0, "..")
from fftindex import index_shot
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import (cell_signature, consensus_cell,
                                index_known_pairangle, reference_lattice, same_lattice)

d = np.load("/sdf/home/s/smarches/lyso_rlp.npz")
rlp, ids = d["rlp"], d["id"]
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
print(f"lysozyme signature: {np.round(cell_signature(LYSO),1)}")

frames = np.unique(ids)
used = nidx = nlyso = nknown = 0
cells = []
for fid in frames[:60]:
    q = rlp[ids == fid]
    if len(q) < 10:
        continue
    used += 1
    qmax = float(np.linalg.norm(q, axis=1).max())
    res = index_shot(q, qmax)
    lb = res.M is not None and same_lattice(res.M, LYSO)
    nidx += res.M is not None
    nlyso += lb
    if res.M is not None:
        cells.append(res.M)
    kn = index_known_pairangle(q, qmax, LYSO, Vref=reference_lattice(LYSO, qmax))
    nknown += kn.M is not None and same_lattice(kn.M, LYSO)
    if used <= 8:
        bc = np.round(np.sort(np.linalg.norm(res.M, axis=0)), 1) if res.M is not None else None
        print(f"  frame {fid}: {len(q)} refl  blind={bc}  lyso?={lb}  known-cell={kn.M is not None and same_lattice(kn.M,LYSO)}", flush=True)

u = max(used, 1)
print(f"\nframes used (>=10 refl): {used}")
print(f"fftindex BLIND: indexed {nidx}/{used} ({100*nidx/u:.0f}%), matches lysozyme {nlyso}/{used} ({100*nlyso/u:.0f}%)")
print(f"fftindex KNOWN-CELL: indexed lysozyme {nknown}/{used} ({100*nknown/u:.0f}%)")
if cells:
    Mc, sup = consensus_cell(cells)
    if Mc is not None:
        print(f"dominant cell: {np.round(np.sort(np.linalg.norm(Mc,axis=0)),1)} A ({sup} agree)")
