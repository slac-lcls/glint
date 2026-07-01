import sys; sys.path.insert(0,"../..")
import numpy as np
from glint import index_shot
from glint.lattice import cell_to_Ar
from glint.multishot import (consensus_cell, cell_signature, same_lattice,
                                index_known_pairangle, reference_lattice)
LYSO=cell_to_Ar(79.02,79.02,37.98,90,90,90)
def read_frames(path):
    lines=open(path).read().split("\n"); fr=[]; i=0
    while i<len(lines):
        if not lines[i].startswith("FRAME"): i+=1; continue
        _,fid,npk=lines[i].split(); npk=int(npk)
        q=np.array([[float(x) for x in lines[i+1+j].split()] for j in range(npk)])
        fr.append((int(fid),q)); i+=1+npk
    return fr
frames=read_frames("frames.txt"); N=len(frames)

# Step 1: blind-index all frames, collect recovered cells (correct AND wrong)
Ms=[]; blind_ok=[]
for fid,q in frames:
    qmax=float(np.linalg.norm(q,axis=1).max())
    r=index_shot(q,qmax)
    blind_ok.append(r.M is not None and same_lattice(r.M,LYSO))
    if r.M is not None: Ms.append(r.M)
print(f"Step1 blind: {sum(blind_ok)}/{N} correct; {len(Ms)} cells recovered (incl. wrong)")

# Step 2: consensus across all recovered cells
Mc,sup=consensus_cell(Ms)
if Mc is None:
    print("consensus: NONE"); sys.exit()
print(f"Step2 consensus: axes {np.round(np.sort(np.linalg.norm(Mc,axis=0)),1)} "
      f"sig {np.round(cell_signature(Mc),1)} support {sup}/{len(Ms)}  is-lyso? {same_lattice(Mc,LYSO)}")

# Step 3: re-index every frame CONSTRAINED to the consensus cell
Vref=reference_lattice(Mc,0.5)
con_ok=0; recovered=0
for (fid,q),b in zip(frames,blind_ok):
    qmax=float(np.linalg.norm(q,axis=1).max())
    kn=index_known_pairangle(q,qmax,Mc,Vref=reference_lattice(Mc,qmax))
    good=kn.M is not None and same_lattice(kn.M,LYSO)
    con_ok+=good
    if good and not b: recovered+=1
print(f"Step3 consensus-constrained re-index: {con_ok}/{N} ({100*con_ok/N:.0f}%); "
      f"recovered {recovered} previously-blind-missed frames")
print(f"\n  blind {sum(blind_ok)}/{N} ({100*sum(blind_ok)/N:.0f}%) -> consensus-constrained "
      f"{con_ok}/{N} ({100*con_ok/N:.0f}%)   [xgandalf 100%]")
