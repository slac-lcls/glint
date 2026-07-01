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
# pass 1: blind index, collect cells
blind=[]; Ms=[]
for fid,q in frames:
    qmax=float(np.linalg.norm(q,axis=1).max())
    r=index_shot(q,qmax); blind.append(r.M)
    if r.M is not None: Ms.append(r.M)
Mc,sup=consensus_cell(Ms)
print(f"consensus is-lyso={same_lattice(Mc,LYSO)} support {sup}/{len(Ms)}")
# pass 2: decide per frame WITHOUT ground truth -- agree-with-consensus, else constrained
nb=nc=nu=0
for (fid,q),Mb in zip(frames,blind):
    qmax=float(np.linalg.norm(q,axis=1).max())
    blind_agree = Mb is not None and same_lattice(Mb,Mc)          # accept blind if it matches consensus
    con = index_known_pairangle(q,qmax,Mc,Vref=reference_lattice(Mc,qmax))
    con_agree = con.M is not None and same_lattice(con.M,Mc)
    accept = blind_agree or con_agree                            # UNION decision (no truth used)
    # score the accepted cell vs LYSO (ground truth, for reporting only)
    truth = (Mb if blind_agree else con.M)
    good = accept and same_lattice(truth,LYSO)
    nb += Mb is not None and same_lattice(Mb,LYSO)
    nc += con_agree and same_lattice(con.M,LYSO)
    nu += good
print(f"  blind alone        : {nb}/{N} ({100*nb/N:.0f}%)")
print(f"  consensus-constr.  : {nc}/{N} ({100*nc/N:.0f}%)")
print(f"  UNION (blind|constr): {nu}/{N} ({100*nu/N:.0f}%)   [xgandalf 100%]")
