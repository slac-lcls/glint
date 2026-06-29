import sys; sys.path.insert(0,"../..")
import numpy as np
from fftindex import index_shot
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import (consensus_cell, same_lattice,
                                index_known_pairangle, index_known_gd, reference_lattice)
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
blind=[]; Ms=[]
for fid,q in frames:
    qmax=float(np.linalg.norm(q,axis=1).max())
    r=index_shot(q,qmax); blind.append(r.M)
    if r.M is not None: Ms.append(r.M)
Mc,sup=consensus_cell(Ms)
print(f"consensus is-lyso={same_lattice(Mc,LYSO)} support {sup}/{len(Ms)}")
Vref=reference_lattice(Mc,0.5)
nb=npa=ngd=nu_pa=nu_gd=0
for (fid,q),Mb in zip(frames,blind):
    qmax=float(np.linalg.norm(q,axis=1).max())
    blind_ok = Mb is not None and same_lattice(Mb,Mc)
    pa=index_known_pairangle(q,qmax,Mc,Vref=reference_lattice(Mc,qmax))
    gd=index_known_gd(q,qmax,Mc,Vref=reference_lattice(Mc,qmax))
    pa_ok = pa.M is not None and same_lattice(pa.M,Mc)
    gd_ok = gd.M is not None and same_lattice(gd.M,Mc)
    nb += Mb is not None and same_lattice(Mb,LYSO)
    npa += pa_ok and same_lattice(pa.M,LYSO)
    ngd += gd_ok and same_lattice(gd.M,LYSO)
    nu_pa += (blind_ok and same_lattice(Mb,LYSO)) or (pa_ok and same_lattice(pa.M,LYSO))
    nu_gd += (blind_ok and same_lattice(Mb,LYSO)) or (gd_ok and same_lattice(gd.M,LYSO))
print(f"  blind alone                 : {nb}/{N} ({100*nb/N:.0f}%)")
print(f"  constrained pairangle alone : {npa}/{N} ({100*npa/N:.0f}%)")
print(f"  constrained GD alone        : {ngd}/{N} ({100*ngd/N:.0f}%)")
print(f"  UNION blind|pairangle       : {nu_pa}/{N} ({100*nu_pa/N:.0f}%)")
print(f"  UNION blind|GD              : {nu_gd}/{N} ({100*nu_gd/N:.0f}%)   [xgandalf 100%]")
