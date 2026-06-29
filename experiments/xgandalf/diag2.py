import sys; sys.path.insert(0,"../..")
import numpy as np
from fftindex import index_shot
from fftindex.lattice import cell_to_Ar
from fftindex.multishot import same_lattice
LYSO=cell_to_Ar(79.02,79.02,37.98,90,90,90)
Vlyso=abs(np.linalg.det(LYSO))
def read_frames(path):
    lines=open(path).read().split("\n"); fr=[]; i=0
    while i<len(lines):
        if not lines[i].startswith("FRAME"): i+=1; continue
        _,fid,npk=lines[i].split(); npk=int(npk)
        q=np.array([[float(x) for x in lines[i+1+j].split()] for j in range(npk)])
        fr.append((int(fid),q)); i+=1+npk
    return fr
print(f"lysozyme axes [38 79 79], V={Vlyso:.0f} A^3")
for fid,q in read_frames("frames.txt"):
    qmax=float(np.linalg.norm(q,axis=1).max())
    r=index_shot(q,qmax)
    if r.M is None or same_lattice(r.M,LYSO): continue
    ax=np.sort(np.linalg.norm(r.M,axis=0)); V=abs(np.linalg.det(r.M))
    print(f"  WRONG f{fid:4d} spots={len(q):3d}: axes={np.round(ax,1)} V={V:5.0f} ({V/Vlyso:.2f}x lyso) inliers={r.n_indexed}/{len(q)}")
