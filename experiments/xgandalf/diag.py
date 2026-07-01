import sys; sys.path.insert(0,"../..")
import numpy as np
from glint import index_shot
from glint.lattice import cell_to_Ar
from glint.multishot import same_lattice
LYSO=cell_to_Ar(79.02,79.02,37.98,90,90,90)
def read_frames(path):
    lines=open(path).read().split("\n"); fr=[]; i=0
    while i<len(lines):
        if not lines[i].startswith("FRAME"): i+=1; continue
        _,fid,npk=lines[i].split(); npk=int(npk)
        q=np.array([[float(x) for x in lines[i+1+j].split()] for j in range(npk)])
        fr.append((int(fid),q)); i+=1+npk
    return fr
frames=read_frames("frames.txt")
none=wrong=ok=0; miss_counts=[]; ok_counts=[]
for fid,q in frames:
    qmax=float(np.linalg.norm(q,axis=1).max())
    r=index_shot(q,qmax)
    good = r.M is not None and same_lattice(r.M,LYSO)
    if good: ok+=1; ok_counts.append(len(q))
    elif r.M is None: none+=1; miss_counts.append(len(q))
    else:
        wrong+=1; miss_counts.append(len(q))
print(f"OK {ok}, NO-CELL {none}, WRONG-CELL {wrong}")
print(f"spots/frame: indexed median {int(np.median(ok_counts))} [{min(ok_counts)}-{max(ok_counts)}]; "
      f"MISSED median {int(np.median(miss_counts))} [{min(miss_counts)}-{max(miss_counts)}]")
# how sparse are the misses?
import numpy as np
allc=sorted([len(q) for _,q in frames])
print("spot-count deciles all frames:", [int(np.percentile(allc,p)) for p in (10,30,50,70,90)])
