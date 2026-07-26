import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("CDIRS","16384")
import numpy as np, torch
sys.path.insert(0,"/sdf/home/s/smarches/git/glint"); sys.path.insert(0,"/sdf/home/s/smarches/git/glint/experiments")
from glint_fast import load, gpass, LYSO
from glint.replica_gpu import index_known_gpu_cell
from kc_batch import index_batch

frames=[q for q in load("frames_cxidb_clean.txt") if len(q)>=6]; n=len(frames)

# --- VALIDATE on first 24: per-frame vs batched ---
sub=frames[:24]
ref=[index_known_gpu_cell(q,LYSO) for q in sub]
bat=index_batch(sub,LYSO)
gr=np.array([gpass(M,q) for M,q in zip(ref,sub)]); gb=np.array([gpass(M,q) for M,q in zip(bat,sub)])
agree=sum(1 for a,b in zip(ref,bat) if (a is not None and b is not None and __import__('glint.multishot',fromlist=['same_lattice']).same_lattice(a,b)) or (a is None and b is None))
print(f"VALIDATION (24 frames):")
print(f"  per-frame gated: frac {gr[:,0].sum()}/24  loose {gr[:,1].sum()}/24")
print(f"  batched   gated: frac {gb[:,0].sum()}/24  loose {gb[:,1].sum()}/24   (cell-agree {agree}/24)")

# --- FULL rate + timing at several batch sizes ---
print(f"\nTIMING on {n} frames:")
for B in (8,16,32,60,120):
    index_batch(frames[:B], LYSO)                      # warmup this shape
    torch.cuda.synchronize(); t0=time.time(); Ms=[]
    for i in range(0,n,B): Ms += index_batch(frames[i:i+B], LYSO)
    torch.cuda.synchronize(); dt=1e3*(time.time()-t0)/n
    g=np.array([gpass(M,q) for M,q in zip(Ms,frames)])
    print(f"  B={B:3d}: {dt:5.2f} ms/frame ({1000/dt:5.0f} f/s)  frac {g[:,0].sum()}/{n} ({100*g[:,0].sum()//n}%)  loose {g[:,1].sum()}/{n} ({100*g[:,1].sum()//n}%)")
print(f"\n  reference: per-frame 17.3 ms | ffbidx 4.4 ms")
