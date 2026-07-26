import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("CDIRS","16384")
import numpy as np, torch
sys.path.insert(0,"/sdf/home/s/smarches/git/glint"); sys.path.insert(0,"/sdf/home/s/smarches/git/glint/experiments")
from glint_fast import load, gpass, LYSO
from glint.hybrid_stream import index_known_fast
frames=[q for q in load("frames_cxidb_clean.txt") if len(q)>=6]; n=len(frames)
index_known_fast(frames[:8], LYSO)                       # warmup
torch.cuda.synchronize(); t0=time.time()
results, stats = index_known_fast(frames, LYSO, batch=32)
torch.cuda.synchronize(); dt=1e3*(time.time()-t0)/n
g=np.array([gpass(r["M"],q) for r,q in zip(results,frames)])
print(f"index_known_fast: {dt:.2f} ms/frame ({1000/dt:.0f} f/s)  n_idx(stats)={stats['n_idx']}")
print(f"  gated: frac {g[:,0].sum()}/{n} ({100*g[:,0].sum()//n}%)  loose {g[:,1].sum()}/{n} ({100*g[:,1].sum()//n}%)")
print(f"  result dicts have keys {sorted(results[0].keys())}, edges={stats['edges']}")
print(f"  EXPECT ~2.4 ms, 73/120 frac, 113/120 loose (== validated prototype)")
