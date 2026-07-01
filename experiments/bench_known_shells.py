"""User idea: known-cell mode = GLINT M1-M6 seeded at the KNOWN |a|,|b|,|c| shells (Fibonacci dirs x
the 2-3 known lengths) instead of the blind 32-shell grid. ~10x fewer seeds (faster M3) + GLINT's robust
refine + coverage-gated scorer. Benchmark rate + ms vs the current ffbidx-style rescue (55%) and blind."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import fftindex.glint_fast as gf
from fftindex.glint_index import sample
from fftindex.glint_fast import index_blind_fast
from fftindex.replica_gpu import index_known_gpu_cell
from fftindex.multishot import same_lattice
DEV = gf.DEV
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)
LYSO = gf.LYSO

L = np.linalg.norm(np.asarray(LYSO, float), axis=0)          # real-space axis lengths (columns)
uL = sorted(set(np.round(L, 1)))                             # unique known shells
print(f"LYSO axis lengths {np.round(L,1)} -> unique shells {uL}")
known_starts = sample(n_dir=2200, lengths=uL).to(DEV)
# also a robust variant: known lengths +/- 2 A (tolerate small cell error)
tolL = sorted(set(np.round(np.concatenate([[l-2, l, l+2] for l in uL]), 1)))
known_starts_tol = sample(n_dir=2200, lengths=tolL).to(DEV)
print(f"known_starts {tuple(known_starts.shape)} | +/-2A {tuple(known_starts_tol.shape)} | blind {tuple(gf.STARTS.shape)}")


def bench(fn, name):
    fn(frames[0]); torch.cuda.synchronize()
    t0 = time.perf_counter(); sl = 0
    for q in frames:
        M = fn(q); sl += (M is not None and same_lattice(M, LYSO))
    torch.cuda.synchronize()
    print(f"  {name:28s} {sl}/{n} = {100*sl//n:3d}%   {1e3*(time.perf_counter()-t0)/n:5.1f} ms", flush=True)


print("=== known-cell modes (rate = same_lattice with LYSO) ===")
bench(lambda q: index_blind_fast(q, starts=known_starts), "GLINT known-shell M1")
bench(lambda q: index_blind_fast(q, starts=known_starts_tol), "GLINT known-shell +/-2A")
bench(lambda q: index_known_gpu_cell(q, LYSO), "current rescue (ffbidx-style)")
bench(lambda q: index_blind_fast(q), "blind (no cell, reference)")
