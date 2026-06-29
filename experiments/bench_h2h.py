"""③ head-to-head throughput+accuracy on the A100. Times the full GLINT pipeline and
reports the fair gated metric, so we can place GLINT on the speed/accuracy Pareto vs
xgandalf (~13 s/frame blind, sparse) and ffbidx (fast but needs a cell).

Stages timed:
  1. GLINT blind FAST   (GPU front-end + GPU-batched M4) -> throughput + gated rate
  2. consensus_cell      (derive the cell from the blind minority; no cell assumed)
  3. replica rescue      (faithful ffbidx replica on the blind FAILURES only)
  => ① hybrid = keep blind successes + rescued failures, all gated.

  python bench_h2h.py [frames.txt] [N]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CDIRS", "16384")
import numpy as np, torch
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex")
sys.path.insert(0, "/sdf/home/s/smarches/git/fftindex/experiments")
from glint_fast import index_blind_fast, load, gpass, matched, LYSO
if os.environ.get("RESCUE", "numpy") == "gpu":
    from replica_gpu import index_known_gpu as replica_rescue
else:
    from replica_v2 import index_known as replica_rescue
from fftindex.multishot import consensus_cell, same_lattice
from fftindex.lattice import cell_to_Ar

DEV = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
    frames = frames[:N]; n = len(frames)
    index_blind_fast(frames[0])                                  # warmup

    # --- Stage 1: blind FAST ---
    if DEV == "cuda": torch.cuda.synchronize()
    t0 = time.time(); blind = [index_blind_fast(q) for q in frames]
    if DEV == "cuda": torch.cuda.synchronize()
    t_blind = time.time() - t0
    g = np.array([gpass(M, q) for M, q in zip(blind, frames)])
    print(f"=== device={DEV}  N={n} ===")
    print(f"[1] GLINT blind FAST : {1e3*t_blind/n:.1f} ms/frame ({n/t_blind:.1f} f/s)  "
          f"gated {g[:,0].sum()}/{n} ({100*g[:,0].sum()//n}%) frac, {g[:,1].sum()}/{n} ({100*g[:,1].sum()//n}%) >=10refl")

    # --- Stage 2: consensus cell (no cell assumed) ---
    t1 = time.time(); Mc, support = consensus_cell([M for M in blind if M is not None])
    t_cons = time.time() - t1
    print(f"[2] consensus cell   : {np.round(np.sort(np.linalg.norm(Mc,axis=0)),1)}  support {support}  "
          f"lyso?={same_lattice(Mc, LYSO)}  ({1e3*t_cons:.0f} ms total)")

    # --- Stage 3: rescue blind FAILURES with the faithful replica ---
    hyb = np.zeros(2, int); n_resc = 0; t2 = time.time()
    for M, q in zip(blind, frames):
        if M is not None and same_lattice(M, Mc):
            hyb += gpass(M, q)
        else:
            n_resc += 1
            hyb += gpass(replica_rescue(q), q)
    t_resc = time.time() - t2
    print(f"[3] replica rescue   : {n_resc} failures rescued, {1e3*t_resc/max(n_resc,1):.0f} ms/rescue "
          f"({t_resc:.1f}s total)")
    print(f"=> ① HYBRID (blind+rescue) gated: frac>=.25 {hyb[0]}/{n} ({100*hyb[0]//n}%)   "
          f">=10refl {hyb[1]}/{n} ({100*hyb[1]//n}%)")
    tot = t_blind + t_cons + t_resc
    print(f"=> END-TO-END: {tot:.1f}s for {n} frames = {1e3*tot/n:.0f} ms/frame ({n/tot:.1f} f/s)")
    print(f"   vs xgandalf-blind ~13000 ms/frame (0.077 f/s) on the same sparse data "
          f"-> ~{13000/(1e3*tot/n):.0f}x faster, comparable gated rate")
