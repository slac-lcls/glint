"""How fast can the GPU xgandalf replica get with GLINT's pruning levers (steps 80->8, fewer
starts)? Sweep (n_dir, dl, steps) on 120 cxidb: start count, ms/frame, blind correct-cell rate.
Baseline (paper config 2200 dir / dl=3 -> 70400 starts, 80 steps) = 2480 ms/f, 61%.

  python sweep_xgspeed.py [frames.txt] [N]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)
import numpy as np, torch
import paper_xg_gpu as xg
from paper_xg_gpu import load, LYSO
from glint.multishot import same_lattice

CONFIGS = [  # (n_dir, dl, steps) -- start count is the bottleneck, so sweep it hard
    (800,  6.0, 8),
    (400,  6.0, 8),
    (200,  6.0, 8),
    (100,  6.0, 8),
    (400,  9.0, 8),     # coarser shells too
    (200,  6.0, 15),    # a few more steps to recover rate
]

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [np.asarray(q, float) for q in load(path) if len(q) >= 6]
    if len(sys.argv) > 2:
        frames = frames[:int(sys.argv[2])]
    n = len(frames)
    sync = xg.DEV == "cuda"
    print(f"N={n}  device={xg.DEV}   [baseline 70400 starts / 80 steps = 2480 ms/f, 61%]")
    print(f"{'n_dir':>6}{'dl':>5}{'steps':>6}{'starts':>8}{'ms/f':>8}{'rate':>7}")
    for ndir, dl, steps in CONFIGS:
        xg.STARTS = xg._starts(n_dir=ndir, dl=dl).to(xg.DEV)
        xg.XGSTEPS = steps
        xg.index_blind(frames[0])                                # warmup this config
        if sync: torch.cuda.synchronize()
        t0 = time.time()
        Ms = [xg.index_blind(q) for q in frames]
        if sync: torch.cuda.synchronize()
        dt = time.time() - t0
        c = sum(M is not None and same_lattice(M, LYSO) for M in Ms)
        print(f"{ndir:>6}{dl:>5.0f}{steps:>6}{xg.STARTS.shape[0]:>8}{1e3*dt/n:>8.0f}{100*c//n:>6}%",
              flush=True)
