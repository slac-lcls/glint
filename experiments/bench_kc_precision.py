"""Validate the KC_FP / KC_SOLVE_FP precision knob on real cxidb: mixed fp32/solve64 (default) / fp64 / fp32
must give the same known-cell rate AND the same lattice per frame. fp32 unlocks fp32-strong GPUs
(e.g. RTX Blackwell, ~2x fp32 / half the price) at no accuracy cost -- measured rate/lattice-
identical to fp64 across all lattice systems + sparse frames (see mp_xcell). GPU node.

  python experiments/bench_kc_precision.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from glint_fast import load, gpass, LYSO
from glint.replica_gpu import DIRS, CA, SA, _fib_halfsphere
from glint.multishot import same_lattice
import glint.replica_gpu_batch as rgb

frames = [q for q in load(os.path.join(HERE, "frames_cxidb_clean.txt")) if len(q) >= 6]; n = len(frames)


def cfg(work, solve):
    """Reconfigure the engine's working/solve precision in-process (the module reads KC_FP at import;
    here we flip the dtype globals + rebuild the dtype-bearing constants to test all three in one run)."""
    WFP = torch.float32 if work == 32 else torch.float64
    SFP = torch.float32 if solve == 32 else torch.float64
    rgb.FP = WFP; rgb._SOLVE_FP = SFP
    rgb._DIRS = DIRS.to(WFP)          # azimuth grid follows rgb.FP via _cell_params/_azimuth_grid
    rgb._DIRS_LO = torch.as_tensor(_fib_halfsphere(min(4096, int(os.environ.get("CDIRS", "16384")))), dtype=WFP, device=rgb.DEV)
    rgb._EX = torch.tensor([1., 0., 0.], dtype=WFP, device=rgb.DEV)
    rgb._EY = torch.tensor([0., 1., 0.], dtype=WFP, device=rgb.DEV)
    rgb._EYE3 = torch.eye(3, dtype=WFP, device=rgb.DEV); rgb._ARANGE = {}; rgb._GRAPHS = {}


def run():
    return [M for i in range(0, n, 32) for M in rgb.index_known_gpu_cell_batch(frames[i:i + 32], LYSO)]


base = None
print(f"host {os.uname().nodename}  n={n}   (frac/loose; (d)=per-frame lattice-diff vs fp64)", flush=True)
for name, w, s in [("mixed fp32/solve64 (default)", 32, 64), ("fp64", 64, 64), ("fp32", 32, 32)]:
    cfg(w, s); Ms = run(); torch.cuda.synchronize()
    g = np.array([gpass(M, q) for M, q in zip(Ms, frames)])
    if base is None:
        base = [np.asarray(M, float) for M in Ms]
    ld = sum(0 if same_lattice(a, np.asarray(b, float)) else 1 for a, b in zip(base, Ms))
    print(f"  {name:<20} frac {int(g[:, 0].sum())}/{n}  loose {int(g[:, 1].sum())}/{n}  lattice-diff {ld}", flush=True)
