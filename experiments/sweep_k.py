"""Sweep the M3 ascent step count K (hand schedule, NO learning) to find the minimum K
that holds blind accuracy -- the free throughput win the unrolling POC surfaced (the
80-step front-end over-iterates)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
sys.path.insert(0, "/sdf/home/s/smarches/git/glint/experiments")
from train_unroll import make_starts, hand_schedule, eval_rate, NDIR_EVAL
from glint_fast import load
from glint_index import DEV

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "frames_cxidb_clean.txt"
    frames = [q for q in load(path) if len(q) >= 6]
    Se = make_starts(NDIR_EVAL)
    z = torch.tensor(0.0, device=DEV)
    print(f"K-sweep (hand schedule) on {len(frames)} frames:")
    for K in (6, 8, 10, 12, 16, 24, 40, 80):
        th = torch.tensor(hand_schedule(K), dtype=torch.float32, device=DEV)
        eval_rate(frames, Se, th, z, K, f"K={K:>2}")
