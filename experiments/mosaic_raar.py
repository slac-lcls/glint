"""RAAR vs GD under MOSAIC broadening (steepest cliff, GENERATION-limited). RAAR helps SELECTION, so the
hypothesis is ~neutral here -- bounds where RAAR helps. All 120 frames perturbed (paired same noise);
blind rate + hybrid n_idx, GD-8 vs RAAR-16, over sigma."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf, glint.glint_index as gi
from glint.glint_fast import index_blind_fast
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)


def blind_and_hybrid(pert, refiner, steps):
    gf.REFINER = refiner; gf.STEPS = steps; gi.BETA = 0.7
    index_blind_fast(pert[0]); torch.cuda.synchronize()
    bl = sum(1 for q in pert if (lambda M: M is not None and same_lattice(M, gf.LYSO))(index_blind_fast(q)))
    _, st = hybrid_index(pert, nbest=3, warmup=False)
    return bl, st["n_idx"], st["support"]


print("sigma     blind GD/RAAR      hybrid GD/RAAR    support GD/RAAR")
for sig in [0.0, 0.0005, 0.001, 0.0015]:
    rng = np.random.default_rng(1)
    pert = [q + rng.normal(0, sig, q.shape) for q in frames]        # paired: same noise both refiners
    gb, gh, gs = blind_and_hybrid(pert, "grad", 8)
    rb, rh, rs = blind_and_hybrid(pert, "raar", 16)
    print(f"{sig:.4f}    {gb:3d} / {rb:3d}         {gh:3d} / {rh:3d}          {gs:3d} / {rs:3d}", flush=True)
