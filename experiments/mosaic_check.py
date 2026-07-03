"""Does the CG rescue refiner (RESCUE_CG) pay off MORE at the basin edge? Stress the 120 real cxidb frames
with mosaic angular jitter (rotate each rlp by a small random angle -> seed dirs land further from truth ->
rescue works nearer its basin edge, where synthetic basin_char2 says CG's wider basin helps). Compare
n_idx for the current RESCUE_CG setting. Run twice (RESCUE_CG unset vs =1) to see the gap vs mosaic sigma."""
import os, sys
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf
from glint.hybrid_stream import hybrid_index

RESCUE_CG = os.environ.get("RESCUE_CG", "0")
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)


def rot(ax, th):
    ax = ax / (np.linalg.norm(ax) + 1e-12); c, s = np.cos(th), np.sin(th); x, y, z = ax
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def jitter(frame, sig_deg, rng):
    if sig_deg <= 0:
        return frame
    return np.stack([rot(rng.normal(size=3), np.deg2rad(sig_deg) * rng.normal()) @ q for q in frame])


print("RESCUE_CG=%s  (STEPS=%s)  mosaic sweep on %d real cxidb frames" % (RESCUE_CG, os.environ["STEPS"], n))
for sig in [0.0, 0.2, 0.3, 0.5]:
    rng = np.random.default_rng(0)
    fr = [jitter(f, sig, rng) for f in frames]
    _, stats = hybrid_index(fr, nbest=3)
    print("  mosaic=%.1f deg:  n_idx=%d/%d  support=%d  n_resc=%d" %
          (sig, stats["n_idx"], stats["n"], stats["support"], stats["n_resc"]))
print("DONE")
