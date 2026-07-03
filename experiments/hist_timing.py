"""Per-frame completion-time instrumentation of the GLINT hybrid on the real 120 cxidb frames, to show
the FAST (blind-solved) vs SLOW (rescued) regimes. Mirrors hybrid_index's own resolution logic, timing
each stage per frame. Emits a compact table (blind_ms, outcome, resc_ms) + the one-time consensus cost;
the plotting is done host-side from this. Run on ampere, STEPS=8."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("STEPS", "8"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
from glint.glint_fast import index_blind_nbest, load
from glint.replica_gpu import index_known_gpu_cell
from glint.multishot import consensus_cell, same_lattice

SYNC = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
frames = [q for q in load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)
NB = 3
index_blind_nbest(frames[0], NB); SYNC()                       # warmup

# (1) per-frame blind nbest time
t_blind = np.zeros(n); NBEST = []
for i, q in enumerate(frames):
    SYNC(); t = time.perf_counter()
    nb = index_blind_nbest(q, NB)
    SYNC(); t_blind[i] = 1e3 * (time.perf_counter() - t)
    NBEST.append(nb)

# (2) one-time consensus barrier
SYNC(); t = time.perf_counter()
Mc, support = consensus_cell([c for nb in NBEST for c, _ in nb])
t_cons = 1e3 * (time.perf_counter() - t)

# (3) per-frame resolution: fast (consensus-consistent nbest hit) vs rescue vs none
outcome = []; t_resc = np.zeros(n)
for i, (q, nb) in enumerate(zip(frames, NBEST)):
    M = None
    for c, _ in nb:
        if Mc is not None and same_lattice(c, Mc):
            M = c; break
    if M is not None:
        outcome.append("fast")
    else:
        SYNC(); t = time.perf_counter()
        Mr = index_known_gpu_cell(q, Mc) if Mc is not None else None
        SYNC(); t_resc[i] = 1e3 * (time.perf_counter() - t)
        outcome.append("resc" if (Mr is not None and same_lattice(Mr, Mc)) else "none")

nf = outcome.count("fast"); nr = outcome.count("resc"); nn = outcome.count("none")
print("N=%d  consensus_barrier_ms=%.1f  support=%s  fast=%d resc=%d none=%d  n_idx=%d" %
      (n, t_cons, support, nf, nr, nn, nf + nr))
print("BLIND_MS_MEDIAN=%.2f  p90=%.2f" % (np.median(t_blind), np.percentile(t_blind, 90)))
print("RESC_MS_MEDIAN=%.2f (over rescued)" % (np.median(t_resc[t_resc > 0]) if nr else 0.0))
# machine-readable dump for host-side plotting
print("DATA_BEGIN")
for i in range(n):
    print("%.3f %s %.3f" % (t_blind[i], outcome[i], t_resc[i]))
print("DATA_END")
print("CONS_MS %.3f" % t_cons)
