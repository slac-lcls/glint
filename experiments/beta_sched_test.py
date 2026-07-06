"""HIO->RAAR: ramp beta from BETA_HI (1.0 = DR/HIO) down to BETA (RAAR) over the RAAR steps -- the standard
phase-retrieval beta-relaxation (explore with reflections early, settle into the data projection late).
Blind + hybrid on cxidb-120, LYSO gate. Compare fixed-beta RAAR (baseline 88/120) vs beta schedules + pure HIO.
  module load pytorch/2.6.0 ; srun ... python beta_sched_test.py [frames_file]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("STEPS", "8")
import numpy as np, torch
import glint.glint_fast as gf, glint.glint_index as gi
from glint.glint_fast import index_blind_fast
from glint.hybrid_stream import hybrid_index
from glint.multishot import same_lattice

path = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)) + "/frames_cxidb_clean.txt"
frames = [q for q in gf.load(path) if len(q) >= 6]; n = len(frames); LYSO = gf.LYSO
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{os.path.basename(path)}: {n} frames  device={dev}  (LYSO gate)", flush=True)


def _set(steps, beta_end, sched, beta_hi, polish):
    gf.REFINER = "raar"; gf.STEPS = steps
    gi.RAAR_ANNEAL = 0                                       # isolate the beta schedule
    gi.BETA = beta_end; gi.BETA_SCHED = sched; gi.BETA_HI = beta_hi; gi.POLISH = polish


def blind(steps, beta_end, sched=0, beta_hi=1.0, polish=0):
    _set(steps, beta_end, sched, beta_hi, polish)
    index_blind_fast(frames[0])
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    b = sum(1 for q in frames if (lambda M: M is not None and same_lattice(M, LYSO))(index_blind_fast(q)))
    if dev == "cuda": torch.cuda.synchronize()
    return b, time.time() - t0


def hyb(steps, beta_end, sched=0, beta_hi=1.0, polish=0):
    _set(steps, beta_end, sched, beta_hi, polish)
    _, st = hybrid_index(frames, nbest=3, warmup=False)
    return st["n_idx"], st["support"]


#      steps  beta_end sched beta_hi polish  label
conds = [
    (16, 0.70, 0, 1.0, 0, "RAAR fixed beta=0.7 (baseline)"),
    (16, 1.00, 0, 1.0, 0, "HIO fixed beta=1.0 (no polish)"),
    (16, 1.00, 0, 1.0, 3, "HIO fixed beta=1.0 + ER polish 3"),
    (16, 0.70, 1, 1.0, 0, "sched 1.0->0.7 (HIO->RAAR)"),
    (16, 0.50, 1, 1.0, 0, "sched 1.0->0.5"),
    (16, 0.60, 1, 0.9, 0, "sched 0.9->0.6"),
    (24, 0.70, 1, 1.0, 0, "sched 1.0->0.7  steps=24"),
]
print(f"{'cond':36s} {'blind':>10s} {'hybrid':>9s} {'support':>8s}  {'s':>5s}", flush=True)
for st, be, sc, bh, po, lab in conds:
    b, dt = blind(st, be, sc, bh, po)
    h, sup = hyb(st, be, sc, bh, po)
    print(f"{lab:36s} {b:3d}/{n} {100*b//n:2d}% {h:4d}/{n}   {sup:4d}    {dt:5.1f}", flush=True)
