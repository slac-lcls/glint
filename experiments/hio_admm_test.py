"""HIO / ADMM / warm-started-SO2D for indexing-as-phase-retrieval, informed by the phase-retrieval notes:
HIO=DR=RAAR(beta=1) is wilder (bigger basin) but needs an ER POLISH; SO2D needs a RAAR WARM-UP (structure
forms only in the basin -- why the cold SO2D collapsed); ADMM is DR-on-the-dual with a penalty rho. Baselines:
GD 84/120, RAAR-16 b0.7 = 88. Same frames + gate as raar_test. Knobs set in-process via glint.glint_index."""
import os, sys, time
sys.path.insert(0, "/sdf/home/s/smarches/git/glint")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
import glint.glint_fast as gf
import glint.glint_index as gi
from glint.glint_fast import index_blind_fast
from glint.multishot import same_lattice
frames = [q for q in gf.load("/sdf/home/s/smarches/git/glint/experiments/frames_cxidb_clean.txt") if len(q) >= 6]
n = len(frames)


def run(refiner, steps, beta=0.7, ws=1.0, warm=0, polish=0, rho=1.0):
    gf.REFINER = refiner; gf.STEPS = steps
    gi.BETA = beta; gi.SO_WS = ws; gi.WARM = warm; gi.POLISH = polish; gi.RHO = rho
    index_blind_fast(frames[0]); torch.cuda.synchronize()          # warmup (per config)
    t0 = time.perf_counter(); sl = 0
    for q in frames:
        M = index_blind_fast(q); sl += (M is not None and same_lattice(M, gf.LYSO))
    torch.cuda.synchronize()
    gi.WARM = 0; gi.POLISH = 0                                       # reset globals between configs
    return sl, 1e3 * (time.perf_counter() - t0) / n


print("config                         blind_rate   ms/frame")
cfgs = [
    ("GD-8",                 dict(refiner="grad", steps=8)),
    ("RAAR-16 b0.7 (ref)",   dict(refiner="raar", steps=16, beta=0.7)),
    ("HIO(b1.0)-16",         dict(refiner="raar", steps=16, beta=1.0)),
    ("HIO(b1.0)-16 +ER3",    dict(refiner="raar", steps=16, beta=1.0, polish=3)),
    ("RAAR-16 b0.7 +ER3",    dict(refiner="raar", steps=16, beta=0.7, polish=3)),
    ("SO2D warm16+8 ws1",    dict(refiner="so2d", steps=8, warm=16, ws=1.0)),
    ("SO2D warm16+8 ws0.5",  dict(refiner="so2d", steps=8, warm=16, ws=0.5)),
    ("ADMM-16 rho1.0",       dict(refiner="admm", steps=16, rho=1.0)),
    ("ADMM-16 rho0.7",       dict(refiner="admm", steps=16, rho=0.7)),
    ("ADMM-16 rho1.0 +ER3",  dict(refiner="admm", steps=16, rho=1.0, polish=3)),
]
for name, kw in cfgs:
    sl, ms = run(**kw)
    print(f"  {name:28s}  {sl}/{n} = {100*sl//n:2d}%    {ms:.1f}", flush=True)
