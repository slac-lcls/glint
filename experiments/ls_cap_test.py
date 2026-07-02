"""Fair retry of the fused line-search M3: is its rate drop a too-tight anti-jump cap / missing momentum,
or the exact line search itself? Rebinds the cap + momentum knobs in-process. The decisive row is
ls + momentum (LS_MOM=0.5): if that recovers ~84/120, the greedy step was the issue (momentum coasting
matters, as with GD); if it still lags, the exact line search is fundamentally wrong for this multimodal
comb. Baseline grad-8 = 84/120 @ ~19 ms. Same 120 cxidb frames + gate as ls_test.py."""
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


def run(refiner, steps, capk=0.5, capq=1.0, mom=0.0):
    gf.REFINER = refiner; gf.STEPS = steps
    gi.LS_CAPK = capk; gi.LS_CAPQ = capq; gi.LS_MOM = mom
    index_blind_fast(frames[0]); torch.cuda.synchronize()          # warmup (per config)
    t0 = time.perf_counter(); sl = 0
    for q in frames:
        M = index_blind_fast(q); sl += (M is not None and same_lattice(M, gf.LYSO))
    torch.cuda.synchronize()
    return sl, 1e3 * (time.perf_counter() - t0) / n


print("refiner steps capk capq  mom   blind_rate   ms/frame")
cfgs = [("grad", 8, {}),
        ("ls", 8, {}),                                   # default cap (the 72/120 we saw)
        ("ls", 8, {"capk": 1.0, "capq": 0.9}),           # relaxed cap (full period, 90th-pct psi)
        ("ls", 8, {"mom": 0.5}),                         # + momentum (greedy-vs-momentum test)
        ("ls", 8, {"capk": 1.0, "capq": 0.9, "mom": 0.5}),  # relaxed + momentum
        ("ls", 4, {"capk": 1.0, "capq": 0.9, "mom": 0.5})]  # the speed case, best config
for r, s, kw in cfgs:
    sl, ms = run(r, s, **kw)
    print(f"  {r:4s}  {s:2d}   {kw.get('capk',0.5):.1f}  {kw.get('capq',1.0):.2f}  {kw.get('mom',0.0):.1f}   "
          f"{sl}/{n} = {100*sl//n:2d}%    {ms:.1f}", flush=True)
