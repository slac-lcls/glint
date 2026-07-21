"""#28: does fusing M3 into one kernel actually pay, and at what accuracy cost?

M3 is 66% of the blind frame and compute-bound, dominated by memory traffic through ~10 S x P
intermediates per step (S = 70,400 starts), 8 steps. glint/fused_m3.py collapses that to one kernel
with T and vel in registers and zero intermediates.

The kernel is NOT bit-exact vs torch -- the per-start reduction over peaks is sequential here and a
matmul there -- so this measures what actually matters, in order of increasing honesty:

  1. M3 stage speed          (the thing being optimised)
  2. refined-vector agreement (how far the two M3s diverge before anything downstream)
  3. END-TO-END blind rate + cell stability on the 120 cxidb frames

(3) is the gate. A stage that is 5x faster while moving the cells is a different indexer, not an
optimisation -- the same trap the TF32 row fell into in blind_precision_sweep.py, where the rate
tied at 85/120 while 8 cells moved.

  python bench_fused_m3.py [frames.txt]
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import glint.glint_fast as gf
import glint.glint_index as gi
from glint.glint_index import refine_vec, invq_weight, STARTS
from glint.multishot import same_lattice
from glint import fused_m3

FR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "frames_cxidb_clean.txt")
frames = [q for q in gf.load(FR) if len(q) >= 6]
n = len(frames)
LYSO = gf.LYSO


def sync():
    if gf.DEV == "cuda":
        torch.cuda.synchronize()


print(f"fused M3 bench, {n} cxidb frames, {gf.DEV}, STEPS={gf.STEPS} tol={gf.TOL} "
      f"starts={tuple(STARTS.shape)}")
print(f"cupy available: {fused_m3._HAVE_CP}   max P for shared mem: {fused_m3.MAX_P}")
usable = [q for q in frames if fused_m3.available(torch.as_tensor(np.asarray(q, float),
                                                                  dtype=torch.float32, device=gf.DEV))]
print(f"frames within the shared-memory bound: {len(usable)}/{n}\n")

# ---- 1 + 2: stage speed and vector agreement -------------------------------------------------
print("--- M3 stage: torch vs fused ---")
print(f"{'peaks':>7}{'torch ms':>10}{'fused ms':>10}{'speedup':>9}{'max|dT|':>11}{'median|dT|':>12}")
for q in sorted(usable, key=len)[:: max(1, len(usable) // 6)][:6]:
    Q = torch.as_tensor(np.asarray(q, float), dtype=torch.float32, device=gf.DEV)
    w = invq_weight(Q); qmax = float(Q.norm(dim=1).max())
    S0 = STARTS.clone()
    Tt = refine_vec(S0.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL); sync()
    Tf = fused_m3.refine_vec_fused(S0.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL); sync()
    t = time.perf_counter()
    for _ in range(5):
        refine_vec(S0.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL)
    sync(); tt = (time.perf_counter() - t) / 5
    t = time.perf_counter()
    for _ in range(5):
        fused_m3.refine_vec_fused(S0.clone(), Q, w, qmax, steps=gf.STEPS, tol=gf.TOL)
    sync(); tf = (time.perf_counter() - t) / 5
    d = (Tt - Tf).norm(dim=1)
    print(f"{len(q):>7}{1e3*tt:10.2f}{1e3*tf:10.2f}{tt/max(tf,1e-9):8.2f}x"
          f"{float(d.max()):11.2e}{float(d.median()):12.2e}")

# ---- 3: end-to-end rate + cell stability (the gate) ------------------------------------------
print("\n--- end-to-end blind: rate + cell stability (the gate) ---")


def blind_all(use_fused):
    """glint_fast does `from glint.glint_index import refine_vec`, so `_refine` resolves the name in
    GLINT_FAST's namespace. Patching glint_index.refine_vec does nothing -- the first run of this
    bench did exactly that and reported a 1.01x end-to-end 'speedup' with 120/120 identical cells,
    which is the tell: a kernel with max|dT| ~ 2.6 cannot leave every cell untouched. Patch gf."""
    orig_gf, orig_gi = gf.refine_vec, gi.refine_vec
    calls = {"fused": 0, "fallback": 0}
    if use_fused:
        def patched(T, Q, w, qmax, steps=80, tol=0.18, sharp_last=25, mom=0.5):
            if fused_m3.available(Q):
                calls["fused"] += 1
                return fused_m3.refine_vec_fused(T, Q, w, qmax, steps, tol, sharp_last, mom)
            calls["fallback"] += 1
            return orig_gf(T, Q, w, qmax, steps, tol, sharp_last, mom)
        gf.refine_vec = patched; gi.refine_vec = patched
    try:
        gf.index_blind_fast(frames[0]); sync()
        calls["fused"] = calls["fallback"] = 0          # discard warmup
        t0 = time.perf_counter()
        Ms = [gf.index_blind_fast(q) for q in frames]
        sync(); ms = 1e3 * (time.perf_counter() - t0) / n
    finally:
        gf.refine_vec, gi.refine_vec = orig_gf, orig_gi
    lat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    return lat, ms, Ms, calls


base_lat, base_ms, base_M, _ = blind_all(False)
fus_lat, fus_ms, fus_M, calls = blind_all(True)
print(f"kernel actually invoked: fused {calls['fused']} / fallback {calls['fallback']} calls "
      f"(must be ~{n} fused, else the patch did not bind)")
ident = sum(int((a is None) == (b is None) and (a is None or np.allclose(a, b, atol=1e-6)))
            for a, b in zip(base_M, fus_M))
close = sum(int((a is None) == (b is None) and (a is None or np.allclose(a, b, atol=1e-3)))
            for a, b in zip(base_M, fus_M))
print(f"{'config':<16}{'same_lattice':>14}{'ms/frame':>11}{'speedup':>9}")
print(f"{'torch M3':<16}{f'{base_lat}/{n}':>14}{base_ms:11.2f}{'--':>9}")
print(f"{'fused M3':<16}{f'{fus_lat}/{n}':>14}{fus_ms:11.2f}{base_ms/max(fus_ms,1e-9):8.2f}x")
print(f"\ncells identical (1e-6): {ident}/{n}   close (1e-3): {close}/{n}")
print("A speedup that moves the cells is a different indexer, not an optimisation.")
