"""#28: what is left on the table for PRECISION in the blind front end?

M3 (66% of the frame) is already fp32, so there is no fp64->fp32 win there. Two knobs remain:

  TF32          torch.backends.cuda.matmul.allow_tf32 -- never set in this repo, so the fp32 matmul
                in objective() runs in true fp32 on the A100. Tensor cores would accelerate it, BUT
                (a) the matmul is K=3, so the stage is elementwise/memory-bound, not FLOP-bound, and
                (b) the objective's `proj - round(proj)` is catastrophic cancellation: with
                |q|.|v| ~ tens, TF32's 10-bit mantissa may not resolve the 0.18 tolerance window.
                Both predict "no gain, possible rate loss". Measured here rather than argued.

  ANNEAL_FP32   M4/M5/M6 run in fp64 by default (glint_fast.ADT); the flag exists but is off. The
                known-cell engine's equivalent (KC_FP=32) was measured rate-identical, so this is
                the more promising of the two.

Reports blind same_lattice rate (the 84-85/120 figure) + ms/frame for each combination.

  python blind_precision_sweep.py [frames.txt]
"""
import os, sys, time, importlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("STEPS", "8")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import glint.glint_fast as gf
from glint.multishot import same_lattice

FR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "frames_cxidb_clean.txt")
frames = [q for q in gf.load(FR) if len(q) >= 6]
n = len(frames)
LYSO = gf.LYSO


def run(label, tf32, anneal_fp32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    gf.ADT = torch.float32 if anneal_fp32 else torch.float64
    gf.index_blind_fast(frames[0])                       # warm
    if gf.DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    Ms = [gf.index_blind_fast(q) for q in frames]
    if gf.DEV == "cuda":
        torch.cuda.synchronize()
    ms = 1e3 * (time.perf_counter() - t0) / n
    lat = sum(int(M is not None and same_lattice(np.asarray(M, float), LYSO)) for M in Ms)
    return lat, ms, Ms


print(f"blind precision sweep, {n} cxidb frames, {gf.DEV}, STEPS={gf.STEPS}")
print(f"default TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}   default ADT: {gf.ADT}\n")
print(f"{'config':<28}{'same_lattice':>14}{'ms/frame':>11}{'vs base':>10}")
base_lat, base_ms, base_M = run("baseline", False, False)
print(f"{'baseline (fp32 M3, fp64 M5)':<28}{f'{base_lat}/{n}':>14}{base_ms:11.2f}{'--':>10}")

for label, tf32, afp32 in [("+ TF32 matmul", True, False),
                           ("+ ANNEAL_FP32 (M4/5/6)", False, True),
                           ("+ both", True, True)]:
    lat, ms, Ms = run(label, tf32, afp32)
    ident = sum(int((a is None) == (b is None) and (a is None or np.allclose(a, b, atol=1e-6)))
                for a, b in zip(base_M, Ms))
    print(f"{label:<28}{f'{lat}/{n}':>14}{ms:11.2f}{100*(base_ms-ms)/base_ms:9.1f}%"
          f"   cells identical to baseline: {ident}/{n}")

run("restore", False, False)
print("\nnote: 'cells identical' is the honest gate -- a rate that happens to tie while the actual")
print("cells move is a different engine, not a free speedup.")
