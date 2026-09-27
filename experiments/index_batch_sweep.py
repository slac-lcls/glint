"""index is 0.430 ms/frame in the DRP table but 0.17 elsewhere. check_numbers.py says why:
0.17 is the B=120 amortization, and the DRP driver ran at B=40 ("index_b40_ms": 0.54). Indexing is
one thread-block per frame for the anneal and refine kernels (obj splits its candidates across
blocks since #165), so B still sets occupancy. Two knobs are therefore available and BOTH are
currently at their slow setting in the measured configuration:
  * B = 40 rather than 120 (B=64 is the driver default but is NOT a knee -- B=120 measures 26%
    faster, 0.214 -> 0.170 ms/frame fp64, so sweep to the full batch rather than stopping at 64)
  * KC_FP as set in the environment (fp32 is the default since 26 Sep 2026; KC_FP=64 for the fp64 path)

Sweep both, and verify fp32 is rate- AND lattice-identical to fp64 rather than trusting the claim.
KC_FP is read at module IMPORT, so this must be one process per value -- an in-process loop would
silently measure the same precision twice.

  KC_FP=64 python index_batch_sweep.py
  KC_FP=32 python index_batch_sweep.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import torch
import glint.glint_fast as gf
from glint.glint_fast import matched
from glint.multishot import same_lattice
import glint.replica_gpu_batch as rgb

LYSO = gf.LYSO
FRAMES = WT + "/experiments/frames_cxidb_clean.txt"
BS = [8, 20, 32, 40, 64, 120]
KC = os.environ.get("KC_FP", "32")
assert (rgb.FP == torch.float32) == (KC == "32"), \
    f"KC_FP={KC} but rgb.FP={rgb.FP} -- module was imported before the env was set"


def main():
    frames = [np.asarray(q, float) for q in gf.load(FRAMES)]
    n = len(frames)
    print(f"KC_FP={KC}  rgb.FP={rgb.FP}  frames={n}")

    rgb.index_fused(frames[:8], LYSO, B=8)          # warm up (JIT / autotune)
    torch.cuda.synchronize()

    print(f"\n{'B':>5s} {'ms/frame':>10s} {'ms/frame min':>13s} {'solved':>8s} {'refl/frame':>11s}")
    out = {}
    for B in BS:
        ts = []
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            Ms = rgb.index_fused(frames, LYSO, B=B)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) / n)
        Ms = [np.asarray(M, float) if M is not None else None for M in Ms]
        ok = sum(1 for M, q in zip(Ms, frames)
                 if M is not None and same_lattice(M, LYSO)
                 and matched(M, q) / len(q) >= 0.25 and matched(M, q) >= 10)
        nref = np.mean([matched(M, q) if M is not None else 0 for M, q in zip(Ms, frames)])
        out[B] = (float(np.median(ts)), float(np.min(ts)), ok,
                  [None if M is None else np.round(np.sort(np.linalg.norm(M, axis=0)), 3).tolist()
                   for M in Ms])
        print(f"{B:5d} {1e3*out[B][0]:10.3f} {1e3*out[B][1]:13.3f} {ok:5d}/{n} {nref:11.1f}")

    best = min(out, key=lambda b: out[b][0]); worst = max(out, key=lambda b: out[b][0])
    print(f"\nbest B={best} at {1e3*out[best][0]:.3f} ms/frame; worst B={worst} at "
          f"{1e3*out[worst][0]:.3f}  -> {out[worst][0]/out[best][0]:.2f}x spread")
    print(f"B=40 (the DRP driver's setting): {1e3*out[40][0]:.3f} ms/frame")
    print(f"B=120: {1e3*out[120][0]:.3f} ms/frame   delta = "
          f"{1e3*(out[40][0]-out[120][0]):+.3f} ms/frame")
    print(f"solve counts identical across B: {len({v[2] for v in out.values()}) == 1}")

    np.save(f"/tmp/idx_cells_{KC}.npy",
            np.array([out[120][3]], dtype=object), allow_pickle=True)
    print(f"\n(cells at B=120 saved to /tmp/idx_cells_{KC}.npy for the fp32-vs-fp64 comparison)")


if __name__ == "__main__":
    main()
