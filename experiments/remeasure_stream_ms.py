"""Re-measure `stream_ms` (the streaming per-frame wall) and check that the stage table decomposes it.

WHY. check_numbers.py records stream_ms=4.16 with a stage table summing to 3.47, leaving 0.69 ms/frame
(17% of the frame) UN-ATTRIBUTED -- large enough that the guard warns on every push. Its own comment
names the suspect: "the in-situ cross-check run points at stream_ms rather than at a missing stage
... Re-measure stream_ms itself before quoting this decomposition stage-by-stage." This is that
re-measurement, made reproducible so the next person does not have to reconstruct the protocol.

It answers three questions, and the first is the one that matters:

  1. WARM-UP. What does one-time cost (CUDA-graph capture, kernel JIT, grid build, first-touch
     allocation) do to a per-frame average? Timed cold vs warmed, single-pass vs tiled. A single pass
     over the 40-frame stack amortises setup over 40 frames; the steady state amortises it away.
  2. PROTOCOL. bench_stream_driver.py's stage table uses `tmin` = MIN over 10 reps, on ONE fixed
     frame, each stage isolated with its own sync. stream_ms is a MEAN over a real stream. Those are
     different protocols, and a sum of best-case minima need not equal a mean. How big is the gap?
  3. DECOMPOSITION. Under each protocol, how much of the wall do the measured stages account for?

RESULT on A100 (sdfampere032/035), main @ f82d0a7, glint_sim 40x1024x1024, B=40, tol=0.002:

    COLD, single pass of 40      13.665 ms/frame      <- setup dominates; NOT a steady state
    COLD, tiled x20 = 800         3.579 ms/frame
    WARMED, single pass of 40     3.496 ms/frame
    WARMED, tiled x20 = 800       3.494 ms/frame      <- steady state
    (an independent run measured 3.662 in the same configuration)

    stage         tmin(min,1 frame)   mean(in-loop)    ratio
    peakfind             1.211 ms        1.274 ms      1.05x
    peaks_to_q           0.083 ms        0.123 ms      1.48x
    predict              0.166 ms        0.224 ms      1.35x     (FACTS predict_ms = 0.16 -- matches)
    integrate            0.330 ms        0.344 ms      1.04x
    index/frame          0.543 ms        0.787 ms      1.45x
    SUM                  2.334 ms        2.751 ms      1.18x

So: min-vs-mean is real but MODEST (1.18x overall), and does NOT explain 0.69 ms. The steady-state
wall measures 3.49-3.66, i.e. 12-16% BELOW the recorded 4.16. Against a 3.5-3.66 wall the stage
table's 3.47 leaves 0.03-0.19 ms (1-5%) un-attributed rather than 0.69 (17%).

NOT ACTED ON. stream_ms is deliberately left at 4.16 in check_numbers.py. Correcting it moves several
headline numbers in GLINT's favour (stream_fps 240->286, peakfind share 28%->33%, the gap to 3500
hits/s 15x->12x, the FPGA ceiling 1.39x->1.50x), which is exactly when to be slowest, and the
provenance of 4.16 -- which GPU, which protocol -- is not recoverable from the code. Confirm on the
hardware the original used before swapping it in.

TWO TRAPS, both of which produced wrong numbers here before being caught:
  * RUN IT ON THE CODE YOU MEAN TO MEASURE. A first attempt ran on a feature branch 44 commits behind
    main, without the fused predict, and read predict=1.175 ms against a true 0.166. That predict
    row is now the harness's own sanity check: if it does not land near FACTS' 0.16, the checkout is
    wrong and every other number is worthless.
  * BATCH THE INDEX. `index_fused([q], Mc, B=1)` per frame measures a call the driver never makes and
    inflates it ~37x. The driver batches at B=40; so does this.

  GLINT_ROOT=/path/to/glint GLINT_SIM=/path/to/sim python remeasure_stream_ms.py
"""
import os, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

G = os.environ.get("GLINT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, G); sys.path.insert(0, os.path.join(G, "experiments"))
import cupy as cp
from glint_fast import cell_to_Ar
import glint.stream_driver as sd
from glint.stream_driver import peaks_to_q, _canonical_axes
from glint.fused_integrate import integrate_fused
import glint.replica_gpu_batch as rgb

SIM = os.environ.get("GLINT_SIM", "/sdf/home/s/smarches/glint_sim")
REPEAT = int(os.environ.get("REPEAT", "20"))
FACTS_STREAM_MS, FACTS_PREDICT_MS = 4.16, 0.16

imgs = np.load(f"{SIM}/images.npy"); t = np.load(f"{SIM}/truth.npz")
N = int(t["det_n"]); pix_mm = float(t["pix_mm"]); dist_mm = float(t["dist_mm"])
wave = float(t["wave_A"]); cell = t["cell"]
panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
               res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
               coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
Mc = cell_to_Ar(float(cell[0]), float(cell[1]), float(cell[2]), 90, 90, 90)
clen = dist_mm / 1000.0
TOL = 0.002


def sync():
    cp.cuda.Stream.null.synchronize()


def new_driver():
    return sd.StreamDriver(Mc, panels, clen, wave, (N, N), dtype=imgs.dtype, B=40, dmin=2.0, tol=TOL)


def wall(repeat, warm):
    """Per-frame wall of the UNINSTRUMENTED driver -- the thing stream_ms is meant to be."""
    drv = new_driver()
    if warm:
        for f in imgs:
            drv.push(f)
        drv.flush(); sync()
    t0 = time.perf_counter()
    for _ in range(repeat):
        for f in imgs:
            drv.push(f)
    drv.flush(); sync()
    return 1e3 * (time.perf_counter() - t0) / (len(imgs) * repeat)


print(f"stack {imgs.shape}, B=40, tol={TOL}, tiled x{REPEAT}\n")
print("1. WARM-UP -- what one-time cost does to a per-frame average")
res = {}
for lbl, rep, warm in (("COLD, single pass of 40", 1, False),
                       (f"COLD, tiled x{REPEAT}", REPEAT, False),
                       ("WARMED, single pass of 40", 1, True),
                       (f"WARMED, tiled x{REPEAT}  (steady state)", REPEAT, True)):
    v = wall(rep, warm); res[lbl] = v
    print(f"   {lbl:40s} {v:7.3f} ms/frame  ({1000/v:5.1f} f/s)")
steady = res[f"WARMED, tiled x{REPEAT}  (steady state)"]
print(f"   {'FACTS stream_ms':40s} {FACTS_STREAM_MS:7.3f} ms/frame  ({1000/FACTS_STREAM_MS:5.1f} f/s)")

# ---- 2/3. the two stage protocols, on the same code and frames -------------------------------
drv = new_driver()
g = cp.asarray(imgs[0])


def tmin(fn, reps=10):
    fn(); sync(); best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync()
        best = min(best, time.perf_counter() - t0)
    return 1e3 * best


pk = drv.finder.find(g); fs = cp.asnumpy(pk["x"]); ss = cp.asnumpy(pk["y"])
q = peaks_to_q(fs, ss, panels, clen, wave); q = q[np.isfinite(q).all(1)]
M = rgb.index_fused([q], Mc, B=1)[0]; Mcan = _canonical_axes(np.asarray(M, float))
pred = drv.grid.predict(Mcan, panels, clen, wave, tol=TOL)
A = {"peakfind": tmin(lambda: drv.finder.find(g)),
     "peaks_to_q": tmin(lambda: peaks_to_q(fs, ss, panels, clen, wave)),
     "predict": tmin(lambda: drv.grid.predict(Mcan, panels, clen, wave, tol=TOL)),
     "integrate": tmin(lambda: integrate_fused(g, pred)),
     "index/frame": tmin(lambda: rgb.index_fused([q] * 40, Mc, B=40)) / 40}

acc = {k: 0.0 for k in A}
nfr = 0
drv2 = new_driver()
for f in imgs:
    drv2.push(f)
drv2.flush(); sync()                                    # warm before timing
t_loop0 = time.perf_counter()
pend_q, pend_g = [], []
for _ in range(REPEAT):
    for f in imgs:
        gg = cp.asarray(f)
        t0 = time.perf_counter(); p = drv2.finder.find(gg); sync()
        acc["peakfind"] += time.perf_counter() - t0
        xf = cp.asnumpy(p["x"]); yf = cp.asnumpy(p["y"])
        t0 = time.perf_counter(); qq = peaks_to_q(xf, yf, panels, clen, wave); sync()
        acc["peaks_to_q"] += time.perf_counter() - t0
        qq = qq[np.isfinite(qq).all(1)]
        if len(qq) < 6:
            continue
        pend_q.append(qq); pend_g.append(gg)
        if len(pend_q) < 40:                            # batch exactly as the driver does
            continue
        t0 = time.perf_counter(); Ms = rgb.index_fused(pend_q, Mc, B=40); sync()
        acc["index/frame"] += time.perf_counter() - t0
        for Mi, gsrc in zip(Ms, pend_g):
            if Mi is None:
                continue
            Mn = _canonical_axes(np.asarray(Mi, float))
            t0 = time.perf_counter(); pr = drv2.grid.predict(Mn, panels, clen, wave, tol=TOL); sync()
            acc["predict"] += time.perf_counter() - t0
            t0 = time.perf_counter(); integrate_fused(gsrc, pr); sync()
            acc["integrate"] += time.perf_counter() - t0
            nfr += 1
        pend_q, pend_g = [], []
sync()
loop_wall = 1e3 * (time.perf_counter() - t_loop0) / max(nfr, 1)
B = {k: 1e3 * v / max(nfr, 1) for k, v in acc.items()}

print(f"\n2. PROTOCOL -- same stages, two ways ({nfr} frames)")
print(f"   {'stage':13s} {'tmin(min,1 frame)':>19s} {'mean(in-loop)':>15s} {'ratio':>7s}")
for k in A:
    print(f"   {k:13s} {A[k]:16.3f} ms {B[k]:12.3f} ms {B[k]/max(A[k],1e-9):6.2f}x")
sa, sb = sum(A.values()), sum(B.values())
print(f"   {'SUM':13s} {sa:16.3f} ms {sb:12.3f} ms {sb/max(sa,1e-9):6.2f}x")

ok = abs(A["predict"] - FACTS_PREDICT_MS) < 0.5 * FACTS_PREDICT_MS
print(f"\n   SANITY: predict {A['predict']:.3f} ms vs FACTS {FACTS_PREDICT_MS} -- "
      f"{'OK' if ok else 'MISMATCH: wrong checkout? every number below is suspect'}")

print(f"\n3. DECOMPOSITION (GPU stages only; FACTS also carries {1.22} ms of host)")
print(f"   steady-state wall            {steady:7.3f} ms   [FACTS stream_ms {FACTS_STREAM_MS}]")
print(f"   tmin stages account for      {100*sa/steady:6.1f} %   ({steady-sa:+.3f} ms un-attributed)")
print(f"   in-loop means account for    {100*sb/loop_wall:6.1f} %  of their own {loop_wall:.3f} ms wall")
print(f"\n   FACTS stage table sums to 3.47 (GPU 2.25 + host 1.22).")
print(f"   Against FACTS' 4.16 that leaves 0.69 ms (17%) un-attributed.")
print(f"   Against the measured {steady:.2f} it leaves {steady-3.47:+.2f} ms "
      f"({100*(steady-3.47)/steady:.0f}%).")
raise SystemExit(0 if ok else 1)
