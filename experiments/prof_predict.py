"""Attribute HKLGrid.predict cost to its substages.

Times, separately and correctly:
  0. R prep (recip_from_M / asarray / counter reset / H2D of R)
  1. _GATE_KERNEL   -- cuda Event around JUST the kernel (GPU time) + host launch cost
  2. n = int(self._gcnt[0])  -- the blocking sync (measured BOTH naively and after an event sync)
  3. cp.asnumpy(self._gout[:n])
  4. np.argsort(..., kind='stable') (+ the row reorder it feeds)
  5. self.g[idx] gather
  6. project_q(...)
  7. structured-array assembly
  8. whole predict()
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import cupy as cp
from glint_fast import cell_to_Ar
import glint.stream_driver as sd
from glint.stream_driver import HKLGrid, _GATE_KERNEL
from glint.predict import recip_from_M, project_q, Z_HAT

SIM = os.environ.get("GLINT_SIM", "/sdf/home/s/smarches/glint_sim")
t = np.load(f"{SIM}/truth.npz")
print("truth.npz keys:", list(t.keys()))
N = int(t["det_n"]); pix_mm = float(t["pix_mm"]); dist_mm = float(t["dist_mm"]); wave = float(t["wave_A"])
cell = t["cell"]
panels = [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]),
               res=1.0 / (pix_mm / 1000.0), cx=-(N / 2.0 - 0.5), cy=-(N / 2.0 - 0.5),
               coffset=0.0, min_fs=0, max_fs=N - 1, min_ss=0, max_ss=N - 1)]
Mc = cell_to_Ar(float(cell[0]), float(cell[1]), float(cell[2]), 90, 90, 90)
clen = dist_mm / 1000.0
DMIN = float(os.environ.get("DMIN", "2.0"))
TOL = float(os.environ.get("TOL", "0.002"))
print(f"cell={np.asarray(cell)}  det_n={N} pix_mm={pix_mm} dist_mm={dist_mm} wave={wave} "
      f"dmin={DMIN} tol={TOL} npanels={len(panels)}")

grid = HKLGrid(Mc, DMIN, gpu=True)
NHKL = grid.g.shape[0]
print(f"gpu={grid.gpu}  nhkl={NHKL}  grid bytes on device = {NHKL*3*8/1e6:.3f} MB (float64)")
print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

# ---- a set of real orientations: true R rotated by random rotations (norm-preserving) ----
R0 = recip_from_M(np.asarray(Mc, float))
rng = np.random.default_rng(0)
NORI = 24


def rand_rot(rng):
    A = rng.normal(size=(3, 3))
    Q, S = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(S))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


ORIS = [R0 @ rand_rot(rng) for _ in range(NORI)]

# =====================================================================================
# instrumented replicas of predict()
# =====================================================================================
qmax2 = grid.qmax * grid.qmax
TPB = 256
BLOCKS = (NHKL + TPB - 1) // TPB


def predict_timed(R, tol, event_sync_kernel):
    """Replica of HKLGrid.predict (gpu branch), line for line, with timings.

    event_sync_kernel=False -> NAIVE: nothing forces the kernel to finish before
        `n = int(self._gcnt[0])`, so that line absorbs the kernel wait (the trap).
    event_sync_kernel=True  -> the kernel is timed with cuda Events and completed
        first, so `n = int(...)` is measured as a bare 4-byte D2H.
    """
    d = {}
    tA = time.perf_counter()
    Rm = np.asarray(R, float)                       # is_recip=True path
    grid._gcnt[0] = 0
    Rf = cp.ascontiguousarray(cp.asarray(Rm).ravel())
    tB = time.perf_counter()
    d["0_prep"] = tB - tA

    if event_sync_kernel:
        ev0 = cp.cuda.Event(); ev1 = cp.cuda.Event()
        ev0.record()
        t0 = time.perf_counter()
        _GATE_KERNEL((BLOCKS,), (TPB,),
                     (grid._ggr, np.int32(NHKL), Rf, np.float64(wave),
                      np.float64(qmax2), np.float64(tol),
                      grid._gout.ravel(), grid._gcnt, np.int32(NHKL)))
        t1 = time.perf_counter()
        ev1.record(); ev1.synchronize()
        d["1_kernel_gpu"] = cp.cuda.get_elapsed_time(ev0, ev1) * 1e-3   # ms -> s
        d["1_kernel_launch_host"] = t1 - t0
    else:
        t0 = time.perf_counter()
        _GATE_KERNEL((BLOCKS,), (TPB,),
                     (grid._ggr, np.int32(NHKL), Rf, np.float64(wave),
                      np.float64(qmax2), np.float64(tol),
                      grid._gout.ravel(), grid._gcnt, np.int32(NHKL)))
        t1 = time.perf_counter()
        d["1_kernel_launch_host"] = t1 - t0

    t2 = time.perf_counter()
    n = int(grid._gcnt[0])                          # BLOCKING SYNC
    t3 = time.perf_counter()
    d["2_blocking_read_n"] = t3 - t2

    C = cp.asnumpy(grid._gout[:n])                  # D2H of survivors
    t4 = time.perf_counter()
    d["3_d2h"] = t4 - t3

    order = np.argsort(C[:, 0], kind="stable")
    t5 = time.perf_counter()
    d["4a_argsort"] = t5 - t4
    C = C[order]
    t6 = time.perf_counter()
    d["4b_reorder_rows"] = t6 - t5

    idx = C[:, 0].astype(np.int64)
    hkl = grid.g[idx]
    q = C[:, 1:4]; qn2 = C[:, 4]; exc = C[:, 5]
    t7 = time.perf_counter()
    d["5_gather_g"] = t7 - t6

    fs, ss, pan = project_q(q, panels, clen, wave)
    t8 = time.perf_counter()
    d["6_project_q"] = t8 - t7

    on = pan >= 0
    out = np.zeros(int(on.sum()), dtype=[("h", int), ("k", int), ("l", int),
                                         ("fs", float), ("ss", float), ("panel", int),
                                         ("exc", float), ("res", float)])
    out["h"], out["k"], out["l"] = hkl[on, 0], hkl[on, 1], hkl[on, 2]
    out["fs"], out["ss"], out["panel"] = fs[on], ss[on], pan[on]
    out["exc"] = exc[on]; out["res"] = 1.0 / np.sqrt(qn2[on])
    t9 = time.perf_counter()
    d["7_struct_assembly"] = t9 - t8
    d["_sum_parts"] = t9 - tA
    d["_n"] = n
    d["_non"] = int(on.sum())
    return d, out


# =====================================================================================
# run
# =====================================================================================
REPS = 10
# warmup
for i in range(5):
    predict_timed(ORIS[i % NORI], TOL, True)
    predict_timed(ORIS[i % NORI], TOL, False)
    grid.predict(ORIS[i % NORI], panels, clen, wave, tol=TOL, is_recip=True)
cp.cuda.Stream.null.synchronize()

KEYS_E = ["0_prep", "1_kernel_gpu", "1_kernel_launch_host", "2_blocking_read_n", "3_d2h",
          "4a_argsort", "4b_reorder_rows", "5_gather_g", "6_project_q", "7_struct_assembly",
          "_sum_parts"]
KEYS_N = ["0_prep", "1_kernel_launch_host", "2_blocking_read_n", "3_d2h",
          "4a_argsort", "4b_reorder_rows", "5_gather_g", "6_project_q", "7_struct_assembly",
          "_sum_parts"]

accE = {k: [] for k in KEYS_E}
accN = {k: [] for k in KEYS_N}
nsurv = []
nonval = []
whole = []

for r in range(REPS):
    for o in ORIS:
        d, out = predict_timed(o, TOL, True)
        for k in KEYS_E:
            accE[k].append(d[k])
        nsurv.append(d["_n"]); nonval.append(d["_non"])

        d2, _ = predict_timed(o, TOL, False)
        for k in KEYS_N:
            accN[k].append(d2[k])

        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        p = grid.predict(o, panels, clen, wave, tol=TOL, is_recip=True)
        cp.cuda.Stream.null.synchronize()
        whole.append(time.perf_counter() - t0)

nsurv = np.array(nsurv); nonval = np.array(nonval)


def stat(v):
    v = np.asarray(v) * 1e3
    return float(np.min(v)), float(np.median(v))


print(f"\nsamples: {NORI} orientations x {REPS} reps = {len(whole)}")
print(f"survivors n: min={nsurv.min()} median={int(np.median(nsurv))} max={nsurv.max()}"
      f"   survivor fraction median = {np.median(nsurv)/NHKL*100:.3f}%")
print(f"on-detector after project_q: median={int(np.median(nonval))}")
nm = float(np.median(nsurv))
print(f"\nBYTES per predict call:")
print(f"  H2D : R = 72 B  (+ 4 B counter reset)")
print(f"  kernel device reads : {NHKL*3*8/1e6:.3f} MB (grid, coalesced), writes {nm*6*8/1e3:.1f} kB")
print(f"  D2H : survivors = {nm*6*8/1e3:.1f} kB  (+ 4 B for the counter read)")

wmin, wmed = stat(whole)
print(f"\n=== A) EVENT-SYNCED decomposition (kernel forced to complete before the host read) ===")
print(f"{'substage':<26}{'min ms':>10}{'median ms':>12}{'% of whole(med)':>18}")
for k in KEYS_E:
    mn, md = stat(accE[k])
    print(f"{k:<26}{mn:>10.4f}{md:>12.4f}{100*md/wmed:>17.1f}%")
print(f"{'WHOLE predict()':<26}{wmin:>10.4f}{wmed:>12.4f}{100.0:>17.1f}%")
smn, smd = stat(accE["_sum_parts"])
print(f"  residual (whole - sum_parts): median {wmed - smd:+.4f} ms  ({100*(wmed-smd)/wmed:+.1f}%)")

print(f"\n=== B) NAIVE decomposition (no event sync -- 'n = int(gcnt[0])' absorbs the kernel) ===")
print(f"{'substage':<26}{'min ms':>10}{'median ms':>12}{'% of whole(med)':>18}")
for k in KEYS_N:
    mn, md = stat(accN[k])
    print(f"{k:<26}{mn:>10.4f}{md:>12.4f}{100*md/wmed:>17.1f}%")
smn2, smd2 = stat(accN["_sum_parts"])
print(f"  residual (whole - sum_parts): median {wmed - smd2:+.4f} ms")

# host tail = everything after the kernel launch returns
tail_keys = ["2_blocking_read_n", "3_d2h", "4a_argsort", "4b_reorder_rows", "5_gather_g",
             "6_project_q", "7_struct_assembly"]
tail_med = sum(stat(accE[k])[1] for k in tail_keys)
kmed = stat(accE["1_kernel_gpu"])[1]
print(f"\nHOST TAIL (event-synced, sum of stages 2-7) median = {tail_med:.4f} ms "
      f"= {100*tail_med/wmed:.1f}% of predict")
print(f"GATE KERNEL GPU time median = {kmed:.4f} ms = {100*kmed/wmed:.1f}% of predict")
print(f"ratio host-tail / kernel = {tail_med/max(kmed,1e-9):.1f}x")

# =====================================================================================
# LEVERS (measured, not implemented in the repo)
# =====================================================================================
print("\n\n================ LEVER 1: is np.argsort avoidable? ================")
# grab one representative survivor block
grid._gcnt[0] = 0
Rf = cp.ascontiguousarray(cp.asarray(np.asarray(ORIS[0], float)).ravel())
_GATE_KERNEL((BLOCKS,), (TPB,), (grid._ggr, np.int32(NHKL), Rf, np.float64(wave),
             np.float64(qmax2), np.float64(TOL), grid._gout.ravel(), grid._gcnt, np.int32(NHKL)))
n = int(grid._gcnt[0])
C0 = cp.asnumpy(grid._gout[:n])
print(f"representative survivor block: n={n}")


def bench(fn, reps=200):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1e3
    return float(ts.min()), float(np.median(ts))


mn, md = bench(lambda: np.argsort(C0[:, 0], kind="stable"))
print(f"  np.argsort(C[:,0],'stable')          min {mn:.4f}  med {md:.4f} ms")
mn2, md2 = bench(lambda: C0[np.argsort(C0[:, 0], kind="stable")])
print(f"  argsort + C[order] row reorder       min {mn2:.4f}  med {md2:.4f} ms")
# alternative: sort the int index only and gather columns lazily
mn3, md3 = bench(lambda: np.argsort(C0[:, 0].astype(np.int64), kind="stable"))
print(f"  argsort on int64 copy                min {mn3:.4f}  med {md3:.4f} ms")
# alternative: no sort at all
mn4, md4 = bench(lambda: C0[:, 0].astype(np.int64))
print(f"  NO SORT (just idx cast)              min {mn4:.4f}  med {md4:.4f} ms")
print(f"  => sort share of predict (median): {100*md2/wmed:.1f}%  ({md2:.4f} of {wmed:.4f} ms)")

# does downstream depend on order?  compare sorted vs unsorted predict output as sets
def predict_nosort(R, tol):
    grid._gcnt[0] = 0
    Rf = cp.ascontiguousarray(cp.asarray(np.asarray(R, float)).ravel())
    _GATE_KERNEL((BLOCKS,), (TPB,), (grid._ggr, np.int32(NHKL), Rf, np.float64(wave),
                 np.float64(qmax2), np.float64(tol), grid._gout.ravel(), grid._gcnt, np.int32(NHKL)))
    n = int(grid._gcnt[0])
    C = cp.asnumpy(grid._gout[:n])
    idx = C[:, 0].astype(np.int64)
    hkl = grid.g[idx]
    q = C[:, 1:4]; qn2 = C[:, 4]; exc = C[:, 5]
    fs, ss, pan = project_q(q, panels, clen, wave)
    on = pan >= 0
    out = np.zeros(int(on.sum()), dtype=[("h", int), ("k", int), ("l", int),
                                         ("fs", float), ("ss", float), ("panel", int),
                                         ("exc", float), ("res", float)])
    out["h"], out["k"], out["l"] = hkl[on, 0], hkl[on, 1], hkl[on, 2]
    out["fs"], out["ss"], out["panel"] = fs[on], ss[on], pan[on]
    out["exc"] = exc[on]; out["res"] = 1.0 / np.sqrt(qn2[on])
    return out


a = grid.predict(ORIS[3], panels, clen, wave, tol=TOL, is_recip=True)
b = predict_nosort(ORIS[3], TOL)
ka = np.sort(a["h"].astype(np.int64) * 10**8 + a["k"] * 10**4 + a["l"])
kb = np.sort(b["h"].astype(np.int64) * 10**8 + b["k"] * 10**4 + b["l"])
print(f"  sorted vs unsorted predict: same length {len(a)==len(b)}, same hkl multiset "
      f"{len(ka)==len(kb) and bool(np.array_equal(ka,kb))}")
mnp, mdp = bench(lambda: predict_nosort(ORIS[3], TOL), reps=60)
print(f"  whole predict WITHOUT sort:          min {mnp:.4f}  med {mdp:.4f} ms   "
      f"(vs {wmin:.4f}/{wmed:.4f} with sort)")

print("\n\n================ LEVER 2: what does project_q cost? ================")
qrep = C0[:, 1:4].copy()
print(f"  project_q at n={len(qrep)} (real size):")
for nn in (1, 10, 100, 500, 1000, len(qrep), 4 * len(qrep), 20 * len(qrep), 100 * len(qrep)):
    qq = np.repeat(qrep, int(np.ceil(nn / len(qrep))) + 1, axis=0)[:nn] if nn > len(qrep) else qrep[:nn]
    mn, md = bench(lambda: project_q(qq, panels, clen, wave), reps=100)
    print(f"    n={nn:>7}   min {mn:8.4f}  med {md:8.4f} ms   ({1e6*md/max(nn,1):.3f} us/elem)")

# fixed-overhead vs arithmetic split: linear fit t = a + b*n over the small-n regime
ns = np.array([1, 10, 100, 500, 1000, len(qrep)])
ts = []
for nn in ns:
    qq = qrep[:nn]
    ts.append(bench(lambda: project_q(qq, panels, clen, wave), reps=100)[1])
ts = np.array(ts)
Adm = np.stack([np.ones_like(ns, float), ns.astype(float)], 1)
coef, *_ = np.linalg.lstsq(Adm, ts, rcond=None)
print(f"  linear fit  t(ms) = {coef[0]:.4f} + {coef[1]*1e3:.6f}e-3 * n")
nrep = len(qrep)
print(f"  at n={nrep}: fixed overhead {coef[0]:.4f} ms ({100*coef[0]/(coef[0]+coef[1]*nrep):.1f}%), "
      f"arithmetic {coef[1]*nrep:.4f} ms ({100*coef[1]*nrep/(coef[0]+coef[1]*nrep):.1f}%)")

# count numpy array ops actually executed
print("\n  numpy array-op count in project_q:")
print("    prologue (once):  asarray, wave*q, +Z_HAT, norm, divide, 3x np.full  = 8 ops")
print("    per panel:        2x np.where + divide (t), X=s_hat*t[:,None],")
print("                      res*X[:,:2], -corner, rhs@inv(A).T, f=+lfls[:,0], s=+lfls[:,1],")
print("                      4 comparisons + 4 logical-and (on), 1 and (take),")
print("                      3 boolean gathers + 3 boolean scatters            = ~25 ops")
print(f"    npanels={len(panels)}  =>  ~{8 + 25*len(panels)} numpy calls per project_q")

# empirical per-call overhead of a bare numpy op at this size
xx = np.empty(nrep)
mn, md = bench(lambda: xx * 2.0, reps=500)
print(f"  bare numpy op (x*2.0) at n={nrep}: med {md*1e3:.3f} us -> "
      f"~{(8+25*len(panels))*md:.4f} ms for {8+25*len(panels)} such ops")
print("\nDONE")
