"""Is predict's cost at DRP scale dominated by FIXED per-call overhead?

The DRP gives each GPU 6 epixUHR ASICs (192x168) = 0.194 Mpix, so it sees ~1/24 the reflections of a
whole-detector frame. Prior profiling at n~740 survivors found predict = 60% host numpy / 39% CuPy
API / 0.9% GPU, with project_q fitting t = 0.0634 + 6.44e-5*n (63% fixed at n=740). If the fixed part
dominates at DRP scale, then predict does NOT shrink proportionally with the panel -- and the
transfer/API-elimination levers become MORE valuable there, not less.

Measures the real HKLGrid.predict decomposition on a true epixUHR-sized panel, and separately as a
function of survivor count, to split fixed from scaling.

  python predict_at_drp_scale.py
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
WT = "/sdf/home/s/smarches/glint_streamfix_wt"
sys.path.insert(0, WT); sys.path.insert(0, WT + "/experiments")
import cupy as cp
from glint.glint_fast import cell_to_Ar
from glint.stream_driver import HKLGrid
from glint.predict import project_q, recip_from_M

LAM, CLEN, RES = 1.322, 0.1, 10000.0
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)


def panel(nfs, nss, name="p0"):
    return [dict(name=name, fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=RES,
                 cx=-(nfs / 2.0 - 0.5), cy=-(nss / 2.0 - 0.5), coffset=0.0,
                 min_fs=0, max_fs=nfs - 1, min_ss=0, max_ss=nss - 1)]


def rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def decompose(grid, pan, Rs, tol):
    """Time predict()'s substages the way the shipped gpu branch runs them."""
    acc = dict(prep=0., kern=0., readn=0., d2h=0., sort=0., gather=0., proj=0., struct=0., whole=0.)
    n_surv = []
    for R in Rs:
        cp.cuda.Stream.null.synchronize(); t_w = time.perf_counter()
        t0 = time.perf_counter()
        nhkl = grid.g.shape[0]; grid._gcnt[0] = 0
        Rf = cp.ascontiguousarray(cp.asarray(R).ravel())
        cp.cuda.Stream.null.synchronize(); acc["prep"] += time.perf_counter() - t0
        from glint.stream_driver import _GATE_KERNEL
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        tpb = 256; blocks = (nhkl + tpb - 1) // tpb
        t0 = time.perf_counter(); ev0.record()
        _GATE_KERNEL((blocks,), (tpb,), (grid._ggr, np.int32(nhkl), Rf, np.float64(LAM),
                     np.float64(grid.qmax ** 2), np.float64(tol),
                     grid._gout.ravel(), grid._gcnt, np.int32(nhkl)))
        ev1.record(); ev1.synchronize(); acc["kern"] += 1e-3 * cp.cuda.get_elapsed_time(ev0, ev1)
        t0 = time.perf_counter(); n = int(grid._gcnt[0]); acc["readn"] += time.perf_counter() - t0
        t0 = time.perf_counter(); C = cp.asnumpy(grid._gout[:n]); acc["d2h"] += time.perf_counter() - t0
        t0 = time.perf_counter(); C = C[np.argsort(C[:, 0], kind="stable")]
        acc["sort"] += time.perf_counter() - t0
        t0 = time.perf_counter(); idx = C[:, 0].astype(np.int64); hkl = grid.g[idx]
        acc["gather"] += time.perf_counter() - t0
        q = C[:, 1:4]; qn2 = C[:, 4]; exc = C[:, 5]
        t0 = time.perf_counter(); fs, ss, pn = project_q(q, pan, CLEN, LAM)
        acc["proj"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        on = pn >= 0
        out = np.zeros(int(on.sum()), dtype=[("h", int), ("k", int), ("l", int), ("fs", float),
                                             ("ss", float), ("panel", int), ("exc", float), ("res", float)])
        acc["struct"] += time.perf_counter() - t0
        acc["whole"] += time.perf_counter() - t_w
        n_surv.append(n)
    k = len(Rs)
    return {a: 1e3 * v / k for a, v in acc.items()}, int(np.median(n_surv))


def main():
    rng = np.random.default_rng(3)
    Rs = [recip_from_M(rot(rng) @ LYSO) for _ in range(30)]
    print(f"{'panel':>16s} {'Mpix':>6s} {'n_surv':>7s} {'predict':>8s}  "
          f"{'proj':>6s} {'sort':>6s} {'d2h':>6s} {'readn':>6s} {'prep':>6s} {'kern':>7s} {'FIXED%':>7s}")
    for label, (nfs, nss), tol in (("epixUHR 6-ASIC", (576, 336), 0.002),    # 0.194 Mpix, the DRP node
                                   ("epixUHR 6-ASIC", (576, 336), 0.006),
                                   ("1024^2", (1024, 1024), 0.002),
                                   ("4096^2", (4096, 4096), 0.002)):
        pan = panel(nfs, nss)
        grid = HKLGrid(LYSO, 2.0, gpu=True)
        d, ns = decompose(grid, pan, Rs, tol)
        mpix = nfs * nss / 1e6
        # per-call fixed: everything that does not scale with survivor count
        fixed = d["prep"] + d["readn"] + d["kern"]
        print(f"{label:>16s} {mpix:6.3f} {ns:7d} {d['whole']:8.3f}  {d['proj']:6.3f} {d['sort']:6.3f} "
              f"{d['d2h']:6.3f} {d['readn']:6.3f} {d['prep']:6.3f} {d['kern']:7.4f} "
              f"{100*fixed/d['whole']:6.1f}%")

    print("\n(n_surv is survivors of the Ewald+qmax gate, BEFORE the on-panel cut; a small panel keeps")
    print(" the same grid work but discards most reflections in project_q, which is the DRP situation.)")


if __name__ == "__main__":
    main()
