"""How many Ewald-gate survivors actually land on ONE DRP node's panel?

predict()'s host tail (D2H, argsort, gather, project_q, struct assembly) processes EVERY survivor of
the reciprocal-space gate, and only afterwards discards the ones that miss the detector. The gate is
panel-independent, so a node owning 6 of ~24 ASICs still pays for all of them. If the on-panel
fraction is small, a PANEL-AWARE enumeration (culling to the node's angular acceptance before the
host tail, rather than after) would cut that tail by 1/fraction -- which is a completely different
argument for the column method than the kernel-cost one, and targets ~90% of the stage instead of 0.9%.

  python onpanel_fraction.py
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


def panel(nfs, nss):
    return [dict(name="p0", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=RES,
                 cx=-(nfs / 2.0 - 0.5), cy=-(nss / 2.0 - 0.5), coffset=0.0,
                 min_fs=0, max_fs=nfs - 1, min_ss=0, max_ss=nss - 1)]


def rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    rng = np.random.default_rng(11)
    Rs = [recip_from_M(rot(rng) @ LYSO) for _ in range(40)]
    grid = HKLGrid(LYSO, 2.0, gpu=True)
    print(f"hkl grid: {grid.g.shape[0]}\n")
    print(f"{'panel (fs x ss)':>18s} {'Mpix':>7s} {'ASICs':>6s} {'survivors':>10s} {'on-panel':>9s} "
          f"{'frac':>7s} {'project_q ms':>13s} {'wasted ms':>10s}")

    # 192x168 per ASIC. 6 ASICs (one DRP GPU) as 3x2; the full 4 Mpix as 24 ASICs (6x4).
    for label, nfs, nss, nasic in (("1 ASIC", 192, 168, 1),
                                   ("6 ASICs = 1 DRP GPU", 576, 336, 6),
                                   ("24 ASICs = full 4Mpix", 1152, 672, 24),
                                   ("whole 4096^2", 4096, 4096, None)):
        pan = panel(nfs, nss)
        ns, non, tproj = [], [], []
        for R in Rs:
            nhkl = grid.g.shape[0]; grid._gcnt[0] = 0
            from glint.stream_driver import _GATE_KERNEL
            Rf = cp.ascontiguousarray(cp.asarray(R).ravel())
            tpb = 256; blocks = (nhkl + tpb - 1) // tpb
            _GATE_KERNEL((blocks,), (tpb,), (grid._ggr, np.int32(nhkl), Rf, np.float64(LAM),
                         np.float64(grid.qmax ** 2), np.float64(0.002),
                         grid._gout.ravel(), grid._gcnt, np.int32(nhkl)))
            n = int(grid._gcnt[0]); C = cp.asnumpy(grid._gout[:n]); q = C[:, 1:4]
            t0 = time.perf_counter(); fs, ss, pn = project_q(q, pan, CLEN, LAM)
            tproj.append(time.perf_counter() - t0)
            ns.append(n); non.append(int((pn >= 0).sum()))
        n_s, n_on = float(np.median(ns)), float(np.median(non))
        f = n_on / max(n_s, 1)
        tp = 1e3 * float(np.median(tproj))
        print(f"{label:>18s} {nfs*nss/1e6:7.3f} {str(nasic):>6s} {n_s:10.0f} {n_on:9.0f} "
              f"{100*f:6.1f}% {tp:13.3f} {tp*(1-f):10.3f}")

    print("\nwasted ms = the project_q time spent on survivors that miss this node's panel.")
    print("The same fraction is wasted in the D2H, argsort, gather and struct assembly.")


if __name__ == "__main__":
    main()
