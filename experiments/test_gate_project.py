"""Guard the fused gate+project path in HKLGrid.predict.

The Ewald+qmax gate is panel-INDEPENDENT, so a node owning a few ASICs of a ~25-panel detector paid
the full host tail (D2H, argsort, gather, project_q, struct assembly) for reflections that then miss
its detector -- 195 survivors for 94 real on-panel reflections. The gate kernel now does the detector
projection itself, so the cull is EXACT and project_q leaves the host entirely.

THE CONTRACT IS WEAKER THAN BIT-IDENTICAL, deliberately, and this test pins exactly how much weaker.
The kernel and numpy evaluate the same expressions in a different order, so:
  * length, and the INTEGER fields h, k, l, panel      -> must be EXACT
  * the float fields fs, ss, exc, res                  -> agree to ~1e-13
1e-13 on a pixel coordinate of order 500 is sub-femtopixel, far below anything that places an
integration box. A reflection within 1e-13 of a panel edge could in principle flip which side it
lands on; the measured rate is 0 over thousands, and the geometric probability is ~1e-16 each.

Also checks the guard: a caller passing DIFFERENT panels than the grid was built for must fall back
to the plain gate + host project_q rather than silently projecting onto the wrong detector.

  python test_gate_project.py       # needs a GPU
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from glint.lattice import cell_to_Ar
from glint.stream_driver import HKLGrid, _panel_geom
from glint.predict import recip_from_M

LAM, CLEN, RES = 1.322, 0.1, 10000.0
LYSO = cell_to_Ar(79.02, 79.02, 37.98, 90, 90, 90)
fails = []


def check(name, cond, msg=""):
    print(f"  {name:44s}: {'PASS' if cond else 'FAIL'}   {msg}")
    if not cond:
        fails.append(name)


def panels(nfs, nss, n=1, gap=0):
    """n panels tiled along fs, each nfs x nss, centred on the beam as a block."""
    out = []
    W = n * nfs + (n - 1) * gap
    for i in range(n):
        x0 = -W / 2.0 + i * (nfs + gap)
        out.append(dict(name=f"p{i}", fs=np.array([1.0, 0, 0]), ss=np.array([0, 1.0, 0]), res=RES,
                        cx=x0, cy=-nss / 2.0, coffset=0.0,
                        min_fs=0, max_fs=nfs - 1, min_ss=0, max_ss=nss - 1))
    return out


def rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


EXACT = ("h", "k", "l", "panel")        # must match bit-for-bit
NEAR = ("fs", "ss", "exc", "res")        # order-of-evaluation differs; 1e-13 tolerated
FTOL = 1e-9                              # 4 orders of margin over the ~1e-13 observed


def same(a, b):
    """predict() outputs equivalent: integer fields exact, float fields within FTOL."""
    if len(a) != len(b):
        return False, f"len {len(a)} vs {len(b)}", 0.0
    for f in EXACT:
        if not np.array_equal(a[f], b[f]):
            return False, f"integer field {f!r} differs in {int(np.sum(a[f] != b[f]))} rows", 0.0
    d = 0.0
    for f in NEAR:
        if len(a):
            d = max(d, float(np.max(np.abs(a[f].astype(float) - b[f].astype(float)))))
    if d > FTOL:
        return False, f"float fields differ by {d:.3e} > {FTOL:.0e}", d
    return True, f"{len(a)} reflections, integer fields exact, floats <= {d:.2e}", d


def main():
    rng = np.random.default_rng(5)
    Rs = [recip_from_M(rot(rng) @ LYSO) for _ in range(60)]

    for label, pan, tol in (("6-ASIC DRP panel", panels(576, 336), 0.002),
                            ("6-ASIC, tol=0.006", panels(576, 336), 0.006),
                            ("1 ASIC", panels(192, 168), 0.002),
                            ("3 separated panels", panels(192, 168, n=3, gap=400), 0.002),
                            ("whole 1024^2", panels(1024, 1024), 0.002)):
        ref = HKLGrid(LYSO, 2.0, gpu=True)                                  # plain gate + host project_q
        fus = HKLGrid(LYSO, 2.0, gpu=True, panels=pan, clen_m=CLEN, wavelength_A=LAM)
        ref.predict(Rs[0], pan, CLEN, LAM, tol=tol, is_recip=True)          # warm up
        fus.predict(Rs[0], pan, CLEN, LAM, tol=tol, is_recip=True)

        ok, msg, dmax, t_r, t_f = True, "", 0.0, 0.0, 0.0
        nr, nf = [], []
        for R in Rs:
            t0 = time.perf_counter(); a = ref.predict(R, pan, CLEN, LAM, tol=tol, is_recip=True)
            t_r += time.perf_counter() - t0
            nr.append(int(ref._gcnt[0]))
            t0 = time.perf_counter(); b = fus.predict(R, pan, CLEN, LAM, tol=tol, is_recip=True)
            t_f += time.perf_counter() - t0
            nf.append(int(fus._gcnt[0]))
            eq, m, d = same(a, b)
            dmax = max(dmax, d)
            if not eq:
                ok = False; msg = m; break
        k = len(Rs)
        print(f"\n--- {label} (tol={tol}) ---")
        print(f"  reaching the host: {np.median(nr):.0f} -> {np.median(nf):.0f}"
              f"   ({100*(1-np.median(nf)/max(np.median(nr),1)):.1f}% fewer)")
        print(f"  predict: {1e3*t_r/k:.3f} -> {1e3*t_f/k:.3f} ms/frame   ({t_r/max(t_f,1e-12):.2f}x)")
        check(f"{label}: equivalent", ok, msg if not ok else f"over {k} orientations, max float diff {dmax:.2e}")

    # ---- guard: different panels than the grid was built for must NOT be projected onto ---------
    pan, other = panels(576, 336), panels(1024, 1024)
    ref = HKLGrid(LYSO, 2.0, gpu=True)
    g = HKLGrid(LYSO, 2.0, gpu=True, panels=pan, clen_m=CLEN, wavelength_A=LAM)
    ref.predict(Rs[0], other, CLEN, LAM, tol=0.002, is_recip=True)
    g.predict(Rs[0], other, CLEN, LAM, tol=0.002, is_recip=True)
    bad = 0
    for R in Rs[:20]:
        x = ref.predict(R, other, CLEN, LAM, tol=0.002, is_recip=True)
        y = g.predict(R, other, CLEN, LAM, tol=0.002, is_recip=True)
        if not same(x, y)[0]:
            bad += 1
    check("mismatched panels fall back safely", bad == 0,
          f"{bad}/20 differ -- must use the plain gate, not project onto the wrong detector")

    check("_panel_geom refuses a panel at/behind the sample",
          _panel_geom([dict(panels(192, 168)[0], coffset=-CLEN)], CLEN) is None)
    check("_panel_geom refuses a degenerate fs/ss basis",
          _panel_geom([dict(panels(192, 168)[0], ss=np.array([1.0, 0, 0]))], CLEN) is None)

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
