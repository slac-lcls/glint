"""Guard the panel-acceptance pre-filter in HKLGrid.predict.

The Ewald+qmax gate is panel-INDEPENDENT, so a node owning a few ASICs of a ~25-panel detector pays
the full host tail (D2H, argsort, gather, project_q, struct assembly) for reflections that then miss
its detector -- measured 99 of 741 kept on a 6-ASIC panel. _panel_cone adds a linear half-space test
in the gate kernel so those are never emitted.

The ONLY thing that makes this safe is that the cone is a NECESSARY condition, never sufficient:
project_q remains the exact on-panel arbiter, so the returned reflections must be BIT-IDENTICAL to
the un-culled path. That is what this checks, on many orientations and several geometries -- a
speedup that changes the output is not a speedup. Also checks the guards: mismatched panels, a
tol wider than the cone was built for, and geometries where the cone should refuse to form.

  python test_panel_cone.py       # needs a GPU
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from glint.lattice import cell_to_Ar
from glint.stream_driver import HKLGrid, _panel_cone
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


def same(a, b):
    """Structured predict() outputs identical, field by field."""
    if len(a) != len(b):
        return False, f"len {len(a)} vs {len(b)}"
    for f in a.dtype.names:
        if not np.array_equal(a[f], b[f]):
            bad = int(np.sum(a[f] != b[f]))
            return False, f"field {f!r} differs in {bad} rows"
    return True, f"{len(a)} reflections identical across {len(a.dtype.names)} fields"


def main():
    rng = np.random.default_rng(5)
    Rs = [recip_from_M(rot(rng) @ LYSO) for _ in range(60)]

    for label, pan, tol in (("6-ASIC DRP panel", panels(576, 336), 0.002),
                            ("6-ASIC, tol=0.006", panels(576, 336), 0.006),
                            ("1 ASIC", panels(192, 168), 0.002),
                            ("3 separated panels", panels(192, 168, n=3, gap=400), 0.002),
                            ("whole 1024^2", panels(1024, 1024), 0.002)):
        ref = HKLGrid(LYSO, 2.0, gpu=True)                                   # no cone
        cul = HKLGrid(LYSO, 2.0, gpu=True, panels=pan, clen_m=CLEN,
                      wavelength_A=LAM, cone_tol=max(tol, 0.006))
        ref.predict(Rs[0], pan, CLEN, LAM, tol=tol, is_recip=True)           # warm up
        cul.predict(Rs[0], pan, CLEN, LAM, tol=tol, is_recip=True)

        ok, nsurv_r, nsurv_c, t_r, t_c, msg = True, [], [], 0.0, 0.0, ""
        for R in Rs:
            t0 = time.perf_counter(); a = ref.predict(R, pan, CLEN, LAM, tol=tol, is_recip=True)
            t_r += time.perf_counter() - t0
            t0 = time.perf_counter(); b = cul.predict(R, pan, CLEN, LAM, tol=tol, is_recip=True)
            t_c += time.perf_counter() - t0
            eq, m = same(a, b)
            if not eq:
                ok = False; msg = m; break
            nsurv_r.append(int(ref._gcnt[0])); nsurv_c.append(int(cul._gcnt[0]))
        cone = cul._cone
        half = np.degrees(np.arccos(np.clip(np.cos(0.0), -1, 1))) if cone is None else None
        k = len(Rs)
        cull = 1.0 - np.median(nsurv_c) / max(np.median(nsurv_r), 1) if nsurv_r else 0.0
        print(f"\n--- {label} (tol={tol}) ---")
        print(f"  cone: {'none' if cone is None else 'axis %s kmin %.5f' % (np.round(cone[0],4), cone[1])}")
        print(f"  gate survivors: {np.median(nsurv_r):.0f} -> {np.median(nsurv_c):.0f}"
              f"   ({100*cull:.1f}% culled)")
        print(f"  predict: {1e3*t_r/k:.3f} -> {1e3*t_c/k:.3f} ms/frame"
              f"   ({t_r/max(t_c,1e-12):.2f}x)")
        check(f"{label}: output identical", ok, msg if not ok else f"over {k} orientations")

    # ---- guards -------------------------------------------------------------------------------
    pan = panels(576, 336)
    other = panels(1024, 1024)
    g = HKLGrid(LYSO, 2.0, gpu=True, panels=pan, clen_m=CLEN, wavelength_A=LAM)
    ref = HKLGrid(LYSO, 2.0, gpu=True)
    a = ref.predict(Rs[0], other, CLEN, LAM, tol=0.002, is_recip=True)
    b = g.predict(Rs[0], other, CLEN, LAM, tol=0.002, is_recip=True)   # DIFFERENT panels -> no cone
    ok, m = same(a, b)
    check("mismatched panels fall back to no culling", ok, m)

    # a cone built at tol=0.006 must still be safe when predict is called with a SMALLER tol
    c6 = HKLGrid(LYSO, 2.0, gpu=True, panels=pan, clen_m=CLEN, wavelength_A=LAM, cone_tol=0.006)
    bad = 0
    for R in Rs[:20]:
        x = ref.predict(R, pan, CLEN, LAM, tol=0.001, is_recip=True)
        y = c6.predict(R, pan, CLEN, LAM, tol=0.001, is_recip=True)
        if not same(x, y)[0]:
            bad += 1
    check("cone built at larger tol is safe at smaller", bad == 0, f"{bad}/20 mismatched")

    # A FLAT panel at positive Zp always subtends less than a hemisphere, so the theta >= pi/2
    # early-out is unreachable for real geometry -- it is defensive only. The property that actually
    # matters for an enormous panel is GRACEFUL DEGRADATION: the cone becomes useless (culls nothing)
    # rather than wrong (culls something real).
    huge = panels(200000, 200000)
    cone = _panel_cone(huge, CLEN, LAM, 0.006)
    gh = HKLGrid(LYSO, 2.0, gpu=True, panels=huge, clen_m=CLEN, wavelength_A=LAM)
    gh.predict(Rs[0], huge, CLEN, LAM, tol=0.002, is_recip=True)
    ref.predict(Rs[0], huge, CLEN, LAM, tol=0.002, is_recip=True)
    bad, culled = 0, []
    for R in Rs[:20]:
        x = ref.predict(R, huge, CLEN, LAM, tol=0.002, is_recip=True); nr = int(ref._gcnt[0])
        y = gh.predict(R, huge, CLEN, LAM, tol=0.002, is_recip=True); nc = int(gh._gcnt[0])
        culled.append(1.0 - nc / max(nr, 1))
        if not same(x, y)[0]:
            bad += 1
    check("hemisphere-wide panel degrades, not breaks", bad == 0 and np.median(culled) < 0.02,
          f"{bad}/20 mismatched, {100*np.median(culled):.1f}% culled "
          f"(cone half-angle {np.degrees(np.arccos(np.clip(cone[0][2],-1,1))) if cone else float('nan'):.1f} deg axis-tilt)")

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
