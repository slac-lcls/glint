#!/usr/bin/env python3
"""PeakFinder8 / RadialIntegrator with a non-finite q, on the GPU (cupy): what test_pf8_nonfinite_q.py checks
on numpy, here on every GPU path the finder has.

The numpy tests cannot reach the GPU code: the default range's finite-q reductions and boolean index on a cupy
array, and pf8's GPU-only paths -- the fused backward kernel (fp64 and fp32), the custom fp32 forward, and the
CUDA-graph replay of the fp32 loop. Each path is built on a q-map with one NaN / +inf / -inf pixel (on the
background and on a peak) and must give the peaks of that pixel masked with its q finite, on the same path:
the same count and npix, centroids, intensities and SNRs within float noise (TOL; GPU sums are not
order-fixed, so bit equality is not expected and is reported, not required). Which paths actually ran is printed and checked, so a geometry that silently falls back cannot pass as
coverage. A q-map with no finite value (a dead panel) must build and find no peaks on every path. fp64 GPU
peaks must match numpy's (same count, centroids within 1e-6 px), and the GPU default range must equal
numpy's.

SKIPS, exit 0, without cupy or a usable GPU (CPU CI). Run it on a GPU node:
  PYTHONPATH=. python experiments/test_pf8_nonfinite_q_gpu.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    cp.zeros(1).sum().item()
except Exception as e:                                   # no cupy, no driver, no device
    print(f"SKIP {os.path.basename(__file__)} -- no usable cupy GPU ({type(e).__name__})")
    sys.exit(0)
import numpy as np
from glint.peakfinder8 import PeakFinder8
from glint.radial import RadialIntegrator

FAILS = []
KEYS = ("x", "y", "intensity", "snr", "npix")


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def host(pk):
    return {k: cp.asnumpy(pk[k]) if isinstance(pk[k], cp.ndarray) else np.asarray(pk[k]) for k in KEYS}


def bitsame(a, b):
    a, b = host(a), host(b)
    return all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes()
               for k in KEYS)


def _sorted(p):
    o = np.lexsort((p["x"], p["y"]))
    return {k: p[k][o] for k in KEYS}


TOL = {"dxy": 1e-4, "dI": 1e-4, "dsnr": 1e-3}   # px, relative, relative


def close(a, b):
    """Same peaks within TOL: count and npix exact, centroids/intensity/SNR within float noise. The GPU sums
    with atomics and cuSPARSE, whose order is not fixed, so two operators that differ only in one masked
    pixel's (empty) column need not agree bit for bit. A real masking error moves the result far past TOL:
    on d21de4c a NaN q gives 0 peaks and +inf hundreds."""
    a, b = _sorted(host(a)), _sorted(host(b))
    if len(a["x"]) != len(b["x"]) or not np.array_equal(a["npix"], b["npix"]):
        return False, f"{len(a['x'])} vs {len(b['x'])} peaks / npix differ"
    if not len(a["x"]):
        return True, "no peaks"
    dxy = max(np.abs(a["x"] - b["x"]).max(), np.abs(a["y"] - b["y"]).max())
    rel = lambda k: float((np.abs(a[k] - b[k]) / np.maximum(np.abs(b[k]), 1e-30)).max())
    d = {"dxy": float(dxy), "dI": rel("intensity"), "dsnr": rel("snr")}
    return all(d[k] <= TOL[k] for k in TOL), ", ".join(f"{k} {v:.1e}" for k, v in d.items())


rng = np.random.default_rng(7)
N = 160
yy, xx = np.mgrid[0:N, 0:N].astype(float)
Q = np.hypot(yy - 80.3, xx - 79.6)                    # beam off the pixel grid
lam = np.full((N, N), 4.0) + 3.0 * np.exp(-((Q - 50.0) ** 2) / 8.0)
PEAKS = [(30, 30), (30, 120), (120, 30), (120, 120), (52, 98)]
for (py, px) in PEAKS:
    lam += 400.0 * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / 1.5)
IMG = rng.poisson(lam).astype(np.float32)
MASK = np.ones((N, N), bool); MASK[:, 70:72] = False  # a caller's mask: a dead column pair
BG_PX, PEAK_PX = (100, 45), PEAKS[0]
VALUES = {"NaN": np.nan, "+inf": np.inf, "-inf": -np.inf}
CONFIGS = {                                           # name -> PeakFinder8 kwargs, and the path flags it must reach
    "fp64 (fused backward)": (dict(), ("_fused",)),
    "fp32 (fused + custom forward)": (dict(dtype=cp.float32), ("_fused", "_fwd_fast")),
    "fp32 CUDA graph": (dict(dtype=cp.float32, graph=True), ("_fused", "_fwd_fast", "_graph_want")),
}


def with_q(px, v):
    q = Q.copy(); q[px] = v
    return q


def masked(mask, px):
    m = np.ones((N, N), bool) if mask is None else mask.copy(); m[px] = False
    return m


def g(a):
    return None if a is None else cp.asarray(a)


IMG_D = cp.asarray(IMG)
print(f"device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, cupy {cp.__version__}")

print("\nA  RadialIntegrator on cupy: the default range is the finite q's")
for tag, v in VALUES.items():
    for px, ptag in ((BG_PX, "background"), (tuple(int(i) for i in np.unravel_index(np.argmax(Q), Q.shape)), "max-q")):
        rg, rc = RadialIntegrator(cp.asarray(with_q(px, v))), RadialIntegrator(with_q(px, v))
        check(f"A q = {tag} at the {ptag} pixel: GPU qmin/qmax == numpy's",
              (rg.qmin, rg.qmax) == (rc.qmin, rc.qmax) and np.isfinite([rg.qmin, rg.qmax]).all(),
              ((rg.qmin, rg.qmax), (rc.qmin, rc.qmax)))
try:
    RadialIntegrator(cp.full((8, 8), cp.nan)); msg = None
except ValueError as e:
    msg = str(e)
check("A an all-NaN cupy q-map without qmin/qmax raises the module's ValueError", msg is not None and "no finite value" in msg, msg)

for cname, (kw, flags) in CONFIGS.items():
    print(f"\nB  {cname}")
    probe = PeakFinder8(cp.asarray(Q), **kw)
    probe.find(IMG_D)
    reached = {f: bool(getattr(probe, f, False)) for f in flags}
    check(f"B {cname}: the path ran ({', '.join(f'{f}={v}' for f, v in reached.items())})", all(reached.values()), reached)
    twin = PeakFinder8(cp.asarray(Q), **kw); twin.find(IMG_D)
    p1, p2 = probe.find(IMG_D), twin.find(IMG_D)
    print(f"  INFO  {cname}: two builds of the SAME finder bit-identical: {bitsame(p1, p2)} ({close(p1, p2)[1]})")
    worst = {}
    for mask, mtag in ((None, "no mask"), (MASK, "caller mask")):
        for px, ptag in ((BG_PX, "background px"), (PEAK_PX, "peak px")):
            ref = PeakFinder8(cp.asarray(Q), mask=g(masked(mask, px)), **kw)
            ref.find(IMG_D); ref_pk = ref.find(IMG_D)           # second call: a graph finder REPLAYS
            for tag, v in VALUES.items():
                f = PeakFinder8(cp.asarray(with_q(px, v)), mask=g(mask), **kw)
                f.find(IMG_D); pk = f.find(IMG_D)
                ok, why = close(pk, ref_pk)
                check(f"B {cname}, {mtag}, q = {tag} on a {ptag} == that pixel masked (within TOL)", ok, why)
                worst[(mtag, ptag, tag)] = why
    print(f"  INFO  {cname}: per-case differences vs that pixel masked: " + "; ".join(sorted(set(worst.values())))[:400])
    for tag, v in VALUES.items():
        try:
            f = PeakFinder8(cp.full((N, N), v), **kw); f.find(IMG_D); n, err = len(host(f.find(IMG_D))["x"]), None
        except Exception as e:
            n, err = None, f"{type(e).__name__}: {e}"
        check(f"B {cname}: every q = {tag} (a dead panel) builds and finds no peaks", n == 0, err or n)

print("\nC  fp64 GPU == numpy, with a non-finite q")
for tag, v in VALUES.items():
    qb = with_q(BG_PX, v)
    gp, cpu = host(PeakFinder8(cp.asarray(qb)).find(IMG_D)), PeakFinder8(qb).find(IMG)
    same_n = len(gp["x"]) == len(cpu["x"])
    d = max(np.abs(np.sort(gp["x"]) - np.sort(cpu["x"])).max(initial=0), np.abs(np.sort(gp["y"]) - np.sort(cpu["y"])).max(initial=0)) if same_n else None
    check(f"C q = {tag}: GPU and numpy agree ({len(gp['x'])} vs {len(cpu['x'])} peaks, max centroid diff {d})",
          same_n and len(gp["x"]) >= len(PEAKS) and d < 1e-6, (len(gp["x"]), len(cpu["x"]), d))

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
