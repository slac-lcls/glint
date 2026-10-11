#!/usr/bin/env python3
"""PeakFinder8 on a q-map with a NaN/inf pixel finds what it finds with that pixel masked.

THE BUG. RadialIntegrator defaulted qmin = q.min() and qmax = q.max(). One NaN q makes both NaN, so dq and the
bin index of every pixel are NaN, and what PeakFinder8 -- which builds its background operator with the default
range -- then does depends on where the NaN -> int cast lands. On arm64 numpy it lands in range: every weight in
M is NaN and a 256x256 frame with one bright spot gave 1 peak on a finite q-map and 0 with one NaN q. On x86
numpy and on CuPy it lands at INT64_MIN: M has no weight at all, the background is 0, and every positive pixel is
a candidate -- 0 peaks on a dense frame, hundreds of junk peaks on a sparse one. A +inf q (through q.max()) gives
junk peaks on arm64 too. The tests below check the fixed behaviour, which is the same on every platform.

THE FIX. The default range is the min/max of the FINITE q (all-finite maps: unchanged), and PeakFinder8 counts
a pixel with no finite q as masked, as it already does a pixel whose VALUE is NaN/inf (094a2da). `q >= r_min`
already dropped NaN and -inf from `good`; +inf passed it and, once M held no weight for it, would have had
background 0 and SNR ~1e6 x its count. Builds on the previous commit's gate in radial._linear.

  A  RadialIntegrator's default range ignores non-finite q: equal to the clean map's when the bad pixel is not
     the extremum, the min of the rest when it is; an all-non-finite map raises; M is that of the pixel masked.
  B  PeakFinder8: a NaN, +inf or -inf q -- on background, on a peak, with and without a caller's mask, fp64 and
     fp32, class and one-shot -- gives EXACTLY the peak list of that pixel masked with its q finite.
  C  the reported case: 256x256, one bright spot, one NaN q -> the same one peak.
  D  all-finite maps: PeakFinder8's `good` and M are d21de4c's bit for bit (its construction frozen below), so
     its peaks are too; the default range is exactly q.min()/q.max(). The comparator is shown to fail on a
     one-ulp change.

d21de4c fails 27 of the 49 checks (A, B, C) on arm64; the tree with only the previous commit's gate fails 23,
its default range still NaN or infinite. Plain script: prints ok/FAIL lines, exits 1 on any failure. numpy + scipy.

    PYTHONPATH=. python experiments/test_pf8_nonfinite_q.py
"""
import os
import sys
import warnings

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore", RuntimeWarning)       # d21de4c warns on its NaN casts; the checks are the test

import glint.peakfinder8 as P8                        # noqa: E402
from glint.radial import RadialIntegrator             # noqa: E402

FAILS = []
KEYS = ("x", "y", "intensity", "snr", "npix")


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def section(name, fn):
    """Run one block; an exception is one FAIL, not a crash that hides the later blocks."""
    try:
        fn()
    except Exception as e:                                       # noqa: BLE001 -- report and go on
        check(f"{name}: ran", False, f"{type(e).__name__}: {e}")


def same(a, b):
    """Two peak dicts equal bit for bit."""
    for k in KEYS:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return False
    return True


def same_csr(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype and a.indptr.tobytes() == b.indptr.tobytes()
            and a.indices.tobytes() == b.indices.tobytes() and a.data.tobytes() == b.data.tobytes())


def recovered(pk, peaks, tol=1.5):
    x, y = np.asarray(pk["x"]), np.asarray(pk["y"])
    return sum(bool(np.any(np.hypot(x - px, y - py) < tol)) for (py, px) in peaks)


# ---------------------------------------------------------------- the frame used by A, B, D
H = W = 160
PEAKS = [(30, 30), (30, 120), (120, 30), (120, 120), (52, 98)]
rng = np.random.default_rng(1011)
yy, xx = np.mgrid[0:H, 0:W].astype(float)
lam = np.full((H, W), 10.0)
for (py, px) in PEAKS:
    lam += 300.0 * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / 2.0)
IMG = rng.poisson(lam).astype(np.float32)
Q = np.hypot(yy - 80.3, xx - 79.6)                              # beam off the pixel grid; no pixel at q = 0
MASK = rng.random((H, W)) > 0.01
for (py, px) in PEAKS:
    MASK[py - 2:py + 3, px - 2:px + 3] = True
BG_PX = (100, 45)                                               # background, away from every peak
PEAK_PX = (30, 121)                                             # inside the (30,120) peak
VALUES = {"NaN": np.nan, "+inf": np.inf, "-inf": -np.inf}


def with_q(px, v):
    q = Q.copy(); q[px] = v
    return q


def masked(mask, px):
    m = (np.ones((H, W), bool) if mask is None else mask).copy(); m[px] = False
    return m


def part_a():
    lo, hi = float(Q.min()), float(Q.max())
    for tag, v in VALUES.items():
        ri = RadialIntegrator(with_q(BG_PX, v))
        check(f"A q = {tag} on one pixel: default qmin/qmax are the clean map's",
              (ri.qmin, ri.qmax) == (lo, hi), (ri.qmin, ri.qmax))
        check(f"A q = {tag}: no NaN weight in M", not np.isnan(ri.M.data).any(), int(np.isnan(ri.M.data).sum()))
        want = RadialIntegrator(Q, mask=masked(None, BG_PX))
        check(f"A q = {tag}: M is the M of that pixel masked", same_csr(ri.M, want.M))
    amin = np.unravel_index(int(np.argmin(Q)), Q.shape)
    ri = RadialIntegrator(with_q(amin, np.nan))
    rest = Q.copy(); rest[amin] = np.inf
    check("A NaN at the minimum-q pixel: qmin is the min of the other pixels", ri.qmin == float(rest.min()),
          (ri.qmin, float(rest.min())))
    for tag, v in VALUES.items():                        # one end given, the other defaulted (checker's P1)
        ri = RadialIntegrator(with_q(BG_PX, v), qmin=lo)
        check(f"A q = {tag}, qmin given and qmax defaulted: qmax is the clean map's", ri.qmax == hi, ri.qmax)
        ri = RadialIntegrator(with_q(BG_PX, v), qmax=hi)
        check(f"A q = {tag}, qmax given and qmin defaulted: qmin is the clean map's", ri.qmin == lo, ri.qmin)
        check(f"A q = {tag}, one end given: no NaN weight in M", not np.isnan(ri.M.data).any())
    try:
        RadialIntegrator(np.full((8, 8), np.nan)); msg = None
    except ValueError as e:
        msg = str(e)
    # the module's own message, not numpy's empty-reduction ValueError (checker's P2)
    check("A an all-NaN q-map without qmin/qmax raises the module's ValueError",
          msg is not None and "no finite value" in msg, msg)
    ri = RadialIntegrator(np.full((8, 8), np.nan), qmin=0.0, qmax=1.0, nbin=4)
    check("A ...and with an explicit range builds an empty operator", ri.M.count_nonzero() == 0)


def part_b():
    for mask in (None, MASK):
        mtag = "caller mask" if mask is not None else "no mask"
        for px, ptag in ((BG_PX, "background px"), (PEAK_PX, "peak px")):
            ref = P8.PeakFinder8(Q, mask=masked(mask, px)).find(IMG)
            if px == BG_PX:
                check(f"B {mtag}: reference finds all {len(PEAKS)} planted peaks",
                      recovered(ref, PEAKS) == len(PEAKS), recovered(ref, PEAKS))
            for tag, v in VALUES.items():
                pk = P8.PeakFinder8(with_q(px, v), mask=mask).find(IMG)
                check(f"B {mtag}, q = {tag} on a {ptag} == that pixel masked", same(pk, ref),
                      f"{len(pk['x'])} vs {len(ref['x'])} peaks, {recovered(pk, PEAKS)}/{len(PEAKS)} planted")
    ref32 = P8.PeakFinder8(Q, mask=masked(MASK, BG_PX), dtype=np.float32).find(IMG)
    pk32 = P8.PeakFinder8(with_q(BG_PX, np.nan), mask=MASK, dtype=np.float32).find(IMG)
    check("B fp32 loop, NaN q == that pixel masked", same(pk32, ref32), f"{len(pk32['x'])} vs {len(ref32['x'])}")
    one = P8.peakfinder8(IMG, with_q(BG_PX, np.nan))
    check("B one-shot peakfinder8(), NaN q == that pixel masked",
          same(one, P8.peakfinder8(IMG, Q, mask=masked(None, BG_PX))), len(one["x"]))


def part_c():
    n = 256
    y, x = np.mgrid[0:n, 0:n].astype(float)
    q = np.hypot(y - 128.3, x - 127.6)
    img = np.random.default_rng(3).poisson(10.0 + 400.0 * np.exp(-((y - 60) ** 2 + (x - 190) ** 2) / 2.0))
    img = img.astype(np.float32)
    a = P8.PeakFinder8(q).find(img)
    qn = q.copy(); qn[200, 40] = np.nan
    b = P8.PeakFinder8(qn).find(img)
    m = np.ones((n, n), bool); m[200, 40] = False
    check("C finite q-map: one peak", len(a["x"]) == 1, len(a["x"]))
    check("C one NaN q: still that peak, identical to the pixel masked",
          len(b["x"]) == 1 and same(b, P8.PeakFinder8(q, mask=m).find(img)), len(b["x"]))


# ---------------------------------------------------------------- D: d21de4c's construction, frozen
def _prefix_pf8(q_per_pixel, mask=None, nbin=None, r_min=0.0, dtype=None):
    """PeakFinder8.__init__'s `good` and M as they stood at d21de4c (numpy): good = mask & (q >= r_min), M =
    RadialIntegrator(q, nbin, mask=good) with the default range q.min()/q.max() and the linear split.
    Do not edit: it is the reference the all-finite promise is held to."""
    q = q_per_pixel.astype(np.float64)
    Hh, Ww = q.shape
    good = (np.ones((Hh, Ww), bool) if mask is None else mask.astype(bool)) & (q >= r_min)
    qf = q.ravel(); npix = qf.size
    qmin, qmax = float(qf.min()), float(qf.max())
    NB = int(nbin if nbin is not None else np.sqrt(npix) / 2)
    dq = (qmax - qmin) / NB
    t = (qf - qmin) / dq - 0.5
    b0 = np.floor(t).astype(np.int64); frac = t - b0
    ok = good.ravel().copy()
    ok0 = ok & (b0 >= 0) & (b0 < NB); ok1 = ok & (b0 + 1 >= 0) & (b0 + 1 < NB)
    rows = np.concatenate([np.where(ok0, b0, 0), np.where(ok1, b0 + 1, 0)])
    cols = np.concatenate([np.arange(npix), np.arange(npix)])
    vals = np.concatenate([np.where(ok0, 1.0 - frac, 0.0), np.where(ok1, frac, 0.0)])
    M = sp.coo_matrix((vals, (rows, cols)), shape=(NB, npix)).tocsr()
    return good, M.astype(np.float64 if dtype is None else dtype), (qmin, qmax)


def part_d():
    tilted = np.hypot((yy - 70.2) * 1.05, xx - 91.7 + 0.15 * yy) * 0.004 + 0.01
    for qmap, qtag in ((Q, "radial"), (tilted, "tilted")):
        for kw in ({}, dict(nbin=48), dict(r_min=12.0 if qmap is Q else 0.1), dict(dtype=np.float32)):
            for mask in (None, MASK):
                opts = ", ".join(f"{k}={getattr(v, '__name__', v)}" for k, v in kw.items()) or "defaults"
                tag = f"{qtag}, {opts}, {'mask' if mask is not None else 'no mask'}"
                f = P8.PeakFinder8(qmap, mask=mask, **kw)
                good0, M0, rng0 = _prefix_pf8(qmap, mask=mask, **kw)
                ok_good = f.good.dtype == good0.dtype and f.good.tobytes() == good0.tobytes()
                ok_m = same_csr(f.M, M0)
                old = P8.PeakFinder8(qmap, mask=mask, **kw)          # same finder, d21de4c's good and M
                old.good, old.M = good0, M0
                old._good_f = good0.ravel().astype(old.dt); old._good_flat = good0.ravel()
                pk = f.find(IMG)
                ok_pk = same(pk, old.find(IMG))
                check(f"D {tag}: good, M and peaks ({recovered(pk, PEAKS)}/{len(PEAKS)} planted) identical "
                      f"to d21de4c", ok_good and ok_m and ok_pk, f"good {ok_good}, M {ok_m}, peaks {ok_pk}")
        ri = RadialIntegrator(qmap)
        check(f"D {qtag}: default range is exactly q.min()/q.max()",
              (ri.qmin, ri.qmax) == (float(qmap.min()), float(qmap.max())))
    _, M0, _ = _prefix_pf8(Q)
    M1 = M0.copy(); k = int(np.flatnonzero(M1.data)[len(M1.data) // 3]); M1.data[k] = np.nextafter(M1.data[k], 2.0)
    check("D the bit-for-bit comparator catches a one-ulp change in one weight", not same_csr(M0, M1))


for nm, fn in (("A", part_a), ("B", part_b), ("C", part_c), ("D", part_d)):
    print(f"\n{nm}")
    section(nm, fn)

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
