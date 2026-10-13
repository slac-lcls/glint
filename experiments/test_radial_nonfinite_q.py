#!/usr/bin/env python3
"""RadialIntegrator: a non-finite q is a masked pixel, and a finite q-map builds the same M bit for bit.

THE BUG. `_linear` cast floor(t) to int64 BEFORE testing that t was finite. Casting NaN to an integer is
undefined: numpy on arm64 gives 0, which is a real bin, so a NaN q at explicit qmin/qmax put its two weights
(1 - frac and frac, both NaN) into M and two bins of I(q) were NaN on every frame. Where the cast lands is the
platform's choice -- CuPy's lands out of range, and the same code is clean there by accident -- so this test
checks the CAST too (part C), not only where it happened to land. `_area` cast in the same place; an infinite
q there gave its four gradient neighbours an infinite half-width, whose interval clip() stretched over every
bin: 780 extra (zero-weight) stored entries from one pixel on the map below. The fix gates both before the
cast.

  A  a NaN / +inf / -inf q at explicit qmin/qmax builds EXACTLY the M of that pixel masked (linear) or of its
     finite-difference stencil masked (area: the pixel and its four neighbours, whose half-width it feeds);
     no NaN weight reaches M, no live bin goes empty, I(q) is finite wherever the clean map's is.
  B  ...and the counts on the ring scene below (NaN weights, extra stored entries).
  C  no non-finite value reaches a float->int cast while M is built (numpy's "invalid value encountered
     in cast"). First shows that this interpreter does warn on such a cast; SKIPs if it does not.
  D  a finite q-map builds the same M, den and I(q) as d21de4c, bit for bit: both splits, default and
     explicit range, with and without a mask, gradient and given half-widths. The pre-fix builder is
     frozen below, as test_peakfinder_mask.py part D freezes the finders' pre-fix arithmetic.

d21de4c fails 12 checks of A, B and C on arm64 (numpy 1.26). On a platform whose cast lands out of range the
linear NaN checks of A and B would pass there by accident; C does not depend on where the cast lands.
Plain script: prints ok/FAIL lines, exits 1 on any failure. numpy + scipy.

    PYTHONPATH=. python experiments/test_radial_nonfinite_q.py
"""
import os
import sys
import warnings

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.radial import RadialIntegrator  # noqa: E402

FAILS = []


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


def same_csr(a, b):
    """Two CSR matrices identical in structure AND stored values, bit for bit (explicit zeros included)."""
    return (a.shape == b.shape and a.indptr.tobytes() == b.indptr.tobytes()
            and a.indices.tobytes() == b.indices.tobytes() and a.data.tobytes() == b.data.tobytes())


def same_bytes(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def scene(N=128, rings=(0.30, 0.55, 0.812, 1.05), sig=0.006):
    """q = |r| * 2/N, a few Gaussian rings."""
    yy, xx = np.mgrid[-N // 2:N // 2, -N // 2:N // 2]
    q = np.sqrt(xx ** 2 + yy ** 2) * (2.0 / N)
    img = np.zeros((N, N))
    for r in rings:
        img += np.exp(-((q - r) ** 2) / (2 * sig ** 2))
    return q, img


Q, IMG = scene()
BAD = (5, 5)                                                     # a pixel near the corner
STENCIL = [(5, 5), (4, 5), (6, 5), (5, 4), (5, 6)]              # the pixel + its four gradient neighbours
KW = dict(nbin=200, qmin=0.0, qmax=1.4)
VALUES = {"NaN": np.nan, "+inf": np.inf, "-inf": -np.inf}


def masked(pixels):
    m = np.ones(Q.shape, bool)
    for p in pixels:
        m[p] = False
    return m


def part_a():
    for split in ("linear", "area"):
        ref = RadialIntegrator(Q, split=split, **KW)
        live = np.isfinite(ref.integrate(IMG)[1])
        cover = [BAD] if split == "linear" else STENCIL
        want = RadialIntegrator(Q, split=split, mask=masked(cover), **KW)
        for tag, v in VALUES.items():
            bad = Q.copy(); bad[BAD] = v
            ri = RadialIntegrator(bad, split=split, **KW)
            check(f"A {split}: q = {tag} builds the M of {'that pixel' if split == 'linear' else 'its stencil'}"
                  f" masked", same_csr(ri.M, want.M),
                  f"nnz {ri.M.nnz} vs {want.M.nnz}, {int(np.isnan(ri.M.data).sum())} NaN weights")
            check(f"A {split}: q = {tag}, no NaN weight in M", not np.isnan(ri.M.data).any(),
                  int(np.isnan(ri.M.data).sum()))
            emptied = int((~np.isfinite(ri.den) & np.isfinite(ref.den)).sum())
            check(f"A {split}: q = {tag} empties no live bin", emptied == 0, emptied)
            I = ri.integrate(IMG)[1]
            check(f"A {split}: q = {tag}, I(q) finite on every live bin", bool(np.isfinite(I[live]).all()),
                  int((~np.isfinite(I[live])).sum()))


def part_b():
    """The counts the bug left on this scene, measured on d21de4c (arm64): 2 NaN weights (linear, NaN);
    780 extra stored entries (area, +inf)."""
    nan_q = Q.copy(); nan_q[BAD] = np.nan
    lin = RadialIntegrator(nan_q, split="linear", **KW)
    check("B linear, NaN q: 0 NaN weights in M", int(np.isnan(lin.M.data).sum()) == 0, int(np.isnan(lin.M.data).sum()))
    inf_q = Q.copy(); inf_q[BAD] = np.inf
    area, ref = RadialIntegrator(inf_q, split="area", **KW), RadialIntegrator(Q, split="area", **KW)
    check("B area, +inf q: no stored entries added", area.M.nnz <= ref.M.nnz, f"+{area.M.nnz - ref.M.nnz}")
    lost = int((np.diff(ref.M.tocsc().indptr) > 0).sum() - (np.diff(area.M.tocsc().indptr) > 0).sum())
    check("B area, +inf q: costs at most its 5-pixel stencil", 0 <= lost <= len(STENCIL), lost)


def _cast_warnings(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with np.errstate(invalid="warn"):
            fn()
    return [str(x.message) for x in w if "cast" in str(x.message)]


def part_c():
    probe = _cast_warnings(lambda: np.floor(np.array([np.nan, np.inf])).astype(np.int64))
    if not probe:
        print("  SKIP  C: this numpy does not warn on a NaN->int64 cast, so the check could not fail here")
        return
    check("C the interpreter warns on a NaN->int64 cast (the check below can fail)", True)
    for split in ("linear", "area"):
        for tag, v in VALUES.items():
            bad = Q.copy(); bad[BAD] = v
            got = _cast_warnings(lambda: RadialIntegrator(bad, split=split, **KW))
            check(f"C {split}: q = {tag} never reaches a float->int cast", not got, got)


# ---------------------------------------------------------------- D: the d21de4c builder, frozen
def _prefix_operator(q_per_pixel, nbin=None, qmin=None, qmax=None, mask=None, norm=None,
                     split="linear", q_halfwidth=None):
    """glint/radial.py RadialIntegrator.__init__ + _linear + _area as they stood at d21de4c (numpy only).
    Returns (M, den, q-centres). Do not edit: it is the reference the finite-q promise is held to."""
    xp = np
    q2d = q_per_pixel
    q = q2d.ravel().astype(xp.float64); npix = q.size
    qmin = float(q.min() if qmin is None else qmin)
    qmax = float(q.max() if qmax is None else qmax)
    NB = int(nbin if nbin is not None else np.sqrt(npix) / 2)
    dq = (qmax - qmin) / NB
    qc = qmin + (xp.arange(NB) + 0.5) * dq
    okmask = None if mask is None else mask.ravel().astype(bool)
    if split == "linear":
        t = (q - qmin) / dq - 0.5
        b0 = xp.floor(t).astype(xp.int64); frac = t - b0
        ok = xp.ones(q.size, bool) if okmask is None else okmask.copy()
        ok0 = ok & (b0 >= 0) & (b0 < NB); ok1 = ok & (b0 + 1 >= 0) & (b0 + 1 < NB)
        rows = xp.concatenate([xp.where(ok0, b0, 0), xp.where(ok1, b0 + 1, 0)])
        cols = xp.concatenate([xp.arange(q.size), xp.arange(q.size)])
        vals = xp.concatenate([xp.where(ok0, 1.0 - frac, 0.0), xp.where(ok1, frac, 0.0)])
    else:
        if q_halfwidth is not None:
            hw = q_halfwidth.ravel().astype(xp.float64)
        else:
            g0, g1 = xp.gradient(q2d.astype(xp.float64))
            hw = 0.5 * (xp.abs(g0) + xp.abs(g1)).ravel()
        hw = xp.maximum(hw, dq * 1e-4)
        qlo = xp.clip(q - hw, qmin, qmax)
        qhi = xp.clip(q + hw, qmin, qmax)
        jlo = xp.clip(xp.floor((qlo - qmin) / dq).astype(xp.int64), 0, NB - 1)
        jhi = xp.clip(xp.floor((qhi - qmin) / dq).astype(xp.int64), 0, NB - 1)
        ok = xp.ones(q.size, bool) if okmask is None else okmask.copy()
        ok &= (qhi > qlo)
        counts = xp.where(ok, jhi - jlo + 1, 0).astype(xp.int64)
        total = int(counts.sum())
        csum = xp.cumsum(counts); ranges = xp.arange(total)
        cols = xp.searchsorted(csum, ranges, side="right")
        within = ranges - (csum - counts)[cols]
        rows = jlo[cols] + within
        elo = qmin + rows * dq; ehi = elo + dq
        ov = xp.clip(xp.minimum(qhi[cols], ehi) - xp.maximum(qlo[cols], elo), 0.0, None)
        vals = ov / (2.0 * hw)[cols]
    M = sp.coo_matrix((vals, (rows, cols)), shape=(NB, npix)).tocsr()
    n = xp.ones(npix, xp.float64) if norm is None else norm.ravel().astype(xp.float64)
    den = M @ n
    den = xp.where(den == 0, xp.nan, den)
    return M, den, qc


def part_d():
    rng = np.random.default_rng(231)
    H, W = 96, 112                                               # not square, so a transpose would show
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    tilted = np.hypot((yy - 40.3) * 1.07, xx - 61.8 + 0.2 * yy) * 0.013 + 0.02
    MASK = rng.random((H, W)) > 0.05
    # ...and one that masks the lowest- and highest-q pixels, so a default range taken over the KEPT pixels
    # instead of all of them (a tempting "nan-safe" variant) would move qmin/qmax and show up here.
    EXT = MASK.copy()
    EXT[np.unravel_index(int(np.argmin(tilted)), tilted.shape)] = False
    EXT[np.unravel_index(int(np.argmax(tilted)), tilted.shape)] = False
    HW = np.full((H, W), 0.004) + 0.002 * rng.random((H, W))
    NORM = 0.5 + rng.random((H, W))
    img = rng.poisson(50.0, (H, W)).astype(np.float32)
    cases = []
    for split in ("linear", "area"):
        for rng_kw in ({}, dict(qmin=0.1, qmax=1.0), dict(nbin=77)):   # default, a range that cuts pixels off
            for mask in (None, MASK):
                cases.append((split, rng_kw, mask, None, None))
        cases.append((split, {}, EXT, None, None))
        cases.append((split, dict(nbin=64), MASK, None, NORM))
    cases.append(("area", dict(nbin=90), MASK, HW, None))
    cases.append(("area", dict(nbin=90, qmin=0.3, qmax=1.2), None, HW, NORM))
    for split, kw, mask, hw, norm in cases:
        mtag = "no" if mask is None else ("q extrema" if mask is EXT else "yes")
        tag = f"{split}, {kw or 'default range'}, mask={mtag}" \
              f"{', given hw' if hw is not None else ''}{', norm' if norm is not None else ''}"
        ri = RadialIntegrator(tilted, mask=mask, norm=norm, split=split, q_halfwidth=hw, **kw)
        M0, den0, qc0 = _prefix_operator(tilted, mask=mask, norm=norm, split=split, q_halfwidth=hw, **kw)
        I0 = (M0 @ img.ravel().astype(np.float64)) / den0
        q1, I1 = ri.integrate(img)
        check(f"D {tag}: M, den, q, I(q) identical to d21de4c",
              same_csr(ri.M, M0) and same_bytes(ri.den, den0) and same_bytes(q1, qc0) and same_bytes(I1, I0),
              f"M {same_csr(ri.M, M0)}, den {same_bytes(ri.den, den0)}, q {same_bytes(q1, qc0)}, "
              f"I {same_bytes(I1, I0)}")
    # ...and the comparator can fail: one stored weight moved by one ulp is caught.
    M0, _, _ = _prefix_operator(tilted, split="linear", nbin=64)
    M1 = M0.copy(); k = int(np.flatnonzero(M1.data)[len(M1.data) // 2]); M1.data[k] = np.nextafter(M1.data[k], 2.0)
    check("D the bit-for-bit comparator catches a one-ulp change in one weight", not same_csr(M0, M1))


for nm, fn in (("A", part_a), ("B", part_b), ("C", part_c), ("D", part_d)):
    print(f"\n{nm}")
    section(nm, fn)

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
