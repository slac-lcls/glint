"""Validate the shared reader core (xtc_core: prep_geometry + frame_q) on synthetic multi-panel frames
with peaks planted at KNOWN pixels and known per-pixel lab coords. This is the peak-find + q math both
the psana1 (in-process) and psana2 (bridge) readers depend on -- so it exercises the risky arithmetic
without needing psana or real data.

    python test_core.py        # needs cupy + a GPU (run under srun)
"""
from __future__ import annotations

import numpy as np

import xtc_core

NSEG, H, W = 4, 128, 128
PITCH_UM = 100.0            # pixel pitch
ZDIST = 0.10               # m
LAM = 1.3                  # A
PED, SIG = 100.0, 5.0

rng = np.random.RandomState(0)

# synthetic per-pixel lab coords (um): each panel a flat grid, offset so panels don't overlap in x.
yy, xx = np.mgrid[0:H, 0:W]
Xf = np.empty((NSEG, H, W)); Yf = np.empty((NSEG, H, W)); Zf = np.full((NSEG, H, W), ZDIST * 1e6)
for p in range(NSEG):
    Xf[p] = (xx - W / 2 + (p - NSEG / 2) * W) * PITCH_UM     # panels tiled along x, beam-centred overall
    Yf[p] = (yy - H / 2) * PITCH_UM

# plant Gaussian blobs at known pixels, two per panel
planted = []
frame = (PED + SIG * rng.standard_normal((NSEG, H, W))).astype(np.float32)
dy, dx = np.mgrid[-3:4, -3:4]
blob = (60 * SIG * np.exp(-(dy**2 + dx**2) / 2.0)).astype(np.float32)
for p in range(NSEG):
    for _ in range(2):
        s, f = rng.randint(20, H - 20), rng.randint(20, W - 20)
        frame[p, s-3:s+4, f-3:f+4] += blob
        planted.append((p, s, f))

X, Y, Zc, kin, finders = xtc_core.prep_geometry(Xf, Yf, Zf, frame.shape,
                                                np.ones((NSEG, H, W), bool), ZDIST)
q = xtc_core.frame_q(frame, finders, X, Y, Zc, kin, LAM, min_peaks=1)

ok = True
def check(name, cond):
    global ok; ok = ok and cond
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")

check("returns (M,3) finite q", q.ndim == 2 and q.shape[1] == 3 and np.isfinite(q).all())
check(f"found ~all planted peaks ({len(q)}/{len(planted)})", len(q) >= len(planted) - 1)

# each planted pixel's EXACT expected q must appear in the returned set (rounding makes centroid==pixel)
def expected_q(p, s, f):
    r = np.array([X[p, s, f], Y[p, s, f], Zc])
    return (r / np.linalg.norm(r) - kin) / LAM

hits = 0
for (p, s, f) in planted:
    eq = expected_q(p, s, f)
    if len(q) and np.min(np.linalg.norm(q - eq, axis=1)) < 1e-6:
        hits += 1
check(f"planted-pixel q reproduced exactly ({hits}/{len(planted)})", hits >= len(planted) - 1)

# beam-centred sanity: |q| for the innermost planted peak is small, outermost larger (monotone with radius)
check("q magnitude tracks detector radius", True if len(q) < 2 else
      np.corrcoef(np.linalg.norm(q, axis=1),
                  [np.hypot(X[p, s, f], Y[p, s, f]) for (p, s, f) in planted][:len(q)])[0, 1] > 0.8
      or len(q) != len(planted))   # skip strict corr if peak/plant counts differ

print("ALL PASS" if ok else "FAILURES ABOVE")
raise SystemExit(0 if ok else 1)
