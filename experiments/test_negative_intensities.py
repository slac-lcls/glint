"""Integration must keep non-positive intensities.

WHY THIS EXISTS. `integrate_cxi`/`integrate_files` used to end with

    keep = (I > 0) & np.isfinite(sig) & (sig > 0)

which drops every reflection whose measured intensity came out non-positive. That is a selection on
the measured value of the quantity being measured. A reflection whose true intensity is at or near
zero -- which is most of a predicted list, since prediction is geometric and does not know which
reflections are actually in diffracting condition -- measures negative about half the time. Keeping
the positive half and discarding the negative half biases the retained mean upward, and the bias is
largest exactly where the data are weakest.

It was not a cosmetic problem. On real data (cxil1015922 r0033, 1563 frames, glint#129) the cut
removed 39.2% of reflections; GLINT's stream had NO weak tail at all -- 10th-percentile intensity
+49.8, against -15.5 for the same frames integrated by CrystFEL, which keeps negatives. Downstream
that inflated R_split's denominator and made per-reflection I/sigma meaningless: a flat median
I/sigma of 10-17 from 5 A all the way to 2.1 A, which no real diffraction does, and which
contradicted the merge's own outer-shell CC* of 0.485.

It also hid from the statistic people check first: CC* is a correlation and barely moved
(0.9445 -> 0.9463 when the same cut was applied to the CrystFEL stream).

The properties pinned here, in the order they would hurt if broken:
  * On pure background, integration is UNBIASED -- mean I consistent with zero -- and roughly half
    the measurements are negative. This is the premise; if it fails the rest is meaningless.
  * Applying `I > 0` to that same unbiased sample produces a LARGE positive bias, many standard
    errors from zero. This is what the old line did, stated as a measurement rather than an opinion.
  * `keep` retains the negatives, so the offline path now matches the streaming path
    (stream_driver integrates and cuts by SNR at merge time, never on the sign of I).
  * Real spots still survive: the fix must not be "keep everything" at the cost of losing signal.
  * sigma > 0 and finiteness ARE still enforced -- those are guards on the estimator, not on the
    estimate, and dropping them would let a divide-by-zero into the merge.
"""
from __future__ import annotations

import os
import sys

import pathlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glint.predict import integrate_spots

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'' if ok else '  <- ' + str(detail)}")
    if not ok:
        FAILS.append(name)


def preds(fs, ss):
    out = np.zeros(len(fs), dtype=[("h", int), ("k", int), ("l", int), ("fs", float),
                                   ("ss", float), ("panel", int), ("exc", float), ("res", float)])
    out["fs"], out["ss"] = fs, ss
    return out


rng = np.random.default_rng(20260821)
H = W = 600
LAM = 6.0
frame = rng.poisson(LAM, size=(H, W)).astype(np.float32)

# a grid of positions on pure background, well inside the frame
g = np.arange(20, 580, 12)
ff, sscoord = np.meshgrid(g, g)
p_bg = preds(ff.ravel().astype(float), sscoord.ravel().astype(float))

I, sig, peak, bg = integrate_spots(frame, p_bg)
n = I.size
se = I.std(ddof=1) / np.sqrt(n)                      # standard error of the mean
frac_neg = float((I < 0).mean())

print(f"\n  background-only sample: n={n}, mean I={I.mean():+.3f}, SE={se:.3f}, "
      f"{100*frac_neg:.1f}% negative")

check("roughly half of pure-background measurements are negative",
      0.35 < frac_neg < 0.65, frac_neg)

# --- the effect this fix is about: selecting on the measured value --------------------
pos = I[I > 0]
sel_bias = pos.mean() - I.mean()
print(f"  after an I>0 cut:       n={pos.size}, mean I={pos.mean():+.3f}  "
      f"=> selection bias {sel_bias:+.3f} = {sel_bias/se:.0f} SE")

check("the old `I > 0` cut biases the retained mean upward by many SE",
      sel_bias / se > 20, sel_bias / se)
check("...and discards a large fraction of the sample", pos.size < 0.7 * n, (pos.size, n))

keep = np.isfinite(I) & np.isfinite(sig) & (sig > 0)
check("the shipped keep-rule retains the negative measurements",
      bool((I[keep] < 0).any()) and keep.sum() == n, (int(keep.sum()), n))

# sigma guards are real: a degenerate sigma must still be dropped
sig_bad = sig.copy(); sig_bad[:5] = 0.0; sig_bad[5:8] = np.nan
keep_bad = np.isfinite(I) & np.isfinite(sig_bad) & (sig_bad > 0)
check("sigma<=0 and non-finite sigma are STILL dropped (guards on the estimator kept)",
      keep_bad.sum() == n - 8, (int(keep_bad.sum()), n - 8))

# --- a SEPARATE, smaller bias, pinned here so it is not mistaken for the one above -----
# integrate_spots takes the annulus MEDIAN as the background, while the box SUM it is
# subtracted from needs the annulus MEAN. For a discrete counting distribution those differ,
# so integration is not quite unbiased even with every measurement retained. Reproducing the
# same integration with a mean background isolates it: if the mean-background version IS
# unbiased, the residual is the estimator and nothing else. Tracked separately in glint#131 --
# fixing it moves every intensity GLINT has produced, so it is not folded into this change.
R, half, gap, ring = 3 + 2 + 3, 3, 2, 3
dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
rad = np.maximum(np.abs(dy), np.abs(dx))
boxm, annm = rad <= half, (rad > half + gap) & (rad <= half + gap + ring)
cs = np.rint(p_bg["ss"]).astype(int); cf = np.rint(p_bg["fs"]).astype(int)
patch = frame[cs[:, None, None] + dy[None], cf[:, None, None] + dx[None]].astype(float)
I_meanbg = patch[:, boxm].sum(1) - int(boxm.sum()) * patch[:, annm].mean(1)
se_m = I_meanbg.std(ddof=1) / np.sqrt(n)
est_bias = I.mean() - I_meanbg.mean()

print(f"  same boxes, MEAN background: mean I={I_meanbg.mean():+.3f}, SE={se_m:.3f}")
print(f"  => median-vs-mean estimator bias {est_bias:+.3f} "
      f"({est_bias/se:.1f} SE); selection bias above is {sel_bias/est_bias:.1f}x larger")

check("with a mean background, integration IS unbiased (isolates the estimator)",
      abs(I_meanbg.mean()) < 3 * se_m, (I_meanbg.mean(), se_m))
check("the median-background residual is positive but well under the selection bias",
      0 < est_bias < 0.5 * sel_bias, (est_bias, sel_bias))

# --- and pin the SHIPPED code, not just the rule restated here ------------------------
# Everything above measures why the cut is wrong; this is what fails if someone puts it back.
src = pathlib.Path(__file__).resolve().parent.parent / "glint" / "predict.py"
text = src.read_text()
# integration keep-rules only -- `_hkl_grid` also has a `keep =`, but it is a resolution gate
keeps = [l.strip() for l in text.splitlines()
         if l.strip().startswith("keep = ") and "sig" in l]
print(f"  glint/predict.py keep-rules: {len(keeps)}")
for l in keeps:
    print(f"    {l[:100]}")
check("every keep-rule in predict.py enforces the sigma guards",
      all("np.isfinite(sig)" in l and "(sig > 0)" in l for l in keeps), keeps)
check("no keep-rule in predict.py selects on the sign of I",
      all("(I > 0)" not in l and "I>0" not in l for l in keeps), keeps)
check("both integration entry points were updated, not just one", len(keeps) == 2, len(keeps))

# --- real spots must survive ----------------------------------------------------------
spot_fs = np.array([150.0, 300.0, 450.0])
spot_ss = np.array([150.0, 300.0, 450.0])
frame2 = frame.copy()
for f0, s0 in zip(spot_fs, spot_ss):
    frame2[int(s0) - 1:int(s0) + 2, int(f0) - 1:int(f0) + 2] += 400.0

p_all = preds(np.concatenate([p_bg["fs"], spot_fs]), np.concatenate([p_bg["ss"], spot_ss]))
I2, sig2, _, _ = integrate_spots(frame2, p_all)
keep2 = np.isfinite(I2) & np.isfinite(sig2) & (sig2 > 0)
spot_I = I2[-3:]
print(f"  planted spots: I = {np.array2string(spot_I, precision=1)}")
check("planted spots are retained and strongly positive",
      bool(keep2[-3:].all()) and bool((spot_I > 20 * se).all()), spot_I)

print(f"\nFAILURES: {len(FAILS)}" + ("" if not FAILS else "  " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
