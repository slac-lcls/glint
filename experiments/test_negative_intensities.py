"""Integration must keep non-positive intensities, and its background must be a robust MEAN.

Two defects, measured on the same synthetic background sample: glint#130 (a selection on the sign
of the measured intensity, +15.9 counts/reflection) and glint#131 (the annulus MEDIAN subtracted
from a box SUM, +3.85 counts/reflection). Both are fixed; this file is what fails if either comes
back. The #130 half is written up first because it is the larger of the two.

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

from glint.predict import BG_MODES, integrate_spots

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

# --- THE BACKGROUND ESTIMATOR (glint#131), a second and separate bias -------------------
# integrate_spots used to take the annulus MEDIAN and subtract it from a box SUM, which needs a
# MEAN. On a discrete counting distribution the median sits below the mean, so I came out biased
# upward by nbox*(mean - median) on EVERY reflection -- worse the sparser the data, because at low
# counts the sample median is frequently a whole integer below the mean. The default is now a
# MAD-clipped mean (`bg_mode="clipmean"`); "median" reproduces every intensity GLINT produced
# before that, and "mean" is the unbiased reference the other two are measured against.
mode_I = {m: integrate_spots(frame, p_bg, bg_mode=m)[0] for m in BG_MODES}
print("\n  background estimator (same boxes, same frame; true I = 0):")
for m in ("mean", "median", "clipmean"):
    Im = mode_I[m]
    sem = Im.std(ddof=1) / np.sqrt(n)
    print(f"    bg_mode={m:9s} mean I = {Im.mean():+7.3f}   SE {sem:.3f}   ({Im.mean()/sem:+6.1f} SE)"
          + ("   <- shipped before #131" if m == "median" else
             "   <- DEFAULT" if m == "clipmean" else "   <- unbiased reference"))

se_ref = mode_I["mean"].std(ddof=1) / np.sqrt(n)
check("the DEFAULT background estimator is the clipped mean, not the median",
      np.array_equal(I, mode_I["clipmean"]) and not np.array_equal(I, mode_I["median"]))
check("with the default background, integration is unbiased",
      abs(mode_I["clipmean"].mean()) < 3 * se_ref, (mode_I["clipmean"].mean(), se_ref))
check("the plain mean is unbiased too (it is the reference, not a candidate)",
      abs(mode_I["mean"].mean()) < 3 * se_ref, mode_I["mean"].mean())
med_bias = mode_I["median"].mean() - mode_I["mean"].mean()
print(f"  => the median background biases I upward by {med_bias:+.3f} counts/reflection "
      f"({med_bias/se_ref:.1f} SE); the clipped mean is within "
      f"{abs(mode_I['clipmean'].mean() - mode_I['mean'].mean())/se_ref:.2f} SE of the reference")
check("...and the median background STILL carries its documented upward bias",
      med_bias / se_ref > 5, med_bias / se_ref)

# Sparse data is where it hurts most: at lambda<1 the sample median is 0 and the mean is not.
sparse = rng.poisson(0.3, size=(H, W)).astype(np.float32)
sp = {m: integrate_spots(sparse, p_bg, bg_mode=m)[0].mean() for m in BG_MODES}
print(f"  sparse frame (lambda=0.3): median {sp['median']:+.2f}, clipmean {sp['clipmean']:+.2f}, "
      f"mean {sp['mean']:+.2f} counts/reflection")
check("the median's bias EXPLODES on sparse data and the clipped mean's does not",
      sp["median"] > 20 * abs(sp["clipmean"]) and abs(sp["clipmean"] - sp["mean"]) < 0.2, sp)

# ...and the robustness the median was there for must survive, or "just use the mean" would do.
# Box untouched, annulus contaminated => the true answer is still 0, and an estimator dragged by
# the contaminant reads NEGATIVE.
dirty = frame.copy()
cs0 = np.rint(p_bg["ss"]).astype(int); cf0 = np.rint(p_bg["fs"]).astype(int)
for s0, f0 in zip(cs0, cf0):
    dirty[s0 + 7 - 1:s0 + 7 + 2, f0 - 1:f0 + 2] += 800.0     # a neighbouring spot in the annulus
    dirty[s0 - 7, f0 + 7] = 65535.0                          # and one saturated pixel
dirt = {m: integrate_spots(dirty, p_bg, bg_mode=m)[0].mean() for m in BG_MODES}
print(f"  contaminated annulus (9-px neighbour + one saturated px): "
      f"median {dirt['median']:+.2f}, clipmean {dirt['clipmean']:+.2f}, mean {dirt['mean']:+.2f}")
check("the clipped mean REJECTS annulus contamination (a plain mean would not do)",
      abs(dirt["clipmean"]) < 5 and abs(dirt["mean"]) > 1000, dirt)

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

# --- the escape hatch has to be REACHABLE, not just implemented -----------------------
# `bg_mode="median"` is the ONLY way to reproduce any intensity GLINT produced before #131.
# A parameter that no shipped route passes is an API-only promise: it reads as an escape hatch
# in review and does nothing for the person holding a pre-#131 dataset. Each route below owns
# one integrate call; every one of them must forward the mode.
ROOT = src.parent.parent
routes = {
    "glint_cli.py --peaks/--images": ("glint/glint_cli.py", "bg_mode=args.bg_mode", 2),
    "stream_driver.py (CPU + GPU)": ("glint/stream_driver.py", "bg_mode=self.bg_mode", 2),
    "glint_xtc.py (raw xtc route)": ("experiments/xtc_bridge/glint_xtc.py", "bg_mode=args.bg_mode", 1),
}
for label, (rel, needle, want) in routes.items():
    got = (ROOT / rel).read_text().count(needle)
    check(f"{label} forwards bg_mode to its integrator", got == want, f"{got} of {want} call sites")

cli = (ROOT / "glint/glint_cli.py").read_text()
check("the CLI exposes --bg-mode with every mode and defaults to clipmean",
      "--bg-mode" in cli and all(f'"{m}"' in cli for m in BG_MODES)
      and 'default="clipmean"' in cli)

lute_model = (ROOT / "lute/glint_index.py").read_text()
check("the LUTE task model has a bg_mode field rendering as --bg-mode",
      "bg_mode:" in lute_model and 'rename_param="bg-mode"' in lute_model)
# ...and the launcher's xtc whitelist, which is where --pf8-min-snr was silently dropped before
check("the LUTE launcher whitelists --bg-mode (else the xtc route drops it)",
      "--bg-mode" in (ROOT / "lute/glint_launch.sh").read_text())

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
