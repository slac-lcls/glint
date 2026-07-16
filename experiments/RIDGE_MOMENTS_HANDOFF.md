# ridge_moments — hand-off note (for Yuan / the CBXD indexing collaboration)

**TL;DR:** a sketch of a per-streak shape descriptor whose main output is a *streak direction*
fitted from the streak's axis. That direction is the exact quantity `cbxd_angles.py`'s pair-angle
indexing runs on — and its bias is the current limiter there.

## Files
- `experiments/ridge_moments.py` — the sketch (this dir).
- Companion general primitive: `peakfinder8(…, moments=True)` in **slac-lcls/drp-benchmarks**
  (`radial_integration/peakfinder8.py`, commit `365a3be`) — per-peak height + intensity-weighted
  covariance (σ_major/σ_minor, θ, ecc). `ridge_moments` is the *curved-streak* version: local
  2nd-moments along the arc → a tangent / width / curvature profile.

## Why it connects to the CBXD work
`cbxd_angles.py` breaks the |G|-magnitude alias (cubic 29³ vs ortho 16/21/25) using the
rotation-invariant pair angle `cos = q_i·q_j / (|q_i||q_j|)`, which samples the reciprocal metric
tensor `g* = BᵀB` rather than just the ring magnitudes. The limiter (per the docstring) is the
per-streak **direction** bias (~0.015 Å⁻¹, tens of degrees at low |G|) → low-|G| pairs get dropped
and you lean on pooling. `ridge_moments` estimates the streak direction by fitting its **axis**
(local moment tangent) instead of inferring it from a centroid, and returns a per-segment **width**
usable as a trust weight. If that lowers the direction bias, you keep lower-|G| pairs → more angular
constraints → denser metric-tensor sampling → cleaner cell at fixed shot count.

## What to try (concrete)
1. In `observed_features`, get each streak's direction from `ridge_descriptor(pixels…)['tangent']`
   instead of (or alongside) the centroid-derived `q`. Plot per-streak direction bias vs |G| against
   the current ~0.015 Å⁻¹ curve — does it shrink, especially at low |G|?
2. If it shrinks, lower `gmin` and re-run the cubic-vs-orthorhombic discrimination — do the extra
   low-|G| pairs improve `fit_score` / the FOM at fixed `nshot`?
3. Use `width` / `eccentricity` to weight or reject streaks (thin, high-ecc = trustworthy axis;
   fat/round = drop).

## Status / caveats (it's a sketch)
- Validated headline only: on a synthetic arc it recovers the true local width where a single
  covariance ellipse is inflated 2.3× by curvature (`__main__` demo).
- The **tangent** (what you'd use) is the robust part; the **curvature/turning** numbers are only
  ballpark — the pixel ordering is a chord-PCA projection that folds as curvature grows. For strongly
  curved streaks, swap in a real ridge tracer (`skimage.skeletonize` + path walk) or a parametric
  conic fit (Fitzgibbon–Pilu–Fisher). Validate the tangent on the actual `simulate()` streaks first.
- Front-end: to pull streak components from peakfinder8 you'd relax its `max_pix` cap and loosen the
  isotropic-ring background (CBXD/Kossel features cross resolution rings).

## The real question
Is the streak-direction bias actually the bottleneck on the low-|G| pairs? If yes, this is cheap to
try. If the bias is dominated by the sim geometry or noise model instead, this won't move the needle
and we should look there. Happy to wire it into `cbxd_angles.py` together.
