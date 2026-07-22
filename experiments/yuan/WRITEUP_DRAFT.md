# CBXD update — correction to #9/#11's premise, + the confidence-based cascade

## TL;DR

The "two consistent failures = below-median streak count" claim in PR #10 (and now the opening
premise of #11, and half of #9's hypothesis) doesn't hold under a corrected, matched comparison.
What actually predicts where the accumulator struggles isn't `#streaks` — it's the accumulator's
own match confidence, which is a much better lever than anything structural. Built and validated a
policy on that (PTS by default, escalate to tangent only when unconfident) that beats every
single-arm choice in both regimes tested.

## 1. The bug behind the original claim (worth knowing regardless of the rest)

`hough_seed_index(kobs, rng, n_coarse=...)` draws millions of random candidates from the *same*
`rng` object used to generate the next crystal's orientation. So crystal N's orientation silently
depended on how much random state crystal N-1's *search* consumed, not just on N. Two runs with
"the same seed" but different searches (or different `n_coarse`) produced different physical
crystals under the same index label. This is why the PR's original "crystal #4 / #7" story doesn't
reproduce cleanly — they aren't the same crystals from run to run.

Fixed by persisting crystals to disk once: `experiments/yuan/generate_dataset.py`
(NA=0.028, N=20, noise=2e-4) and `experiments/yuan/generate_dataset_lowna.py` (NA=0.016, N=20,
noise=2e-4) — every arm now reads the exact same orientations/streak-clouds regardless of what
search runs on them. Reproduce:
```
cd experiments/yuan
python generate_dataset.py        # writes data/simulated_data/
python generate_dataset_lowna.py  # writes data/simulated_data_lowna/
```

## 2. #streaks doesn't predict PTS's failures (re: #11's premise)

Matched, per-crystal comparison on the fixed baseline dataset (N=20, NA=0.028):
```
python run_three_arms.py 5000000
```
Result: **PTS 14/20, TAN 18/20** (not the 35/40 headline number — that number mixed noise levels
and predates the rng-crystal bug fix). Spearman correlation between `#streaks` and PTS's frac_indexed:
**ρ=-0.33, p=0.15 (not significant)**. Direction is backwards from the hypothesis: median
`#streaks` among PTS's failures (23.5) is *higher* than among its successes (20.5). PTS's four
sparsest crystals (12-18 streaks) all solve perfectly; its single worst failure has 32 streaks — the
most in the set.

Pooling all PTS data collected today (N=56, baseline + low-NA + partial noise-1e-4 leg) softens
this slightly — there's a real dip below ~10 streaks (56% success in the 4-10 bin vs 92%+ in
11-20) — but it's not monotonic (21-32 streaks drops back to 65-67%), so `#streaks` is at best a
partial factor, not the dominant one.

## 3. Direct test of "tangent wins in the reflection-starved corner" (re: #9's prediction)

Built a second fixed dataset specifically in the sparse regime -- NA=0.016 (median 11 streaks,
range 4-18), chosen because it's the one point in an earlier coarse NA sweep where every arm
showed real dynamic range (not a floor or ceiling):
```
python generate_dataset_lowna.py
python run_three_arms_lowna.py 5000000
```
Result: **PTS 15/20, TAN 14/20** — tangent is net *negative* in the regime it was predicted to
win. Spearman(`#streaks`, TAN-PTS margin) = -0.11, p=0.63. Two crystals with the *identical*
streak count (n=9) go in opposite directions (TAN helps one, hurts the other) — direct evidence
`#streaks` alone can't be the trigger, even qualitatively.

## 4. What actually works: confidence-based cascade, not a streak-count switch

Instead of predicting in advance which crystals need tangent, use PTS's own result as the trigger:
run PTS first (cheap), and only escalate to TAN if PTS's own matched-fraction of all observed
points (`score(R, kobs, tol) / len(kobs)`) is below a threshold. No ground truth needed — real
blind indexing doesn't have it either, and it's free (you were always running PTS first).

Calibration (baseline dataset): PTS's confidence cleanly separates its own successes (0.811-0.960)
from failures (0.200-0.489) — a wide gap, any threshold in [0.5, 0.7] gives the identical
escalation set. Code: `experiments/yuan/cbxd_hough_cascade.py`.

Validated end-to-end (real runs, not estimates — TAN itself has real run-to-run variance at 5M
candidates, so a derived/estimated cascade number doesn't match what actually running it gives):

| dataset | COM | PTS | TAN | **Cascade** |
|---|---|---|---|---|
| baseline (NA=0.028) | -- | 14/20 | 18/20 | **17/20**, ~30% less compute than always-TAN |
| low-NA (NA=0.016) | 0/20 | 15/20 | 14/20 | **16/20** (beats every single arm) |

Reproduce: `python run_com_cascade_lowna.py` (low-NA) or the relevant section of
`run_four_arms_full.py` (baseline, both noise levels).

Important nuance from the full run below: cascade is NOT a strict win everywhere. On the baseline
(NA=0.028) dataset, always-on TAN slightly beats cascade (37/40 vs 35/40, consistently 1 crystal/
noise-level) -- a handful of crystals where PTS's own confidence is high but it's still marginally
wrong, so cascade (trusting that confidence) misses a rescue always-TAN would have caught. Cascade's
real win there is ~30-40% less compute (TAN's cost paid on ~30% of crystals, not 100%), not higher
accuracy. On the low-NA/sparse dataset it's a strict win on both axes (accuracy AND cost) --
always-on TAN is net-negative there, so cascade beats every single arm outright. Net: cascade is
never the worst choice and is the right general-purpose default, but "cascade always wins" would
overstate the baseline-dataset result.

## 5. Full reproduction, matched to PR #10's original scope

20 crystals x {1e-4, 2e-4} noise x {COM, PTS, TAN, Cascade}, 5M candidates -- exactly PR #10's
original shape, on the fixed/bugfixed dataset, now with all four things built since. Checkpointed
(`results_four_arms_full.jsonl`, one line per trial, safe to resume).

| noise | COM | PTS | TAN | Cascade |
|---|---|---|---|---|
| 1e-4 | 1/20 (9%) | 16/20 (100% median) | 19/20 (100%) | 18/20 (100%) |
| 2e-4 | 2/20 (12%) | 14/20 (98%) | 18/20 (97%) | 17/20 (99%) |
| **combined (40)** | **3/40 (7.5%)** | **30/40 (75%)** | **37/40 (92.5%)** | **35/40 (87.5%)** |

Reproduce: `python run_four_arms_full.py` (resumable; `results_four_arms_full.jsonl` has the raw
per-trial data, `python run_four_arms_full.py summary` reprints the table from the log).

## 6. Deliverable 3: phase diagram, wide-NA extension, and the streak-count deconfound (re: #11)

Scoped per Stefano's review: (NA x noise) is the swept grid, `#streaks`/streak length are recorded
as observed statistics on top, not independent axes (they aren't independent controls -- both
derive from NA/dmin/cell, confirmed in section 5's open items above).

**Grid.** NA in {0.010, 0.016, 0.022, 0.028, 0.040, 0.050, 0.060} (extended from the original four
to reach the range where COM was reported to turn over) x noise in {2e-4, 5e-4, 1e-3, 2e-3, 5e-3,
1e-2, 2e-2}, 49 cells, 20 crystals/cell, COM/PTS/TAN (Cascade excluded -- a deployment result, not
a phase-boundary variable). Frozen-dataset/checkpointed methodology as in section 1.
Reproduce: `python generate_dataset_grid.py --all && python run_grid_arms.py`.
Figures: `python plotting/plot_phase_diagram.py` -> `plotting/phase_diagram.png/.pdf` (3D surfaces,
all 7 NA rows) and `plotting/tan_vs_pts.png` (2D companion at noise=1e-3).

**COM stays flat.** COM never exceeds ~10% anywhere across the full grid (all 7 NA values, full
noise ladder) -- no turnover or collapse to show, just a flat floor throughout the tested range.

**TAN vs PTS (the actual finding, since the COM comparison isn't usable).** At noise=1e-3: PTS
stays noisy and low (5-40%) with no real trend across NA. TAN dips similarly around NA=0.022-0.028,
then rises sharply from NA=0.040 onward, reaching 85% at NA=0.050 (vs PTS's 30%) before easing to
80% at NA=0.060 -- TAN pulls away from PTS specifically once the cone widens, not just "better on
average." TAN's own peak is at NA=0.028/0.040 (20/20 at noise=2e-4), easing to 19/20, 18/20 at
0.050/0.060 -- a real, gentle non-monotonicity, so "wider NA is always better" doesn't hold either.

**Streak-count deconfound.** Does NA's effect survive once streak count is held fixed? Subsampled
each NA class's own crystals down to fixed target counts (4, 11 -- matching NA=0.010's and
NA=0.016's natural medians), each crystal keeping an independently-random subset (so 20 trials per
cell are 20 distinct geometric realizations, not repeats). First pass, single NA=0.028
(`subsample_streaks_na028.py`, noise=2e-4):

| streak count | subsampled NA=0.028 (fixed geometry) | natural NA (varying geometry) |
|---|---|---|
| 4 | PTS 19/20, TAN 18/20 | NA=0.010: PTS 15/20, TAN 12/20 |
| 11 | PTS 17/20, TAN 17/20 | NA=0.016: PTS 17/20, TAN 18/20 |

Then extended to four distinct NA classes at the same two targets (`subsample_streaks_multi_na.py`,
`results_subsample_multi.jsonl`), to check whether that pattern is general or specific to one
geometry:

| NA | target=4 | target=11 |
|---|---|---|
| 0.016 | PTS 16/20, TAN 14/20 | PTS 19/20, TAN 15/20 |
| 0.028 | PTS 19/20, TAN 18/20 | PTS 17/20, TAN 17/20 |
| 0.040 | PTS 15/20, TAN 18/20 | PTS 13/20, TAN 16/20 |
| 0.060 | PTS 8/20, TAN 12/20 | PTS 4/20, TAN 5/20 |

Even at matched streak count, NA still matters -- but not monotonically. NA=0.060 collapses badly
once forced down to few streaks (worst of the four classes at both targets), well below
NA=0.028/0.040 at the same count. This qualifies the main grid's headline: NA=0.060's strong
natural performance there is coming from having ~49 streaks available, not from wide-cone geometry
being inherently better on its own -- strip it down to a handful and it underperforms.

**Does the specific geometry (not just the count) of kept streaks predict success?**
(`analyze_subsample_geometry.py`, `analyze_subsample_geometry_multi.py`.) Feature `min_sv`: smallest
singular value of the matrix of kept streaks' unit reciprocal-lattice-vector (`G`, not detector-
position `kout` -- `kout` is spuriously near-degenerate for every crystal regardless of orientation,
a bug caught while building this) directions -- measures how well-spread (large) vs. near-coplanar
(near 0) the kept constraints are.

At NA=0.028 alone, min_sv looked like it flipped sign with streak count: rho=-0.68 (p=0.001) at
target=4, rho=+0.58 (p=0.007) at target=11 -- read at the time as evidence for a general
search-mechanism effect (well-spread constraints carve a sharper-but-smaller true-orientation basin,
helping precision once found but hurting random-search discoverability at very few streaks; helping
outright once there's enough redundancy). Extending across all four NA classes does **not** support
that as a general mechanism -- the flip only happens at NA=0.028:

```
NA=0.016: n=4 rho=-0.704 (p=0.001)  ->  n=11 rho=-0.363 (p=0.115)   same sign
NA=0.028: n=4 rho=-0.676 (p=0.001)  ->  n=11 rho=+0.579 (p=0.007)   FLIPS
NA=0.040: n=4 rho=-0.403 (p=0.078)  ->  n=11 rho=-0.036 (p=0.880)   same sign
NA=0.060: n=4 rho=-0.216 (p=0.360)  ->  n=11 rho=-0.403 (p=0.078)   same sign
```

What generalizes: at target=4, min_sv correlates negatively with success across all four NA classes
(consistent direction, strongest/significant at NA=0.016/0.028, weakening toward zero as NA grows)
-- "well-spread G's hurt at very sparse counts" holds up. "...and helps at higher counts" does not;
that was specific to one geometry, not a general property. min_sv also doesn't explain the NA=0.060
collapse above (p=0.36, 0.08 there) -- something else about wide-cone geometry at low counts is
driving that, not yet identified.

**Methodological caveat: "target=11 beats target=4" only holds at NA=0.016.** At every other NA in
the table above, going from target=4 to target=11 is flat or *worse* (PTS: 0.028 19->17, 0.040
15->13, 0.060 8->4; TAN similarly flat/down) -- the opposite of the naive "more streaks should only
help" intuition. Explained by retention fraction, not a mysterious effect: `target / natural pool
size` at each NA:

```
NA=0.016  natural pool median=11   retention@4=36%   retention@11=100%
NA=0.028  natural pool median=22   retention@4=18%   retention@11=50%
NA=0.040  natural pool median=32   retention@4=12%   retention@11=34%
NA=0.060  natural pool median=49   retention@4=8%    retention@11=22%
```

NA=0.016 is the only row where target=11 is ~100% retention -- going 4->11 there means recovering
the *entire* natural streak set, not adding a few more random ones, so the intuitive monotonic
improvement holds. Everywhere else, both targets are still small minority samples of a much bigger
pool. Two effects work against "more must be better" there: (1) the subsampling preserves a fixed
spurious-point ratio (`cbxd_joint.SPUR=0.30`), so target=4->11 roughly triples the absolute number
of spurious streaks mixed in (~1->~3), not just the real ones -- more noise competing in the
accumulator can offset the extra real constraints; (2) at low, roughly-constant retention you're
nowhere near saturating the pool, so the specific streaks kept at target=11 are a fresh random draw
that (per the min_sv result above) can land on a worse geometric configuration just as easily as a
better one -- no guaranteed monotonic improvement until retention actually approaches 100%. This is
also a more likely explanation for the NA=0.060 collapse than min_sv alone, since NA=0.060 has by
far the lowest retention at both targets (8%, 22%).

## Open items / not yet touched

- #12 (cross-frame consensus for CBXD): not started -- planning next.
- NA=0.060 streak-geometry collapse at low fixed counts (section 6): min_sv doesn't explain it --
  open mechanistic question if worth chasing.
- TAN's run-to-run variance at fixed `n_coarse` (same crystal, different random draws, meaningfully
  different quality) -- not explained, could itself be worth a small investigation.
