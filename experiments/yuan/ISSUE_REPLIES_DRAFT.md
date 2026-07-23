# Draft reply — phase diagram refinement (issue #11), ready to post pending review

Pushed the three fixes you asked for. Numbers and figures below; happy to adjust before this goes
up if anything reads wrong.

## 1. Extended NA to 0.040/0.050/0.060

Full grid now: NA in {0.010, 0.016, 0.022, 0.028, 0.040, 0.050, 0.060} x the same noise ladder,
same frozen-dataset/checkpointed methodology (`generate_dataset_grid.py` + `run_grid_arms.py`,
now 49 cells). One thing worth flagging directly rather than glossing over: **COM never rises above
~10% anywhere in the tested range** -- it sits flat at 0-10% across all 7 NA values and the full
noise ladder, so there's no COM turnover/collapse to show on this grid, just a flat floor.

## 2. Does TAN survive where the simpler method struggles?

Couldn't test this against COM directly (see above), but a cleaner, equally strong version of the
question showed up against PTS instead -- and this is the actual finding worth leading with
(figure: `experiments/yuan/plotting/tan_vs_pts.png` -- attach when posting, `gh`/API can't inline
a local image into a comment automatically).

At noise=1e-3: PTS stays noisy and low (5-40%) across the whole NA range, no real trend. TAN dips
similarly around NA=0.022-0.028, then **rises sharply and monotonically from NA=0.040 onward**,
reaching 85% at NA=0.050 (vs PTS's 30%) before easing slightly to 80% at NA=0.060. That's a
specific, falsifiable mechanism -- TAN pulls away from PTS exactly once the cone gets wide enough
-- not just "TAN is better on average." Also modestly answers your "does the NA axis turn over"
concern: TAN peaks at NA=0.028/0.040 (20/20 at noise=2e-4) then eases to 19/20, 18/20 -- a real,
if gentle, non-monotonicity, so a reader won't extrapolate "wider is always better."

Full 3D phase diagram (all 7 NA rows, COM/PTS/TAN): `experiments/yuan/plotting/phase_diagram.png`
/ `.pdf`, regenerated from `plot_phase_diagram.py` which now reads `NA_GRID` directly from
`generate_dataset_grid.py` (single source of truth, won't drift if the grid extends again).

## 3. Streak-count deconfound

Subsampled crystals down to fixed streak-count targets (4, 11 -- matching NA=0.010's and
NA=0.016's natural medians), holding count fixed while varying geometry. First pass, single
NA=0.028 (`subsample_streaks_na028.py`), vs. the natural low-NA cells:

| streak count | subsampled NA=0.028 (fixed geometry) | natural NA (varying geometry) |
|---|---|---|
| 4 | PTS 19/20, TAN 18/20 | NA=0.010: PTS 15/20, TAN 12/20 |
| 11 | PTS 17/20, TAN 17/20 | NA=0.016: PTS 17/20, TAN 18/20 |

Extended to four distinct NA classes at the same two targets (`subsample_streaks_multi_na.py`) to
check whether "geometry matters at low counts" is general or specific to NA=0.028:

| NA | target=4 | target=11 |
|---|---|---|
| 0.016 | PTS 16/20, TAN 14/20 | PTS 19/20, TAN 15/20 |
| 0.028 | PTS 19/20, TAN 18/20 | PTS 17/20, TAN 17/20 |
| 0.040 | PTS 15/20, TAN 18/20 | PTS 13/20, TAN 16/20 |
| 0.060 | PTS 8/20, TAN 12/20 | PTS 4/20, TAN 5/20 |

Even holding count fixed, NA still matters, but not monotonically: NA=0.060 collapses badly once
forced down to few streaks -- worst of the four classes at both targets, well below NA=0.028/0.040
at the same count. Reads as: NA=0.060's strong performance in the main grid (section 2) is coming
from having ~49 streaks available, not from wide-cone geometry being inherently better -- strip it
down and it underperforms.

Also checked whether the *specific* geometry of kept streaks (not just the count) predicts success,
via `min_sv` (spread of kept streaks' reciprocal-lattice-vector directions). At NA=0.028 alone this
looked like a sign flip with count (rho=-0.68 at n=4 -> rho=+0.58 at n=11), which we initially read
as a general search-mechanism effect. Extending across all four NA classes shows the flip is
specific to NA=0.028, not general -- at n=4, well-spread G's hurt success across all four NA
classes (consistent negative correlation, weakening toward zero as NA grows), but the "helps at
higher counts" half doesn't generalize. Also doesn't explain the NA=0.060 collapse above (p=0.36,
0.08) -- open question if worth chasing.

One methodological wrinkle worth flagging: `target=4 -> target=11` isn't a clean "more streaks"
comparison everywhere. It only behaves that way at NA=0.016, because target=11 there is ~100% of
that NA's natural streak pool (median 11) -- everywhere else, both targets are still small,
roughly-fixed-fraction random draws from a much bigger pool (NA=0.028: 18%->50% retention; 0.040:
12%->34%; 0.060: 8%->22%), so going from 4 to 11 doesn't reliably help -- it can be flat or worse
(PTS at NA=0.028/0.040/0.060 all go *down* from target=4 to target=11). Two reasons: the subsampling
preserves a fixed spurious-point ratio, so more target also means proportionally more spurious
streaks mixed in, and at low retention the extra streaks are still a fresh random draw that can land
on a worse geometric configuration just as easily as a better one. Likely a better explanation for
the NA=0.060 collapse than the min_sv geometry feature alone, since NA=0.060 has by far the lowest
retention at both targets.

Reproduce everything: `pytest test_cbxd_hough.py test_cbxd_arms.py` first (fast sanity check), then
`run_grid_arms.py` (resumable, checkpointed), `subsample_streaks_na028.py` +
`subsample_streaks_multi_na.py` for the deconfound, `analyze_subsample_geometry_multi.py` for the
geometry check, `plotting/plot_phase_diagram.py` for both figures.

---

# Draft reply — cross-frame consensus (issue #12), ready to post pending review

TL;DR up front since this isn't the result the hypothesis predicted: **naive pooling doesn't
replicate the SFX "pooling doubles the effect" result for CBXD, at full statistical power.** TAN
has a small but real-looking exception, with a mechanism, below.

## 1. Two mechanisms, both mirroring existing validated code, not invented from scratch

**Data-level pooling** (`generate_dataset_multishot.py`, `run_multishot.py`): concatenate M
independent shots of the *same* crystal (same true orientation, independent noise + spurious-point
draws each shot -- modeling repeat exposures) into one bigger point cloud, run the existing
single-shot accumulator (`cbxd_hough_gpu.hough_seed_index`, unchanged) on the pool. This is exact
for the vote-count objective -- `score(R, concat(shots), tol) == sum(score(R, shot_m, tol))`,
proven in `test_cbxd_multishot.py::test_multishot_pooling_exact` -- so "pool streaks within a shot"
and "pool shots" are literally the same accumulator operation, just on a bigger point cloud.

**Result-level consensus** (`cbxd_orientation_consensus.py`): mirrors `glint/multishot.py`'s
`consensus_cell` directly -- solve each shot *independently* (own search, own points, no data
shared), collect the M orientation hypotheses, group by rotational proximity (geodesic angle <
`ang_tol_deg`, same greedy single-link-to-representative clustering as `_grp_reduced`), return the
dominant cluster if it clears `min_support`. This is the actual mechanism behind the SFX result
this issue is asking us to replicate, not a lookalike.

Crystals throughout are the exact same 20 from `data/grid/orientations_na0.028.npz` deliverable 3
already validated (`generate_dataset_multishot.py` reuses that pool directly, not a fresh draw --
`test_cbxd_multishot.py::test_shot_zero_matches_grid_cell` checks shot 0 is byte-identical to the
cached grid cell at every noise level).

## 2. Headline: PTS + data-pooling is flat at every noise level, full N=20/n_coarse=5M

Same candidate density deliverable 3's own headline numbers use (not the faster 1M we piloted
with first -- see the methodology note in section 4). Noise in {2e-4, 5e-4, 1e-3, 2e-3} (the
2e-4/5e-4/1e-3 range plus the 2e-3 WALLED point from the deliverable-3 grid), M in {1, 2, 4, 8},
NA=0.028:

| noise | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|
| 2e-4 (baseline) | 13/20, 74% mean | 15/20, 80% | 13/20, 73% | 15/20, 79% |
| 5e-4 | 8/20, 54% mean | 8/20, 52% | 8/20, 52% | 7/20, 49% |
| 1e-3 (transition) | 0/20, 40% mean | 1/20, 44% | 2/20, 41% | 0/20, 39% |
| 2e-3 (walled) | 0/20, 38% mean | 0/20, 37% | 0/20, 34% | 0/20, 32% |

Flat everywhere, no exceptions. The walled point stays walled at every M -- pooling doesn't retreat
the boundary deliverable 3 mapped. Full data: `results_multishot.jsonl`.

## 3. Result-level consensus: negative for the weak arm, inconclusive for the strong one

Small test (8 crystals, noise=1e-3, M in {2,4,8}), comparing COM (arm 1, centroid-only -- no
cross-streak pooling, deliberately the weak per-shot signal analogous to a single sparse SFX
frame) and PTS, each alone vs. with consensus:

| method | M | has consensus | success \| has answer | mean frac \| has answer |
|---|---|---|---|---|
| COM + consensus | 2 | 0/8 | -- | -- |
| COM + consensus | 4 | 3/8 | 0/3 | 7% |
| COM + consensus | 8 | 5/8 | 0/5 | 5% |
| PTS + consensus | 2 | 2/8 | 1/2 | 67% |
| PTS + consensus | 4 | 6/8 | 2/6 | 59% |
| PTS + consensus | 8 | 8/8 | 0/8 | 41% |

**COM + consensus is a clean negative.** Even when independent COM searches *do* cluster
(support>=2), the cluster is on the wrong orientation (5-7% mean frac). Mechanistic reason: COM
essentially never finds the true orientation as a viable candidate on its own at this noise (4-8%
mean frac even at M=1, consistent with section 1's "COM never rises above ~10%" finding). SFX
consensus works because a sparse frame's blind search *does* often land on the true cell as one of
a few candidates (FFT peaks carry real structure even from few points); CBXD's COM analog doesn't
have that property here, so there's nothing for consensus to lock onto.

**PTS + consensus is inconclusive, not negative** -- we used a fixed `min_support=2` regardless of
M, which is too weak a bar at large M (2-of-8 agreeing by accident is much easier than 2-of-2), so
coverage climbed to 8/8 by M=8 while conditional accuracy *degraded* (67% -> 59% -> 41%). That's
consistent with a threshold that needs to scale with M (majority-style, not a fixed constant), not
with the mechanism itself failing. Not retested with a fixed threshold yet.

## 4. TAN + data-pooling: a small-N hint of a real effect, with a mechanism

Same data-pooling mechanism as section 2, arm 3 (TAN) instead of PTS. Small test only (4 crystals,
noise=1e-3, not yet scaled to N=20):

| | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|
| PTS (same 4 crystals) | 58% mean | 59% | 49% | 51% |
| TAN | 61% mean | 49% | 60% | **62%** |

TAN ends ~10 points above PTS at M=8 on these crystals, with a per-crystal story worth telling:
one crystal (index 3 in the pool, 19 streaks/shot) has a genuine systematic rival orientation
~152-157deg from truth that PTS's points-only vote count locks onto at *every* M -- its vote count
scales in exact lockstep with the true orientation's as M grows (19->38->78->154, doubling right
along with the pooled data), so pooling reinforces the ambiguity equally on both sides and can
never break the tie. The true orientation's neighborhood never even reaches the points-only coarse
top-10 at any M (stuck 44-48deg away). Adding the tangent bonus rescues it *immediately* -- already
true at M=1, not something pooling is needed for -- pulling a candidate 1.15deg from truth into the
top-10 at every M, because the rival orientation coincidentally satisfies the Bragg-plane residual
test at many points but predicts the *wrong local arc curvature* there (tangent_bonus ~27 for the
rival vs. ~109 for the true orientation). That's exactly the curvature-breaks-point-ambiguity
mechanism issue #9 predicted TAN for.

Caveat this needs before it's a real result: the win here looks like it's mostly from tangent
information being available at all (present at M=1), not from pooling per se -- M's role may just
be averaging the tangent estimate over more shots. Worth scaling to N=20/full noise ladder (like
PTS got) before calling this a confirmed effect; held at N=4 for now.

## 5. Bottom line

Neither mechanism replicates the SFX pooling result cleanly for CBXD as-is. PTS + data-pooling is a
confirmed flat/negative result at full power. Result-level consensus is a clean negative for the
weak arm (COM) and inconclusive for the strong one (PTS, methodology bug). TAN + data-pooling has
the most promising signal but at 1/5th the sample size of the PTS result, and the mechanism we can
already see (crystal 3's case study) suggests the win is about *curvature information*, not
*pooling*, per se -- M might be a red herring for TAN too, and the real lever might just be "use
the tangent bonus." Next step, if this is worth chasing further: scale the TAN test to N=20 and
check whether the crystal-3-style systematic-rival mechanism is common enough across the pool to
explain the aggregate gap, or whether it's a lucky N=4 draw.

Reproduce: `pytest test_cbxd_multishot.py` (pooling-exact + byte-identity golden checks) first,
then `generate_dataset_multishot.py` (frozen dataset, reuses `data/grid/orientations_na0.028.npz`)
+ `run_multishot.py` (PTS, resumable/checkpointed) for section 2, `run_orientation_consensus_test.py`
for section 3, `run_tan_pooling_test.py` for section 4. `diag_multishot_coarse.py` /
`diag_ambiguity_scan.py` are the standalone diagnostics behind the systematic-rival-orientation
finding.
