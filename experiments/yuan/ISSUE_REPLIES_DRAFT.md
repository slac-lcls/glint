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
