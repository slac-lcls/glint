# Convergent-beam CBXD — step 1 (forward sim + arc accumulator)

**What this is.** The step-1 scaffold for blind convergent-beam (CBXD) indexing: a pure-numpy forward
simulator + the Kossel-arc accumulator objective. Code: `cbxd_twocolor.py` (extends the single-colour
`cbxd_joint.py` overlay). Steps 2–4 are Yuan's (below).

**Re-cast, and why.** This file first argued that *two colours move the blind wall*, measured at one
split — 17.5/15.0 keV, i.e. Δ*E*/*E* = 15.4% — and reported as though two colours were the lever.
Adding a `cover` mode that measures reciprocal-space coverage as pure geometry shows the lever is
**convergence**, and that the two-colour term is a bonus whose size depends entirely on how the split
compares to the convergence half-angle. The original wide-split results are all still here and all
still stand; what changed is what they are evidence *for*.

## Physics — convergence first

A parallel monochromatic beam samples the Ewald sphere, a surface: measure zero in three dimensions,
so the reciprocal-lattice difference cloud stays rank-deficient along the beam. Convergence fixes
that. Tilting **k**ᵢₙ over a cone of half-angle NA sweeps the sphere through a **shell** of radial
width ≈ *K*·NA, and two things follow:

- every reflection inside the shell is excited, and
- each is excited along an **arc** (the Kossel circle `k_out·Ĝ = |G|/2`, `|k_out| = K`, swept as the
  incident direction moves over the cone), whose *direction* encodes the out-of-plane σ₃ axis a
  parallel-beam still cannot constrain.

At the sweep's mean energy of 16.25 keV, *K* = 1.31 Å⁻¹, so NA = 20 mrad gives a shell ≈ 0.026 Å⁻¹
thick — about two-thirds of a reciprocal-cell spacing for a 25 Å axis. That is the whole CBXD
premise, and it is continuous and tunable.

A second photon energy adds a second, **discrete** sphere at Δ*K*/*K* = Δ*E*/*E* (because *K* = *E*/*hc*).
It reaches new reflections only once it lands *outside* the shell convergence has already swept:

> **Coverage threshold — Δ*E*/*E* ≳ NA.** Below it the two spheres overlap inside one convergence
> shell and the second colour re-excites the reflections the first already had.

The accumulator's own colour bookkeeping has a *different* and much lower threshold. It separates the
colours with the K-scaled direction gate `kz > K·cos(NA)`, which keeps working until
`K2 > K1·cos(NA)`:

> **Bookkeeping limit — Δ*E*/*E* ≈ 1 − cos(NA) ≈ NA²/2.** At NA = 20 mrad that is 2×10⁻⁴, a **hundred
> times** below the coverage threshold.

Between the two, the accumulator labels every point by colour and reports twice the votes at truth
while the second sphere is contributing nothing. That gap is the trap this file walked into.

## (A) Coverage — the measurement that re-cast the file

`cbxd_twocolor.py cover [na] [ncry]` — pure geometry, no noise, no scoring, no refinement. `union` is
the distinct reflections reached by **either** colour; `arcs` counts *excitations*, which is what the
original write-up reported doubling (25 → 52). Median over 16 crystals.

| Δ*E*/*E* | NA=14 mrad | NA=20 | NA=28 | NA=40 | arcs/col1 (all NA) |
|---|---|---|---|---|---|
| 0.15385 | **1.63×** | **1.37×** | **1.23×** | **1.17×** | ~2.00× |
| 0.08 | 1.26 | 1.23 | 1.09 | 1.08 | ~2.00 |
| 0.04 | 1.09 | 1.10 | 1.06 | 1.03 | 2.00 |
| 0.02 | 1.05 | 1.03 | 1.04 | 1.01 | 2.00 |
| 0.01 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 |
| 0 | 1.00 | 1.00 | 1.00 | 1.00 | 2.00 |

Two readings, both load-bearing:

1. **Even at 15.4%, "twice the arcs" is 1.2–1.6× the reflections.** The arc count doubles at *every*
   split down to and including zero, because a duplicate excitation is still an excitation. Arc count
   is not a coverage measure.
2. **The two-colour gain shrinks as NA grows** — 1.63× at 14 mrad down to 1.17× at 40 mrad — exactly
   as the shell picture requires: a wider shell has already swallowed the second sphere.

And the convergence lever measured the same way, single colour (`cover na`), distinct reflections:

| NA (mrad) | 14 | 20 | 28 | 40 | 56 |
|---|---|---|---|---|---|
| reflections | 8 | 14 | 20 | 32 | 44 |
| vs previous | — | 1.75× | 1.43× | 1.60× | 1.37× |

**A ~1.4× step in NA beats the entire 2.5 keV split, at every NA.** Convergence is also continuously
tunable, where the split is set by what the source can be made to do.

## (B) The vote-count trap

`cbxd_twocolor.py split [na] [ncry]` sweeps Δ*E*/*E* at fixed convergence and scores the same three
configs on accumulator votes. Votes at truth, median over 6 crystals, NA = 20 mrad, 3000 decoys:

| Δ*E*/*E* | 1-col/1-sphere | 2-col/1-sphere (naive) | 2-col/2-sphere | coverage gain |
|---|---|---|---|---|
| 0.15385 | 11 | 10 | **24** | 1.37× |
| 0.04 | 11 | 10 | 23 | 1.10× |
| 0.01 | 11 | 10 | 23 | 1.00× |
| 0.002 | 11 | 11 | **23** | **1.00×** |
| 0 | 11 | 22 | 22 | 1.00× |

The two-sphere config still reports **more than twice** the naive config's votes at Δ*E*/*E* = 0.002,
where `cover` says it reaches 1.00× the reflections. The curve is flat over three decades of split.
Vote count measures how many points got booked, not how much reciprocal space was reached — and when
points are duplicated, those are different things.

The last row is the anchor: at Δ*E*/*E* = 0 exactly, `K1 == K2`, one sphere *is* both spheres, and the
naive config's penalty vanishes discontinuously (10 → 22). A fine sweep locates that transition where
the algebra says it should be:

| Δ*E*/*E* | 2e-3 | 1e-3 | 5e-4 | **2e-4** | 1e-4 | 5e-5 | 0 |
|---|---|---|---|---|---|---|---|
| naive votes | 10 | 10 | 12 | **15** | 20 | 20 | 22 |

1 − cos(20 mrad) = 2.0×10⁻⁴, and 15/22 is half way. The bookkeeping limit is confirmed, and it sits a
hundred-fold below where the physics stops paying.

## (C) The wide-split results — retained, re-labelled

Everything below was measured at Δ*E*/*E* = 15.4% (cell 16/21/25 Å ortho, d_min 3.5 Å, noise 2e-4) and
is unchanged. Read it as *what a wide split buys*, not as a general two-colour result; per (A) the
coverage gain behind these numbers is 1.2–1.6×, not 2×.

Three configs isolate *information* from *method*:
1. **1-col data / 1-sphere** — single-colour baseline
2. **2-col data / 1-sphere** — naive: ignore the 2nd colour (its arcs become noise)
3. **2-col data / 2-sphere** — the two-Ewald accumulator

**Objective tower — `contrast 0.028 6` (NA=28 mrad, same crystals across configs, 3000 decoys):**

| config | #arcs | truth votes | best decoy |
|---|---|---|---|
| 1-col / 1-sphere | 25 | 24 | 5 |
| 2-col / 1-sphere (naive) | 52 | **22** | 6 |
| 2-col / 2-sphere | 52 | **51** | 11 |

> **Corrected.** This table previously read 29/27/6, 59/26/6, 59/56/11, and those numbers do not
> reproduce — `contrast 0.028 6` gives the values above. It is not a regression from the `set_beam`
> refactor: the pre-edit file checked out of git returns the same corrected numbers to the digit, so
> the old table came from a code state or an `ncry` that is no longer recoverable. The shape of the
> result is unchanged (the naive config scores like single colour; two-sphere roughly doubles the
> tower), only the digits. The other two tables were re-run and both hold: `capture 0.020 16`
> reproduces to the digit, and `blind`'s NA=20 mrad / 6k column reproduces exactly
> (4/10 61%, 2/10 19%, 3/10 41%, λ 100%). The two remaining `blind` columns were not re-run.

**Capture radius — `capture` (NA=20 mrad, 16 crystals): fraction recovered to <1° vs seed error**

| config | #arcs | 3° | 6° | 10° |
|---|---|---|---|---|
| 1-col / 1-sphere | 13 | 5/16 | 4/16 | 1/16 |
| 2-col / 1-sphere (naive) | 29 | 4/16 | 3/16 | 2/16 |
| 2-col / 2-sphere | 29 | **10/16** | **6/16** | **4/16** |

**Blind end-to-end — `blind`** (random-SO(3) seeder → anneal; symmetry-aware success, <2° mod 222).
Matched candidate budget — success / (median indexed fraction); median λ-acc for 2-sphere:

| config | NA=20mrad, 6k | NA=20mrad, 15k | **NA=14mrad, 30k** (single-colour starved) |
|---|---|---|---|
| 1-col / 1-sphere | 4/10 (61%) | 6/10 (93%) | 6/12 (88%) |
| 2-col / 1-sphere (naive) | 2/10 (19%) | 3/10 (29%) | 8/12 (**43%**) |
| 2-col / 2-sphere | 3/10 (41%) | 7/10 (92%) | **10/12 (94%), λ 100%** |

The wall-moving column is the low-NA one, and note *why* it is the low-NA one: (A) says the
two-colour coverage gain is largest exactly where NA is smallest (1.63× at 14 mrad). Two colours
substitute for convergence you do not have. Where NA is already generous the substitution is not
worth much — which is the same statement as the shrinking gain down the first row of (A).

The **naive** config's defining, regime-independent flaw is that it structurally indexes only ~half
the pattern (idx 19–43% everywhere); its orientation recovery is erratic.

> Absolute blind rate is gated by the **placeholder random-SO(3) seeder** (fraction of SO(3) within a
> ~6° basin ≈ 6e-5), which is what Yuan's step-2 orientation accumulator replaces. Two evaluation
> subtleties that bit us: (i) success must be judged **modulo the crystal point group** (a correct
> solution can sit 180° from truth for this ortho cell); (ii) the indexed *fraction* is not comparable
> across configs (the naive config's ceiling is ~50%).

## (D) Capture radius — the same crossover, from an independent metric

`splitcap` runs the split sweep scored on capture radius rather than votes: the metric duplicated
points cannot inflate, and therefore the one that *should* corroborate (A) — the second sphere adding
nothing below Δ*E*/*E* ≈ NA should show up as a basin that stops being wider than single colour's.

**Read cell by cell it shows nothing.** At NA = 20 mrad the counts bounce between 1/16 and 10/16 with
no visible trend in any config. Counting noise on 16 Bernoulli trials is ±3 and the per-cell effect is
smaller than that, which is exactly why (A) was measured as geometry instead.

**Pooled over the three NA and the three seed perturbations — 144 trials a cell — it appears:**

| Δ*E*/*E* | 1col/1sph | 2col/1sph | 2col/2sph | 2sph − 1col |
|---|---|---|---|---|
| 0.15385 | 35/144 | 28/144 | 51/144 | +11.1 pts |
| 0.08 | 29/144 | 24/144 | 54/144 | +17.4 |
| 0.04 | 24/144 | 32/144 | 45/144 | +14.6 |
| 0.02 | 32/144 | 23/144 | 46/144 | +9.7 |
| 0.01 | 39/144 | 19/144 | 48/144 | +6.2 |
| 0.005 | 41/144 | 22/144 | 43/144 | +1.4 |
| 0.002 | 43/144 | 22/144 | 35/144 | −5.6 |
| 0 | 41/144 | 34/144 | 34/144 | −4.9 |

The two-sphere arm on its own: **150/432 (34.7%) at wide splits (≥0.04) against 112/432 (25.9%)
at narrow ones (≤0.005), Fisher exact two-sided p = 0.006.** The advantage decays through the same
Δ*E*/*E* ≈ 0.01–0.02 band where (A)'s coverage gain reaches 1.00×, and goes negative below it. Two
independent metrics — reflections reached, and how far a seed can start and still converge — put the
crossover in the same place.

**One caveat on the difference column.** The single-colour baseline is not flat: it *rises* from 20.4%
at wide splits to 28.9% at narrow ones. The sweep holds the MEAN energy fixed, so the 1-colour arm's
own energy slides from 17.5 to 16.25 keV across the rows, and while `cover` says its reflection count
barely moves (14→14), its capture radius evidently does. So part of the decay in `2sph − 1col` is the
baseline climbing rather than the two-sphere arm falling. The two-sphere arm's own wide-vs-narrow
decline is what carries the result, and it is stated above without reference to the baseline.

## Split of work

- **Step 1 (done, this file):** forward sim + arc accumulator + (A)–(D). Self-contained on a Mac:
  `python3 cbxd_twocolor.py {cover|coverna|split|splitcap|contrast|capture|blind|sweep} [na] [ncry]`.
- **Step 2 (Yuan — the money figure):** replace the random seeder with a real orientation
  **accumulator** (arc-Hough over SO(3), her PR#10 lineage); blind-recovery curves at **matched
  candidate budget across (NA, noise)** (folds into deliverable #11). Per (A), NA is the axis that
  moves the wall; a second colour enters as a bonus term whose size is set by Δ*E*/*E* ÷ NA.
- **Step 3 (Yuan):** per-peak λ assignment + within-shot consistency filter (`assign_colour` shows the
  mechanism). **Gated:** first establish that the source's Δ*E*/*E* exceeds 1 − cos(NA). Below that the
  stub still returns a label and the label is meaningless — it is fitting the same sphere twice.
- **Step 4 (Yuan):** run on real convergent-beam frames, which needs geometry refinement +
  peakfinder8 first (on real data geometry, not the beam, is the barrier).

## Notes / gotchas

- **The trap in one line:** the accumulator's colour bookkeeping outlives the second sphere's
  usefulness by a factor of ~100 in Δ*E*/*E*, so a vote-count metric reports a two-colour win long
  after there is nothing to win. Measure coverage, not votes, when the claim is about information.
- Ratios must be taken **per crystal and then median'd**. The median of a union over the median of a
  colour-1 count is not a ratio, and it moved `union/col1` by ~0.1 while I had it the wrong way round.
- `set_beam` rebuilds `HS`, `KS_1`, `KS_2` **and** `CONFIGS`, because `CONFIGS` captures the two colour
  lists by reference; rebinding the lists alone leaves the sweep silently running the old beam.
- The inherited `cbxd_joint.refine` used `scipy` Nelder-Mead from `x0=0`, whose default simplex is
  ~1e-4 rad — it could not move degrees. Fixed by pairing each anneal σ with an **explicit shrinking
  angular simplex** (`_SCHED`); that is what gives the refiner a multi-degree capture radius.
- The centroid seeder's `tol_c` must be **well below the reciprocal-node spacing** (~0.049 here) or
  every orientation matches some node and the seeder is blind (was 0.05 → fixed to 0.02).
- **Wide-cone seeder breakdown (step-2 spec item).** The centroid seeder assumes each arc's centroid
  ≈ its reciprocal node G (parallel-beam approx). That holds for narrow cones and **breaks for wide
  ones**: long arcs pull the centroid off G, so blind recovery collapses — indexed ~22% at NA=50 mrad
  vs 94% at NA=25 mrad. Step 2 should seed from the fitted Kossel-circle centre or the arc endpoints.
  Note the tension this creates with (A): coverage says push NA up, the seeder says it breaks there,
  and resolving that is the step-2 job.
- Realism knobs still to add: monoclinic cell, mosaicity, per-reflection convergence as a q-disk
  rather than a smear, and a source-realistic Δ*E*/*E* rather than the 15.4% default (which
  `set_beam` now makes a one-line change).

## Figures (`cbxd_twocolor_figs.py` regenerates them; PNGs committed alongside)

- `cbxd_ridges.png` — a simulated shot with each reflection drawn as a **ridge** (in-cone Kossel arc)
  + a tangent bar. Single-colour vs two-colour; the two-colour panel shows the near-parallel ridge
  pairs (76→156 ridges). Note per (A) that the pairing is the point: at 15.4% the pairs are *near*
  parallel because they are largely the same reflections twice.
- `cbxd_solved.png` — a **solved shot**: blind-recovered orientation (0.4° from truth, 94% indexed),
  predicted ridges rebuilt from the recovered R overlaid on the observed data.
