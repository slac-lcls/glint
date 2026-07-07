# Research plan — position-aware coherent cluster-FFT seeding (dense front end)

> Parallel research directions extending the 2026-07-06 dense seed-combine study. Origin: S. Marchesini's
> questions on how the per-cluster FFTs are summed, cluster selection, grouping, and aliasing.
> **Verified baseline (this study, n=150, 10 dense cells ×15 rot, A100):** pooled per-cluster peak-pick = 100%;
> full-coherent phased-sum single-volume = 97.3% at ~1.5× faster M1 + 7× fewer seeds; random grouping /
> partial-coherence = NEGATIVE (monotone worse: G1 97.3 > G2 95.3 > G3 92.7); L∞/Lp phase-blind = ~82–96%
> (coherence is the lever, not the Lp choice). Wired: `COHERENT_SEEDS` option (default off). `tricl` is the
> irreducible hard cell for every summed variant. See memory `glint-dense-coherent-seeds`.
>
> **KEY CAVEAT that motivates this plan:** the grouping negative used **random** groups (`gid = k//s` over the
> randomly-ordered center list). Position-structured grouping is UNTESTED — and physically should behave
> differently. This plan tests it. Harness to reuse: `experiments/dense_seed_group.py` (gen_clouds + run_core +
> is_correct == glint_cells.py:14; group_coherent). Coherent seed fn shipped: `glint_fast._cluster_fft_seeds_coherent`.

## Axis A — sampling / wraparound in the OUTPUT (real) space
**CORRECTION (user):** the aliasing is in the OTHER space. We deposit the peaks in RECIPROCAL space (`rho`,
spacing `dq`) and FFT to `F(x)` in REAL space; the "input autocorrelation must fit in half the grid" pad rule
is the phase-retrieval oversampling condition (object in real space) and does NOT apply here. The real effects,
all in the real-space OUTPUT `F(x)`:
1. **Wraparound / period** = `1/dq = 2·fov`: a lattice vector `|x| > fov` folds to the opposite side. Already
   cut at `0.9·fov` — but check it's tight enough for the long-axis cells (large_tet 150 Å vs fov=280).
2. **Resolution** `dx = 2·fov/n_grid` (≈4.2 Å at 96/200): two lattice vectors closer than `dx` merge in the
   output → they can't be separated as distinct seeds. Matters for near-degenerate axes.
3. **Deposit quantization** — `round(q_j/dq)` snaps each peak to the `dq` grid (≤`dq/2` position error) → a
   real-space PHASE error `δ(x·q) ∝ dq·|x|` that GROWS with `|x|` → decoheres/broadens the LONG-vector peaks
   (the paper's "CIC digitization error ∝ δq·x kills the long axis"). THIS, not input-padding, is the real limiter.
- **A1. Map it.** Per cell log the fov-fill and, for the true long axis, the peak SNR/width vs `|x|` — confirm
  the long-axis decoherence is quantization (2) not wraparound (1).
- **A2. Cut the quantization.** Sub-pixel / soft deposit (trilinear CIC or Gaussian splat instead of `round`),
  and/or finer `dq` (larger `n_grid` at fixed fov). Predict: sharper long-axis peaks → recovers the long-axis
  cells for ALL variants (esp. coherent, whose phase-back is most quantization-sensitive).
- **A3. fov/`n_grid` tradeoff sweep** — period (fov) vs resolution (`dx`) vs quantization; find the knee per cell.
- **A4. Free win check:** apply the winning deposit to the shipped POOLED path too (independent of the option).

## Axis B — position-aware CLUSTER selection (how centers + members are chosen)
Current: uniform-random centers over peaks + 300 nearest neighbours (a local patch).
- **B1. Coverage-spread centers** (farthest-point / k-means++ over peak positions) → tile reciprocal space,
  avoid redundant overlapping clusters. vs random. Metric: seed COMPLETENESS (fraction of true lattice vectors
  with a seed within tol) + rate.
- **B2. Density-adaptive `cpts`** — cluster radius set by local density (fixed radius, variable count) instead
  of fixed 300 nearest. Keeps each cluster's spatial extent bounded (ties into Axis A padding).
- **B3. Shell-stratified clusters** — one cluster per resolution shell; does per-shell coherent seeding surface
  weak high-|q| vectors better?

## Axis C — position-aware GROUPING (the main new hypothesis)  ★
The random-grouping negative may reverse under spatial structure. Physics (user): **adjacent** clusters summed
coherently TILE into a larger coherent aperture → sharper, higher-resolution peak (helps long/weak vectors);
**distant** clusters summed coherently interfere → sidelobe RIPPLES → false peaks.
- **C1. Coherent-adjacent groups:** k-means the CENTER positions into G spatial groups; coherent-sum within
  each (contiguous aperture), combine across groups incoherently (L1/max). Sweep G. Predict: recovers tricl /
  anisotropic cells that full-random-coherent misses (aperture synthesis on the weak axis).
- **C2. Coherent-spread groups (proper Welch):** each group INTERLEAVES distant clusters (each group ≈ a
  full-aperture independent look); combine incoherently across → variance/ripple reduction. Compare C1 vs C2 vs
  random (this study's negative) vs full-coherent (97.3%).
- **C3. Adjacency-weighted coherence:** continuous version — weight each cluster's phase-back contribution by a
  kernel in center-distance, so nearby clusters add coherently and far ones are down-weighted (soft aperture).
- Metric: rate (esp. per-cell tricl/thaum/ortho_lg recovery) + M1 + seeds, **n≥150** (the verified-stats lesson).

## Axis D — aperture / resolution analysis (mechanism)
- **D1.** Quantify the coherent-peak FWHM vs group adjacency (C1) — does contiguous tiling measurably sharpen
  the autocorrelation peak (∝ 1/aperture)? Does a sharper seed cut downstream M3 refine steps?
- **D2.** Ripple map: for spread groups (C2), image the |sum| sidelobe structure — confirm distant-cluster
  interference creates the false peaks, and that incoherent across-group averaging suppresses them.

## Axis E — coherence annealing / homotopy (user: "the selector could switch incoherent→coherent along the way")
A continuation on a coherence parameter rather than a fixed choice: DETECT broad/robust, then SHARPEN to
coherent for precise localization — coarse-to-fine across the incoherent↔coherent axis.
- **E1. Two-stage detect→localize:** peak-DETECT candidate regions on a robust volume, then re-LOCALIZE each
  on the coherent volume `|Σ e^{iφ}F_c|` (sharp) before seeding. NB the robust detector should be the POOLED
  per-cluster field (verified 100%), NOT incoherent `Σ|F_c|` — the data show incoherent BURIES weak vectors
  (82%, background pileup), so "incoherent = robust" is false here; pooled is the robust stage.
- **E2. Blend homotopy:** `V(t) = (1−t)·Σ|F_c| + t·|Σ e^{iφ}F_c|`, sweep `t: 0→1` and track peaks by
  continuation (follow each maximum as the volume sharpens). Does the path reach the coherent optima while
  keeping the incoherent detections that coherent-cold misses (tricl)?
- **E3. Fold into M3.** GLINT's `refine_vec` ALREADY anneals a smooth→sharp objective (OBJSIG / cos² sharpen at
  the end). Extend that schedule to drive the seed-combine coherence in lockstep, so front-end coherence and
  refiner sharpness harden together — the seed-domain instance of the soft→hard schedule (memory
  `constraint-hardening-schedule`; the kick pays only while the restoring constraint is co-active + hardening).
- Metric: does annealing beat both fixed endpoints (pooled 100% / coherent 97%) on rate×speed, esp. recover tricl?

## Axis F — weighted FFT deposit (~1/|q|, xgandalf-style)  [user question]
Current: `_cluster_fft_seeds` deposits `torch.ones` — every peak weighted EQUALLY in the autocorrelation, NO
1/|q| weight. (Downstream M2/M3 DOES weight: `invq_weight(Q)` ≈ `QPOW=1.0` ~1/|q| on the ORIGINAL q; xgandalf's
1/|q|² measured WORSE in GLINT — notes_m2_tuning.) So only the SEEDING is unweighted.
- **F1. Weight the deposit** `scatter_add_(…, w_j)` with `w_j ∝ 1/|q_j|^p`, `p∈{0,0.5,1,2}` — using the
  TRUE-ORIGIN `|q_j|` (resolution), NOT the centroid-recentered local `|q−q0|`. Emphasizes strong low-res peaks
  in the autocorrelation. Predict: may sharpen toward the true SHORT lattice vectors, or hurt by down-weighting
  the high-q fine structure. Sweep p, rate/M1/per-cell.
- **F2. Recentering note:** peaks are recentered by CENTROID for POSITION only (local grid + |F| translation-
  invariance); the weight must use true-origin |q| (resolution), never the recentered local magnitude.
- **F3.** Composes with A (soft deposit) and E (annealing) — weighted + sub-pixel splat is one op; and a
  `p`-schedule is another annealing knob (heavy low-res detect → flat high-res localize).

## Execution / parallelism
Independent axes → run concurrently as self-contained GPU sweeps (S3DF ampere / NERSC), each reusing the
`dense_seed_group.py` harness. Each sweep: seeds → identical GLINT core → rate/M1/seeds/per-cell misses.
**Verification protocol (hard-learned this session):** n=20 fabricates spurious optima (±2–3 cloud rotation
noise invented BOTH the L∞ and random-G2 "wins"); confirm any claimed optimum at **n≥150** with a fresh
high-rep run before believing a ≤3-cloud difference. Report per-cell misses, not just aggregate rate.

## Deliverable / exit
- If a position-aware strategy (likely C1 coherent-adjacent, or A2 padding) robustly beats full-coherent —
  especially recovering `tricl` toward 100% at the coherent speed — wire it as a new `COHERENT_SEEDS` mode and
  update memory. Otherwise document as clean negatives (like random grouping).
- Axis A (padding) is the highest-EV / lowest-risk (could improve the shipped pooled path itself); do it first.
- Scope: dense/rotation front end only; the sparse SFX Fibonacci path (GLINT's headline niche) is untouched.
- Credit: coherent phased-sum = user's phase question (the verified win); position-aware grouping + anti-alias
  padding = user's follow-up hypotheses (this plan).
