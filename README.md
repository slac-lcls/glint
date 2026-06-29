# fftindex — sparse single-shot crystallography indexing by 3D FFT

**Thesis.** Index sparse single-shot (SFX/nanocrystal) diffraction by taking the
**3D FFT of the reciprocal peak cloud** and reading direct-lattice vectors off the
transform — and survive the sparse regime (where the 3D FFT normally starves) by
replacing hand-tuned peak detection with a **learned peakfinder on the FFT volume**.

This keeps the sparse-data target of *Compressive Auto-Indexing in Femtosecond
Nanocrystallography* (arXiv:1011.3072) but swaps its basis-pursuit recovery for a
3D FFT, and swaps thresholded peak detection for a PeakNet-style learned detector
(arXiv:2303.15301), applied **downstream on the transform** to find the axes.

## Why 3D FFT, and how it relates to mosflm/DPS

The mosflm/DPS "1D-FFT-over-directions" method and the 3D FFT are the *same object*.
By the Fourier projection-slice theorem, projecting the cloud onto a direction `u`
and 1D-FFT'ing it equals the central ray `x = s·u` of the full 3D transform:

    F̂[ proj_u(ρ) ](s) = ρ̂(s·u).

So the 1D method just samples the one 3D volume `ρ̂` along rays through the origin.
The full volume retains every pairwise phase coincidence a single ray discards —
which is exactly the information the sparse regime needs, and why a learned detector
on the *volume* is the natural upper envelope of all 1D-projection indexers.
(`transform.central_rays` reads those rays out of the same volume → the free 1D
baseline.)

## Layout

    fftindex/
      lattice.py     cell <-> basis conventions, SO(3) sampling
      simulate.py    monochromatic single-shot forward model (-> g_i + ground truth)
      transform.py   fft_volume() : the indexing volume;  central_rays() : 1D baseline
      peakfind.py    find_peaks_classical() — baseline; learned detector slots in here
      features.py    peak_features() : per-peak fringe/DPS-consistency/sharpness
      dataset.py     make_dataset() : labeled (features, true-axis?) from the sim
      detector.py    LearnedPeakFinder : re-rank candidate peaks by a trained model
      cnn_dataset.py make_cnn_sample() : (volume, peakness-heatmap) pairs (numpy)
      cnn.py         UNet3D + CNNPeakFinder : 3D-CNN peakfinder (torch, guarded)
      index.py       estimate_grid_n() + search_basis() + refine() : volume -> (M,hkl)
      metrics.py     score() : lattice-match success / wrong-index / no-index
    experiments/
      demo.py                  smoke test + indexing-rate-vs-spot-count sweep
      baseline_curve.py        honest characterization (classical, robust selection)
      train_detector.py        train + save the feature re-ranker (detector_rf.joblib)
      learned_vs_classical.py  head-to-head: learned re-ranker vs classical
      symmetry_eval[_hard].py  cross-symmetry generalization (mild / hard setting)
      train_cnn.py             train the 3D-CNN peakfinder (Perlmutter; see docs/)

## Status (v0 baseline — honest lattice-match metric)

Success = recovered basis matches the true lattice by an **integer, unimodular**
transform `T = M_true⁻¹ M_rec` (|det T|=1). This rejects the spurious sub-lattices
that a naive "fraction of spots indexed" test counts as wins (`metrics.py`).

Reported as solved% / wrong-index% / no-index% (`experiments/baseline_curve.py`),
where wrong-index = returned a confident-but-wrong lattice (the dangerous case).

**Robust indexer** (parsimony selection + fail-safe rejection):

| cut | result |
|---|---|
| vs spots (clean) | solved 0% @15 · 35% @30 · 75% @40 · 95% @60 · 100% @90 |
| vs jitter σ (n=60) | solid to σ=0.001, then **cliff**: 10% @0.002, 0% @0.004 |
| vs spurious frac (n=60) | solved **85% @0.1** · 75% @0.2 · 55% @0.4 (wrong ≈0%) |

The spurious cut is the headline: a naive max-inlier selection collapsed to **15%**
at 10% spurious; parsimony selection recovers it to **85%**.

What worked / what didn't this pass:
- **Parsimony selection IS the spurious fix** — among cells indexing ≥70% of spots,
  take the *smallest volume* (coarsest reciprocal, fewest predicted reflections).
  Maximizing raw inliers rewards an over-fit lattice that absorbs spurious spots.
- **Fail-safe rejection works** — `min_inlier_frac` returns "no index" instead of a
  confident wrong cell; wrong-index% is ~0 across the board.
- **Difference-vector (Patterson) seeding was NEUTRAL** (seed on==off everywhere) —
  the real-space |F(x)| peaks already supply the winning hypotheses. Kept in
  `seed.py`, `seed_diff=False` default, for degraded-peak/real-data regimes.
- **Runtime fixed** — normal-equations refit (no SVD `matrix_rank`/`pinv`) +
  setting `OMP_NUM_THREADS` *before* importing numpy: sweep `sys` time 10m → 4s.
- **Cost**: clean-*sparse* regressed slightly (n=40: 95%→75%) — a robustness/accuracy
  trade. **Jitter cliff (σ≈0.0015) unchanged** — it's a peak-precision problem, not
  selection; the ML peakfinder / better localization is the lever there.

Earlier hard-won facts: CIC deposit not Gaussian (envelope exp(-2π²σ²|x|²) crushes
long axes); peak height = DPS fringe score so pick basis by amplitude not length;
absolute (not relative) inlier tolerance; the direct-FT localizer is redundant with
refine()'s LSQ fit (`localize=False`).

## Learned peakfinder (feature re-ranker) — validates the thesis

`detector.LearnedPeakFinder` runs the classical finder for a generous candidate
pool, then re-ranks by a RandomForest trained on `features.peak_features`
(amp / inlier_frac / med_resid / len / sharpness / harmonic support). Held-out
per-peak AP = 0.999. Plugged into `index_shot`, it lifts exactly the two failure
modes selection couldn't (`experiments/learned_vs_classical.py`, solved%):

| failure mode | classical | learned |
|---|---|---|
| sparse floor n=25 / n=30 / n=40 | 15 / 35 / 75 | **50 / 70 / 100** |
| jitter cliff σ=0.0015 / 0.002 | 45 / 10 | **90 / 50** |
| spurious 0.2 / 0.4 | 75 / 55 | **95 / 95** |

So a learned re-ranker pulls true axes back into `search_basis`'s top-k when
jitter/sparsity/spurious bury them in raw amplitude — the 3D FFT survives the sparse
regime once the peakfinder is learned. This is the feature-based proxy; it de-risks
(and shares dataset + interface with) the eventual PeakNet-style 3D CNN.

### Adaptive grid — the 96 Å ceiling, lifted

The real-space grid spans only |x| ≤ `n/(4·qmax)` ≈ **96 Å at n=128**, so any cell
axis longer than that fell off the volume and was unindexable (41% of a 30–120 Å
range). `transform.estimate_grid_n` now sizes the grid from the data (longest axis ≈
1/3rd-smallest pairwise spot difference), and `index_shot` **escalates** to
`n_max=256` once if the estimate is short (the longest axis often has no observed
adjacent pair). Result: large orthorhombic cells (70–120 Å) **0/12 → 12/12**;
clean full-range indexing **75–100% across all symmetries**.

### Cross-symmetry generalization (memorization probe resolved; learned win is regime-dependent)

`lattice.random_cell` (7 symmetry classes) + `make_dataset(symmetry="random")` train a
mixed-cell model; `experiments/symmetry_eval.py` compares per symmetry over the full
30–120 Å range (random cell per trial, **mild** setting n=45, σ=0.001, 10% spurious):

| symmetry | classical | learn-mixed |
|---|---|---|
| monoclinic *(sheared)* | 47% | **80%** (+33) |
| tetragonal | 53% | **73%** (+20) |
| hexagonal | 67% | **80%** (+13) |
| orthorhombic | 60% | 60% |
| cubic | 67% | 53% (**−14**) |
| trigonal | 67% | 53% (**−14**) |
| triclinic *(sheared)* | 80% | 60% (**−20**) |

At this **mild** setting the learned re-ranker is roughly a wash (little headroom),
and noisy enough to show apparent regressions. The **hard** setting
(`symmetry_eval_hard.py`, n=30, σ=0.0015, 20% spurious) is the tiebreaker:

| symmetry | classical | learn-mixed |
|---|---|---|
| monoclinic *(sheared)* | 13% | **27%** (+13) |
| triclinic *(sheared)* | 20% | **33%** (+13) |
| hexagonal | 20% | **33%** (+13) |
| tetragonal | 20% | **27%** (+7) |
| cubic / orthorhombic / trigonal | 33 / 20 / 27% | 33 / 20 / 27% (±0) |

**Verdict: where there is headroom, learn-mixed ≥ classical everywhere — it never
hurts and helps most on sheared/hexagonal cells.** The mild-setting cubic/trigonal
"regression" does *not* reproduce, so it was low-headroom variance (~±12% at 15
trials), not a real failure. Memorization stays resolved — with mixed training
`med_resid`+`amp`+`inlier_frac` (consistency) outweigh `len` (0.62 vs 0.29). Absolute
rates are low at this setting because sparse + jittery + spurious + *unknown cell* is
genuinely hard — that residual is the 3D-CNN's target.

### 3D-CNN peakfinder (built for Perlmutter)

PeakNet-style 3D U-Net that segments a *peakness heatmap* over the FFT volume
(`cnn.py`), trained on (volume, heatmap) pairs from the same simulator
(`cnn_dataset.py`) — Gaussian blobs at the true lattice vectors, loss = BCE + soft
Dice. `CNNPeakFinder` matches the `index_shot` peakfinder interface (resamples the
volume to the model grid, maps peaks back through the physical extent, so auto/
escalated grids work). Torch is a **guarded** import — the repo runs without it.

**Trained and tested on a Perlmutter A100** (`module load pytorch/2.8.0`; 12 epochs,
1500 vols/epoch, n=96, ~47 s/epoch; loss 1.27 → 0.57). Plugged into `index_shot`
(`cnn_vs_classical.py`), the trained CNN peakfinder **dominates the hard regime** —
random cells 30–64 Å:

| setting | classical | **CNN** |
|---|---|---|
| clean n=40 | 95% | **100%** |
| **n=30, σ=0.0015, 20% spurious** | 25% | **100%** |
| n=25, 10% spurious | 40% | **90%** |

The n=30 hard case (25% → 100%) is exactly where the *feature* re-ranker only reached
~30%: a true 3D detector denoises the volume so cleanly that `search_basis` is trivial.
`solved` requires a lattice-match, so these are correct cells. (Bug found in the run:
inference must normalize the volume to [0,1] like training — without it the model
saturates to all-zeros. Fixed in `CNNPeakFinder`.)

v1 caveats: trained/tested on cells ~30–64 Å (fit n=96); 12-epoch demo model; not yet
tested on large cells or real data. The numpy dataset generator is also validated
locally (heatmap peaks land on true lattice vectors, `|M⁻¹x−round|≈0.014`). NERSC
recipe: `docs/perlmutter_cnn.md`.

### Bravais centering (FCC, BCC, base-centered)

`simulate_shot(centering=...)` applies the systematic-absence rule (`I`: h+k+l even;
`F`: hkl same parity; `C`: h+k even) and sets ground-truth `M` to the **primitive**
lattice — the lattice the observed spots actually form (a rhombohedral cell for
FCC/BCC), which is what the indexer recovers and what `score` checks. Earlier tests
covered the 7 crystal *systems* but only primitive (P); "cubic" was simple cubic.

Centered lattices index *at least as well as* primitive — their generic rhombohedral
primitive cell dodges simple-cubic's high-symmetry degeneracy (`centering_eval.py`,
mild setting: classical BCC/FCC 100% vs simple-cubic 85%). On Perlmutter at the **hard**
setting (n=30, σ=0.0015, 20% spurious; `cnn_centering_eval.py`):

| centering | classical | CNN |
|---|---|---|
| P (simple cubic) | 0% | **65%** |
| I (BCC) | 40% | **90%** |
| F (FCC) | 60% | **95%** |

The CNN handles FCC/BCC excellently and rescues the hardest case (simple cubic 0→65%).
FCC (primitive α=60°, *outside* the CNN's training α-range) scored highest — peak
detection in the volume is angle-agnostic, so the out-of-distribution worry didn't
materialize. (Not yet implemented: R-centering / rhombohedral hexagonal setting.)

### Multi-shot consensus (sparsity-floor lever) — works

The sparsity floor (n<25) is the one regime nothing else moved. Many shots share ONE
cell whose metric tensor is rotation-invariant, so `multishot.py` pools them: index
each independently, find the cell the indexed minority agree on (`consensus_cell`, via
a rotation-invariant length-spectrum signature), then re-index every shot against that
cell with a **taketwo-style pair-angle matcher** (`index_known_pairangle`): match
observed spot *pairs* to the known reciprocal lattice by length+angle, solve the
orientation, and vote. It's peak-independent (no FFT volume) so it indexes far below
the single-shot floor, and returns the known cell by construction. Fixed cell, 50
shots, jitter 0.001, 10% spurious:

| n_spots | single-shot | consensus | known-cell rescue |
|---|---|---|---|
| 18 | 28% | correct (6/50) | **56%** |
| 22 | 50% | correct (24/50) | **94%** |
| 28 | 70% | correct (34/50) | **94%** |
| 35 | 78% | correct (36/50) | **100%** |

Two findings: (1) **the consensus cell emerges from the indexed minority** — at n=18,
72% of shots fail individually yet the shared cell is recovered *correctly*; (2) once
the cell is known, the pair-angle matcher **rescues nearly everything down to n≈18**
(28→56, 50→94).

**Bootstrap test** (`bootstrap.py`, shot-count sweep at extreme sparsity) sharpens the
limit into *two* floors:
- **Cell recovery is a shot-count knob.** n=15 (single-shot 5.5%): consensus forms and
  grows with N (support 3→7 as N=50→150) — more shots ⇒ the cell is recovered.
- **Per-shot rescue has its own floor.** Knowing the cell, the matcher still needs
  enough spots: rescue is 94% @n=22, 56% @n=18, **only 20% @n=15**.
- **n=12 (single-shot 1%): no consensus even at 200 shots** (support stalls at 2) —
  single-shot bootstrap can't start when single-shot ≈ 0.

So it's a knob for *cell recovery* but not a free pass for *per-shot indexing*; the
genuinely extreme regime (n≤12) needs a method that doesn't bootstrap from any single
shot at all — i.e. a **joint fit of the shared metric tensor across all shots
simultaneously** (full SO(3) multi-shot synchronization).

Note: an earlier FFT-peak `index_known` (length-spectrum filter) half-worked and
regressed at n≥28 — superseded by the pair-angle matcher, which sidesteps lattice-
equality, so full Niggli reduction turned out unnecessary for the rescue.

### Joint multi-shot cell recovery — reaches n≤6, but powder-limited

The fix for the n≤12 bootstrap wall: don't bootstrap from any single shot. Every spot
gives `|q|² = hᵀMh` (rotation-invariant), so pooling `|q|²` over many shots is a
**powder-like spectrum** of allowed reflection lengths; `joint.py` fits the shared
metric tensor to it with **no per-shot indexing** (and no FFT — just simulate + pool).

It genuinely reaches the extreme regime: for a favorable tetragonal cell it recovers
the cell at **n=4 and n=6 spots/shot** — where single-shot indexing is **0%** and the
bootstrap is impossible — by pooling ~1500 shots. Nothing else gets near n≤6.

**Honest limit: this is the powder-indexing problem and inherits its hard cases.** It
nails cubic a=48, tetragonal 55/80, and the n=4–6 case above, but slips for large cubic
(a=65: `(A*,A*,A*)` vs `(A*,2A*,2A*)` ambiguity) and at n=8–12 of the same cell.

**Tried the angular fix** (`fit_joint`, pooled `(|qᵢ|²,|qⱼ|²,qᵢ·qⱼ)` triples to re-rank
candidates) — it helped partially (recovers the first axis on large cubic) but **does
not resolve it**, for two measured reasons: (1) the correct cell isn't even in the
powder top-k — the upstream over-prediction penalty is miscalibrated for *dense*
spectra (a real M20/DICVOL-class FOM problem), so re-ranking has nothing to fix; (2)
the discriminating configurations are *absent* in sparse data — 0 equal-shortest-length
pairs at 90° in 778 short pairs, because two `a*`-type vectors rarely co-occur in one
sparse shot. So the angular idea is sound but starved of data at high sparsity, and the
powder FOM is the bigger blocker.

Net: the joint pooled method **works for small / higher-symmetry cells down to n≈4–6**
and is the only thing that reaches there, but robust general-cell joint indexing is the
full powder-indexing problem (a known-hard field) — not a quick win.

## Next (open)

1. **Joint indexing is now a scoped research problem**, not a tweak: a real powder FOM
   (de Wolff M20 / DICVOL/McMaille-class peak extraction + scoring) for the dense-cell
   cell-finding, with the angular triples as a tie-breaker where data permits. Worth it
   only if extreme-sparsity multi-shot (n≤8) is a target use case.
2. Lower-hanging: scale the CNN (longer training, multi-scale past n=96); GPU
   `fft_volume`→cuFFT; real CXI data via PeakNet upstream; pink-beam forward model.
2. **Scale the CNN win** — longer training, past the n=96 cell-size scope (multi-scale),
   re-run cross-symmetry with the CNN.
3. GPU `fft_volume`→cuFFT; tighten residual clean wrong-index; real CXI data via
   PeakNet upstream; pink-beam forward model.
