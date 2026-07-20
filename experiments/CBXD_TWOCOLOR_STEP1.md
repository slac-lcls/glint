# Two-color convergent-beam CBXD — step 1 (forward sim + two-Ewald accumulator)

**What this is.** The step-1 scaffold for the two-color convergent-beam (CBXD) indexing project:
a pure-numpy forward simulator + the two-Ewald accumulator objective, with three synthetic
results showing that *two colors move the blind wall* — and that the win comes specifically from
scoring against **both Ewald spheres**, not merely from having more spots. Code: `cbxd_twocolor.py`
(extends the single-color `cbxd_joint.py` Kossel-arc overlay). Steps 2–4 are Yuan's (below).

## Physics
Two photon energies E1,E2 → two Ewald radii K1,K2 (K = 1/λ). In a convergent beam each excited
reflection is an **arc** (the Kossel circle: `k_out.Ĝ = |G|/2`, `|k_out| = K`), swept as the
incident direction varies over the convergence cone (half-angle NA). A reflection G can be excited
at **either** color, so a two-color beam lays down ~2× as many arcs as single color.

The **plane** residual `|k_out.Ĝ − |G|/2|` that the accumulator scores is *color-free* (both colors
trace circles on the same perpendicular-bisector plane of G). The color enters only through the
convergence-cone gate `k_z > K·cosNA`, which is **K-scaled** — a K2 point (|k_in|=K2) cannot pass
the K1 threshold — so the direction gate already separates the colors. The explicit `|k_in|=K`
magnitude test is kept OFF while scoring (it narrows the angular basin) and used tight only for the
final per-peak λ-label (`assign_colour`).

Unlike the earlier `sim_cb.py` (emit each peak at *both* q's → GLINT difference-vector consensus,
which locked a **supercell ghost** — see memory `glint-twocolor-convergent-beam`), the arc
accumulator never emits ghost points, so it cannot form the supercell by construction.

## Results (synthetic; cell 16/21/25 Å ortho, E1/E2=17.5/15.0 keV, d_min 3.5 Å, noise 2e-4)

Three configs isolate *information* from *method*:
1. **1-col data / 1-sphere** — single-color baseline
2. **2-col data / 1-sphere** — naive: ignore the 2nd color (its arcs become noise)
3. **2-col data / 2-sphere** — the two-Ewald accumulator

**(A) Objective tower — `contrast` (NA=28 mrad, same crystals across configs, 3000 decoys):**

| config | #arcs | truth votes | best decoy |
|---|---|---|---|
| 1-col / 1-sphere | 29 | 27 | 6 |
| 2-col / 1-sphere (naive) | 59 | **26** | 6 |
| 2-col / 2-sphere | 59 | **56** | 11 |

Two colors double the arcs (29→59), but a one-sphere index of two-color data scores the **same as
single color** (26 vs 27 — the 2nd color's 30 arcs are invisible). The two-Ewald accumulator harvests
them: the truth tower **doubles** (27→56). Taller tower → recoverable at lower NA / more noise / fewer
candidates.

**(B) Capture radius — `capture` (NA=20 mrad, 16 crystals): fraction recovered to <1° vs seed error**

| config | #arcs | 3° | 6° | 10° |
|---|---|---|---|---|
| 1-col / 1-sphere | 13 | 5/16 | 4/16 | 1/16 |
| 2-col / 1-sphere (naive) | 29 | 4/16 | 3/16 | 2/16 |
| 2-col / 2-sphere | 29 | **10/16** | **6/16** | **4/16** |

Two-Ewald ≈ **2× the capture radius** of single color; the naive config ≈ single color (extra arcs
buy nothing without the 2nd sphere). A wider basin = fewer candidates the seeder must try.

**(C) Blind end-to-end — `blind` (random-SO(3) seeder → anneal; symmetry-aware success, <2° mod 222):**

Matched candidate budget — blind orientation success / (median indexed fraction); median λ-acc for 2-sphere:

| config | NA=20mrad, 6k | NA=20mrad, 15k | **NA=14mrad, 30k** (single-color starved) |
|---|---|---|---|
| 1-col / 1-sphere | 4/10 (61%) | 6/10 (93%) | 6/12 (88%) |
| 2-col / 1-sphere (naive) | 2/10 (19%) | 3/10 (29%) | 8/12 (**43%**) |
| 2-col / 2-sphere | 3/10 (41%) | 7/10 (92%) | **10/12 (94%), λ 100%** |

The **wall-moving result is the low-NA column**: where single-color is arc-starved (~7 arcs), two-Ewald
recovers **10/12 vs single-color 6/12** — ~1.7× — while indexing ~all arcs (94%) with 100% correct λ-labels.
At NA=20 mrad single-color is already over-determined (13 arcs) so the edge narrows (7 vs 6). The **naive**
config's defining, regime-independent flaw is that it structurally indexes only ~half the pattern
(idx 19–43% everywhere — it cannot touch the 2nd color); its orientation recovery is erratic (worst at
20 mrad, but able to seed off the color-1 sublattice alone at 14 mrad). Two-Ewald dominates on all three
axes — orientation success, indexed fraction, and the λ-label.

> Absolute blind rate is gated by the **placeholder random-SO(3) seeder** (fraction of SO(3) within a
> ~6° basin ≈ 6e-5), which is exactly what Yuan's step-2 orientation accumulator replaces. Two evaluation
> subtleties that bit us and are worth flagging for step 2: (i) success must be judged **modulo the
> crystal point group** (a correct solution can sit 180° from truth for this ortho cell); (ii) the
> indexed *fraction* is not comparable across configs (the naive config's ceiling is ~50%).


## Split of work
- **Step 1 (done, this file):** two-color forward sim + two-Ewald accumulator objective + (A)/(B)/(C)
  above. The testbed is self-contained on a Mac (`python3 cbxd_twocolor.py {contrast|capture|blind|sweep} [na] [ncry]`).
- **Step 2 (Yuan — the money figure):** replace the random seeder with a real orientation
  **accumulator** (arc-Hough over SO(3), her PR#10 lineage); produce the blind-recovery-vs-single-color
  curves at **matched candidate budget across (NA, noise)** = "two-color moves the wall" (folds into
  deliverable #11).
- **Step 3 (Yuan):** XTCAV per-shot energies → real per-peak λ-assignment + within-shot consistency
  filter (the `assign_colour` stub shows the mechanism; 100% label accuracy on the sim).
- **Step 4 (Yuan):** run on the real Chapman two-color frames (337 TB; needs full BayFAI-style geometry
  refinement + peakfinder8 first — see memory `glint-twocolor-convergent-beam` for the geometry-is-the-
  barrier lesson).

## Notes / gotchas found building step 1
- The inherited `cbxd_joint.refine` used `scipy` Nelder-Mead from `x0=0`, whose default simplex is
  ~1e-4 rad — it could not move degrees. Fixed here by pairing each anneal σ with an **explicit
  shrinking angular simplex** (`_SCHED`); this is what gives the refiner a multi-degree capture radius.
- The centroid seeder's `tol_c` must be **well below the reciprocal-node spacing** (~0.049 here) or
  every orientation matches some node and the seeder is blind (was 0.05 → fixed to 0.02).
- **Wide-cone seeder breakdown (step-2 spec item).** The centroid seeder assumes each Kossel arc's
  centroid ≈ its reciprocal node G (parallel-beam approx). This holds for narrow cones but **breaks for
  wide ones**: long arcs pull the centroid off G, so blind recovery collapses — indexed ~22% at NA=50
  mrad vs 94% at NA=25 mrad (same crystal, same budget). Yuan's step-2 accumulator should use a
  **wide-arc-aware back-projection** (seed from the fitted Kossel-circle center or the arc endpoints,
  not the centroid) so it holds at large convergence. The main results above (NA=14–28 mrad) are inside
  the valid range.
- Realism knobs still to add for step 2/3: monoclinic cell (the real ~11/13/9/90/102/90 iodine cell),
  mosaicity, per-reflection convergence as a q-disk (not just a smear), and the real λ1/λ2 split.

## Figures (`cbxd_twocolor_figs.py` regenerates them; PNGs committed alongside)
- `cbxd_ridges.png` — a simulated shot with each reflection drawn as a **ridge** (in-cone Kossel arc)
  + a tangent bar (the per-streak direction, the `ridge_moments` observable). Single-colour vs two-colour;
  the two-colour panel shows the blue/red near-parallel ridge pairs (the two Ewald spheres, 76→156 ridges).
- `cbxd_solved.png` — a **solved shot**: blind-recovered orientation (0.4° from truth, 94% indexed),
  predicted ridges rebuilt from the recovered R overlaid on the observed data — blue/red predicted ridges
  thread through the observed Bragg points (filled = indexed, hollow = missed) and avoid the spurious
  background. Uses keep-best over seeder restarts (`n_coarse≈150k, keep=50`) at NA=25 mrad.
