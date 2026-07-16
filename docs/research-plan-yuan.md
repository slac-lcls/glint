# GLINT research plan — Yuan

A focused, self-contained plan. The broad map for the whole team is [`../ROADMAP.md`](../ROADMAP.md);
the "what is indexing" primer is [`onboarding.md`](onboarding.md) §0; the algorithm write-up is the
paper (*Architecture*, *Candidate generation*, *Throughput*, *Outlook* sections).

Two thrusts, both playing to a maths + GPU strength, and both circling an idea you already know from
the sync / Gramian work on sharpy: **a weak per-instance signal, pooled across many instances, becomes
a strong consensus.** In indexing that principle is what breaks the single-frame ceiling; in CBXD it is
(we think) what breaks the *blind* wall. So this is the same music in a new key.

---

## Thrust A (flagship) — break the *blind* convergent-beam (CBXD) wall

**The problem, in one picture.** Instead of a parallel beam, focus a *cone* of incident directions onto
the crystal (convergent-beam X-ray diffraction, CBXD — Chapman group, arXiv:2602.14402). Now each
reflection is excited not at a single point but along a **Kossel circle**: the intersection of the Ewald
sphere \(|k|=1/\lambda\) with the Bragg plane \((q-G/2)\cdot G=0\). The lens aperture admits the arc of
that circle whose incident wavevector lies in the convergence cone, so every spot becomes a curved
**streak** (this is Fig. 4 of the paper — the simulated pattern we just put back; the figure lives with
the paper repo as `fig_cbxd.png`, ask Stefano for the generator).

**Why it's a beautiful inverse problem.** A single still gives you only the thin Ewald slice — a 2-D
shadow of a 3-D lattice, which is exactly why blind SFX indexing has a ceiling. A CBXD streak carries
*more*: its **curvature** (the sagitta of the admitted arc) encodes the out-of-plane component of the
reflection — the third dimension a point-spot throws away. In principle a single convergent-beam shot
could fix orientation **and** cell with no prior. In practice:

- **Known-cell CBXD is already solved** (97% at the true orientation, `experiments/cbxd_joint.py`).
- **Blind CBXD is walled** by per-streak precision: the sagitta scales like \(\sim\!\text{NA}^2\), so
  fitting one streak's curvature is hopelessly noise-sensitive at realistic detector resolution
  (it collapses to 0% once streak-point noise exceeds \(\sim\!2\times10^{-5}\,\text{Å}^{-1}\)).

**The idea that should break it (your part).** Don't fit streaks one at a time. **Pool them.** Search
the orientation so that *all* predicted Kossel arcs overlay *all* observed streaks at once — a
**generalized Hough accumulation over arcs**, the convergent-beam analog of GLINT's gridless direct-sum
objective \(\sum_i w_i\cos(2\pi\,x\cdot q_i)\). Each streak votes for the orientations consistent with
it; the true orientation is the peak in the accumulator where thousands of streaks agree. Pooling
regularizes the weak per-streak curvature the same way cross-frame consensus regularizes weak single
frames. The evidence it works: the joint objective already stays robust to
\(2\times10^{-4}\,\text{Å}^{-1}\) streak-point noise **at the true orientation** (97% of streak points,
zero spurious) — an order of magnitude past where per-streak fitting dies. The open part is doing that
*blind*: turning the same pooling into a genuine orientation+cell **search** with no prior, then
characterizing where it succeeds.

**Concrete deliverables (each is a paper-figure-sized result):**

1. **Forward model.** Reproduce/clean the CBXD streak simulator (the generator behind `fig_cbxd.png`):
   cell + orientation + NA + energy → list of streaks (arc points, per-point λ, pupil coordinate).
   Land it as `experiments/cbxd_forward.py` with a golden test. *(Warm-up; ~days.)*
2. **The GPU arc-Hough accumulator.** Build `distinct` orientation voting over Kossel arcs on the GPU —
   batched over candidate orientations × streaks, the arc-curvature analog of `index_blind_fast`'s
   cosine sum. This is the core, and it is a lovely GPU-parallel voting kernel: your wheelhouse.
   In the reflection-starved corner (few streaks / small cells), add each streak's **tangent** as an
   extra per-streak vote weight — sketch + note in `experiments/ridge_moments.py` +
   `experiments/RIDGE_MOMENTS_HANDOFF.md` (robust tangent only, *not* curvature); see #9.
3. **Blind solvability phase diagram.** The clean, publishable result: map, over (NA, streak length,
   streak-point noise, #streaks), the boundary between *blind-solvable* and *walled*. Where does the
   third dimension become recoverable? A crisp phase boundary is exactly the kind of figure a
   mathematician's eye makes rigorous.
4. **Consensus for CBXD.** Test whether cross-frame consensus lifts blind CBXD as it lifts blind SFX —
   i.e. does pooling *across shots* stack with pooling *across streaks*? (Hypothesis: yes, and it is the
   same theorem twice.)

**Why it's enticing:** it is genuinely *open* (blind CBXD is unsolved), it has a clean geometric
formulation, a GPU-voting heart, and a likely-publishable phase-boundary. It is Chapman-group-adjacent
(CBXD is their program), and it is the figure we deliberately kept in the paper as the outlook — so
there is a natural home for the result.

**Start here:** paper *Outlook: convergent-beam streaks* section (the geometry, the sagitta argument,
the joint-fit sketch) → `fig_cbxd.png` and its generator → ROADMAP track 5. The known-cell scaffold is now **in the repo** —
`experiments/cbxd_joint.py` (forward model `simulate` / `rand_rot` / `B` / `HS` + the joint Kossel-overlay
solver `score` / `refine` / `seed_index`); `cbxd_multishot.py`, `cbxd_angles.py`, `cbxd_ransac.py` all
import it. (`fig_cbxd.png` and its generator still live with the paper repo — ask Stefano for those.)

---

## Thrust B — make M1–M6 faster (the optimizer + GPU thrust)

The pipeline is fast (~21 ms/frame blind on one A100, ~550× xgandalf at matched accuracy) but the
profile says it is **host-bound, not compute-bound**: end-to-end GPU busy ≈ 59%, so ~40% is Python /
launch-gap idle. There is structured maths *and* GPU engineering here, and every win is measurable
(ms/frame, before/after pie). See [`onboarding.md`](onboarding.md) §5 for the validation gates and the
memory note on the profile (M1–M6 breakdown, STEPS=8).

**What's measured (so you don't re-derive it):** M3 refine is compute-saturated (~3.3 k-starts/ms,
flat); M2/M4 host round-trips already removed (on-device, bit-exact); the N-best reduce loop already
moved to GPU (150→40 ms); M5 anneal has a **6.1 ms launch-latency floor** (15 sequential 3×3 solves that
underoccupy a sparse frame); a naive multi-stream runner only bought ~10–15% (data-dependent syncs
serialize the issue). So the *remaining* levers, roughly in order of appeal:

1. **CUDA-graph the M5 anneal.** Capture the fixed 15-iteration solve loop in a CUDA graph to collapse
   the launch floor. Deferred once because static-shape padding collided with the `cnt≥6` inlier guard —
   a bounded, well-posed GPU problem (pad triplet/peak buckets, mask the guard). Likely the cleanest win.
2. **Concurrency that actually fills the 40% idle.** The naive stream runner stalled on data-dependent
   syncs; a *batched* front end (pad peaks to `Pmax`, stack `B` frames, run M2/M3/M5 as `(B,·)` ops) is
   the more promising shape. This is a real optimization problem, not a knob-turn.
3. **The M3 objective landscape — the maths.** M3 ascends the multimodal cosine comb
   \(\sum_i w_i\cos(2\pi\,x\cdot q_i)\) by momentum gradient. Newton **failed** (its 2nd-order steps
   basin-jump the comb from imperfect seeds); plain nonlinear-CG was a wash. The open question is
   structural: this objective is an *almost-periodic* function whose gradient flow has a lattice of
   maxima; can you design a **preconditioned / trust-region** ascent that respects that structure (the
   Hessian is a weighted \(\sum q_i q_i^\top\cos(\cdot)\) — a metric-tensor-shaped operator) and reaches
   the same basins in fewer steps? M3 is compute-saturated, so fewer steps = proportional time cut. This
   is the part that is genuinely *your* kind of problem.
4. **Smarter seeds (M1).** The seed grid is a uniform Fibonacci sphere. Can a data-driven seeding
   (difference-vector-informed, or importance-sampled toward where peaks predict axes) hold the rate
   with far fewer starts? Ties directly to the consensus view: the true axes are the basins that attract
   the most starts, so seeding *toward* likely basins should preserve them while cutting the count.

**Deliverables:** a re-profiled M1–M6 pie (before/after), each committed lever validated to hold the
rate (blind ~71% gated, hybrid ~96%) bit-exact where deterministic, and — the ambitious one — a short
note on the M3 objective's landscape + a better-conditioned ascent. Target: push toward the MHz-rate SFX
regime LCLS-II needs (many A100s, real-time indexing).

**Start here:** `glint/glint_fast.py` (`index_blind_fast`, `anneal_batch_t`, `index_blind_nbest`),
`glint/glint_index.py` (M4/refine), the `experiments/` profiler + occupancy scripts, and the paper's
*Throughput* section. Validate on S3DF ampere (onboarding §6).

---

## How the two thrusts connect (and to your sharpy work)

Both are the **pooling / consensus** theme you already work on: CBXD pools *streaks* to regularize a weak
per-streak curvature; consensus pools *frames* to regularize a weak per-frame lattice; and the sync /
Gramian machinery on sharpy pools *overlaps* to regularize a weak per-frame phase. Same idea, three
domains — which is why this is a natural extension of what you're already good at, not a cold start.
If Thrust A's arc-Hough and the sparse direct-sum objective end up sharing a GPU voting kernel, so much
the better.

*Ownership is a proposal, not an assignment — open in the Issues, revise freely.*
