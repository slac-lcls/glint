# GLINT roadmap

High-level directions — the map. Concrete, assignable work lives as **GitHub Issues** (labelled by
track); ownership is discussed there, not fixed here. How we work: [`CONTRIBUTING.md`](CONTRIBUTING.md)
+ [`docs/onboarding.md`](docs/onboarding.md). Yuan's focused plan (CBXD + throughput):
[`docs/research-plan-yuan.md`](docs/research-plan-yuan.md).

The engine is stable (blind serial + dense rotation, CrystFEL drop-in). Work splits into a
**real-data / productization** side, an **algorithms** side, and a small set of **exploratory
regimes** where the same gridless-objective + GPU-consensus machinery may extend beyond monochromatic
serial/rotation data. *Suggested* leads are in brackets — a proposal to react to, not an assignment.

## 1. Validation & real data  *(suggested lead: Mona)*
Run GLINT blind on real SFX datasets and judge output against established indexers.
- End-to-end on a real dataset: cell + merge (CC\*/Rsplit) vs cctbx.xfel / DIALS / CrystFEL.
- Extend `--fromfile` → CrystFEL-refine → `partialator` beyond ProK / lysozyme (≥2 more proteins),
  including the hexagonal `cxidb_62` (NERSC) as a second real protein.
- Characterize where GLINT wins / ties / needs work across cell types and sparsity — a clean
  "when to reach for GLINT" table.

## 2. Productization (LCLS)  *(suggested lead: Mona / Stefano)*
- Wire the LUTE `GLINTIndexer` task (`lute/`) into a real SFX DAG — it fills the gap that LUTE's
  CrystFEL builds lack (none is compiled with FFBIDX).
- Packaging / deployment polish; documented recipes; the adaptive front end + native `--integrate`
  path exposed and smoke-tested.

## 3. CBXD — blind convergent-beam  *(flagship; suggested lead: Yuan)*
Convergent-Beam X-ray Diffraction (Chapman group, arXiv:2602.14402): a cone of incident directions, so
each reflection is a Kossel-circle **streak** rather than a point — and the streak's curvature carries
the out-of-plane information a single still lacks. Known-cell indexing is solved; **blind** orientation
+ cell is walled by per-streak precision. The idea: pool streaks into a **generalized-Hough / joint
fit** (the convergent-beam analog of the direct-sum objective + consensus). Full plan, deliverables, and
the solvability-phase-diagram target in [`docs/research-plan-yuan.md`](docs/research-plan-yuan.md),
Thrust A. This is the figure kept in the paper's *Outlook* — a live direction with a home for the result.

## 4. Throughput & the optimizer  *(suggested lead: Yuan)*
Audit and speed up M1–M6 + consensus + rescue. Full plan in
[`docs/research-plan-yuan.md`](docs/research-plan-yuan.md), Thrust B. Background: the paper's
*Architecture*, *Accuracy-ceiling*, and *Throughput* sections. Concrete experiments, several of them from
a re-audit of the "attic" where we pruned in the wrong regime or at a fixed setting. Priority now sits
with the two **accuracy-neutral** engineering levers (NUFFT dense-M1, CUDA-graph M5): the two
accuracy-coupled ones (STARTS-down, CG) were tested 2026-07-01 and confirmed non-free (below).

- **NUFFT the dense cluster-seeding** *(highest-value; a prune in the wrong regime)*. NUFFT (cuFINUFFT)
  was benchmarked only on the *sparse* path, where M1 is ~0 ms and it can't help. On the *dense* path M1
  (the cluster-FFT) is **30 ms = 62%** of the frame — the whole dense bottleneck — and NUFFT was never
  evaluated there. Swap the 28 gridded cluster-FFTs for a cuFINUFFT and re-profile.
- **CUDA-graph the M5 anneal** *(deferred, not disproven; ~29% of the sparse frame)*. M5 is launch-bound
  at a 6.1 ms floor (15 sequential 3×3 solves). Capture the fixed-iteration loop in a CUDA graph; the
  padding that collides with the `cnt≥6` inlier guard is solvable by masking the guard on padded entries.
  Then re-test fp32 M5 *on top* (it was negative only because it was launch-bound — the two were coupled).
- ~~**Sweep M1 STARTS down.**~~ **TESTED 2026-07-01 (A100, `experiments/starts_sweep.py`) — no free
  lunch.** Blind rate drops monotonically as the grid thins: n_dir 2200/1600/1100/700 → 84/80/76/65 of
  120 at 19.2/16.6/15.3/13.8 ms. The 70,400 starts are *load-bearing for generation* (consistent with the
  generation-limited diagnosis), not redundant. **Follow-up ANSWERED (`experiments/hybrid_starts.py`):
  the consensus + rescue fully absorb it** — the *hybrid* rate holds at **115–118/120 down to n_dir 700**
  (a 3× smaller grid) as the rescue picks up what blind drops (n_resc 26→38). So the blind grid can be set
  2–3× smaller with **no end-to-end accuracy cost** — a safe latency/memory knob. Not a wall-time win,
  though: the saved blind compute is offset by more rescues, so end-to-end ms is ~flat.
- ~~**M3 second-order / line-search refiners.**~~ **CLOSED — six optimizers now agree GD STEPS=8 is the
  M3 optimum.** CG tested (`cg_test.py`): momentum-GD Pareto-dominates it at every step count and equal
  wall-time (grad-4 = 76 @15.3 ms vs cg-4 = 67 @15.7; grad-8 = 84 @19.7). The elegant one — an **exact
  fused-phase 1-D Newton line-search** (`refine_vec_ls`, `ls_test.py`/`ls_cap_test.py`): along a direction
  the cos objective is closed-form `S(a)=Σ wm cos(φ+aψ)`, so f/f'/f'' are matmul-free once the phases are
  computed — was built and swept (cap + momentum knobs). Verdict **negative**: ls-8 = 72, relaxing the
  anti-jump cap gives nothing (72), momentum recovers only part (77), all still < GD's 84 and at ~2× cost
  (the direction's `ψ=d·q` is a *second* matmul, so it was never gradient-cost). Root cause is the same
  across grad/cg/bb/lm/newton/ls: **greedy per-step optimality lands more starts in spurious maxima on the
  multimodal comb; momentum-GD's gentle, non-greedy, schedule-annealed ascent is the actual mechanism.**
  GD STEPS=8 stands.
- **`torch.compile` fusion** on the M3 gradient / anneal normal-equations — untried, modest expected gain.
- GPU-batch the known-cell rescue's candidate search to close the last ~4× gap to ffbidx.

## 5. Exploratory regimes — beyond monochromatic serial/rotation  *(open; ideas welcome)*
The paper's *Outlook* frames these; they are speculative but share GLINT's core — a gridless objective
over peak **positions** with unit weight, GPU-parallel, pooled by consensus. None is committed work yet;
they are here to be argued about.

- **Powder / 1-D auto-indexing (→ Rietveld).** A powder pattern is the full spherical average: all
  orientation information is gone and only the shell radii \(|G_{hkl}|\) survive as a 1-D list of
  d-spacings. Auto-indexing then means recovering the six cell parameters from that list — i.e. fitting
  the **reciprocal metric tensor** \(Q(h,k,l)=h^2A+k^2B+l^2C+hkD+klE+hlF\) so every observed \(|q_n|^2\)
  is a near-integer quadratic form (the ITO / TREOR / DICVOL / McMaille problem). This is a massively
  parallel search over a 6-parameter metric — exactly GLINT's multi-start-on-GPU shape, with the cosine
  objective replaced by a metric-residual score. GLINT would be the auto-indexer; **Rietveld**
  (GSAS-II / FullProf) is the established whole-profile *refinement* it feeds — the powder analog of how
  the SFX path feeds `partialator`. Honest scope: no orientation, harder degeneracies (dominant zones,
  impurity lines); a GPU-DICVOL is the concrete first experiment.
- **Laue / pink-beam — the fat Ewald sphere.** A polychromatic beam turns the Ewald sphere into a thick
  shell between \(\lambda_{\min}\) and \(\lambda_{\max}\), so one shot samples much *more* of the 3-D
  lattice — the same "fat slice buys out-of-plane information" argument the paper already makes for
  bandwidth and for CBXD. The catch is the per-spot wavelength unknown (a reflection fixes the
  *direction* of \(q\) but not its radius until \(\lambda\) is chosen — the harmonic degeneracy). This
  is `pinkIndexer`'s home turf (already in our literature table); GLINT's angle is that each spot still
  pins the *direction* of q and the fat slice adds many more constraints per shot, so it may *raise* the
  blind ceiling rather than lower it. A clean first experiment: simulate a pink-beam still and test
  whether the extra bandwidth lifts blind indexing as the geometry predicts. This is also the regime to
  **revive the reverse/Chamfer selection cost**: it was negative on thin-slice data but our own notes
  flag that as premature — it needs the 3-D information a fat slice provides and was confounded by
  missing weak peaks. Re-test it where the geometry supports it (`experiments/reverse_cost.py`,
  `fat_ewald.py`).

---
*Directions are open and discussed in the Issues. Ownership is proposed here, decided there.*
