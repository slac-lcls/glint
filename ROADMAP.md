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
Audit and speed up M1–M6 + consensus + rescue. The pipeline is host-bound (~59% GPU busy). Open levers:
CUDA-graph the M5 anneal (collapse the 6.1 ms launch floor), a batched concurrent-frame front end, and —
the mathematical part — a better-conditioned M3 ascent on the almost-periodic cosine objective (Newton
basin-jumps; CG was a wash). Also: GPU-batch the known-cell rescue's candidate search to close the last
gap to ffbidx; trim the M1 seed grid. Full plan in [`docs/research-plan-yuan.md`](docs/research-plan-yuan.md),
Thrust B. Background: the paper's *Architecture*, *Accuracy-ceiling*, and *Throughput* sections.

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
  is `pinkIndexer`'s home turf (already in our literature table); GLINT's angle is that the direct-sum
  objective is naturally radius-tolerant along a fixed direction, and the fat slice should *raise* the
  blind ceiling rather than lower it. A clean first experiment: simulate a pink-beam still and test
  whether the extra bandwidth lifts blind indexing as the geometry predicts.

---
*Directions are open and discussed in the Issues. Ownership is proposed here, decided there.*
