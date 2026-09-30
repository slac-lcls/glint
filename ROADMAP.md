# GLINT roadmap

High-level directions — the map. Concrete, assignable work lives as **GitHub Issues** (labelled by
track); ownership is discussed there, not fixed here. How we work: [`CONTRIBUTING.md`](CONTRIBUTING.md)
+ [`docs/onboarding.md`](docs/onboarding.md).

The engine is stable (blind serial + dense rotation, CrystFEL drop-in). Work splits into a
**real-data / productization** side, an **algorithms** side, and a small set of **exploratory
regimes** where the same gridless-objective + GPU-consensus machinery may extend beyond monochromatic
serial/rotation data. *Suggested* leads are in brackets — a proposal to react to, not an assignment.

Settled questions — the optimizer verdicts, the phase-retrieval family, the pruning experiments that
went the other way — live in [`docs/results.md`](docs/results.md), so this file stays a map of what is
*open*. Measured numbers are in `experiments/check_numbers.py` (`--facts`); quote them from there.

## 1. Validation & real data  *(suggested lead: Mona)*
Run GLINT blind on real SFX datasets and judge output against established indexers.
- **[#1]** End-to-end on a real dataset: cell + merge (CC\*/Rsplit) vs cctbx.xfel / DIALS / CrystFEL.
- **[#2]** Extend `--tofile` → CrystFEL-refine → `partialator` beyond ProK / lysozyme (≥2 more
  proteins), including the hexagonal `cxidb_62` (NERSC) as a second real protein.
- Characterize where GLINT wins / ties / needs work across cell types and sparsity — a clean
  "when to reach for GLINT" table.

## 2. Productization (LCLS)  *(suggested lead: Mona / Stefano)*
- **[#3]** Wire the LUTE `GLINTIndexer` task (`lute/`) into a real SFX DAG — it fills the gap that
  LUTE's CrystFEL builds lack (none is compiled with FFBIDX).
- **Landed since August:** the `--integrate` path and the `--images` front end are exposed and smoke-tested;
  the streaming driver has a named cell registry and a per-frame event trace (#199), pixel and peaks-in replay
  under a CrystFEL geometry (#200, #203), a merge class chosen per sample (#186), an opt-in chance floor on the
  live gate fitted on a lattice-free null (#214, `null_floor=`), and two recorded replays with provenance in
  [`docs/streaming_replay.md`](docs/streaming_replay.md).
- **Open on the driver:** the chance floor in the published arm, re-fitted per detector and peak finder (its
  constants are cxidb-17's); index before compress on the DRP — ring slots for hits only and pixels kept for the retroactive rescues shipped in #212,
  the reducer-side wiring has not; the adaptive-effort arms rerun at the batch size the cost table was measured
  at (#213 checked the mechanism at B=20); deployment recipes for `effort=` per beamline rate. The option map
  is [`docs/stream_driver_options.md`](docs/stream_driver_options.md).

## 3. CBXD — blind convergent-beam  *(flagship; suggested lead: Yuan)*
Convergent-Beam X-ray Diffraction (Chapman group, arXiv:2602.14402): a cone of incident directions, so
each reflection is a Kossel-circle **streak** rather than a point — and the streak's curvature carries
the out-of-plane information a single still lacks. Known-cell indexing is solved; **blind** orientation
+ cell is walled by per-streak precision. The idea: pool streaks into a **generalized-Hough / joint
fit** — the convergent-beam analog of the direct-sum objective + consensus. This is the figure kept in
the paper's *Outlook*.

- **[#6]** C1 — reproduce the streak sim + probe the blind streak-precision wall.
- **[#9]** Add the `ridge_moments` tangent as a per-streak vote weight in the accumulator.
- **[#11]** C3 — blind solvability phase diagram (NA × streak precision).
- **[#12]** C4 — does cross-frame consensus stack with cross-streak pooling?
- **[PR #10]** GPU arc-Hough accumulator (draft).
- **Landed — two-color step 1** (PR #13, [`experiments/CBXD_TWOCOLOR_STEP1.md`](experiments/CBXD_TWOCOLOR_STEP1.md)).
  A second colour gives a second Ewald sphere; scoring against both roughly **doubles the truth tower**
  (27 → 56 votes) for ~**2× the capture radius**, and in the low-NA regime where a single colour starves
  it recovers **10/12 vs 6/12** blind, with 100% correct λ-labels.

## 4. Throughput & the optimizer  *(suggested lead: Yuan)*
Background: the paper's *Architecture*, *Accuracy-ceiling* and *Throughput* sections. The M3 refiner
and the known-cell rescue have both since been fused — see [`docs/results.md`](docs/results.md) for
what closed, and for the levers that were tried and rejected. The live front is now peakfinding.

- **[#4]** A1 — review the M1–M6 pipeline + optimizers; propose improvements.
- **[#7]** **NUFFT the dense cluster-seeding** *(highest-value open lever)*. cuFINUFFT was benchmarked
  only on the *sparse* path, where M1 is ~0 ms and cannot help. On the *dense* path M1 (the cluster-FFT)
  is **62% of the frame** — the whole dense bottleneck — and NUFFT was never evaluated there.
- **Peakfinding is the current wall** (PR #41). Profiled at 4096², the reduction was 47% of `find()`
  and is now a single fused pass (1.45–2.13×, bit-identical). In the streaming driver it is the
  **largest single stage by a wide margin**: 1.16 ms against `predict`'s 0.16 ms, a **7.3×** gap.
  *This line previously said the two were "tied" and pointed the next lever at `predict`. That rested
  on `predict_ms = 1.11`, which never reproduced — re-measured 2026-08-01, the pre-#68 code gives 0.42
  and PR #68 (merged) takes it to 0.16.* Two things still qualify the lever: removing peakfind
  **entirely** bounds out at **1.39×** (4.16 → 3.00 ms/frame), and 0.69 ms/frame of the wall is
  currently un-attributed, so `stream_ms` itself wants re-measuring. `python experiments/check_numbers.py`
  prints both as advisories.
- **Device-resident streaming driver** (PR #19, merged): peakfind → index → integrate → running merge,
  pixels never leaving the GPU. The running accumulator reproduces the batch `merge_stats.py` math
  exactly at every I/σ floor. It is **not a live merge** — `--facts` carries the current gap.
- **Adaptive effort** (#213, merged): the known-cell search depth follows the hit rate, and the spare GPU
  time buys a chance-controlled deep search on the misses (#208, #211). Measured as a mechanism at B=20 on one
  A100; the cost table it reasons with is B=120. Open: `tiers=` measured per batch size and per GPU. The live
  gate's chance floor (#214, `null_floor=`) landed as an opt-in; on by default is the step that makes the
  driver's `indexed` count mean what it says.
- **Selection, not search** (#215, open): keep each frame's best-matching consensus-consistent candidate
  instead of the first that fits; the joint multi-shot experiments found the gain in the selection, not in
  coupling the cells.
- `torch.compile` fusion on the M3 gradient / anneal normal-equations — untried, modest expected gain.

## 5. Exploratory regimes — beyond monochromatic serial/rotation  *(open; ideas welcome)*
The paper's *Outlook* frames these; they are speculative but share GLINT's core — a gridless objective
over peak **positions** with unit weight, GPU-parallel, pooled by consensus. None is committed work;
they are here to be argued about.

- **Powder / 1-D auto-indexing (→ Rietveld).** A powder pattern is the full spherical average: only the
  shell radii survive, so auto-indexing means recovering the six cell parameters by fitting the
  **reciprocal metric tensor** *Q(h,k,l) = h²A + k²B + l²C + hkD + klE + hlF*, so that every observed
  |q|² is a near-integer quadratic form (the ITO / TREOR / DICVOL / McMaille problem) — a massively
  parallel search over a 6-parameter metric, exactly GLINT's multi-start-on-GPU shape. The interesting
  regime is **low symmetry**, where distinct (hkl) no longer coincide in q so every line independently
  constrains the metric. **Shipped as a package**:
  [`experiments/powder_index/`](experiments/powder_index/) — all 7 crystal systems, de Wolff M20 figure
  of merit, centering / reflection conditions, seed-and-verify plus simulated annealing for triclinic,
  with tests and a GSAS-II benchmark harness. Its README lists what is left: no zero-point handling,
  pseudo-symmetric equal-edge cells still fail, not real-data-hardened, GSAS-II head-to-head not yet
  run. The dependency-free radial front end lives in `slac-lcls/drp-benchmarks/radial_integration`.
- **[#8] Laue / pink-beam — the fat Ewald sphere.** A polychromatic beam turns the Ewald sphere into a
  thick shell, so one shot samples much *more* of the 3-D lattice — the same "a fat slice buys
  out-of-plane information" argument the paper makes for bandwidth and for CBXD. The catch is the
  per-spot wavelength unknown: a reflection fixes the *direction* of q but not its radius until λ is
  chosen. This is `pinkIndexer`'s home turf; GLINT's angle is that the fat slice adds many more
  constraints per shot, so it may *raise* the blind ceiling rather than lower it. First experiment:
  simulate a pink-beam still and test whether the extra bandwidth lifts blind indexing as the geometry
  predicts. This is also the regime in which to revive the reverse/Chamfer selection cost — negative on
  thin-slice data, but that test needed the 3-D information a fat slice provides
  (`experiments/reverse_cost.py`, `fat_ewald.py`).

---
*Directions are open and discussed in the Issues. Ownership is proposed here, decided there.*
