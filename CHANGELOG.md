# Changelog

Pull-request numbers refer to `slac-lcls/glint`. Measured figures live in `experiments/check_numbers.py --facts` and
in the pages it guards; this file names what changed, not how much.

## v0.1.0 — first tagged release (date: the tag)

The state of the code behind *Real-time blind indexing by cross-frame consensus* (arXiv:2609.07722) and the first
public release under the licence in `LICENSE.md`.

### Blind indexing and consensus
- Gridless multi-start front end (M1–M6) with the fused M3 kernel on by default (#30, #39); the M3 peak cap and a
  degenerate-intensity guard (#37); step-count and peak-weight effects measured and bounded (#125, #126).
- Cross-frame consensus: order-independent voting by default (#102), a degenerate cell no longer aborts the vote
  (#109), merged groups recounted against the locked representative (#187), the lock keyed on pool share,
  runner-up margin and alias tightness (#83, #87, #122).
- Alias gate scores frames rather than the pooled cloud and guards the primary lock as well as relocks (#111,
  #112); HNF sublattice enumeration corrected (#179).
- Opt-in escalation: a deeper known-cell search on frames that still fail the gate, accepted only against the
  frame's own azimuth-scrambled copies, batched (#208, #211); opt-in best-matching candidate selection (#215).
- `same_lattice` made symmetric (#197); `misorientation_deg` compares lattices, not bases (#207); one Bravais-aware
  axis standardizer for frames and the reference cell (#185).

### Known-cell engine
- CUDA-graphed, then fully fused known-cell registration (`index_fused`) with on-device staging (#14, #16);
  candidate axis split across blocks (#167); fp32 working precision by default with `KC_FP=64` to restore fp64
  (#15, #209).
- Full azimuth turn on oblique cells (#22); both handednesses seeded on oblique cells and the reference setting
  recovered (#225).
- The coarse anchor-direction grid used only where it resolves the cell (L0·qmax ≤ 25), the full grid elsewhere
  (#227); cached fp32 CUDA graphs keep the cell parameters they were captured against (#229).

### Streaming driver (`glint.stream_driver.StreamDriver`)
- Device-resident ring: peak finding, batched indexing, integration against the still-resident pixels and a
  running merge (#19), with the merge scatter-add on the GPU (#52) and a fused prediction gate (#50).
- Blind warm-up with a sequential-stop consensus and adaptive re-lock (#54); peak-triaged, fanned-out warm-up and
  double-hit detection (#56, #99, #100); warm-up rescue and the watchdog's individual rescue; miss-buffer rescue;
  a dead fan-out degrades instead of crashing (#156).
- Opt-in retry cascade on gate-failing frames (#145); merge under a chosen Laue class (#186); named cell registry,
  per-frame event trace and peaks-in ingest (#199); best-fit cell assignment (#204); per-lattice scoring of
  double hits (#206); ring slots for hits only and pixels kept for the retroactive rescues (#212); adaptive effort
  (`effort=`), the known-cell depth following the hit rate with a chance-controlled deep search on the misses
  (#213); an opt-in chance floor on the live gate fitted on a lattice-free null (#214); alias-gate refusals reported
  in `stats()` (#164).
- Live detector-geometry refinement as a running accumulator, diagnostic only (#55); CrystFEL stream output with
  per-frame drift and provenance, observed peaks on request, per-frame completeness flag.
- The committed recorder and two recorded replays with provenance (#199, #200, #201, #202, #203, #205); an empty
  CrystFEL event id (`Event: //`) names no frame rather than a bad one (#228).

### Front ends, command line and pipelines
- `glint` command: peaks + geometry, pre-bridged q, or raw images through GLINT's own peak finder; `--integrate`,
  `--tofile` (#43), `--escalate` (#208), `--gate` with an optional chance floor (#216, #224), `--select` (#215);
  solution files in the reference cell's setting (#221).
- Non-finite and zero-length q rows dropped wherever q enters an indexer (#226). `import glint` loads its public
  names on first use, so NumPy-only submodules import without SciPy (#234); `h5py` declared as a dependency
  (#169).
- CrystFEL geometry bridge with one geometry core behind both entry points (#20, #25); the `data =` key forwarded
  to integration (#155); un-assembled multi-panel stacks integrated slab-locally (#157).
- Peak finders: peakfinder8 with the absolute ADU floor and interior ASIC seams masked (#108, #127), peakfinder9,
  and the adaptive dual-threshold finder with a fused one-pass reduction and sync-free labelling (#41, #51, #174);
  masked and non-finite pixels take no part in the v4, pf9 and pf8 finders' arithmetic, and an opt-in per-panel
  finder searches a multi-panel slab one panel at a time (#231, #232).
- LUTE task `GLINTIndexer` with integration and a self-contained front end (#23), raw xtc as a third frame source
  (#88), the activate-installation trap fixed (#141), and a measured status record (`lute/STATUS.md`, #220).
- xtc ingestion for LCLS-I and LCLS-II in one command, MPI-sharded, with a start-up geometry manifest (#47, #48,
  #104, #105); its `--integrate` predicts each event at its own wavelength, in CrystFEL's frame, on one detector
  plane (#233).

### Integration and merge
- Fused GPU box integration, bit-exact on detector dtypes (#17, #18); robust-mean background and event-aware
  integration (#142); non-positive intensities kept (#132).
- The live merge refuses frames with no measured mean intensity and keeps I ≤ 0 (#230).
- Symmetry-constrained refinement and a partiality merge model (#53, synthetic).

### Validation, numbers and reproducibility
- CPU test suites on GitHub Actions, mirrored locally by `experiments/run_ci_locally.py` (#107, #124, #165).
- `experiments/check_numbers.py`: the FACTS table and the guard over the README, the docs, the manuscript and the
  decks (#24 and many follow-ups); `REPRODUCING.md` maps every table and figure to what regenerates it (#161).
- The strict gate canonicalised in one place (#170, #172); goldens with a shared cases module (#135).
- Docs: onboarding with the blind-vs-known-cell walkthrough and the streaming driver as a fourth path, the
  driver's option reference checked against the constructor by CI (#21, #217, #223).

### Licensing and release
- BSD-3-Clause licence with the SLAC/DOE enhancements grant-back, SLAC copyright notice, third-party notices with
  verified upstream terms and data provenance (#114, #133, #150, #166, #193); funding acknowledgment in TT&SP's
  wording (#194, #195).

### Research scripts (not shipped in the wheel)
- Powder auto-indexing package and pipeline (#177); FPGA-grade peak emission study (#189). Convergent-beam
  (CBXD) research scripts and results (#10, #13, #149) are still present in the source tree.
