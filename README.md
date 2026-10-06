# GLINT — a fast GPU-native blind crystallography indexer

GLINT indexes sparse single-shot serial-crystallography (SFX) diffraction **blind** (no unit
cell supplied) on the GPU. It proposes candidate real-space axes from a gridless objective,
anneals cells, keeps the *N*-best hypotheses per frame, derives the unit cell across frames by
**consensus**, and rescues the remaining frames with a cell-general GPU known-cell indexer. It
ingests exactly what a CrystFEL / LUTE peak search emits and writes a CrystFEL `.stream`, so it
drops into the existing CrystFEL-based merging flow (`partialator`).

On one NVIDIA A100, over 480 sparse cxidb-17 lysozyme frames, GLINT **matches** the strongest blind
indexer we tested — 361 of 480 frames against xgandalf's 350 at the same gate, a difference that is
not significant (McNemar *p* = 0.18) — at **~450×** the throughput. Deriving the cell by consensus
rather than being handed it costs nothing at that gate (361 against 357 with the cell supplied) and
about ten frames at the looser ≥10-reflection bar (458 against 468 of 480).

**New here?** [`docs/onboarding.md`](docs/onboarding.md) has a short primer on *what crystallographic
indexing is and what GLINT does*, plus how to set up, run, and contribute.

**Checking the paper?** [`REPRODUCING.md`](REPRODUCING.md) maps every table and figure to the
script, data and command that regenerates it — including what reproduces on a laptop from data
committed here, and what does not reproduce from this checkout at all.

## The streaming driver, replayed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/media/streaming_cxidb17_480_dark.gif">
  <img alt="Recorded replay of the GLINT streaming driver over the 480-frame cxidb-17 lysozyme run: the driver schematic with the blind warm-up, consensus, device ring, batched known-cell indexing and watchdog rescue, the recovered cell, and the stream composition chart, with the strict indexed count ending at 331 of 480" src="docs/media/streaming_cxidb17_480_light.gif" width="900">
</picture>

A recorded replay of the shipped `glint.stream_driver.StreamDriver`, cold-started with no cell — not a
live beamline, not a simulation. Input: the 480-frame extension of the cxidb-17 lysozyme run (CXIDB 17,
Boutet *et al.* 2012, CC0) as CrystFEL peakfinder8 peaks from an offline `indexamajig` pass, converted to
q at one fixed wavelength — the paper's primary streaming set, not the published 120-frame subset. Arm:
constructor defaults except `B=20` and `min_inliers=10`, plus the two published opt-ins `warmup_rescue`
and `adaptive_relock`; recorded on one A100 (other boxes differ by a frame). The cell locks at frame 5;
the watchdog rescues six frames; near event 364 the adaptive re-lock adds a second cell that is spurious —
not lysozyme — and uncounted. Result: **331/480 indexed at the strict correct-lattice bar, 69.0%**, the top
of the paper's 67–69% band on this set (the 67% is the same driver at its shipped defaults). Frame rate is
a display choice. Provenance, per-frame semantics and stills: [`docs/streaming_replay.md`](docs/streaming_replay.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/media/streaming_multicell_dark.gif">
  <img alt="Recorded replay of the GLINT streaming driver over a two-species stream, real lysozyme and Proteinase K peak lists interleaved by a planted schedule: the driver schematic, two cell cards named from a roster after the driver finds each cell, the planted schedule with its playhead, and the per-cell composition chart over a ribbon of the planted species, with the strict count ending at 582 of 900" src="docs/media/streaming_multicell_light.gif" width="900">
</picture>

The same driver, two species, one stream: real lysozyme frames (the cxidb-17 extension above) and real
Proteinase K frames (CXIDB 45, Masuda *et al.* 2017, CC0; the deposit's top-170 peak lists, the same lists
DIALS was given in the paper) interleaved by a seeded schedule — the interleaving is the one planted thing
on the page, and it is drawn as such. Cold start, same arm, plus a 64-frame rescue buffer and a *roster* that
names a cell after the driver finds it, never before. The consensus locks lysozyme at frame 5; when
Proteinase K blends in, its misses vote a second cell and at frame 146 the driver adds it, names it from
the roster and re-indexes the 10 buffered misses against it — one re-lock, no alias; the watchdog rescues 15
frames on its own. Scored against each frame's own species: **582 of 900** at the strict bar — lysozyme
309 of 454 (311 alone on the same frames), Proteinase K 273 of 446 (375 alone). The gap is mostly one
mechanism, shown rather than hidden: 85 Proteinase K frames were claimed first by the lysozyme cell —
first-fit over the active cells, on 170-peak frames — and none of them pass the strict bar there. Index-only
(no pixels); the GIF plays every second frame, 2× real time.
[`docs/streaming_replay.md`](docs/streaming_replay.md#two-species-replay--the-cell-registry-at-work).

Everything the driver does beyond the base path is opt-in and was measured before it shipped: the warm-up
and re-lock rescues above, a retry cascade on the misses, per-lattice scoring of double hits, best-fit
assignment between cells, a named cell registry with a per-frame event trace, ring slots for hits only, and
`effort=`, which makes the known-cell search depth follow the hit rate and spends the spare GPU time on a
chance-controlled deep search of the misses (glint#213). Every constructor option, by topic, with its default
and the pull request that measured it: [`docs/stream_driver_options.md`](docs/stream_driver_options.md).

## Install

```bash
pip install -e .        # from a checkout; CPU works, CUDA is used automatically if available
```

Requires Python ≥ 3.9 and `numpy`, `scipy`, `torch`, `h5py` (installed automatically).

## Command-line

```bash
# what a CrystFEL / peakfinder8 run emits: a peak-search stream + a .geom
glint --peaks peaks.stream --geom detector.geom -o indexed.stream

# or pre-bridged reciprocal q-vectors (FRAME blocks, 3 cols, 1/Angstrom)
glint --qframes frames.txt -o indexed.stream
```

Options: `--cell "a b c al be ga"` (known cell, skip consensus) · `--nbest N` (multi-hypothesis
consensus, default 3) · `--mode auto|sparse|dense` (dense self-indexes each frame and refuses `--cell`,
`--nbest`, `--select`, `--escalate` and `--cascade`; auto stays sparse when `--cell` is given) · `--escalate`
(a deeper known-cell search on the frames that still fail the gate, accepted only against the frame's own
azimuth-scrambled copies; off by default) ·
`--gate none|strict|floor` (write a frame as a crystal only if it passes the paper's scoring bar; `floor` also
requires `--floor NAME|a,b[,c]` for a dataset-specific chance floor; default none) ·
`--integrate` (real I/σ) · `--tofile` (hand orientations to CrystFEL for the refined merge) ·
`--device cpu|auto` · `-N` (limit frames). `--images raw.cxi --geom detector.geom` runs GLINT's own peak
finder on the pixels instead of reading a peak stream; on this route it runs on the host CPU (numpy/scipy),
not the GPU (the GPU finder is used by the xtc route, `experiments/xtc_bridge`, and by `StreamDriver`).

## LUTE pipeline

GLINT ships LUTE Task and DAG definitions in [`lute/`](lute/), so it runs as a drop-in replacement
for `CrystFELIndexer` in the LCLS SFX workflow:

    PeakFinderSFX -> [GLINTIndexer] -> StreamFileConcatenator -> PartialatorMerger -> HKLManipulator

The reason it exists: the CrystFEL builds LUTE runs (0.10.2 by default) lack FFBIDX support, so
`indexamajig --indexing=ffbidx` fails there; S3DF's separate fast-feedback build is not one LUTE
uses. GLINT adds GPU blind indexing to LUTE, emitting a CrystFEL `.stream` the downstream stages
already understand.

**Pick a merge route.** GLINT's default stream carries the cell and the per-frame orientation, with
placeholder `I=0.00` intensities — everything a *refiner* needs and nothing a *merger* does. One of
the two settings below turns it into a mergeable dataset; which one you want depends on whether you
want CrystFEL in the pipeline:

* **`integrate: true`** — no CrystFEL step. GLINT predicts and box-integrates its own reflections
  and writes real I/σ, so the stream flows straight through the concatenator to
  `PartialatorMerger`. Prediction and the box gather are host numpy on this route (the fused GPU
  integrator is `StreamDriver`'s). This is the configuration of the validated end-to-end run.
* **`tofile:`** — hand the orientations to `indexamajig --indexing=file`, added as a task between
  `GLINTIndexer` and `StreamFileConcatenator`, so CrystFEL's prediction refinement imposes the
  lattice symmetry. Note that `tofile:` alone is not enough: without the added task the DAG
  concatenates the placeholder stream.

Which merges *better* is not settled — see
[glint#129](https://github.com/slac-lcls/glint/issues/129), where the only head-to-head ran the two
routes at unmatched integration settings. Matching them closed the CC\* gap, and the R_split
difference that remains is explained by multiplicity and a selection cut rather than by intensity
quality. Choose on dependencies, not on an expected quality ranking.

Configuration for both, including the traps worth knowing on the `tofile:` route, is in
[`lute/README.md`](lute/README.md). Install the Task into a LUTE tree with
[`lute/install_into_lute.sh`](lute/install_into_lute.sh).
[`lute/STATUS.md`](lute/STATUS.md) is the evidence behind this integration: every measurement with
its run, the negative results, and the assumptions that have never been tested.

## Library

```python
from glint.geom import parse_geom, read_crystfel_peaks, peaks_to_q   # CrystFEL .geom + peaks -> q
from glint.hybrid_stream import hybrid_index                         # blind -> consensus -> rescue
from glint.stream import write_stream                                # results -> CrystFEL .stream
from glint.stream_driver import StreamDriver                         # streaming: push frames, batched index+integrate
```

The blind front-end (`glint.glint_fast.index_blind_nbest`), the consensus
(`glint.multishot.consensus_cell`), and the GPU known-cell rescue
(`glint.replica_gpu.index_known_gpu_cell`) are all individually importable. The `experiments/`
directory holds the research scripts and diagnostic harness (not shipped in the wheel).

## Roadmap & contributing

Open directions live in [`ROADMAP.md`](ROADMAP.md): real-data validation and LCLS productization,
faster M1–M6, breaking the **blind convergent-beam (CBXD)** wall, and exploratory extensions to
**powder** (metric-tensor auto-indexing → Rietveld) and **Laue / pink-beam** (the fat Ewald sphere) —
all resting on the same gridless objective + GPU consensus. Settled questions and negative results are
in [`docs/results.md`](docs/results.md). How to work in the repo (branches/PRs, envs, validation gates)
is in [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/onboarding.md`](docs/onboarding.md).

## Origins

GLINT's blind sparse indexer is a gridless **multi-start optimizer + cross-frame selector** (not an
FFT). The **3-D-FFT** method it grew out of (the *fftindex* prototype — a learned peakfinder + multi-shot
consensus on the transform volume) is now the **dense-data** (rotation / many-peak) front end. That
research write-up is preserved in [`docs/lineage.md`](docs/lineage.md).

## Copyright

Copyright (c) 2026, The Board of Trustees of the Leland Stanford Junior University, through SLAC National Accelerator Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.
GLINT was developed at SLAC with support from DOE's Basic Energy Sciences and Advanced
Scientific Computing Research programs, including the ILLUMINE project. Work at SLAC National
Accelerator Laboratory is supported by the U.S. Department of Energy under contract
DE-AC02-76SF00515.

**Usage restrictions.** Neither the name of the Leland Stanford Junior University, SLAC National
Accelerator Laboratory, U.S. Department of Energy nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written permission.

**Licence.** GLINT is released under the terms in [`LICENSE.md`](LICENSE.md) (BSD 3-Clause with a
SLAC Enhancements grant-back). Third-party material carrying its own terms is recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The notice above is also in
[`COPYRIGHT`](COPYRIGHT) as a standalone file, which is where the DOE contract number lives.
