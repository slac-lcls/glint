# GLINT — a fast GPU-native blind crystallography indexer

GLINT indexes sparse single-shot serial-crystallography (SFX) diffraction **blind** (no unit
cell supplied) on the GPU. It proposes candidate real-space axes from a gridless objective,
anneals cells, keeps the *N*-best hypotheses per frame, derives the unit cell across frames by
**consensus**, and rescues the remaining frames with a cell-general GPU known-cell indexer. It
ingests exactly what a CrystFEL / LUTE peak search emits and writes a CrystFEL `.stream`, so it
drops into the existing CrystFEL-based merging flow (`partialator`).

On one NVIDIA A100, over 120 sparse cxidb-17 lysozyme frames, GLINT indexes blind
*above* xgandalf's rate (76% vs 71% at the same gate) at ~450× the throughput, and with cross-frame
consensus indexes 97% of frames blind (≥10 reflections).

**New here?** [`docs/onboarding.md`](docs/onboarding.md) has a short primer on *what crystallographic
indexing is and what GLINT does*, plus how to set up, run, and contribute.

## Install

```bash
pip install -e .        # from a checkout; CPU works, CUDA is used automatically if available
```

Requires Python ≥ 3.9 and `numpy`, `scipy`, `torch` (installed automatically).

## Command-line

```bash
# what a CrystFEL / peakfinder8 run emits: a peak-search stream + a .geom
glint --peaks peaks.stream --geom detector.geom -o indexed.stream

# or pre-bridged reciprocal q-vectors (FRAME blocks, 3 cols, 1/Angstrom)
glint --qframes frames.txt -o indexed.stream
```

Options: `--cell "a b c al be ga"` (known cell, skip consensus) · `--nbest N` (multi-hypothesis
consensus, default 3) · `--mode auto|sparse|dense` · `--integrate` (real I/σ) · `--tofile` (hand
orientations to CrystFEL for the refined merge) · `--device cpu|auto` · `-N` (limit frames).

## LUTE pipeline

GLINT ships LUTE Task and DAG definitions in [`lute/`](lute/), so it runs as a drop-in replacement
for `CrystFELIndexer` in the LCLS SFX workflow:

    PeakFinderSFX -> [GLINTIndexer] -> StreamFileConcatenator -> PartialatorMerger -> HKLManipulator

The reason it exists: none of LUTE's bundled CrystFEL builds are compiled with FFBIDX support, so
`indexamajig --indexing=ffbidx` fails and GPU fast-feedback-style indexing is unavailable in LUTE
today. GLINT fills that gap, emitting a CrystFEL `.stream` the downstream stages already understand.

> ⚠️ **The default stream is orientation-only and is *not* mergeable.** Every reflection carries
> placeholder `I=0.00 sigma(I)=0.00`, so feeding it through the concatenator to `PartialatorMerger`
> merges zeros. To get a real dataset:
>
> * **`tofile:` + an added `indexamajig --indexing=file` task between `GLINTIndexer` and
>   `StreamFileConcatenator`.** CrystFEL's prediction refinement imposes the lattice symmetry, and
>   this gives the **better merge**. Setting `tofile:` alone is not enough — the DAG above would
>   still concatenate the placeholder stream. Two measured traps: the added task must reuse the
>   stored peaks (`peaks: cxi`), or `indexamajig` validates the solutions against its own re-found
>   peaks and rejects them; and CrystFEL **0.12.0's** `--indexing=file` is broken ("Failed to
>   prepare indexing method" before any frame) — use 0.11.1.
> * **`integrate: true` on the raw-images route** (`--images`, GLINT's event-aware `integrate_cxi`
>   path — the configuration of the validated end-to-end run): GLINT box-integrates its own
>   reflections and writes real I/sigma, and the stream flows through the concatenator to the
>   merger with no CrystFEL step.
>
> ⚠️ Do **not** combine `integrate: true` with the `PeakFinderSFX` peaks path on stacked
> multi-event `.cxi`: that route's integrator is not event-aware and silently integrates every
> frame against event 0 of its file ([#136](https://github.com/slac-lcls/glint/issues/136)).
> (`image_dir` is also required there — the config is rejected without it — but supplying it does
> not fix the event addressing.)

Install the Task into a LUTE tree with [`lute/install_into_lute.sh`](lute/install_into_lute.sh);
[`lute/README.md`](lute/README.md) has the configuration, and
[`lute/STATUS.md`](lute/STATUS.md) is the honest account of what is measured, what is assumed and
what has never been run. Check it for current readiness rather than relying on a count here — its
summary and its per-item sections do not presently agree with each other.

## Library

```python
from glint.geom import parse_geom, read_crystfel_peaks, peaks_to_q   # CrystFEL .geom + peaks -> q
from glint.hybrid_stream import hybrid_index                         # blind -> consensus -> rescue
from glint.stream import write_stream                                # results -> CrystFEL .stream
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

COPYRIGHT (c) SLAC National Accelerator Laboratory. All rights reserved. This work is supported
[in part] by the U.S. Department of Energy, Office of Basic Energy Sciences under contract
DE-AC02-76SF00515.

**Usage restrictions.** Neither the name of the Leland Stanford Junior University, SLAC National
Accelerator Laboratory, U.S. Department of Energy nor the names of its contributors may be used to
endorse or promote products derived from this software without specific prior written permission.

**Licence.** GLINT is released under the terms in [`LICENSE.md`](LICENSE.md) (BSD 3-Clause with a
SLAC Enhancements grant-back). Third-party material carrying its own terms is recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The notice above is also in
[`COPYRIGHT`](COPYRIGHT) as a standalone file, which is where the DOE contract number lives.
