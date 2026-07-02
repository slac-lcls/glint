# GLINT — a fast GPU-native blind crystallography indexer

GLINT indexes sparse single-shot serial-crystallography (SFX) diffraction **blind** (no unit
cell supplied) on the GPU. It proposes candidate real-space axes from a gridless objective,
anneals cells, keeps the *N*-best hypotheses per frame, derives the unit cell across frames by
**consensus**, and rescues the remaining frames with a cell-general GPU known-cell indexer. It
ingests exactly what a CrystFEL / LUTE peak search emits and writes a CrystFEL `.stream`, so it
drops into the existing CrystFEL-based merging flow (`partialator`).

On one NVIDIA A100, over 120 sparse cxidb-17 lysozyme frames, GLINT matches the blind indexing
rate of xgandalf at ~550× the throughput, and with cross-frame consensus indexes 96% of frames
blind (≥10 reflections).

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
consensus, default 3) · `--mode auto|sparse|dense` · `--integrate` (real I/σ) · `--fromfile` (hand
orientations to CrystFEL for the refined merge) · `--device cpu|auto` · `-N` (limit frames).

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
all resting on the same gridless objective + GPU consensus. Yuan's focused plan (CBXD + throughput) is
[`docs/research-plan-yuan.md`](docs/research-plan-yuan.md). How to work in the repo (branches/PRs, envs,
validation gates) is in [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/onboarding.md`](docs/onboarding.md).

## Origins

GLINT's blind sparse indexer is a gridless **multi-start optimizer + cross-frame selector** (not an
FFT). The **3-D-FFT** method it grew out of (the *fftindex* prototype — a learned peakfinder + multi-shot
consensus on the transform volume) is now the **dense-data** (rotation / many-peak) front end. That
research write-up is preserved in [`docs/lineage.md`](docs/lineage.md).
