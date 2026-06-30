# GLINT: a fast GPU-native blind crystallography indexer

*Status report — validated on one NVIDIA A100 (S3DF `ampere`), 120 sparse cxidb-17
lysozyme frames + 60 rich DIALS frames. All numbers below are measured, gated, and
reproduced from the experiment scripts in `experiments/`.*

## 1. Summary

GLINT indexes sparse single-shot diffraction **blind** (no unit cell supplied) by a
modular GPU pipeline: ascend a continuous lattice objective to candidate real-space
axes, assemble + anneal cells, derive the unit cell across frames by consensus, and
rescue blind failures with a GPU known-cell indexer.

The contribution is a point on the speed/accuracy frontier that neither incumbent
occupies: **blind, at xgandalf's accuracy, ~340× faster, and ~95% index-identical to
ffbidx when both solve.**

| indexer | mode | indexing rate | ms/frame | frames/s |
|---|---|---|---|---|
| **GLINT-①** | **blind** | **~71% / 94% ≥10 refl** | **34** | **30** |
| xgandalf | blind | 71% | 11,542 | 0.087 |
| ffbidx | known-cell | 75% | 4.4 | 226 |
| GLINT (known-cell mode) | cell given | matches ffbidx | 16.5 | 60 |

*(120 sparse cxidb frames, one A100, same gate: correct lattice AND indexes ≥25% of
spots / ≥10 reflections. "GLINT-①" = blind + consensus cell + GPU rescue. Rich DIALS-60:
GLINT-① = 47 ms/frame, 21 f/s, 100/100.)*

## 2. Architecture (M1–M6)

The pipeline is six swappable modules, built end-to-end in PyTorch (hence differentiable
and GPU-batchable). There are two paths:

- **Blind path** (`glint_fast.py::index_blind_fast`) — the contribution. ffbidx cannot
  run blind, so these modules are xgandalf-lineage + original, not ffbidx.
- **Known-cell path** (`replica_gpu.py`) — a faithful GPU port of ffbidx, used only to
  rescue blind failures.

| module | ffbidx | GLINT-blind | provenance |
|---|---|---|---|
| M1 sample | lengths given | Fib-sphere × length shells 30–126 Å | original (blind length search) |
| M2 objective | inlier + trimmed-log2 | Σ(1/\|q\|)·cos(2π q·v), cos² sharpen | xgandalf |
| M3 refine | cos ascent | cos ascent + momentum (8 steps) | shared (same) |
| M4 assemble | rotation search (given cell) | triplet enum of converged maxima | original |
| M5 anneal | ifss | residual-threshold anneal, GPU-batched | TORO/ffbidx math, GLINT engineering |
| M6 score | inlier + trimmed-log2 | coverage-gated defect + primitivize | original |
| consensus | — (per-frame) | derive cell across frames | original — no ffbidx analog |

Only **M3 (cos ascent)** and **M5 (ifss)** are literally shared crystallography. The
known-cell rescue path *is* ffbidx (replicated, then GPU-batched).

## 3. Throughput: the engineering wins

The blind pipeline went from **2342 ms/frame (scalar) → 34 ms/frame (~69×)**, all
validated bit-equal or better in accuracy:

| lever | before | after | speedup |
|---|---|---|---|
| M4 triplet anneal: numpy loop → one batched GPU solve | 2247 ms | 10 ms | **220×** |
| M3 ascent steps 80 → 8 (front-end over-iterated) | 98 ms | 27 ms | **3.6×** |
| Known-cell rescue: numpy → GPU (`replica_gpu`) | 127 ms | 16.5 ms | **7.7×** |

The original "GPU indexer" spent 96% of its time in a numpy `for`-loop over candidate
triplets; batching that loop is the single biggest win. After it, the front-end ascent
dominated, and a step-count sweep (a by-product of an algorithm-unrolling experiment)
showed it over-iterated 10×.

## 4. Accuracy: the ceiling, and why every single-frame lever fails

Blind indexing on sparse cxidb saturates at **~71% gated** (reachable ceiling ~76%).
An oracle diagnostic shows the gap is **28% generation-miss** (true axes absent from the
candidate set) vs only **6–8% selection-miss**. Every attempt to push past this failed,
all for the same reason — **the data is spurious-limited**: on a sparse single shot the
spurious-to-true peak ratio caps blind indexing, and richer candidates/peaks/scorers add
spurious signal faster than true signal.

| lever tried | result | why it fails |
|---|---|---|
| denser M1 sampling | no ceiling lift | cos-objective maxima source saturated |
| more candidates (NTOP 30→60) | solve 69→64% | more spurious cells fool the selector |
| projection / Patterson seeds | ceiling +2%, solve −7% | adds spurious axes faster than true |
| M4 TORO joint-basis (lattice reduction) | 0–9% | shortest-vector reduction is noise-dominated |
| scorer: inlier-count | 32% | degenerate cells over-index |
| scorer: pure-defect | 58% | spurious tight cells index few spots tightly |
| **scorer: coverage-gated defect** | **69% (best)** | — |
| algorithm unrolling (learned M3 schedule) | 35% | self-sup loss rewards basin collapse |
| reverse/Chamfer cost (predict→observed) | 48/30/23% | every metric gameable; Ewald fit (λ=1.322 Å) works but doesn't help |
| recover below-threshold peaks | 50→26% | sub-threshold set is mostly noise |

**Conclusion:** the ~71% single-frame ceiling is neither a scoring nor a missing-data
deficiency. The ~100 strongest peaks is near-optimal; the peakfinder threshold discards
noise, not signal. This is why xgandalf, ffbidx, and GLINT all tie near 71–75% on a
single frame.

## 5. What breaks the ceiling: multi-frame consensus (①)

A single frame is spurious-limited; **multiple frames are not**. GLINT-① indexes every
frame blind, derives the unit cell from the indexed minority by consensus (no cell
assumed), keeps blind successes, and rescues failures with the GPU known-cell indexer
seeded by the derived cell. This lifts the gated rate to **94% ≥10 refl (sparse) /
100% (rich)** — the result that ties SOTA, now at 30 f/s.

GLINT-① and ffbidx agree on **95%** of jointly-solved frames (median 100% of shared
spots get identical Miller indices, up to the inevitable handedness convention); GLINT's
*independent* blind path reproduces ffbidx's indices on **70/72** frames. So the rate
parity is the same crystallography, not coincidence.

**How far consensus reaches under severe mosaicity.** Broadening the peaks (δ(q·v) ~ σ|v|
on the 79 Å axis) collapses the single-frame rate, but consensus over many frames recovers
the cell *as long as the truth still surfaces in some frames' candidate set* (its **reach**).
At σ=0.0015 only 9% of frames index and only 11% even surface the correct cell in their top-3,
yet pooling ~40–80 frames recovers it at 87–100% (the few frames that surface it accumulate
support while spurious cells scatter below the support-3 floor). At σ≥0.0020 the **reach
falls to zero** — the long-axis phase is destroyed beyond candidate *generation*, the cell
never appears even as a hypothesis, and consensus is powerless. The recoverability cliff is
sharp between σ=0.0015 and 0.0020. Throughout, a present cell is also the *dominant* cluster
(absence, not out-voting, is the failure mode), so the only lever beyond the cliff is raising
reach at generation, not a better consensus vote (`severe_mosaic_consensus.py`).

## 6. Reproducibility (S3DF)

- Environment: `ana-4.0.58-py3-minipytorch` (torch 2.1.0).
- From `ssh sdfiana001` (SLURM at `/opt/slurm`):
  `srun -p ampere -A lcls:default@ampere -q preemptable --gres=gpu:a100:1 -t N <PY> experiments/<script>.py frames_cxidb_clean.txt 120`
- Key scripts: `glint_fast.py` (blind, env: `STEPS`/`NTOP`/`KEEP`/`SCORER`),
  `replica_gpu.py` (GPU known-cell), `bench_h2h.py` (end-to-end, `RESCUE=gpu`),
  `compare3.py` (3-way), `compare_ffbidx.py` (index agreement),
  `oracle_blind.py`/`sweep_k.py`/`reverse_cost.py`/`index_sw.py` (diagnostics).
- Code + frames under `/sdf/home/s/smarches/git/fftindex`.

## 7. Open levers

- **Throughput (real):** the M1 start grid (70,400 vectors) is the last front-end knob;
  GPU-batching the rescue's c-candidate search would close the remaining gap to ffbidx.
- **Accuracy (bounded):** single-frame blind is at its ceiling — do not chase it. A
  learned M6 ranker is the only untested form, capped at the ~6% selection-miss.
- The genuine accuracy lever is **more / better-overlapped frames** for consensus, not
  any single-frame change.
