# GLINT — contributor onboarding

Welcome. GLINT is a fast, GPU-native crystallography indexer (blind serial + dense rotation).
This page gets a new collaborator from zero to a validated change. See also the top-level
[`README.md`](../README.md) (what GLINT does + the library API) and [`CONTRIBUTING.md`](../CONTRIBUTING.md)
(the short version of the rules below).

## 0. Background — what indexing is, and what GLINT does

*(If you already do crystallography, skip to §1. This is a primer for a maths / GPU reader.)*

A crystal is a 3-D periodic lattice of molecules. Illuminate it with X-rays and it diffracts: the
detector records a set of bright spots (**Bragg peaks**). After correcting for the detector geometry
and wavelength, each spot maps to a point **q** in 3-D *reciprocal space*, and those points lie on a
lattice — the reciprocal of the crystal's real-space lattice.

**Indexing is the inverse problem:** given the cloud of measured points {q_i}, recover the lattice
that generated them. Every spot is an integer combination of three unknown basis vectors,

    q_i = h_i·a* + k_i·b* + l_i·c*,     (h_i, k_i, l_i) ∈ ℤ³,

so indexing = find the basis B = (a*, b*, c*) — equivalently the **unit cell** (three lengths + three
angles) and the crystal's **orientation** R ∈ SO(3) — and assign each spot its integer (h, k, l). It's
lattice-basis recovery from a noisy point cloud: a cousin of lattice reduction and integer least
squares. The cell is orientation-invariant (a Gram / metric-tensor quantity); the orientation is the
rotation on top.

**Why it's hard in serial crystallography (SFX):** each shot is a single *still* from a crystal in a
random, unknown orientation, so you see only the thin curved slice of the reciprocal lattice that meets
the **Ewald sphere** — typically **a few tens to a couple of hundred spots** (GLINT's own cxidb corpus
stratifies sparse `<70` / moderate `70–150` / dense `>150`; the front end's floor is `--min-peaks 6`, and
no peak cap is applied by default: `--top-peaks 0` keeps every peak, and the optional `M3_CAP`
only cheapens the M3 ascent without dropping peaks from scoring), with noise and spurious peaks, and (the hard case) often
**no unit cell known in advance** — "blind". Many classical indexers need the cell; blind +
sparse is where most methods fall over.

**What GLINT does — a multi-start optimizer + selector.** For the sparse blind case, GLINT is best read
as *massively parallel multi-start optimization*: it seeds **many** candidate directions distributed over
the sphere, ascends each on a **gridless** scoring objective — a direct sum $\sum_i w_i \cos(2\pi\,x\cdot
q_i)$ that peaks when $x$ is a true lattice vector — to its nearest continuous maximum, and then
**selects** the basis vectors that many starts *and* many frames converge on (the consensus / parsimony
step). No FFT, no grid: the true lattice vectors are simply the basins that attract the most starts.
For **dense** data (full / partial rotations, many peaks) it instead seeds from a **3-D FFT** of local
peak clusters — the transform method the project was originally built on (see [`lineage.md`](lineage.md)).
Both front ends feed one refine → assemble → anneal → score core plus a GPU known-cell rescue, all
GPU-batched (the blind pass at ~39 frames/s on one A100, the batched known-cell engine at ~5900 frames/s at B=120), emitting an oriented lattice per frame → CrystFEL `.stream` → merge.
Where the maths lives: `glint/glint_fast.py` (seed grid + objective + ascent — the *optimizer*),
`glint/multishot.py` (cross-frame consensus — the *selector*), `glint/lattice.py` (cell ↔ basis, SO(3)).
The paper's *Architecture* and *Candidate-generation* sections are the fuller treatment.

### Further reading — the indexing literature

GLINT reuses the field's strongest ideas; these are the primary sources, grouped by method family (the
same taxonomy as the paper's landscape table). Start here to place GLINT in context:

- **1-D-FFT / projection (DPS):** Steller, Bolotovsky & Rossmann, *J. Appl. Cryst.* **30**, 1036 (1997);
  MOSFLM — Battye *et al.*, *Acta Cryst.* **D67**, 271 (2011); *labelit* — Sauter, Grosse-Kunstleve &
  Adams, *J. Appl. Cryst.* **37**, 399 (2004).
- **3-D FFT of the peak cloud** *(GLINT's dense front end; the fftindex origin — [`lineage.md`](lineage.md))*:
  DIALS `fft3d` — Winter *et al.*, *Acta Cryst.* **D74**, 85 (2018); cctbx — Grosse-Kunstleve *et al.*,
  *J. Appl. Cryst.* **35**, 126 (2002).
- **Difference vectors:** DirAx — Duisenberg, *J. Appl. Cryst.* **25**, 92 (1992); TakeTwo — Ginn *et al.*,
  *Acta Cryst.* **D72**, 956 (2016); CrystFEL `asdf` — White *et al.*, *J. Appl. Cryst.* **45**, 335 (2012).
- **Sampling / optimization** *(GLINT's family for the sparse blind path)*: xgandalf — Gevorkov *et al.*,
  *Acta Cryst.* **A75**, 694 (2019); pinkIndexer — Gevorkov *et al.*, *Acta Cryst.* **A76**, 121 (2020);
  TORO — Gasparotto *et al.*, *J. Appl. Cryst.* **57**, 931 (2024), doi:10.1107/S1600576724003182;
  *fast feedback indexer* (ffbidx) — PSI software.
- **Merging / post-refinement:** *partialator* — White, *Phil. Trans. R. Soc. B* **369**, 20130330 (2014).
- **The fftindex lineage specifically:** *Compressive Auto-Indexing in Femtosecond Nanocrystallography*,
  arXiv:1011.3072; PeakNet (learned peak finding), arXiv:2303.15301.
- **Polychromatic / averaged regimes** *(exploratory — [`../ROADMAP.md`](../ROADMAP.md) track 5)*:
  Laue / pink-beam is `pinkIndexer`'s regime (above); powder auto-indexing — ITO (Visser, *J. Appl.
  Cryst.* **2**, 89, 1969), TREOR (Werner, Eriksson & Westdahl, *J. Appl. Cryst.* **18**, 367, 1985),
  DICVOL (Boultif & Louër, *J. Appl. Cryst.* **24**, 987, 1991), McMaille (Le Bail, *Powder Diffr.*
  **19**, 249, 2004); Rietveld refinement — Rietveld, *J. Appl. Cryst.* **2**, 65 (1969), GSAS-II —
  Toby & Von Dreele, *J. Appl. Cryst.* **46**, 544 (2013).
- **Recent / adjacent:** Nasser *et al.*, *Robust Indexing for Challenging Serial X-ray Diffraction
  Patterns* (2025, symmetry-aware lattice decoding, small-N); CBXD — Li *et al.*, arXiv:2602.14402 (2026).

New collaborators: the direction map is [`../ROADMAP.md`](../ROADMAP.md) — open work, per track, with
issue numbers. Questions already settled (and the levers that did not pay) are in
[`results.md`](results.md).

## 1. Repository & sync model

**GitHub is the source of truth:** `git@github.com:slac-lcls/glint.git` (private, in the `slac-lcls` org).
Everyone — laptops, S3DF, NERSC — clones from and pushes to GitHub. There is no other "canonical" copy.

```bash
git clone git@github.com:slac-lcls/glint.git
cd glint
```

- Work on a **feature branch**, open a **pull request**, let CI go green, get a quick review, then merge to
  `main` (`main` is the shared trunk; keep it green). Small fixes go the same way: the suites and the number
  guard run on the pull request, and that is where a pin or a figure gets checked.
- **S3DF and NERSC are already wired to GitHub** (SSH). On those machines just `git pull` / `git push`
  like anywhere else; the local repo is `~/git/glint` on S3DF.

## 2. Environments

GLINT is pure Python (numpy/scipy/torch/h5py); CUDA is used automatically when present, CPU otherwise.

| where | setup |
|---|---|
| **laptop / any** | `pip install -e .`  (gets you the `glint` command + `import glint`) |
| **S3DF** | `source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh && conda activate ana-4.0.58-py3-minipytorch` |
| **NERSC (Perlmutter)** | `module load pytorch/2.6.0` |

> Naming note: the package is `glint` (`import glint`); `import fftindex` still works as a back-compat alias.

## 3. Layout

```
glint/          the engine — one file per stage (import glint.<mod>)
  glint_fast.py         M1–M3 blind front end (seed grid, score, gradient ascent); the strict-gate constants
  glint_index.py        M4 assembly / refine
  fused_m3.py, fused_kernels.py, fused_integrate.py
                        the cupy RawKernels behind M3, the batched known-cell engine and the integrator
  replica_gpu.py        per-frame GPU known-cell rescue (ffbidx-style)
  replica_gpu_batch.py  the batched, fused known-cell engine (index_fused; the deep search's depth arguments)
  multishot.py          cross-frame consensus (derive the cell; reject non-crystals) — the batch path
  running_consensus.py  the same consensus one frame at a time, with the sequential lock (streaming)
  hybrid_stream.py      offline orchestration: blind → consensus → rescue (→ escalation → cascade)
  stream_driver.py      the streaming driver: device ring, blind warm-up and lock, batched known-cell,
                        the rescue ladder, cell registry, running merge, effort policy
                        (options: docs/stream_driver_options.md)
  retry_cascade.py      the retry arms the driver and the offline measurement share; escalate_batch
  warmup_batch.py       peak-triaged, fanned-out blind warm-up
  multilattice.py       double-hit primitives: deflate, second-lattice verdict, azimuth scramble
  alias_gate.py, spurious_meter.py
                        lock-time alias gate; spurious-peak meters and the lock-quality null
  lattice.py            cell ↔ basis, SO(3), Laue classes, axis standardization
  geom.py, lute_bridge.py
                        CrystFEL .geom + peaks → reciprocal q (the offline and the driver bridges)
  peakfinder_v4.py, peakfinder8.py, peakfinder9.py, radial.py, ring_mask.py
                        device peak finders (the driver uses v4) and the radial primitive
  predict.py            spot prediction + box integration (→ real I/σ); the driver's .stream writer
  geom_refine.py        running detector-geometry refinement (diagnostic)
  stream.py             results → CrystFEL .stream (offline)
  cascade.py            optional external fallback — shells out to any indexer binary speaking
                        the FRAME-in / basis-out protocol (real ffbidx + real xgandalf drivers
                        in experiments/xgandalf/)
  glint_cli.py          the `glint` command
  index.py, transform.py, peakfind.py, localize.py, seed.py, cnn*.py, detector.py, features.py
                        the FFT-volume lineage and the learned peak finder (docs/lineage.md)
  gpu_pool.py, device_merge.py, joint.py, partiality.py, refine_sym.py, simulate.py, synth_sfx.py
                        idle-GPU pool, device-side merge, joint cell recovery, merge models, simulators
experiments/    research scripts, the CPU suites CI runs, results notes (NOT shipped in the wheel)
lute/           LUTE `GLINTIndexer` task (drop-in for CrystFELIndexer in the SFX DAG)
docs/           this file, the driver's option reference, the replay provenance, settled results, lineage
```

> **Two indexers — don't start from `demo.py`.** `experiments/demo.py` calls `glint.index_shot`
> (`glint/index.py`), the *legacy* **FFT-volume** route: 3-D FFT of the whole peak cloud → peak-pick →
> basis search — the original `fftindex` idea, kept for pedagogy (and its lineage into the dense front
> end; see [`lineage.md`](lineage.md)). The **production** SFX indexer is the modular **M1–M6** pipeline
> listed above (`glint_fast` → `hybrid_stream`, driven by `glint_cli.py`) — that's what the paper and the
> LUTE DAG use. Read `index.py` to grok the FFT-of-peak-cloud concept, but work against the M1–M6 path.

### The two paths — blind vs known cell

The most useful thing to understand about GLINT: **blind and known-cell indexing are different problems,
not the same problem at two speeds.**

| | unknowns | the search |
|---|---|---|
| **Blind** | 3 free vectors — lengths *and* directions *and* mutual angles | generate candidate vectors, then assemble a basis out of them |
| **Known cell** | 3 rotation DOF — lengths and angles are given | rotate a known basis until it fits |

Blind indexing is *combinatorial*; known-cell is *registration*. Per frame that is worth ~1.6× on its
own — blind 26 ms vs the per-frame rescue `index_known_gpu_cell` at 16.5 ms. The dramatic numbers belong
to the *batched, fused* known-cell engine (`replica_gpu_batch.index_fused`): 0.33 ms/hit at B=32 and
0.17 ms/hit at B=120 (fp64, one A100), amortized over a batch, not the per-frame rescue this diagram shows.
More important than either is that
the known-cell pass indexes frames the blind pass could not (see "why rescue works", below).

```text
                          ┌─────────────────────────────────┐
  frames: list of N×3     │  hybrid_index()                 │  hybrid_stream.py
  rlp arrays  ───────────▶└─────────────────────────────────┘
                                      │
        ╔═════════════════════════════▼══════════════════════════════╗
        ║  PASS 1 — BLIND, every frame     index_blind_nbest(q)      ║ glint_fast.py
        ╠════════════════════════════════════════════════════════════╣
        ║ M1  70,400 seeds = 2200 Fibonacci dirs                     ║ glint_index.sample()
        ║       × 32 length shells (30…123 Å, 3 Å step)              ║  ← lengths SEARCHED
        ║ M3  gradient ascent on all 70,400 at once (STEPS=8)        ║
        ║ M2  dedup to distinct maxima → KEEP 44 → NTOP 30 vectors   ║
        ║ M4  ALL triplets C(30,3)=4060, det-filtered → ~1,200 bases ║
        ║ M5  residual-threshold anneal (ANNEAL_ITERS=3), batched    ║
        ║ M6  coverage-gated score → GPU metric-dedup to distinct    ║
        ║     ↳ Buerger reduce + primitivize on the reps             ║
        ║     ↳ up to nbest=3 DISTINCT cells + scores  (n-best)      ║
        ╚════════════════════════════════════════════════════════════╝
                                      │  per-frame hypotheses
                                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │  CONSENSUS over the POOLED hypotheses                       │ multishot.py
        │  consensus_cell() → (Mc, support)                           │
        │  truth recurs across frames, aliases scatter → run cell Mc  │
        └────────────────────────────────────────────────────────────┘
                                      │  Mc — or None if nothing reaches
                                      │  min_support=3 (→ per-frame top-1)
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
   top-1 already consensus-consistent        no consistent hypothesis →
        → take it, FREE (not counted)        ╔══════════════════════════════════╗
   a LOWER-ranked N-best one is              ║ PASS 2 — KNOWN-CELL RESCUE       ║ replica_gpu.py
        → promote it, FREE (n_nbest)         ╠══════════════════════════════════╣
                                             ║ anchor: 16,384 dirs at FIXED     ║  ← length GIVEN
                                             ║   length L0 (shortest axis)      ║
                                             ║   → top120 → refine → NC=16      ║
                                             ║ axis1: 360 azimuths over a HALF  ║  ← angle GIVEN
                                             ║   turn of the cone at angle c01  ║
                                             ║   → top-8 by inlier count        ║
                                             ║ axis2: NOT SEARCHED — solved in  ║  ← _third_axis()
                                             ║   closed form from the metric    ║
                                             ║ → 16×8 = 128 bases, anneal+score ║
                                             ╚══════════════════════════════════╝
                                                            │
                                              accept only if same_lattice(Mr, Mc)
                                                     (stats n_resc)
```

Note the asymmetry: blind explores **70,400** seeds and assembles up to **4,060** triplets, of which the
det filter leaves **~1,200** to actually anneal and score; the rescue explores **128** candidate bases, and
one of its three axes is never searched at all — `_third_axis()` places it analytically from the metric
constraints `a2·e0 = L2·c02`, `a2·e1 = L2·c12`, `|a2| = L2`, with handedness inherited from the reference
cell (`sign(det)` of Mc's columns *after* they are sorted shortest-first, in `_axes_from_cell`).

Two scoping notes that bite if you skip them:

- The azimuth grid is **per-cell** (`_azimuth_grid(c01)`). A perpendicular pair (`c01 = 0`) gets
  `linspace(0, π, 360)` — half the cone at 0.5° steps, which is complete there because `θ+π` maps `a1`
  to `−a1`, the same lattice vector. An **oblique** pair gets `linspace(0, 2π, 360)` instead: `θ+π` then
  maps `a1` to `2·c01·L1·ĉ − a1`, which is *not* `−a1`, and the `−a1` partner lies on the supplementary
  cone `x·ĉ = −L1·c01` that a half sweep never visits. Same sample count either way, so the full turn
  costs nothing (23.9 vs 24.0 ms/frame) — it trades 0.5° → 1° step for the missing half, which the
  annealer absorbs. Sweeping half a cone on oblique cells was a real bug, worth 28–37 points; see the
  resolved entry under "Good first tasks".
- The 16,384 → 4,096 adaptive anchor grid (`_adaptive_dirs`, angles within 2° of 90°; rate-neutral on
  orthogonal cells, ~8 pts worse on triclinic, `KC_ADAPTIVE_DIRS=0` forces full) lives in
  **`replica_gpu_batch.py`** and applies to the batched family only. The per-frame rescue in
  `replica_gpu.py` always sweeps the full 16,384-dir grid.

**Why the rescue recovers frames the blind pass missed.** It is *not* a smarter search. It shares the
blind pass's annealer routine (`anneal_batch_t`, imported from `glint_fast`) but runs it on a longer
schedule (`thr0=0.30, contract=0.82, max_iter=20`, plus a polish pass, vs the blind pass's
`0.25 / 0.85 / ANNEAL_ITERS=3`), and its scoring is its own: `objective_t` in `replica_gpu.py` (unweighted
inlier count + trimmed-log2 defect) rather than the blind pass's `|q|⁻¹`-weighted `objective` +
coverage-gated `score_batch_t`. It wins because it has **information the frame does not contain**,
imported from the other frames.

A sparse still is rank-deficient: at the spot counts of §0 many bases fit about equally well. Blind
indexing then fails in two distinguishable ways that the rescue kills separately:

- **Generation miss** — the true axes never appear among the 30 candidate vectors, so no triplet can span
  the true cell. Better scoring cannot help: it was never a candidate. The rescue doesn't need them to
  appear, because it *constructs* bases at the known lengths and angles. It is not exhaustive, though: the
  anchor is still picked by an inlier-score argmax (16,384 dirs → top-120 → refine → greedy 0.985 dedup →
  NC=16), and only the top-8 sampled azimuths survive per anchor, so a true axis that scores badly on one
  frame can still miss the 128-basis set. What it removes is the need for the *blind* candidate pool to
  have contained the true axes.
- **Selection miss** — the true cell *is* generated but loses the argmax to a spurious or alias cell that
  happens to score better on that one frame. The rescue only ever *seeds* bases with the correct metric, so
  the impostor is not in the set to be chosen; the score no longer has to identify the right *cell*, only
  the right *orientation*. (The anneal that follows is an unconstrained 3×3 refit, so the winner can drift
  off Mc — that is what the final `same_lattice(Mr, Mc)` gate is for.)

Two mechanisms, in cost order — the first is free, and worth understanding before you optimise the second:

1. **N-best re-selection** (`n_nbest`) — costs *nothing*. The blind pass already computed up to 3 cells per
   frame; if top-1 isn't consensus-consistent but #2 or #3 is, take that one. Pure selection fix on
   already-generated hypotheses. (Only *promotions* are counted: `n_nb += (c is not t1)`.)
2. **Known-cell GPU rescue** (`n_resc`) — actual re-indexing against `Mc`. This is what fixes generation
   misses.
3. Optional external cascade (`cascade.py`) on whatever still fails — it shells out to any indexer binary
   speaking the FRAME-in / basis-out protocol; reference drivers for real ffbidx and real xgandalf are in
   `experiments/xgandalf/`.

Every accepted rescue is gated by `same_lattice(Mr, Mc)`, so a frame either comes back on the consensus
lattice or stays unindexed — the rescue cannot pollute the run with a different cell.

**Which entry point to call:**

| function | file | use when |
|---|---|---|
| `hybrid_index()` | `hybrid_stream.py` | default — the whole ladder, fully blind |
| `hybrid_index(..., Mc_known=Mc)` | `hybrid_stream.py` | cell known, want max **accuracy** (still runs the N-best pass) |
| `index_known_fast()` | `hybrid_stream.py` | cell known, want max **throughput** — batched, skips the blind pass |
| `index_fused()` / `index_all_graph()` | `replica_gpu_batch.py` | the batched known-cell engine itself (fused CUDA kernels / CUDA graph) |
| `index_known_gpu_cell()` | `replica_gpu.py` | one frame against one cell (what the rescue calls) |
| `dense_index()` | `hybrid_stream.py` | rotation/dense data — self-indexes per frame, no consensus needed |
| `hybrid_index(..., escalate=True)` | `hybrid_stream.py` | offline, plus a deep known-cell search on the frames that still fail the gate, accepted only against the frame's own scrambled copies (glint#208, #211) |
| `escalate_batch()` | `retry_cascade.py` | that deep search as a batched, null-controlled arm — what the driver's `effort=` calls on the misses |
| `StreamDriver` | `stream_driver.py` | **streaming** — frames pushed one at a time, blind warm-up and lock, batched known-cell, the rescue ladder, running merge; the fourth path below |

The third path, `dense_index`, needs no consensus at all: a dense rotation cloud is already 3-D complete,
so each frame self-indexes via the local-cluster-FFT front end (`index_blind_cluster_seeded`).

### The fourth path — the streaming driver

`hybrid_index` sees all frames at once. `glint.stream_driver.StreamDriver` sees them one at a time, which
changes three things and adds a fourth:

- **The cell is discovered while the data arrive.** With `Mc=None` the first frames are indexed blind and vote
  in a running consensus (`running_consensus.py`, the incremental twin of `consensus_cell`); the cell locks when
  the vote is unambiguous — about six frames on cxidb-17 — and the driver switches to the batched known-cell
  engine for the rest of the run. Hand it a cell and it starts there.
- **Batches are a ring on the device.** `B` frames sit in preallocated device memory; when the ring is full they
  are indexed as one `index_fused` batch and each is integrated against its still-resident pixels, then folded
  into a running merge (completeness, CC½, CC*, R_split under a chosen Laue class).
- **Recovery is opt-in.** `warmup_rescue` re-indexes startup frames when the initial cell locks. For locked
  batches, misses go through `per_lattice`, the deep search enabled by `effort=`, `retry_cascade`, and finally
  the `adaptive_relock` watchdog (with `rescue_buffer` for retroactive re-indexing), in that order. Each
  mechanism shipped with a CPU suite and a measured number; off, the base path is byte-identical.
- **Effort follows the hit rate** (`effort=dict(rate_hz=..., n_gpu=1)`, glint#213). The GPU time per hit is
  `n_gpu / (rate × hit rate)`; the known-cell depth is the deepest tier that fits, and what is left buys a deep
  search on the misses that must beat the frame's own scrambled copies (glint#211's null). Decisions at flush
  boundaries, logged, reproducible from the record.

Two cautions travel with it. The driver's live accept gate (`min_inliers`, `min_inlier_frac`) is looser than
the paper's strict bar, and on lattice-free frames it accepts about half of the sparse ones, so a yield read off
`indexed` is a live-gate yield: compare the recorder's strict column (glint#214 adds the calibrated floor). And
the effort policy's tier costs are per-hit costs at B=120; at the B=20 the replays use the same search is dearer
per hit, so pass `tiers=` measured at the batch size you run.

Run it through the recorder rather than by hand — it is the driver's public surface (`push_q`, `events`) with
strict scoring per species and a regression gate:

```bash
# the published 480-frame arm, as a gate (S3DF, one A100; ~/q480_fix.txt is the lysozyme q list)
PYTHONPATH=. python experiments/record_stream_replay.py --input lyso=~/q480_fix.txt --B 20 --dmin 2.0 \
    --warmup-rescue --adaptive-relock --min-inliers 10 --out replay480.json \
    --expect 333/480 --expect-wresc 10 --expect-relock 1
# the same arm with the effort policy at a 2 kHz budget on one GPU
PYTHONPATH=. python experiments/record_stream_replay.py --input lyso=~/q480_fix.txt --B 20 --dmin 2.0 \
    --warmup-rescue --adaptive-relock --min-inliers 10 --effort '{"rate_hz": 2000, "n_gpu": 1}' \
    --out replay480_effort.json
```

The record's header carries the counters and the effort log; per frame, the event trace says which cell took
the frame, at what inlier fraction, and by which mechanism. Every option, with its default and the pull request
that measured it: [`stream_driver_options.md`](stream_driver_options.md). Two recorded runs, with what each
animation frame means: [`streaming_replay.md`](streaming_replay.md).

## 4. Run it

```bash
# peaks + geometry → indexed CrystFEL stream (what a peakfinder8 / LUTE run emits)
python -m glint.glint_cli --peaks peaks.stream --geom detector.geom -o indexed.stream
# pre-bridged q-vectors instead of peaks:
python -m glint.glint_cli --qframes frames.txt -o indexed.stream
```

Useful flags: `--cell "a b c al be ga"` (known cell) · `--nbest N` (consensus hypotheses) ·
`--mode auto|sparse|dense` · `--escalate` (a deep known-cell search on the frames that still fail the gate,
accepted only against the frame's own scrambled copies; glint#208) · `--cascade <driver>` (external fallback) ·
`--integrate` (real I/σ) · `--tofile <sol>` (hand orientations to CrystFEL for the refined merge) ·
`--device cpu|auto`. `--images raw.cxi --geom detector.geom` runs GLINT's own peak finder on the pixels
instead of reading a peak stream.

## 5. Validate before you push ("definition of done")

- **The CPU suites CI runs must pass, locally, in an environment as poor as CI's:**
  ```bash
  python experiments/run_ci_locally.py               # every CPU step of ci.yml, with torch/cupy/numba blocked
  python experiments/run_ci_locally.py --check-sync  # its list matches ci.yml (CI runs this check too)
  ```
  That is about fifty suites — the smoke test (`test_cli_smoke.py`, 6/6), the consensus and gate pins, every
  driver mechanism with a fit-oracle indexer (`test_stream_*.py`, `test_cell_registry.py`, `test_per_lattice.py`,
  `test_stream_effort.py`), and the number guard (`check_numbers.py`) over the README and the docs. A suite
  added to `ci.yml` but not to the runner's list fails `--check-sync`; a figure that contradicts
  `check_numbers.py --facts` fails the guard.
- **No indexing-rate regression on the GPU.** Any change to the front end / consensus / rescue / driver must
  hold the rate on the 120 sparse cxidb frames — blind 77% (92/120) at the strict ≥25%-of-spots bar, and 96%
  (115/120) at the correct-lattice bar — 100% on the 10-cell dense sweep, and the streaming arm on the 480
  (`record_stream_replay.py --expect 333/480 --expect-wresc 10 --expect-relock 1`, §3). The `experiments/`
  harness has the scripts (run on GPU via `srun`, below). If a change is meant to be bit-identical, verify it is.
- **Constructor pins.** `StreamDriver.__init__` is not keyword-only, so a new option is appended, never
  inserted, and the positional pins in `test_cell_registry.py`, `test_per_lattice.py` and
  `test_lock_gate_wiring.py` are extended; the option gets a row in
  [`stream_driver_options.md`](stream_driver_options.md), which `test_stream_driver_options_doc.py` checks
  against the signature.
- Keep the change scoped; put throwaway analysis in `experiments/`, not the shipped `glint/` package.

## 6. Where the compute + data live

- **S3DF GPU:** `srun -p ampere -A lcls:default@ampere -q preemptable --gres=gpu:a100:1 --pty bash`
- **NERSC GPU:** `srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 30 --pty bash`
- **Datasets:** cxidb-17 lysozyme + cxidb-45 Proteinase K (S3DF; its 290-frame merge is
  CC* 0.90, R_split 31%, ⟨I/σ⟩ 7.8), cxidb_62 hexagonal (NERSC) —
  ask for exact paths; they're the standard test/benchmark sets.

## 7. Good first tasks

**Crystallography / real data**
- Run GLINT blind on a real SFX dataset you know and sanity-check the cell + merge vs cctbx.xfel/DIALS.
- Extend the `--fromfile` → CrystFEL-refine → `partialator` merge to more proteins beyond ProK/lysozyme.
- Exercise the LUTE `GLINTIndexer` task in a real SFX DAG (`lute/`), where LUTE's CrystFEL builds lack FFBIDX.

**Streaming driver** (§3, the fourth path)
- Measure `tiers=` for `effort=` at another batch size or another GPU (the shipped table is B=120 on an A100)
  and rerun `experiments/effort_gpu.sbatch` at that B; the recorder's effort log is the readout.
- A two-species *pixel* stream: the recorded mixture is q-level because the CXIDB 45 deposit is MPCCD data,
  so two species on one detector would be new evidence for the cell registry.

**GPU / algorithms**
- Throughput: the *known-cell* engine is batched and fused down to sub-ms (`index_fused`, PR #16) and now
  beats ffbidx; the *blind* front end is not — the same treatment on M1/M3 (and trimming the 70,400-seed
  start grid) is the open lever.
- A symmetry-constrained (Bravais) GPU orientation refiner, or the learned CNN peakfinder on the FFT volume.
- ~~**Open question** — the rescue's half-turn azimuth sweep on oblique cells.~~ **RESOLVED 2026-07-19:
  it was a bug, and it cost a lot.** `replica_gpu.py` built the second axis over `TH = linspace(0, π, 360)`,
  half the cone. With `a1 = L1·(c01·ĉ + s01·(cosθ·u + sinθ·v))`, `θ → θ+π` sends `a1 → 2·c01·L1·ĉ − a1`,
  which equals `−a1` (same lattice vector, half sweep complete) **only when `c01 = 0`**. For an oblique
  pair the `−a1` partner sits on the supplementary cone and was never generated, so the true axis was
  missed for ~50% of orientations — by the full cone offset (10° triclinic, 20° rhombohedral-oblique),
  far outside the annealer's basin. Measured on A100 over 24 random triclinic cells × 60 orientations
  (`experiments/azimuth_oblique.py`), half → full turn at matched sample count:

  | `\|c01\|` | dense half → full | still half → full |
  |---|---|---|
  | 0.00–0.02 | 98.3% → 100% | 89.2% → 96.7% |
  | 0.02–0.05 | 71.7% → 100% | 92.5% → 97.1% |
  | 0.05–0.10 | 70.0% → 98.9% | 92.1% → 99.6% |
  | 0.10–0.20 | 66.3% → 100% | 89.2% → 97.3% |
  | 0.20–1.00 | 62.7% → 99.8% | 88.3% → 95.0% |

  Even ~1° of obliquity costs ~28 points. Fixed by `_azimuth_grid(c01)` (half turn iff perpendicular,
  else full turn at the *same* sample count — cost-neutral; doubling `NANG` instead buys nothing).
  Perpendicular cells keep the half turn and are **bit-identical**, verified elementwise on the 120
  sparse cxidb frames across per-frame, batched and fused (`experiments/azimuth_validate.py`).

  Two traps this exposed, worth internalising before you trust a cross-cell benchmark:
  - `_axes_from_cell` sorts axes by **length**, so `c01` is the angle between the two *shortest* axes.
    `gen_cells`' monoclinic case (60/70/90, β=105°) has `c01 = 0` and is unaffected — cell obliquity is
    **not** the obliquity of the swept pair.
  - `trig_ob` (rhombohedral, all angles 100°) scored 100% even with the half sweep, because its
    symmetry-equivalent anchors supply a covered-half alternative. **Symmetry masked the bug**; only the
    low-symmetry cells ate it. A 10-cell set with one true triclinic case is thin cover for this class of
    error.

## 8. The paper

The write-up (IUCr Acta A style) lives separately (LaTeX → Overleaf). Ask for access if you want to
contribute text/figures; the code repo and the paper are kept in step but are different repositories.
