# Reproducing the results in the GLINT paper

This maps the tables and figures of *Blind indexing by cross-frame consensus* to what regenerates
them: the script, the data, the hardware, and the command. It is written for a referee or a reader
who has the repository and wants to check something specific.

Coverage is deliberately uneven. Artifacts a reader is likely to want to check are worked through;
a few are recorded only as not reproducible here. Two tables carry no entry because there is
nothing to run: `tab:modules` describes the source layout and is checked by reading the modules it
names, and `tab:realindex` gathers rows stated elsewhere in the paper, each covered under its own
dataset below.

**Read this first.** The results divide into four tiers by what they cost to check, and the cheap
tiers are not token gestures. The paper's central claim — that the unit cell can be recovered from
sparse still frames with no cell supplied — reproduces on a laptop in about two minutes, and the
xgandalf comparison it is measured against reproduces *exactly*, in about two seconds, from data
committed here. Start there.

| tier | what it needs | what it covers |
|---|---|---|
| **0** | nothing | every published number, with its provenance, and an automated consistency check |
| **1** | this checkout, CPU only | GLINT's own blind indexing on the benchmark, the xgandalf rows of the headline table, the consensus barrier, the regression suite |
| **2** | a CUDA GPU + public CXIDB data | the timing tables, and the benchmark at speed |
| **3** | SLAC S3DF / NERSC accounts | the real-data merges and the three statistics figures |

Some artifacts are **not reproducible from this checkout at all**. They are listed explicitly in
[Not reproducible here](#not-reproducible-from-this-checkout) rather than left for you to discover.

Tables are referenced by their LaTeX labels (`tab:summary`, …) because the printed numbers shift
with typesetting; each entry says what the table reports.

---

## Tier 0 — check the numbers without running anything

Every measured number in the paper is registered in one table, each with a comment naming the run,
job ID or PR that produced it.

```bash
python3 experiments/check_numbers.py --facts
```

Prints 134 key/value rows. Requires only the standard library — no install, no dependencies. This
is the authoritative source: where a number in the paper and a number here disagree, this file is
what was measured.

The same tool also reads documents and fails on values it knows to be retired, on claim shapes the
measurements no longer support, and on arithmetic that does not follow from the table (percentages
recomputed from their counts, speedups from their timings):

```bash
python3 experiments/check_numbers.py /path/to/glint_rewrite_JAC_refined.tex
```

It is not a spell-checker for numbers in general — it enforces the specific values and framings
this project has gotten wrong before, which is why several rules carry a comment explaining the
mistake they exist to prevent.

## Tier 1 — laptop, CPU only, no downloads

Everything here runs from a clean checkout on CPU. The scoring and gate-arithmetic commands need
only `numpy` and `scipy`; running GLINT itself additionally needs `torch` (CPU build is fine).

```bash
python3 -m pip install -e .        # installs numpy/scipy/torch -- do this in a fresh environment
```

That install is what brings in the dependencies. The `PYTHONPATH=.` on the commands below only
makes the checkout importable -- `experiments/` is deliberately not shipped in the wheel, so the
research scripts run from the checkout rather than site-packages -- but it installs nothing, so a
fresh environment still needs the line above.

### The xgandalf rows of `tab:summary` — exact, ~2 seconds

`tab:summary` is the head-to-head indexing comparison on the 120-frame cxidb-17 benchmark. The
xgandalf arms were run with the production `libxgandalf` from CrystFEL 0.12.0 at CrystFEL's own
default settings, and **their per-frame solutions are committed here**, so the scoring reproduces
without an xgandalf build, without the raw data and without a GPU:

```bash
cd experiments/xgandalf && PYTHONPATH=../.. python3 score_xg_gate.py
```

```
xgandalf BLIND      N=120: correct-lattice 94/120=78%  >=25%(Table1) 86/120=72%  ...
xgandalf KNOWN-CELL N=120: correct-lattice 111/120=92%  >=25%(Table1) 96/120=80%  ...
```

The `>=25%(Table1)` column is the paper's gate — a correct reduced cell **and** at least 25% of the
frame's observed peaks indexed **and** at least 10 reflections. `86/120 = 72%` is the paper's
xgandalf blind rate, reproduced exactly. That the three gates disagree (78% / 72% / 78%) is the
point of quoting the bar with the number: the same run scores differently under each, which is why
comparisons across papers that do not state their acceptance rule are not comparable.

### GLINT's own blind indexing — ~2 minutes, no GPU

GLINT-①, the arm reported throughout, is `glint.hybrid_stream.hybrid_index(frames,
Mc_known=None)`: N-best blind, consensus over the pooled N-best, the best N-best cell consistent
with it, then cell-general rescue. It runs on the committed q-vectors with no GPU:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python3 experiments/score_glint_gate.py -N 120
```

`CUDA_VISIBLE_DEVICES=` is what actually forces CPU here: the module-level device is chosen at
import, before any flag is read, so on a GPU host the command would otherwise use CUDA. (The CLI's
`--device cpu` sets the same variable for you; either is fine, but set one deliberately if you mean
to test the CPU path.)

The scorer applies `tab:summary`'s gate -- correct reduced cell **and** at least 25% of observed
peaks indexed **and** at least 10 reflections -- reading its tolerance from the xgandalf scorer
above so the two arms cannot drift apart. Add `--cell` for the known-cell arm.

Measured here on an M-series laptop, **~115 s**:

```
GLINT-(1) BLIND  N=120: correct-lattice 115/120=96%  >=25%(Table1) 91/120=76%  >=10refl 115/120=96%
  consensus cell edges [37.8 78.6 78.9]  support 90  (refused: False)
```

The cell is recovered blind: `[37.8 78.6 78.9]` against lysozyme's 79.0/79.0/38.0 A, derived from
the frames alone with no cell supplied. That is the paper's central claim, reproducible on a laptop.

**One frame short of the table, and here is why.** The paper's blind row is `92/120` at this gate;
this CPU run gives `91`. That is expected, and it is worth understanding before you conclude
anything from a re-run of your own:

* **It is not run-to-run noise.** Two identical CPU runs return `91` bit-for-bit, with the same
  consensus cell and the same support.
* **It is not code drift.** `91` comes back identically at HEAD, at the pre-binarisation commit
  `8da091b`, and at `7ca3b49` -- the very commit where the `92` was recorded. The pipeline has not
  changed its answer.
* **It is the gate sitting on a knife-edge.** Two of the 120 frames are within 0.002 of the 0.25
  boundary and are **one peak** from crossing it: frame 118 at 79/318 = 0.2484 and frame 89 at
  34/137 = 0.2482. One peak moving across the 0.15 hkl-residual tolerance -- routine between CUDA
  and CPU kernels -- moves the published count by one.

So the table's `92` is an A100 measurement of a quantity that is device-sensitive at the ±1 level,
and `91` on CPU is the same result, not a contradiction. (Independently, `azimuth_validate.py`'s
reconciliation block records `93` for this arm at this gate -- the same effect in the other
direction.) What is *not* device-sensitive, and is what the row is really claiming, is the
correct-lattice count: **115/120** here, every run, every commit tested.

Note separately that `91/120` is also a real published value -- the *known-cell* row -- which the
paper's SI explains as two pipelines at one gate. That is a different distinction from this one,
and the numerical coincidence is unfortunate; do not merge the two stories.

Note also that this is *not* `compare3.py`'s blind-top-1 arm -- the two agree on the 120-frame
subset and differ by 15 frames at n=480 (361 vs 346), so the distinction only becomes visible on
the larger set.

To write a CrystFEL stream instead of a score:

```bash
PYTHONPATH=. python3 -m glint.glint_cli --qframes experiments/frames_cxidb_clean.txt \
    -N 120 --device cpu -o indexed.stream
```


### The consensus barrier, cached vs uncached

The recorded N-best pool for the 120-frame benchmark is committed (`nbest_120.npz`, 360 hypotheses
= 120 frames × top-3), so the consensus reduction runs with no GPU and no re-indexing. This script
times the shipped (cached) path against a reconstruction of the pre-cache one, both in the same
run — the only way to check a speedup whose baseline no longer exists in the code:

```bash
PYTHONPATH=. python3 experiments/consensus_barrier_ab.py 5      # 5 = timing repetitions
```

Its argument is the repetition count, **not** a pool size: it always consensus-es all 360
hypotheses. The gate's accept/refuse arithmetic is separately checkable from the same file, but no
committed script sweeps it — the `min_frac`/`min_lead` decision is exercised by
`experiments/test_consensus_gate.py` on its own fixtures instead.

### The regression suite

```bash
PYTHONPATH=. python3 experiments/run_ci_locally.py
```

24 steps, ~3 minutes, no GPU and no data (each test builds its own inputs). This is the same set
CI runs on every push. Passing establishes that the CPU-reachable half of the engine is intact —
the alias gate, peak-finder thresholding, ASIC seam masking, negative-intensity handling,
event-addressed and multi-panel integration, the streaming driver's gates and fan-out guards, and
the number guard's own self-tests. It does **not** exercise the GPU indexing kernels.

### The vector figures — needs the manuscript sources, not this repository

Figures 2 (two paths), 4 (streaming schedule), 5 (pipeline) and 6 (Ewald construction) are TikZ:
drawn in the manuscript sources, not generated from data, so there is no script and no dataset
behind them. They rebuild by compiling the paper — but the `.tex` sources live in the **manuscript**
repository, not this one, so this step needs whatever the journal supplies you rather than a clone
of GLINT:

```bash
cd <manuscript sources> && pdflatex glint_rewrite_JAC_refined.tex
```

Figure 6 additionally exists there as a standalone document (`glint_ewald.tex`) that compiles on
its own.

## Tier 2 — a CUDA GPU and public data

### The data

Five CXIDB accessions, all CC0, from `https://www.cxidb.org/data/<id>/`:

| entry | contents | used for |
|---|---|---|
| **17** | lysozyme, LCLS-CXI, CSPAD | the principal sparse-still benchmark (120- and 480-frame sets) |
| **45** | Proteinase K, SACLA MPCCD | the DIALS head-to-head; 907 readable frames |
| **62** | hexagonal *P*6, SACLA MPCCD | the large-cell / indexing-ambiguity dataset |
| **61**, **83** | POMGnT1; β-lactamase (EuXFEL AGIPD) | transfer across detectors |

**Size warning:** cxidb-17 is deposited as nine per-run tarballs totalling roughly 315 GB. The
120-frame benchmark does not need any of them — it is committed here as reciprocal-space vectors
(`experiments/frames_cxidb_clean.txt`, FRAME blocks of 3 columns in Å⁻¹), which is what every
indexer in `tab:summary` was actually fed. Download the raw data only if you want to re-derive the
peak lists themselves.

The Jungfrau-4M lysozyme dataset (`cxil1015922` r0033) is **LCLS experimental data and is not
public**.

Timing tables (`tab:throughput`, `tab:stages`) are per-row measurements from
`experiments/bench_kc_graph.py`, `bench_fused.py`, `index_batch_sweep.py` and `profile_glint.py`.
They were measured on one A100, so numbers will not transfer to a different GPU.

**Two of those four are not runnable as-is**: `index_batch_sweep.py` hard-codes a worktree path and
`profile_glint.py` hard-codes both its repository root and its input, all under `/sdf/home/`, with
no path argument. Edit the constants at the top or treat those two rows as facility-only.

Several entries in the FACTS table are *derived* rather than measured (frames/s from ms/frame,
speedups from timing pairs) and so need no independent measurement at all — the arithmetic guard
checks them against each other, and `--facts` marks which is which in its provenance comments.

## Tier 3 — SLAC S3DF / NERSC

The real-data merge tables (`tab:realmerge`) and the three statistics figures (7, 8, and the phase
diagram) depend on data or scripts that live on facility filesystems.

The Jungfrau-4M row runs through the LUTE SFX DAG with GLINT substituted for `CrystFELIndexer`
(`lute/glint_dag.yaml`, `integrate: true`); see [`lute/README.md`](lute/README.md) for
configuration and [`lute/STATUS.md`](lute/STATUS.md) for the measurement record. **The merge
protocol is part of the measurement**: partialator with the partiality model, `--iterations` at the
CrystFEL default, and the resolution cutoff stated in the table caption. Changing any of them
reproduces a different number — an earlier version of this project published a value that could not
be reproduced for exactly this reason (it had been merged at `--iterations=1`), so if your numbers
disagree, check the protocol before the code.

Figures 7 and 8 are rendered by `exp1_fig_notitle.py` and `adv_fig_notitle.py` on S3DF. Their
generators write to filenames that **differ from the names the paper uses**, so searching for the
paper's filename finds nothing; search for a panel title or legend label instead.

## Not reproducible from this checkout

Stated plainly, because a wrong guess costs more than a gap:

- **Figure 1** (`figures/Recreated/Main_draft_v0.png`) — no generator exists in either repository.
  It is a schematic; the row (b) panels beside it were produced from cxidb-17 data.
- **`tab:cxidb62`** (the hexagonal large-cell comparison) — no script, log or peak list for
  cxidb-62 is committed, and there is no FACTS entry for its rates. Note also that the xgandalf,
  TORO and Nasser figures in that table are **those works' own published numbers under their own
  acceptance conventions**, not measurements made here; the caption says so, and it is not a
  controlled head-to-head like `tab:summary`.
- **The cxidb-45 and cxidb-17 merge rows** of `tab:realmerge` — the indexing side is reproducible,
  but no committed script runs the merges themselves; the route is documented in prose only.
- **The `asdf` and `mosflm` rows** of `tab:summary` — the input builder (`experiments/make_cxi.py`)
  and the scorer are committed, but the harness that drove those two indexers is not.

If you need one of these, ask — in most cases the run exists and only the driver is missing from
the repository.

## A note on why the numbers are guarded

Several values in this paper moved during its preparation, and two of them moved *silently*: a
percentage pair stayed written in prose after the underlying counts changed, and a merge statistic
was published from an under-converged run. Both were caught by re-deriving numbers from their
sources rather than by re-reading the text. That is what `experiments/check_numbers.py` mechanizes,
and it is the reason Tier 0 is the first thing this document offers: the fastest way to check this
paper is not to rerun it, but to ask the repository what it measured.
