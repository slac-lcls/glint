# Running GLINT at S3DF — from zero to an indexed `.stream`

For people who did not write GLINT: beamline staff putting it into production, and anyone
testing it on their own data. It assumes an S3DF account with an LCLS data allocation and
nothing else.

> **Verification status: executed.** The Track A / A1 walkthrough below was run end to end on
> an S3DF A100 (job `36862739`, `sdfampere024`, GLINT `af147ad`) exactly as printed, and
> reproduced the reference result: **54 frames with ≥ 6 peaks, 50 indexed (92%)**. Every flag and
> field name is additionally checked against the code by script. Tracks A2/A3 and Track B are
> checked against the code but have **not** been executed here.

> **Access.** `slac-lcls/glint` is private pending SLAC's institutional software-release review.
> If you cannot clone it, ask Stefano Marchesini for repository access; there is no public
> download yet.

---

## Which track do you want?

| | **Track A — direct run** | **Track B — LUTE DAG** |
|---|---|---|
| purpose | try GLINT on a dataset | production pipeline |
| gets you | one `.stream` | `.stream` → merged `.hkl` |
| needs | a clone + an `srun` | a LUTE install |
| setup traps | none | `activate_installation` (see Track B) |
| start here if | you are evaluating GLINT on your own data | you are deploying at a beamline |

**Testing GLINT on your data? Use Track A.** It is the same indexer — LUTE only wraps it — and
it skips the one setup step known to fail silently. Move to Track B once it works.

---

## Before you start

```bash
ssh sdfiana027     # then: kinit
```

> **Use an LCLS analysis node (`sdfiana*`), not the general login pool.** `ssh s3df` lands on a
> node that carries neither the LCLS data mounts nor Slurm — measured on `sdflogin002`, where
> `/sdf/data/lcls/ds/mfx/` does not exist and there is no `sinfo`/`sbatch` on `PATH`. The data
> "not found" there is a missing mount, not missing data: the same path resolves normally on
> `sdfiana027`. If your experiment directory appears to be gone, check the host before you
> conclude the run was purged.

Get an interactive GPU node. Everything below assumes you are **on** it, not on a login node:

```bash
srun -p ampere -A lcls:prjdat21 --gres=gpu:a100:1 -c 8 --mem=96G -t 2:00:00 --pty bash
```

The `ampere` partition is busy and several nodes sit in `maint`/`drain` at any time; if `srun`
does not return promptly, submit with `sbatch` instead of waiting interactively.

Set up the environment. These two lines are the whole environment story:

```bash
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
```

`ana-4.0.58-py3-minipytorch` and `ana-4.0.59-py3-minipytorch` are the **only** two ana releases
carrying torch at all. 4.0.58 is the default and the newer torch (2.1.0).

> **If `psana.Detector()` raises a `KeyError` that looks like a typo in your detector name**, it
> probably is not. 4.0.58 cannot parse `Jungfrau.ConfigV4` and *silently drops* the detector.
> Switch releases: `conda activate ana-4.0.59-py3-minipytorch` (torch 1.11.0 — GLINT supports it).

Confirm the GPU is actually yours before you spend a run on it:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

`cuda True` is required. If it prints `False` you are on a login node or the `--gres` did not land.

Then clone and enter the repo — **run everything from the repo root**, which is how `import glint`
resolves (GLINT is not pip-installed into the ana envs):

```bash
git clone git@github.com:slac-lcls/glint.git
cd glint
```

---

## Track A — index a dataset directly

GLINT takes **exactly one** frame source. Pick the row that matches what you have:

| you have | use | entry point |
|---|---|---|
| a raw LCLS run | `--exp` + `--run` | `experiments/xtc_bridge/glint_xtc.py` |
| a `.cxi` of images | `--images` | `python -m glint.glint_cli` |
| a CrystFEL peak `.stream` | `--peaks` | `python -m glint.glint_cli` |

### A1 · Raw xtc — start here if you have an experiment and a run number

```bash
python -u experiments/xtc_bridge/glint_xtc.py \
    --exp mfxx49820 --run 16 \
    --det MfxEndstation.0:Epix10ka2M.0 \
    --zdist 0.102973 \
    --wavelength 1.290757 \
    --geom /sdf/data/lcls/ds/mfx/mfxx49820/results/btx/geom/r0016.geom \
    --psana 1 \
    --min-peaks 6 \
    --max-events 200 \
    -o glint.stream
```

Three flags people get wrong:

* **`--zdist` is in METRES**, and it is required. It is the sample–detector distance. psana's
  per-pixel Z is not used.
* **`--det` is the psana detector name**, not a nickname — e.g.
  `MfxEndstation.0:Epix10ka2M.0`. Get it from `detnames exp=<exp>:run=<run>`.
* **`--geom` is optional to the CLI but you almost always need it.** Without it, psana's own
  pixel coordinates are used, and on most detectors they are not refined well enough to index.
  Use your beamline's refined geometry.

`--wavelength` may be omitted, in which case the per-event photon energy is read from
`--energy-det` (default `ebeamh`). Pass it explicitly if you know it — it is one less thing
that can be wrong.

### A2 · CXI images

```bash
python -u -m glint.glint_cli \
    --images /path/to/run.cxi \
    --geom /path/to/detector.geom \
    --peakfinder stored \
    -o glint.stream
```

`--peakfinder stored` reuses the peakfinder8 peaks already in the `.cxi`. Use `v4` to have
GLINT find peaks itself instead; on this route that runs on the host CPU (numpy/scipy), not on
the GPU.

> Leave `--top-peaks` **unset**. It is a guard against a finder over-finding on background, not
> a speedup. Measured on 120 real frames: `200` costs ~2 points of correct-lattice rate, `100`
> costs ~11, `50` collapses it.

### A3 · An existing peak stream

```bash
python -u -m glint.glint_cli \
    --peaks /path/to/peaks.stream \
    --geom /path/to/detector.geom \
    -o glint.stream
```

### Known cell vs blind

Everything above runs **blind** — GLINT determines the unit cell itself by pooling weak lattice
hypotheses across frames. If you already know the cell, pass it and GLINT registers orientations
against it instead, which is much faster:

```bash
    --cell 78.6 78.6 37.9 90 90 90        # a b c (A) alpha beta gamma (deg)
```

Blind is the interesting mode and the one to test first: if GLINT finds *your* cell without being
told it, that is the strongest signal the run is working.

---

## Did it work?

GLINT prints a progress line and a summary. Check three things, in this order.

**1 · Did frames survive the peak-finder?** The log reports frames with at least `--min-peaks`
peaks. If this is near zero, nothing downstream can work and the problem is peak-finding or
geometry — not indexing.

**2 · How many indexed?** Count crystals in the stream:

```bash
grep -c 'Begin crystal'  glint.stream      # indexed
grep -c 'Begin chunk'    glint.stream      # frames offered
```

**3 · Is the cell right?** Read the cells GLINT found:

```bash
grep 'Cell parameters' glint.stream | head
```

They should cluster tightly on one cell. A blind run whose cells scatter has not converged.

> **These print in nanometres**, as CrystFEL writes them — `Cell parameters 3.82629 7.92156
> 7.98653 nm, ...`. Multiply by 10 before comparing with the ångström values quoted below and in
> the literature. GLINT's own summary line reports the consensus cell in Å, so the two differ by
> a factor of ten by design, not by error.

**Reference — what a healthy small run looks like** (`mfxx49820` r0016, the A1 command above
with `--max-events 200`), and the independent reproduction of it:

| | recorded (job `34240308`) | reproduced (job `36862739`) |
|---|---|---|
| events read | 200 | 200 |
| frames with ≥ 6 peaks | 54 | **54** |
| **indexed** | **50 (92%)** | **50 (92%)** |
| consensus cell (Å) | 38.3 / 79.2 / 79.9 | 38.5 / 79.1 / 79.6 |
| truth | 38.4 / 79.3 / 79.5 | — |
| wall time | — | 60 s on one A100 |

The counts reproduce exactly. The cell differs in the third significant figure and both runs sit
within ~0.4 Å of truth — blind consensus re-derives the cell from the data each time, so expect
that last digit to move; expect the *counts* to be stable.

Your yield will differ — 92% is a bright lysozyme run. What should *not* differ is the shape:
peaks found on most hits, a cell that clusters, and a cell that matches whatever you know about
your sample.

### The big geometry block is a diagnostic, not an error

Before indexing, the xtc route prints a geometry provenance report ending in a verdict —
`CORROBORATED`, `DISAGREE`, or `UNVERIFIED` — plus a `VERDICT: WARN ... geometry is NOT verified`
manifest. **It does not stop the run**, and on the reference dataset above it prints `DISAGREE`
while indexing 92%. Read it like this:

| verdict | meaning | what to do |
|---|---|---|
| `CORROBORATED` | your `.geom` agrees with psana's deployed geometry | nothing |
| `DISAGREE` | your `.geom` differs from psana's | **usually correct** — see below |
| `UNVERIFIED` | you passed no `.geom`; nothing to compare against | **this is the risky one** |

**`DISAGREE` is normally the outcome you want.** psana's deployed geometry is often the *unrefined*
starting calibration, while the refinement everyone actually trusts lives only in a `.geom` and is
never written back. If you supplied a refined `.geom`, it is *supposed* to differ — that is why you
supplied it. This is not hypothetical on the reference run: against psana's own geometry, blind
indexing on `mfxx49820` r0016 locks a **wrong doubled-*c* cell** at support 23/6294 and reports
success; the refined `.geom` takes support to 832 and the cell to the truth.

What the report adds is *which kind* of difference it found:

* a **global scale / distance** term is benign — `--zdist` exists to set exactly that, and it moves
  every panel together;
* a **per-panel shape** term (tilts, offsets) is what `--zdist` cannot absorb and what stops blind
  indexing converging. The discriminator is inter-panel dispersion relative to the size of the
  error, not "could one scale factor explain it" — a real per-quadrant error is still ~54%
  absorbable by a single scale and would be waved through.

**`UNVERIFIED` deserves more caution than `DISAGREE`.** A wrong geometry does not crash and does not
look wrong: it quietly moves every *q*, and blind indexing converges on something else and reports
it confidently. If you see `UNVERIFIED`, supply a refined `.geom` before believing a cell.

> **The stream always says `lattice_type = triclinic, centering = P`.** GLINT imposes no
> symmetry. This is not a failure to detect your space group — symmetry is supplied downstream
> (`partialator -y`). Do not read the header as GLINT's opinion about your crystal.

---

## Getting a mergeable dataset

**The default stream is orientation-only.** Every reflection carries placeholder
`I = 0.00, sigma(I) = 0.00`. It is what a refiner needs, not what a merger needs — sending it
straight to `partialator` merges zeros. Pick one of two routes.

### Route 1 — GLINT integrates (no CrystFEL indexing step)

Add `--integrate` and GLINT predicts and box-integrates its own reflections, writing real
I/sigma into the stream:

```bash
python -u -m glint.glint_cli \
    --peaks peaks.stream --geom detector.geom \
    --integrate \
    --image-dir /path/to/images \
    --int-dmin 2.0 \
    --int-tol 0.002 \
    -o glint.stream
```

> ⚠ **Pass `--int-tol 0.002` explicitly.** The CLI default is `0.006`, and it measurably
> over-predicts: on real data 0.006 gave 2017 reflections of which only 449 sat on a real peak
> (78% background) and **CC1/2 fell to 0.04**, against **0.28** at 0.002. The LUTE Task overrides
> the default for you; a hand-rolled command line does not. This is the single easiest way to get
> a quietly bad merge.

`--image-dir` is required with `--peaks` (GLINT needs the pixels to integrate) and ignored with
`--images`.

Then merge — **no `indexamajig` step**:

```bash
partialator -i glint.stream -o merged.hkl -y 422 --iterations=1
```

Set `-y` to your point group explicitly. The stream header says triclinic/P regardless.

### Route 2 — hand the orientations to CrystFEL

```bash
    --tofile glint.sol --lattice tPc
```

then `indexamajig --indexing=file --fromfile-input-file=glint.sol --tolerance=10,10,10,3 ...`.
This buys CrystFEL's prediction refinement, which imposes lattice symmetry; GLINT's own
integrator does not.

Two traps on this route, both of which fail in a way that does not point at the cause:

* **The `indexamajig` step must reuse the stored peaks** (`peaks: "cxi"`). Left unset it runs its
  own peak search and then validates GLINT's solutions against peaks GLINT never saw — reported
  as `N processed, 0 indexable`, which reads like a bad solution file.
* **CrystFEL 0.12.0's `--indexing=file` is broken** (`Failed to prepare indexing method
  file-nolatt-nocell`, before reading a frame). The same `.sol` works on **0.11.1**. LUTE configs
  pin 0.12.0 by default.

**Which route merges better is unresolved** — the only head-to-head ran at unmatched integration
settings, and matching them closed the gap. Choose on dependencies, not on an expected quality
win: Route 1 removes CrystFEL from the pipeline; Route 2 keeps symmetry refinement.

---

## Run it on YOUR data

Take the A1 command and swap these. Everything else can stay.

| swap | where to get it | gets it wrong how |
|---|---|---|
| `--exp`, `--run` | you know these | — |
| `--det` | `detnames exp=<exp>:run=<run>` | `KeyError` that looks like a typo |
| `--zdist` | your beamline; **metres** | indexes nothing, or a stretched cell |
| `--geom` | your refined CrystFEL `.geom` | peaks found, nothing indexes |
| `--wavelength` | run metadata; else omit and let `--energy-det` read it | systematically wrong cell |
| `--cell` | only if you already know it | — |

Then work up in three steps rather than launching a full run:

1. **`--max-events 200`.** Confirm peaks are found and a cell appears. Seconds to minutes.
2. **`--max-events 2000`.** Confirm the blind cell is stable and the yield holds.
3. **Full run**, as a batch job rather than an interactive `srun`.

If step 1 finds peaks but indexes nothing, suspect `--geom` and `--zdist` before suspecting
GLINT — a wrong detector distance produces a self-consistent, wrong q-space.

### Peak-finder thresholds

Defaults are `v4` and are reasonable. If you are tuning:

* **`--min-peaks` is coupled to the threshold.** What collapses yield is frames dropping below
  `min-peaks` and never being offered — not the threshold degrading solutions. Tune them together.
* **`pf8` is not calibrated for your detector until you calibrate it.** The calibrated object is
  the *pair* (`--thr-*` ADU floor, `--pf8-min-snr`), never the SNR alone. The shipped
  `PF8_MIN_SNR = 15` is well above where practitioners run this hardware (3.5–6). On Jungfrau 16M
  with interior ASIC seams masked the knee is at 6–8 and the corrected pair is (110 ADU, 8) —
  **but that pair is only valid with the mask**, which is not yet wired into any ingest path.
  The default `v4` finder is unaffected by all of this.

---

## Track B — the production LUTE DAG

Replaces `CrystFELIndexer` in the standard SFX pipeline:

    PeakFinderSFX → [GLINTIndexer] → StreamFileConcatenator → PartialatorMerger → HKLManipulator

Worth knowing why this exists: none of LUTE's bundled CrystFEL builds are compiled with FFBIDX, so
`indexamajig --indexing=ffbidx` errors out — GPU fast-feedback indexing is simply unavailable in
LUTE today.

### Install

```bash
./lute/install_into_lute.sh [/path/to/lute_new/lute]      # default ~/git/lute_new/lute
```

This copies `glint_index.py` into `lute/io/models/`, exports it, and registers
`GLINTIndexer = Executor("IndexGLINT")`.

### ⚠ The one step that fails silently

`activate_installation` derives `PYTHONPATH` from whatever `python3` is ambient on `$PATH`. A LUTE
install can carry several `install/lib/pythonX.Y` trees side by side and leave some **unpopulated**.
If the ambient `python3` lands on an empty one, activation reports success and exports a PYTHONPATH
into the void. Every task then dies later with `ModuleNotFoundError: No module named
'launch_scripts'` or a bare return code `127` — far from the cause.

**Immediately after sourcing `activate_installation`, and before submitting**, check the tree that
is actually active:

```bash
python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
ls "$LUTE/install/lib/python<that version>/site-packages/launch_scripts"    # must exist
```

`install_into_lute.sh` runs this at install time and refuses to install (exit 3) if it finds the
trap — but that is **install time, not launch time**, and the recipe below sources `psconda.sh`
after that point, which can select a different interpreter. The check above is the one that counts.

### Run

```bash
ssh sdfiana027 ; kinit
cd $LUTE
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
source install/bin/activate_installation
# ---- run the launch_scripts check here ----
submit_launch_slurm.sh $(which launch_slurm) -e <exp> -r <run> \
    -W $(pwd)/glint_dag.yaml -c $(pwd)/glint_config.yaml --partition=ampere
```

Use `glint_dag_images.yaml` instead if you are feeding `.cxi` images — it has no `FindPeaksSFX`
node, since GLINT takes the peaks itself (`stored`, or `v4`/`pf9` peak-finding on the host CPU of
the GLINT node, not on the GPU).

`lute/glint_config.yaml` is the annotated config; copy the `FindPeaksSFX` and downstream merge
sections from the standard `peakfinder8_lcls2` example.

---

## Known gaps

Things an external user will reasonably expect and not find:

* **`--gpu-calib` is not reachable from LUTE.** It exists only as a CLI flag on `glint_xtc.py`;
  there is no `IndexGLINTParameters` field for it, and `glint_launch.sh`'s xtc flag whitelist does
  not carry it, so it cannot be passed through the Task. It is also **psana1-only** — the flag
  exits with an error under `--psana 2`. On the direct route it is a real option and reproduces
  `det.calib` bit-identically; `det.calib` is 97.7% of the per-event time budget, so this matters
  for throughput.
* **psana2 (`--psana 2`) has never been run** through this integration.
* **MPI sharding (`glint_xtc_mpi.py`) has never been run** through this integration.
* **The `peaks` and `images` routes were not re-run** when the xtc route was added; they are
  pre-existing and untouched, not freshly validated.

---

## Verifying this document

Track A/A1 was executed on 2026-09-03: S3DF job `36862739`, node `sdfampere024`
(A100-SXM4-40GB), GLINT `af147ad`, env `ana-4.0.58-py3-minipytorch` (torch 2.1.0, CUDA available).
The A1 command was run verbatim as printed above — **no `--calib-dir`**, confirming it is not
needed on this dataset — and took 60 s wall.

To re-verify after a change:

```bash
ssh sdfiana027 ; kinit
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
python -c "import torch; assert torch.cuda.is_available()"
# then the A1 command, then the three checks in "Did it work?"
# expect: 54 frames with >=6 peaks, 50 indexed
```

Expect the counts to reproduce exactly and the cell's third significant figure to move. If the
counts differ, that is a real change and this document needs correcting.

## Where to look next

* [`README.md`](README.md) — how the Task is wired, and every trap found while wiring it.
* [`STATUS.md`](STATUS.md) — the measurement record: what was run, on which job, with the negative
  results and the assumptions that have never been tested.
* [`glint_config.yaml`](glint_config.yaml) — annotated LUTE config.
* `experiments/check_numbers.py --facts` — the pinned numbers, with their protocols. Quote from
  here rather than from prose.
