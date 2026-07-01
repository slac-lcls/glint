# GLINT — contributor onboarding

Welcome. GLINT is a fast, GPU-native crystallography indexer (blind serial + dense rotation).
This page gets a new collaborator from zero to a validated change. See also the top-level
[`README.md`](../README.md) (what GLINT does + the library API) and [`CONTRIBUTING.md`](../CONTRIBUTING.md)
(the short version of the rules below).

## 1. Repository & sync model

**GitHub is the source of truth:** `git@github.com:slac-lcls/glint.git` (private, in the `slac-lcls` org).
Everyone — laptops, S3DF, NERSC — clones from and pushes to GitHub. There is no other "canonical" copy.

```bash
git clone git@github.com:slac-lcls/glint.git
cd glint
```

- Work on a **feature branch**, open a **pull request**, get a quick review, then merge to `main`.
  (`main` is the shared trunk; keep it green.) For tiny, low-risk fixes among the core team, a direct
  push to `main` is fine — use judgement.
- **S3DF and NERSC are already wired to GitHub** (SSH). On those machines just `git pull` / `git push`
  like anywhere else; the local repo is `~/git/glint` on S3DF.

## 2. Environments

GLINT is pure Python (numpy/scipy/torch); CUDA is used automatically when present, CPU otherwise.

| where | setup |
|---|---|
| **laptop / any** | `pip install -e .`  (gets you the `glint` command + `import glint`) |
| **S3DF** | `source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh && conda activate ana-4.0.58-py3-minipytorch` |
| **NERSC (Perlmutter)** | `module load pytorch/2.6.0` |

> Naming note: the package is `glint` (`import glint`); `import fftindex` still works as a back-compat alias.

## 3. Layout

```
glint/          the engine — one file per stage (import glint.<mod>)
  glint_fast.py     M1–M3 blind front end (seed grid, score, gradient ascent)
  glint_index.py    M4 assembly / refine
  replica_gpu.py    GPU known-cell rescue (ffbidx-style)
  multishot.py      cross-frame consensus (derive the cell; reject non-crystals)
  hybrid_stream.py  orchestration: blind → consensus → rescue (→ optional cascade)
  geom.py           CrystFEL .geom + peaks → reciprocal q
  predict.py        spot prediction + integration (→ real I/σ)
  stream.py         results → CrystFEL .stream
  cascade.py        optional external fallback (ffbidx / xgandalf-clone)
experiments/    research scripts + the validation harness (NOT shipped in the wheel)
lute/           LUTE `GLINTIndexer` task (drop-in for CrystFELIndexer in the SFX DAG)
docs/           this file + notes
```

## 4. Run it

```bash
# peaks + geometry → indexed CrystFEL stream (what a peakfinder8 / LUTE run emits)
python -m glint.glint_cli --peaks peaks.stream --geom detector.geom -o indexed.stream
# pre-bridged q-vectors instead of peaks:
python -m glint.glint_cli --qframes frames.txt -o indexed.stream
```

Useful flags: `--cell "a b c al be ga"` (known cell) · `--nbest N` (consensus hypotheses) ·
`--mode auto|sparse|dense` · `--cascade <driver>` (external fallback) · `--integrate` (real I/σ) ·
`--fromfile <sol>` (hand orientations to CrystFEL for the refined merge) · `--device cpu|auto`.

## 5. Validate before you push ("definition of done")

- **CPU smoke test must pass** — no GPU needed, runs anywhere:
  ```bash
  python experiments/test_cli_smoke.py     # expect ALL PASS (6/6)
  ```
- **No indexing-rate regression.** Any change to the front end / consensus / rescue must hold the
  rate on the 120 sparse cxidb frames (blind ~71% gated; hybrid consensus ~96% at ≥10 reflections)
  and 100% on the 10-cell dense sweep. The `experiments/` harness has the scripts (run on GPU via
  `srun`, below). If a change is meant to be bit-identical, verify it is.
- Keep the change scoped; put throwaway analysis in `experiments/`, not the shipped `glint/` package.

## 6. Where the compute + data live

- **S3DF GPU:** `srun -p ampere -A lcls:default@ampere -q preemptable --gres=gpu:a100:1 --pty bash`
- **NERSC GPU:** `srun -A lcls_g -C gpu -q interactive -N1 -n1 --gpus 1 -t 30 --pty bash`
- **Datasets:** cxidb-17 lysozyme + cxidb-45 Proteinase K (S3DF), cxidb_62 hexagonal (NERSC) —
  ask for exact paths; they're the standard test/benchmark sets.

## 7. Good first tasks

**Crystallography / real data**
- Run GLINT blind on a real SFX dataset you know and sanity-check the cell + merge vs cctbx.xfel/DIALS.
- Extend the `--fromfile` → CrystFEL-refine → `partialator` merge to more proteins beyond ProK/lysozyme.
- Exercise the LUTE `GLINTIndexer` task in a real SFX DAG (`lute/`), where LUTE's CrystFEL builds lack FFBIDX.

**GPU / algorithms**
- Throughput: GPU-batch the known-cell rescue's candidate search to close the gap to ffbidx; trim the
  M1 start grid.
- A symmetry-constrained (Bravais) GPU orientation refiner, or the learned CNN peakfinder on the FFT volume.

## 8. The paper

The write-up (IUCr Acta A style) lives separately (LaTeX → Overleaf). Ask for access if you want to
contribute text/figures; the code repo and the paper are kept in step but are different repositories.
