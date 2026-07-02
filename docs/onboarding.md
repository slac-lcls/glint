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
the **Ewald sphere** — typically **~15–60 spots**, sparse, with noise and spurious peaks, and (the hard
case) often **no unit cell known in advance** — "blind". Many classical indexers need the cell; blind +
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
GPU-batched (hundreds of frames/s), emitting an oriented lattice per frame → CrystFEL `.stream` → merge.
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

New collaborators: the focused, self-contained research plan (blind CBXD + M1–M6 throughput) is
[`research-plan-yuan.md`](research-plan-yuan.md); the full direction map is [`../ROADMAP.md`](../ROADMAP.md).

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
