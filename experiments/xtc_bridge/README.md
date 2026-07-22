# GLINT ← xtc, one command (LCLS-I *and* LCLS-II)

GLINT can already peak-find and index its own frames from a `.cxi` (`--images`). This reads **`xtc`
directly**, GPU peak-finds, and indexes — for data that never becomes a `.cxi` — as a **single
command**, for both data eras.

## The two eras are not symmetric

| data | framework | where the reader runs | bridge? |
|---|---|---|---|
| **xtc1** (LCLS-I: MFX/CXI, the bulk of existing data) | psana1 | **in the GLINT env** — psana1 + torch coexist | **no** — in-process |
| **xtc2** (LCLS-II) | psana2 | conda2 — numpy 2.3, no torch | **yes** — [envbridge](https://github.com/slac-lcls/drp-benchmarks/tree/main/envbridge) |

psana1 lives in the GLINT env (`ana-4.0.58`) alongside torch, so xtc1 is read **in one interpreter, no
bridge** — the simpler path, and where most data is. xtc2's psana2 needs numpy 2.3 and cannot co-import
with torch, so its reader runs in conda2 and only the q-vectors cross the pickle-free wire. Both paths
share the same peak-find + q core and end at the same GLINT blind `hybrid_index` → CrystFEL `.stream`.

## Run

```bash
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch

# xtc1 (LCLS-I) — in-process, the default
python glint_xtc.py --exp <exp> --run <run> --det jungfrau --zdist 0.246 -o out.stream

# xtc2 (LCLS-II) — over the bridge (needs envbridge, below)
python glint_xtc.py --exp <exp> --run <run> --zdist 0.246 --psana 2 -o out.stream
```

`--psana 2` needs envbridge in the GLINT env:
`pip install 'git+https://github.com/slac-lcls/drp-benchmarks.git#subdirectory=envbridge'`.
Add `--cell "a b c al be ga"` for known-cell; `--reader-env` picks the conda2 env (default `xpp_drp_gpu_311`).

## Pieces

| file | runs in | does |
|---|---|---|
| `glint_xtc.py` | conda1 (torch) | driver + `--psana 1\|2` dispatch → `hybrid_index` → CrystFEL `.stream` |
| `xtc_core.py` | conda1 or conda2 | shared peak-find + q core (identical for both eras) |
| `xtc_qreader_psana1.py` | conda1 (in-process) | psana1 reader — `DataSource("exp=X:run=N")`, `det.calib`, `det.coords_*` |
| `xtc_qreader.py` | conda2 (via bridge) | psana2 reader — `DataSource(exp=,run=)`, `det.raw.calib`, `GeometryAccess` |
| `test_core.py` | GPU | shared core on synthetic frames with peaks at known positions (no psana) |
| `end_to_end_test.py` + `_sim_qreader.py` | torch env | the bridge call + driver indexing on synthetic q (no psana) |

## Verified (S3DF, A100)

- **Shared core** `test_core.py`: peaks planted at known pixels → the exact expected q is reproduced,
  covering the arithmetic both eras use.
- **psana1 reader** imports in the GLINT env (its psana1 API resolves).
- **End-to-end** `end_to_end_test.py`: the bridge call + `hybrid_index` recover a planted lysozyme cell
  30/30, psana-free.
- **envbridge** (its home): conda1 numpy 1.26 ↔ conda2 numpy 2.3, float64 bit-exact.

## ⚠ The real-data gate (unchanged, both eras)

The synthetic tests don't exercise the psana **geometry**, the one real risk — identical for psana1 and
psana2 because both default to the PSANA coordinate frame:

* psana per-pixel **Z is nominal** — `--zdist` overrides it; a wrong value scales every `|q|`.
* coords are the **PSANA frame** (cframe=0), correctly handed for GLINT's blind indexing but **not**
  CrystFEL's lab frame — don't feed these orientations to `indexamajig --indexing=file` unfixed.
* `calib()` is DAQ panel order, coords are geometry-file order; the size assert catches a mismatch, not
  a **permutation**. Verify segment order against `det.image()` on first real data. Cell params are
  mirror-invariant, so a handedness flip would pass a cell-agreement check — confirm chirality once.

**The gate:** on a run that also has a trusted `.geom`, index the same events both ways and confirm cell
+ per-frame q agree.

## Known simplifications (correctness-first)

* One reader call returns the whole run's q at once (consensus wants the pooled set anyway); a
  batched/streaming reader is future work.
* Per-panel peak-find, one `PeakFinderV4` each. Peak centroids rounded to the nearest pixel for the
  coord lookup.
