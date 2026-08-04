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

## Scale out (MPI) — how a LUTE/slurm task runs it

`glint_xtc_mpi.py` is the same body sharded across ranks: each rank owns event `i` where `i % nranks
== rank` (round-robin, so indexable frames that cluster in time still spread evenly), reads + indexes
its shard into a partial `.stream`, and rank 0 concatenates. Same flags as `glint_xtc.py`, both eras:

```bash
# -n = ranks = event shards. Give each rank ONE GPU so its peak-find + index (and, for xtc2, the
# conda2 bridge worker it spawns and shares an env with) land on a distinct device.
srun -n 8 --gpus-per-task=1 python glint_xtc_mpi.py --exp <exp> --run <run> --zdist 0.246 -o out.stream            # xtc1
srun -n 8 --gpus-per-task=1 python glint_xtc_mpi.py --exp <exp> --run <run> --zdist 0.246 --psana 2 -o out.stream  # xtc2
```

The global event index is written into each chunk, so per-rank chunks never collide and the merge is a
plain header-once concatenation. GPU pinning: if the launcher already gives each rank one device
(`--gpus-per-task=1` / `--gpu-bind`), the wrapper leaves it alone; otherwise pass `--gpus-per-node N`
and each rank is pinned to `local_rank % N` **before** torch/cupy import and before the bridge worker
spawns (so xtc2's inherited-env worker shares the same GPU). For xtc2 the MPI is only at the conda1
level — each rank calls its own conda2 bridge worker with its shard, so there is no MPI inside conda2.

### Related: the online/streaming counterpart (LCLStreamer)

This wrapper is the **file/batch** scale-out. The **online** counterpart is
[LCLStreamer](https://github.com/lclstream/lclstreamer) (V. Mariani *et al.*, SLAC): a **producer**
(MPI on CPU nodes) reads raw `xtc`, serialises batches with `HDF5BinarySerializer`, and **pushes them
over the network** (ZeroMQ `PUSH`/`PULL`) to a **GPU consumer**. Note it covers **both eras** —
`Psana1EventSource` and `Psana2EventSource` are first-class — so it is *not* a psana2-only tool. A
[Jungfrau GPU writeup](https://confluence.slac.stanford.edu/spaces/~ajshack/pages/672473679/Jungfrau+GPU+Computing+with+LCLStreamer)
(A. Shackelford) records ~1.4 GB/s for Jungfrau.

**The difference from envbridge is the node boundary.** LCLStreamer decouples the reader from the GPU
env *across nodes*; envbridge decouples them *within a single node*, in-process. Complementary, not
competing: envbridge sends tiny **q** after an in-reader peak-find (low bandwidth, but the reader itself
needs a GPU); LCLStreamer streams **raw frames** and the consumer does the compute. A streaming GLINT is
the natural LCLStreamer **consumer** — raw frame → peak-find → index on one GPU — reusing its
`Psana1DetectorInterface`/`Psana2DetectorInterface` producer instead of our conda2 reader; that is the
device-resident streaming driver at network scale.

## Pieces

| file | runs in | does |
|---|---|---|
| `glint_xtc.py` | conda1 (torch) | driver: `build_parser`/`read_qframes`/`index_and_write`, `--psana 1\|2` dispatch → `hybrid_index` → `.stream` |
| `glint_xtc_mpi.py` | conda1 (torch) | MPI wrapper: shard events over ranks, one partial stream each, rank 0 concatenates |
| `xtc_core.py` | conda1 or conda2 | shared peak-find + q core + `event_in_shard` (identical for both eras) |
| `xtc_qreader_psana1.py` | conda1 (in-process) | psana1 reader — `DataSource("exp=X:run=N")`, `det.calib`, `det.coords_*` |
| `xtc_qreader.py` | conda2 (via bridge) | psana2 reader — `DataSource(exp=,run=)`, `det.raw.calib`, `GeometryAccess` |
| `test_core.py` | GPU | shared core on synthetic frames with peaks at known positions (no psana) |
| `test_shard.py` | CPU (CI-able) | sharding partition (disjoint/complete/balanced) + stream merge, no psana/MPI/GPU |
| `end_to_end_test.py` + `_sim_qreader.py` | torch env | the bridge call + driver indexing on synthetic q (no psana) |

## Verified (S3DF, A100)

- **Shared core** `test_core.py`: peaks planted at known pixels → the exact expected q is reproduced,
  covering the arithmetic both eras use.
- **psana1 reader** imports in the GLINT env (its psana1 API resolves).
- **End-to-end** `end_to_end_test.py`: the bridge call + `hybrid_index` recover a planted lysozyme cell
  30/30, psana-free.
- **Sharding + merge** `test_shard.py`: round-robin ownership is a disjoint + complete + balanced
  partition (nranks 1–64); the stream merge keeps one header + every chunk, tolerates empty shards, and
  preserves global event ids — all psana/MPI/GPU-free.
- **envbridge** (its home): conda1 numpy 1.26 ↔ conda2 numpy 2.3, float64 bit-exact.

## The real-data gate — xtc2 VALIDATED, xtc1 pending

The synthetic tests don't exercise the psana **geometry**, the one real risk. Status:

**xtc2 / psana2 — validated on real data.** On a real psana2 Jungfrau16M run whose deployed psana
geometry is a LUTE/BayFAI-refined fit, the reader's per-pixel coord→`|q|` reproduces the full trusted
geometry to **max 0.025 %, median 0.011 %**: the deployed `calibconst` geometry *is* the refined fit (so
no geometry override is needed), the constant-`--zdist` override is an excellent approximation (per-pixel
Z spread ~0.1 mm), and real frames peak-find to sane ring `|q|`. The reader is **self-consistent** —
coords (`get_pixel_coords`) and data (`raw.calib`) are both psana-native order, so there is no internal
segment permutation.

**xtc1 / psana1 — still pending** a run that has *both* raw xtc and a trusted refined geometry staged
(the obvious calibration candidate's raw data is on tape, not disk).

**Residuals (both eras):**

* psana per-pixel **Z is nominal** — `--zdist` overrides it; a wrong value scales every `|q|`. Source it
  from the geometry's refined distance (e.g. a `.poni` `Distance:`).
* coords are the **PSANA frame** (cframe=0), correctly handed for GLINT's blind indexing but **not**
  CrystFEL's lab frame — don't feed these orientations to `indexamajig --indexing=file` unfixed.
* **handedness/chirality** is mirror-invariant in both `|q|` and cell params, so a powder/geometry check
  cannot catch a global mirror — confirm chirality once on a real crystal (for blind indexing it is just
  the enantiomorph, resolvable downstream).
* if you ever reconcile with an *external* `.geom` (different segment order), the size assert catches a
  size mismatch, not a **permutation** — verify against `det.raw.image()`.

## Known simplifications (correctness-first)

* Within one rank, the reader returns that shard's whole q at once (consensus wants the pooled set
  anyway). Scale-out is by event sharding across ranks (`glint_xtc_mpi.py`); a *streaming* reader that
  overlaps read with index inside a rank is still future work.
* Per-panel peak-find, one `PeakFinderV4` each. Peak centroids rounded to the nearest pixel for the
  coord lookup.
