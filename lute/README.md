# GLINT as a LUTE Task (`IndexGLINT` / `GLINTIndexer`)

A drop-in alternative to `CrystFELIndexer` in the LUTE SFX DAG:

    PeakFinderSFX -> [GLINTIndexer] -> StreamFileConcatenator -> PartialatorMerger -> HKLManipulator

**Why GLINT in LUTE.** None of LUTE's bundled CrystFEL builds (0.10.2 default ... 0.12.0) are compiled
with FFBIDX -- `indexamajig --indexing=ffbidx` errors "compiled without FFBIDX support". So GPU
fast-feedback-style indexing is simply *unavailable* in LUTE today. GLINT fills that gap: GPU blind
indexing + cross-frame consensus, ~10^2-10^3x faster, emitting the same CrystFEL `.stream` format the
rest of the DAG consumes.

> **The default stream is ORIENTATION-ONLY and is NOT mergeable.** Every reflection carries
> placeholder `I=0.00 sigma(I)=0.00`. Choose one of:
>   * `integrate: true` -- GLINT predicts and box-integrates its own reflections and writes real
>     I/sigma, so the stream goes straight to `PartialatorMerger` with no CrystFEL step; or
>   * `fromfile:` -- hand the orientations to `indexamajig --indexing=file` (below). CrystFEL's
>     prediction refinement imposes the lattice symmetry, and this still gives the **better merge**.
>
> Feeding the default stream to partialator merges zeros.

On sparse real data GLINT recovers ~1.5x more frames than xgandalf and, via the `fromfile` route,
merges to a more complete / higher-signal dataset.

## Install
    ./install_into_lute.sh [/path/to/lute_new/lute]     # default ~/git/lute_new/lute
Copies `glint_index.py` -> `lute/io/models/`, exports it, and registers
`GLINTIndexer = Executor("IndexGLINT")` in `managed_tasks.py`. Edit `executable` in `glint_index.py`
(or `glint_launch.sh`) if the GLINT repo path differs. GLINTIndexer runs on a **GPU partition** (see
`glint_dag.yaml`) and the launcher activates the GLINT torch env.

## Run (mirrors the standard SFX functional test)
    ssh sdfiana027 ; kinit
    cd $LUTE ; source .../psconda.sh ; source install/bin/activate_installation
    submit_launch_slurm.sh $(which launch_slurm) -e <exp> -r <run> \
        -W $(pwd)/glint_dag.yaml -c $(pwd)/glint_config.yaml --partition=ampere

## Best merge: hand CrystFEL the refined solution
Set `fromfile:` (+ `lattice: tPc` for tetragonal) in the `IndexGLINT` config; GLINT emits a
`--indexing=file` solution, then:
    indexamajig --indexing=file --fromfile-input-file=glint.sol --tolerance=10,10,10,3 ...
CrystFEL's refiner imposes the lattice symmetry -> best merge (validated: beats xgandalf on cxidb-17).

`lattice:` applies **only** to the `--fromfile` solution file. The GLINT stream header always reports
`lattice_type = triclinic / centering = P`, so set partialator's point group explicitly (`-y`) in the
`PartialatorMerger` config rather than relying on the header.

## CrystFEL-free merge: integrate on the GPU
Set `integrate: true` and GLINT predicts + box-integrates its own reflections, writing real I/sigma
straight into the stream -- no `indexamajig` step. Needs image data: with `peaks` also set
`image_dir`; with `images` the frames are already at hand.

    integrate: true
    image_dir: "{{ work_dir }}/images"   # required with `peaks`; ignored with `images`
    int_dmin: 2.0
    int_tol: 0.002    # the model's default; the CLI's 0.006 over-predicts (CC1/2 0.04 vs 0.28)

The integration itself is cheap (the whole-frame float64 upcast that used to dominate it is gone, and
the box sum is a fused GPU kernel). **Trade-off:** the `fromfile` route above still merges better,
because CrystFEL's prediction refinement imposes the lattice symmetry. Use `integrate` when you want a
GPU pipeline with no CrystFEL dependency; use `fromfile` when merge quality is what matters.

## Self-contained front end: drop FindPeaksSFX
Set `images` (raw `.cxi` or a `.list`) instead of `peaks` and GLINT peak-finds on the GPU itself, so
the DAG loses its 73-core CPU peak-finding node. `glint_dag_images.yaml` is that DAG.

    images: "{{ work_dir }}/run.cxi"
    peakfinder: "stored"   # v4 | pf9 | stored -- `stored` reuses the .cxi's own peakfinder8 peaks
    top_peaks: 100         # images only; ~100 strongest is the measured sweet spot

`peaks` and `images` are mutually exclusive (the model rejects both, or neither, at config time).
