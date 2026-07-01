# GLINT as a LUTE Task (`IndexGLINT` / `GLINTIndexer`)

A drop-in alternative to `CrystFELIndexer` in the LUTE SFX DAG:

    PeakFinderSFX -> [GLINTIndexer] -> StreamFileConcatenator -> PartialatorMerger -> HKLManipulator

**Why GLINT in LUTE.** None of LUTE's bundled CrystFEL builds (0.10.2 default ... 0.12.0) are compiled
with FFBIDX -- `indexamajig --indexing=ffbidx` errors "compiled without FFBIDX support". So GPU
fast-feedback-style indexing is simply *unavailable* in LUTE today. GLINT fills that gap: GPU blind
indexing + cross-frame consensus, ~10^2-10^3x faster, emitting the same CrystFEL `.stream` the rest of
the DAG consumes. On sparse real data GLINT also recovers ~1.5x more frames than xgandalf and, refined
through partialator, merges to a more complete / higher-signal dataset.

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
