# GLINT as a LUTE Task (`IndexGLINT` / `GLINTIndexer`)

A drop-in alternative to `CrystFELIndexer` in the LUTE SFX DAG:

    PeakFinderSFX -> [GLINTIndexer] -> StreamFileConcatenator -> PartialatorMerger -> HKLManipulator

**Why GLINT in LUTE.** None of LUTE's bundled CrystFEL builds (0.10.2 default ... 0.12.0) are compiled
with FFBIDX -- `indexamajig --indexing=ffbidx` errors "compiled without FFBIDX support". So GPU
fast-feedback-style indexing is simply *unavailable* in LUTE today. GLINT fills that gap: GPU blind
indexing + cross-frame consensus, ~10^2-10^3x faster, emitting the same CrystFEL `.stream` format the
rest of the DAG consumes.

## Is it ready?

Partly, and [**STATUS.md**](STATUS.md) is the honest answer -- what is measured, what is assumed, and
what has never been run. It tracks
[the seven things that stand between this and production](STATUS.md#the-seven-things-that-stand-between-this-and-production),
struck through as they close:

| | | |
|---|---|---|
| 1 | ~~no test for the LUTE task model~~ | **done** -- tests, and CI runs six CPU-only files on every push (glint#107) |
| 2 | ~~`--pf8-min-snr` unreachable~~ | **done** |
| 3 | ~~emitted `.stream` not mergeable~~ | **done**, verified on real data (needs `integrate: true`, below) |
| 4 | ~~150 ms/event, 98% of it CPU calibration~~ | **done** -- `--gpu-calib`, byte-identical stream, 4.1x end to end |
| 5 | ~~no ana env satisfies both psana and torch~~ | **done** -- GLINT runs on torch 1.11, so both do |
| 6 | ~~`PF8_MIN_SNR = 15` is detector-specific~~ | **done as a reframing** (glint#110) -- the calibrated object is the PAIR (`threshold`, `min_snr`) |
| 7 | ~~geometry provenance is silent when wrong~~ | **done** -- startup check, three states, never silent |

All seven are now closed. Item 6 closed as a measurement that **reframed the question**: `min_snr`
is not the calibrated quantity on its own. With the beamline's 110 ADU floor (`thr_adu`, glint#108)
the shipped 15 sits near the knee on raw Jungfrau **16M** (`mfx101555026` r0013, scored against the
beamline's own event-mapped hit list), while a *floor-less* ladder on a kept Jungfrau run puts the
knee at **6--10**, with practitioners running 3.5--6. So calibrate the pair (`threshold`,
`min_snr`) per detector before `peakfinder: pf8` is used on new hardware; it does not affect the
default `v4`.

Two cautions, both recorded in [STATUS.md](STATUS.md): the earlier conclusion that Jungfrau
"tolerates" a high cut came from **a run the beamline discarded**, which passed everything at every
threshold — that framing is inverted, not merely refined. And `min_peaks` is coupled: what collapses
yield is frames falling below it, not the threshold degrading solutions. Tune the two together.

> **The default stream is ORIENTATION-ONLY and is NOT mergeable.** Every reflection carries
> placeholder `I=0.00 sigma(I)=0.00`. Choose one of:
>   * `integrate: true` -- GLINT predicts and box-integrates its own reflections and writes real
>     I/sigma, so the stream goes straight to `PartialatorMerger` with no CrystFEL step; or
>   * `tofile:` -- hand the orientations to `indexamajig --indexing=file` (below). CrystFEL's
>     prediction refinement imposes the lattice symmetry, and this still gives the **better merge**.
>
> Feeding the default stream to partialator merges zeros.

> **Renamed:** `tofile:` was `fromfile:`. GLINT *writes* that file; the old name came from CrystFEL's
> reader flag (`--fromfile-input-file`) and so read backwards from the GLINT side. Existing configs
> keep working — `fromfile:` is still accepted and maps to `tofile:` — but setting both is an error.

On sparse real data GLINT recovers ~1.5x more frames than xgandalf and, via the `tofile` route,
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

> **Trap: `activate_installation` can silently point at an empty python tree (glint#128).**
> It derives `PYTHONPATH` from whatever `python3` is ambient on `$PATH`, not a pinned one. A LUTE
> install can carry several `install/lib/pythonX.Y` trees side by side (e.g. 3.9, 3.11, 3.12) and
> leave some unpopulated -- if the ambient `python3` resolves to one of those, activation reports
> nothing wrong and exports a `PYTHONPATH` into the void. Every LUTE task then dies later with
> `ModuleNotFoundError: No module named 'launch_scripts'` or a bare subprocess return code `127`,
> far from this cause -- `install/bin/launch_slurm` and `install/bin/submit_slurm` also bake in a
> shebang pinned to a specific ana release, so which python3 is ambient in *your* shell may not
> even be the one that matters. **Before sourcing `activate_installation`**, confirm the tree it
> will pick actually contains `launch_scripts`:
>     python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
>     ls "$LUTE/install/lib/python<that version>/site-packages/launch_scripts"   # must exist
> `./install_into_lute.sh` now runs this check automatically against `$LUTE` and refuses to install
> (exit 3) if it finds the trap -- pass `--skip-activation-check` if you already have a workaround
> (e.g. the shim noted in glint#128) or want to install anyway. This cannot be fixed from the GLINT
> side; `lute/upstream_activate_installation.patch` is a draft fix to propose upstream.
>
> **The check is install-time and best-effort, not a runtime guarantee.** It samples `python3` from
> the shell running `install_into_lute.sh`, not the shell that later actually launches a job. The
> `Run` recipe above sources `psconda.sh` *after* that point, which can select a different
> interpreter -- and therefore a different, possibly empty, tree -- than whatever was ambient at
> install time. GLINT's own scripts are never in that launch chain before the failure (see #128's
> analysis), so we cannot check the true launch-time interpreter from here; the guard instead prints
> a population map of every `lib/pythonX.Y` tree the install carries, so an empty tree a *different*
> runtime `python3` could land on stays visible even when the install-time one happens to be fine.
> **The interpreter active at LAUNCH time (i.e. right after `source install/bin/activate_installation`
> in the recipe above) is what actually governs** -- verify that one directly with the `python3 -c ...`
> / `ls ... launch_scripts` check a few lines up, immediately before `submit_launch_slurm.sh`.
> The check also only fires when `activate_installation` still contains the ambient-derivation
> pattern quoted above (`sys.version_info.major` interpolated into a `site-packages` path); a
> patched upstream that no longer derives the tree from the ambient interpreter will not trip it.

## Best merge: hand CrystFEL the refined solution
Set `tofile:` (+ `lattice: tPc` for tetragonal) in the `IndexGLINT` config; GLINT emits a
`--indexing=file` solution, then:
    indexamajig --indexing=file --fromfile-input-file=glint.sol --tolerance=10,10,10,3 ...
CrystFEL's refiner imposes the lattice symmetry -> best merge (validated: beats xgandalf on cxidb-17).

`lattice:` applies **only** to the `--tofile` solution file. The GLINT stream header always reports
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
the box sum is a fused GPU kernel). **Trade-off:** the `tofile` route above still merges better,
because CrystFEL's prediction refinement imposes the lattice symmetry. Use `integrate` when you want a
GPU pipeline with no CrystFEL dependency; use `tofile` when merge quality is what matters.

## Self-contained front end: drop FindPeaksSFX
Set `images` (raw `.cxi` or a `.list`) instead of `peaks` and GLINT peak-finds on the GPU itself, so
the DAG loses its 73-core CPU peak-finding node. `glint_dag_images.yaml` is that DAG.

    images: "{{ work_dir }}/run.cxi"
    peakfinder: "stored"   # v4 | pf9 | stored -- `stored` reuses the .cxi's own peakfinder8 peaks
    # top_peaks: 200        # images only -- OPTIONAL, and NOT a speedup: it is a guard against a
                             # finder over-finding on background. Leave UNSET by default. Measured on
                             # 120 real cxidb frames (glint#137): top_peaks 200 costs ~2 points of
                             # correct-lattice, 100 costs ~11, 50 collapses the rate -- see the field
                             # doc in glint_index.py. The earlier "~100 is the sweet spot" claim here
                             # predates #35/#33, which fixed top_peaks to actually truncate on this
                             # (now-default) `stored` path; on this corpus it no longer holds.

`peaks` and `images` are mutually exclusive (the model rejects both, or neither, at config time).
