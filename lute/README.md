# GLINT as a LUTE Task (`IndexGLINT` / `GLINTIndexer`)

A drop-in alternative to `CrystFELIndexer` in the LUTE SFX DAG:

    PeakFinderSFX -> [GLINTIndexer] -> StreamFileConcatenator -> PartialatorMerger -> HKLManipulator

**Why GLINT in LUTE.** The CrystFEL builds LUTE runs lack FFBIDX: 0.10.2 (LUTE's default) predates
it, and 0.12.0 and 0.13.0 report "compiled without FFBIDX support". S3DF's separate fast-feedback build
(`/sdf/group/lcls/ds/tools/crystfel-fast-feedback-indexer`) is not one LUTE's tasks use. GLINT adds GPU
blind indexing + cross-frame consensus, ~10^2-10^3x faster, emitting the same CrystFEL `.stream` format
the rest of the DAG consumes.

## What is validated

The DAG runs end to end on real data and produces a mergeable dataset. On `cxil1015922` r0033
(Jungfrau-4M lysozyme, 1563 frames) it indexes 1506 blind (96%) and merges 1482 of them to
CC\* 0.915 / R_split 31.6% / ⟨I/σ⟩ 7.7 at 2.1 Å. On sparse cxidb-17 frames GLINT-① indexes 361 of 480 blind
against xgandalf's 350 at the same gate — a match, not a lead (McNemar *p* = 0.18).

Those merge numbers come from job `35507050` and are pinned, with their protocol, in
`experiments/check_numbers.py` (`--facts`); quote them from there.
[**STATUS.md**](STATUS.md) holds the evidence for the *LUTE integration itself* — every measurement
with the run that produced it, the negative results, and, the part worth reading before trusting a
number here, the assumptions that have never been tested. Two of those bound what is below:

* **`peakfinder: pf8` is not calibrated for your detector until you calibrate it.** The calibrated
  object is the *pair* (`thr_adu`, `min_snr`), not `min_snr` alone: peakfinder8 applies an absolute
  ADU floor on top of the relative SNR test, and every practitioner value is quoted with one. The
  shipped `PF8_MIN_SNR = 15` is above where practitioners run this hardware (3.5–6). Measured on
  Jungfrau 16M with the interior ASIC seams masked, the knee is at `min_snr` 6–8 and the corrected
  pair is (110 ADU, 8) — but that pair is **only valid with the mask**, and `asic_seam_mask` is not
  yet wired into any ingest path, so the default stays 15 until both move together
  ([glint#127](https://github.com/slac-lcls/glint/issues/127), STATUS.md item 6). The default
  finder, `v4`, is unaffected.
* **`min_peaks` is coupled to the threshold.** What collapses yield is frames dropping below
  `min_peaks` and never being offered — not the threshold degrading solutions. Tune the two
  together.

## Choose a merge route

The default stream carries the cell and the per-frame orientation with placeholder `I=0.00`
intensities: what a refiner needs, not what a merger needs. Set one of:

* **`integrate: true`** — GLINT predicts and box-integrates its own reflections and writes real
  I/sigma, so the stream goes straight to `PartialatorMerger` with no CrystFEL step.
* **`tofile:`** — hand the orientations to `indexamajig --indexing=file` (below), so CrystFEL's
  prediction refinement imposes the lattice symmetry.

Which of the two merges *better* is **unresolved**: the only head-to-head
([glint#129](https://github.com/slac-lcls/glint/issues/129)) ran them at unmatched integration
settings, and matching those closed the CC\* gap. Choose on dependencies.

Both are configured below. (`tofile:` was once called `fromfile:`; the old name is still accepted
and maps to it, because GLINT *writes* that file while CrystFEL's reader flag is what it was named
after. Setting both is an error.)

## Install
    ./install_into_lute.sh [/path/to/lute_new/lute]     # default ~/git/lute_new/lute
Copies `glint_index.py` -> `lute/io/models/`, exports it, and registers
`GLINTIndexer = Executor("IndexGLINT")` in `managed_tasks.py`. The installed copy's default
`executable` is this checkout's `glint_launch.sh`; if the checkout moves, inspect the generated
path change and re-run the installer with `--force`, or set `executable` in the config. The repo copy has no default, so a copy made by hand fails validation
until `executable` is set. GLINTIndexer runs on a **GPU partition** (see
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

## Merge through CrystFEL: hand it the refined solution
Set `tofile:` (+ `lattice: tPc` for tetragonal) in the `IndexGLINT` config; GLINT emits a
`--indexing=file` solution, then:
    indexamajig --indexing=file --fromfile-input-file=glint.sol --tolerance=10,10,10,3 ...
CrystFEL's refiner imposes the lattice symmetry, which GLINT's own integrator does not. That is the
concrete thing this route buys; it is not established that the merge comes out better (glint#129).

`lattice:` applies **only** to the `--tofile` solution file. The GLINT stream header always reports
`lattice_type = triclinic / centering = P`, so set partialator's point group explicitly (`-y`) in the
`PartialatorMerger` config rather than relying on the header.

**Two traps on this route**, both hit in the first end-to-end DAG run
([glint#3](https://github.com/slac-lcls/glint/issues/3), 2026-08-21) and each of which fails in a way
that does not point at its cause:

* **The added `indexamajig` task must reuse the stored peaks** (`peaks: "cxi"`). Left unset, it runs
  its own peak search, then validates GLINT's solutions against peaks GLINT never saw and rejects
  them — reported as `1563 processed, 0 indexable`, which reads like a bad solution file.
* **CrystFEL 0.12.0's `--indexing=file` is broken**: `Failed to prepare indexing method
  file-nolatt-nocell`, before any frame is read. The same `.sol` on **0.11.1** indexes normally.
  LUTE configs pin 0.12.0, so this is the version you get by default. (Recorded also in
  [glint#129](https://github.com/slac-lcls/glint/issues/129), where it is a confound on the
  route-comparison numbers.)

## CrystFEL-free merge: native integrate
Set `integrate: true` and GLINT predicts + box-integrates its own reflections, writing real I/sigma
straight into the stream -- no `indexamajig` step. Needs image data: with `peaks` also set
`image_dir`; with `images` the frames are already at hand.

    integrate: true
    image_dir: "{{ work_dir }}/images"   # required with `peaks`; ignored with `images`
    int_dmin: 2.0
    int_tol: 0.002    # the model's default; the CLI's 0.006 over-predicts (CC1/2 0.04 vs 0.28)

The integration itself is cheap: the whole-frame float64 upcast that used to dominate it is gone
(~105x on a 16 Mpix frame), and what remains is a gather over the predicted boxes. Prediction and
that gather run on the host in numpy on this route — the fused GPU box-integration lives on the
streaming driver's device path, not here. **Trade-off, stated as what is actually known:** `tofile` buys CrystFEL's
prediction refinement, which imposes the lattice symmetry; `integrate` buys a pipeline with no
CrystFEL dependency and, on the one comparison run, ~5x the observations per crystal. Their merge
quality has NOT been separated -- glint#129's head-to-head used unmatched integration settings, and
once matched the CC\* difference closed. Do not pick one expecting a quality win.

## Self-contained front end: drop FindPeaksSFX
Set `images` (raw `.cxi` or a `.list`) instead of `peaks` and GLINT takes the peaks itself, so the
DAG loses its separate FindPeaksSFX node. `glint_dag_images.yaml` is that DAG. With `peakfinder: v4`
or `pf9` the peak-finding runs on the host CPU of the GLINT node (numpy/scipy, one frame at a time
in one process), not on the GPU and not spread over the 73 cores the FindPeaksSFX node had; budget
the run's time for that. `stored` finds no peaks: it reuses the ones already in the `.cxi`.

    images: "{{ work_dir }}/run.cxi"
    peakfinder: "stored"   # v4 | pf9 | stored -- `stored` reuses the .cxi's own peakfinder8 peaks
    # top_peaks: 200        # images only -- OPTIONAL, and NOT a speedup: it is a guard against a
                             # finder over-finding on background. Leave UNSET by default. Measured on
                             # 120 real cxidb frames (glint#137): top_peaks 200 costs ~2 points of
                             # correct-lattice, 100 costs ~11, 50 collapses the rate -- see the field
                             # doc in glint_index.py. The earlier "~100 is the sweet spot" claim here
                             # predates #35/#33, which fixed top_peaks to actually truncate on this
                             # (now-default) `stored` path; on this corpus it no longer holds.

`peaks`, `images` and `exp` (raw xtc, routed to `experiments/xtc_bridge/glint_xtc.py`) are the three
frame sources, and the model requires **exactly one** at config time — they are alternatives, not
layers. The xtc route is where the `pf8` calibration caveat above applies; see
[STATUS.md](STATUS.md) for what has and has not been run on it.
