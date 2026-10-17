# Configuration, sourced by every script in this folder. Paths and settings only; no logic beyond two helpers.
# Every input path here was checked on sdfiana027 on 2026-10-15/16.
# No defaults: VERIFY_DIR, VERIFY_GLINT and VERIFY_OUT must be exported (../README.md).

J2_HOME=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/merges_tofile_vs_crystfel   # scripts (this folder)
GLINT_TREE=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}   # GLINT 95d51ce, plain tree with a COMMIT file, never modified
OUT=${J2_OUT:-${VERIFY_OUT:?set VERIFY_OUT to the output root}/merges_tofile_vs_crystfel}   # all outputs (J2_OUT: dry run only)
CFBIN=/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin              # the CrystFEL of every LUTE r51/r194 run (lute.db executables)
TU=/sdf/group/lcls/ds/tools/lute/test_utilities/sfx            # LUTE test geometry + cells (md5 = the copies the runs used)
BTX=/sdf/group/lcls/ds/tools/btx/stable                         # btx 0f1af1d (2024-03-03): rebuilds the r0016 .cxi
BTX_ENV=ana-4.0.47-py3                                          # btx's own psana1 env (btx scripts/elog_submit.sh)

# Datasets that run. cxidb-45 is NOT here: its images are on NERSC only.
DATASETS="r51 r194 r0016"

# mfx100848724 r51, Jungfrau-16M, tetragonal lysozyme. Events 0..17,871 -> 11,058 frames with >= 10 peaks
# (LUTE FindPeaksSFX round 7, 1 Oct 2026, amax_thr 1000; the .cxi is a virtual dataset over per-rank files).
r51_LIST=/sdf/data/lcls/ds/prj/prjdat21/scratch/lute_sfx_val7/J2_r51_30k/sfx_mfx100848724_r51_glint_peaks/sfx_mfx100848724_r51_glint.list
r51_GEOM=$TU/mfx100848724_356mm_refined.geom
r51_CELLFILE=$TU/lyso_tetragonal_p43212.cell
r51_CELL="79.17 79.17 37.96 90.00 90.00 90.00"
r51_LATT=tPc
r51_SYM=4/mmm
r51_IDXX=""                                   # extra indexamajig flags, EVERY arm of r51: none (LUTE)

# mfx101343025 r194, Jungfrau-16M, monoclinic lysozyme P2_1. Only the first 3,000 events exist as a .cxi:
# 278 frames with >= 10 peaks above 5000 ADU (LUTE round 5, 29 Sep 2026). Small; labelled as such.
r194_LIST=/sdf/data/lcls/ds/prj/prjdat21/scratch/lute_sfx_val5/sfx_mfx101343025_r194_glint/sfx_mfx101343025_r194_glint_peaks/sfx_mfx101343025_r194_glint.list
r194_GEOM=$TU/mfx101343025_r0139_refined.geom
r194_CELLFILE=$TU/maybe_lyso.cell
r194_CELL="28.00 62.50 60.90 90.00 90.80 90.00"
r194_LATT=mPb
r194_SYM=2/m_uab
r194_IDXX=""                                  # none (LUTE)

# mfxx49820 r0016, Epix10ka2M, tetragonal lysozyme. btx's .cxi was purged with the experiment scratch; job 0
# (j2_r0016_cxi.sbatch) rebuilds it with btx's own PeakFinder and r0016 settings for psana events 0..5,999.
# Geometry and cell are btx's (results/btx/geom/r0016.geom, cell/sample2.cell). Unrefined psana geometry is
# not used anywhere in this check.
R16_EVENTS=6000
R16_RANKS=16
r0016_LIST=$OUT/r0016/cxi/r0016.lst
r0016_GEOM=/sdf/data/lcls/ds/mfx/mfxx49820/results/btx/geom/r0016.geom
r0016_CELLFILE=/sdf/data/lcls/ds/mfx/mfxx49820/results/btx/cell/sample2.cell
r0016_CELL="79.327 79.461 38.406 90.111 89.981 90.083"
r0016_LATT=tPc
r0016_SYM=4/mmm
# btx indexed r0016 with --no-revalidate. With revalidation on, CrystFEL 0.10.2 keeps 5-22 of the stored peaks of
# the dry-run frames and indexes 1 of 9; without it 7 of 9 (dry run, 16 Oct). Applied to every r0016 arm.
r0016_IDXX="--no-revalidate"

# The LUTE functional-test settings (lute.db of rounds 4/5/7). Integration and merging are the same in every
# arm and every dataset; only the orientation source and the acceptance checks differ.
INTEG="--integration=rings-nocen --fix-divergence=0.0 --int-radius=3,4,6 --int-diag=none --push-res=infinity"
CF_INDEX="--indexing=xgandalf --tolerance=5,5,5,1.5 --multi"           # LUTE IndexCrystFEL (CrystFEL test)
# CFnr: sensitivity arm on r51/r194 only (CF + --no-revalidate): is CrystFEL's low r51 rate its revalidation?
CFNR_DATASETS="r51 r194"
FILE_INDEX="--indexing=file --tolerance=10,10,10,3"                     # LUTE IndexCrystFEL (GLINT test) = tofile route
# B0 / N0 only. The cell check stays ON: it reindexes every solution to the reference setting (GLINT's blind
# monoclinic .sol has a length-only setting) and a shuffled orientation keeps its cell, so it still passes.
NOCHECK="--no-check-peaks --no-refine --no-retry"
NOPEAKS="--no-peaks-in-stream"                                          # B0, N, N0: peaks not needed, saves disk
MERGE="--model=unity --min-measurements=2 --max-rel-B=100.0"           # LUTE MergePartialator (+ -y SYM -j MERGE_J)
MERGE_J=1              # FIXED for every merge. partialator 0.10.2's crystal rejections depend on -j AND on
                       # the chunk layout for -j > 1 (r194 val5: -j 72 -> CC 0.257 / 70 OK; -j 8 and -j 1 ->
                       # CC 0.017 / 52 OK; r51 val7 identical at all -j). -j 1 is the pre-registered argv and is
                       # layout-independent, so it is the only setting that means the same thing on both runs.
NODCC="--no-deltacchalf"                                               # secondary protocol, common sets only
NSPLIT=10                                                              # paired half-split orders, seeds 1..10
NULL_SEEDS="1 2 3"                                                     # shuffle seeds for N and N0

# GLINT environment: the one GLINT's own sbatch files use (experiments/live_gate_gpu.sbatch).
# psconda.sh and the conda activate.d hooks read unset variables, so `set -u` is lifted around them
# (with -u on, the activate-binutils hook aborts the calling script silently).
conda_env() {   # ENV [PYTHONPATH]
    local had_u=0; case $- in *u*) had_u=1; set +u ;; esac
    source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
    conda activate "$1" >/dev/null 2>&1
    [ "$had_u" = 1 ] && set -u
    export PYTHONPATH=$2
    [ "${CONDA_DEFAULT_ENV:-}" = "$1" ] || { echo "conda_env: activation of $1 failed" >&2; return 1; }
}
glint_env() { conda_env ana-4.0.58-py3-minipytorch "$GLINT_TREE"; }
btx_env()   { conda_env "$BTX_ENV" "$BTX"; }

# Run a command, logging its exact argv first (commands.log is the provenance record of every step).
run() {
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-interactive}" "$*" >> "$OUT/commands.log"
    "$@"
}
# indexamajig 0.10.2 refuses to overwrite a stream ("Refusing to overwrite stream", exit 1, no Final: line).
# On a rerun the old file is moved aside, never deleted.
aside() { for f in "$@"; do [ -e "$f" ] && mv "$f" "$f.prev_$(date -u +%Y%m%dT%H%M%SZ)"; done; return 0; }
