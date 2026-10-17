# Configuration, sourced by every script in this folder. Paths and settings only.
# Every input path was checked to exist on sdfiana027 on 2026-10-16.
# No defaults: VERIFY_DIR, VERIFY_GLINT and VERIFY_OUT must be exported (../README.md).

TOOLS_HOME=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r51_merge_vs_frames   # scripts (this folder)
GLINT_TREE=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}   # GLINT 95d51ce, plain tree with a COMMIT file, never modified
OUT=${VERIFY_OUT:?set VERIFY_OUT to the output root}/r51_merge_vs_frames   # all outputs
CFBIN=/sdf/group/lcls/ds/tools/crystfel/0.10.2/bin              # the CrystFEL of every r51 merge (LUTE rounds 2-7) and of merges_tofile_vs_crystfel
TU=/sdf/group/lcls/ds/tools/lute/test_utilities/sfx            # LUTE test cells (md5 = the copies the r51 runs used)

# mfx100848724 r51 (Jungfrau-16M, tetragonal lysozyme P4_3 2_1 2)
SYM=4/mmm
CELLFILE=$TU/lyso_tetragonal_p43212.cell                       # md5 dfb8e692b12ce33ca3d1c1dc4d75639a

# Stage A (the recorded run): CrystFEL's own indexing of r51, LUTE SFX round 4 (29 Sep 2026): 12,000 events -> 8,770 frames
# with >= 10 peaks -> 280 crystals on 251 frames (xgandalf, --multi, cell check on). The stream LUTE's
# MergePartialator read.
VAL4=/sdf/data/lcls/ds/prj/prjdat21/scratch/lute_sfx_val4/sfx_mfx100848724_r51_crystfel
A_INPUTS="r51CF:$VAL4/mfx100848724_r51_crystfel_all.stream"
A_LUTE_HKL=$VAL4/mfx100848724_r51_crystfel.hkl                  # LUTE's own merge of the same stream (reproduction check)

# Stage B (not run; only after merges_tofile_vs_crystfel): its two arms of r51 on the whole run (11,058 frames).
#   CF  = CrystFEL's own indexing; GLb = GLINT 95d51ce blind orientations -> indexamajig --indexing=file (tofile route)
MERGES_OUT=$VERIFY_OUT/merges_tofile_vs_crystfel
B_INPUTS="r51CF_merges:$MERGES_OUT/r51/CF.stream r51GLb_merges:$MERGES_OUT/r51/GLb.stream"

# The sweep
FRACTIONS="0.10 0.20 0.35 0.50 0.75 1.00"  # of the frames with a crystal; nested within a seed. The low end is
                                          # dense on purpose: the dry run found no CC1/2 from partialator at 25 frames
SEEDS="1 2 3 4 5 6 7 8"                  # random orders; seed 0 = the stream as written (100% only)
NULLS="all shell"                         # (I, sigma) permuted within each crystal: over all its reflections / within
                                          # blocks of 20 neighbours in resolution
NSHELLS=10
NPAR=8                                    # merges at a time
NJ=4                                      # threads per merge (NPAR*NJ = cpus-per-task)

# The merge protocol: LUTE MergePartialator of the r51 runs (lute.db, task 4) = MERGE of merges_tofile_vs_crystfel, + -y SYM -j NJ.
MERGE="--model=unity --min-measurements=2 --max-rel-B=100.0"
# The second merger: process_hkl as GLINT's shipped experiments/phaseC_merge.py calls it (+ --odd-only/--even-only).
MC="--scale"

# GLINT environment: the one GLINT's own sbatch files use (experiments/live_gate_gpu.sbatch). Used here for numpy only.
# psconda.sh and the conda activate.d hooks read unset variables, so `set -u` is lifted around them.
glint_env() {
    local had_u=0; case $- in *u*) had_u=1; set +u ;; esac
    source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
    conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
    [ "$had_u" = 1 ] && set -u
    export PYTHONPATH=$GLINT_TREE
    export PYTHONDONTWRITEBYTECODE=1                       # no __pycache__ in the (95% full) home
    [ "${CONDA_DEFAULT_ENV:-}" = ana-4.0.58-py3-minipytorch ] || { echo "glint_env: activation failed" >&2; return 1; }
}

# Run a command, logging its exact argv first (commands.log is the provenance record of every CrystFEL call).
run() {
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-interactive}" "$PWD" "$*" >> "${CMDLOG:-$OUT/commands.log}"
    "$@"
}
