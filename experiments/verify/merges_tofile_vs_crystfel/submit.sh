#!/bin/bash
# Submits the five Slurm jobs of this check with their dependencies, from an S3DF login node.
# Needs VERIFY_GLINT, VERIFY_DIR and VERIFY_OUT exported (../README.md).
#   0 merges_r0016_cxi  CPU 16 ranks, <= 1 h       rebuild btx's r0016 .cxi (events 0..5,999), ~20 GB
#   1 merges_glint   GPU, 1 A100, <= 4 h        GLINT solution files (blind GLb, known-cell GLk)   (afterok 0)
#   2 merges_cf      CPU 64 cores, <= 3 h        CrystFEL's own indexing (arm CF)                    (afterok 0)
#   3 merges_tofile  CPU 32 cores, <= 4 h        tofile arms GLb/GLk, nulls N, B0, N0                (afterok 1, 2)
#   4 merges_merge   CPU 64 cores, <= 6 h        merges + RESULTS.md                                  (afterok 3)
set -euo pipefail
export PATH=/opt/slurm/slurm-curr/bin:$PATH
H=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/merges_tofile_vs_crystfel
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/merges_tofile_vs_crystfel
mkdir -p "$O/logs"; chmod 700 "$O"
cd "$O"                         # the jobs run here: relative --output=logs/..., and indexamajig.<pid>/ in the cwd
z=$(sbatch --parsable --chdir="$O" "$H/r0016_cxi.sbatch")
g=$(sbatch --parsable --chdir="$O" --dependency=afterok:$z "$H/glint_solutions.sbatch")
c=$(sbatch --parsable --chdir="$O" --dependency=afterok:$z "$H/cf_index.sbatch")
t=$(sbatch --parsable --chdir="$O" --dependency=afterok:$g:$c "$H/tofile_index.sbatch")
m=$(sbatch --parsable --chdir="$O" --dependency=afterok:$t "$H/merges.sbatch")
echo "$(date -u +%FT%TZ) merges_r0016_cxi $z  merges_glint $g  merges_cf $c  merges_tofile $t  merges_merge $m" | tee -a "$O/jobs.txt"
