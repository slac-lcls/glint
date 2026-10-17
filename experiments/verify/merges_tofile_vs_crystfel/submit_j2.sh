#!/bin/bash
# Submits the five Slurm jobs of this check with their dependencies, from an S3DF login node.
# Needs VERIFY_GLINT, VERIFY_DIR and VERIFY_OUT exported (../README.md).
#   0 j2_r16cxi  CPU 16 ranks, <= 1 h       rebuild btx's r0016 .cxi (events 0..5,999), ~20 GB
#   1 j2_glint   GPU, 1 A100, <= 4 h        GLINT solution files (blind GLb, known-cell GLk)   (afterok 0)
#   2 j2_cf      CPU 64 cores, <= 3 h        CrystFEL's own indexing (arm CF)                    (afterok 0)
#   3 j2_tofile  CPU 32 cores, <= 4 h        tofile arms GLb/GLk, nulls N, B0, N0                (afterok 1, 2)
#   4 j2_merge   CPU 64 cores, <= 6 h        merges + RESULTS.md                                  (afterok 3)
set -euo pipefail
export PATH=/opt/slurm/slurm-curr/bin:$PATH
H=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/merges_tofile_vs_crystfel
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/merges_tofile_vs_crystfel
mkdir -p "$O/logs"; chmod 700 "$O"
cd "$O"                         # the jobs run here: relative --output=logs/..., and indexamajig.<pid>/ in the cwd
z=$(sbatch --parsable --chdir="$O" "$H/j2_r0016_cxi.sbatch")
g=$(sbatch --parsable --chdir="$O" --dependency=afterok:$z "$H/j2_glint.sbatch")
c=$(sbatch --parsable --chdir="$O" --dependency=afterok:$z "$H/j2_cf.sbatch")
t=$(sbatch --parsable --chdir="$O" --dependency=afterok:$g:$c "$H/j2_tofile.sbatch")
m=$(sbatch --parsable --chdir="$O" --dependency=afterok:$t "$H/j2_merge.sbatch")
echo "$(date -u +%FT%TZ) j2_r16cxi $z  j2_glint $g  j2_cf $c  j2_tofile $t  j2_merge $m" | tee -a "$O/jobs.txt"
