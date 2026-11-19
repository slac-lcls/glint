#!/bin/bash
# Run "$@" on one GPU of this node. A no-op when the job sees one GPU (the per-rule slurm jobs of the
# perlmutter profile); on a whole node (node_gpu.sbatch, local snakemake with --resources gpu=4) it
# takes the first GPU whose lock file it can flock, so concurrent rules land on different GPUs.
n=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$n" -le 1 ] || { [ -n "$CUDA_VISIBLE_DEVICES" ] && [[ "$CUDA_VISIBLE_DEVICES" != *,* ]]; }; then
    exec "$@"
fi
while true; do
    for i in $(seq 0 $((n - 1))); do
        exec {fd}>"${TMPDIR:-/tmp}/glint_gpu_${SLURM_JOB_ID:-$$}.$i.lock"
        if flock -n $fd; then
            CUDA_VISIBLE_DEVICES=$i "$@"
            exit $?
        fi
        exec {fd}>&-
    done
    sleep 5
done
