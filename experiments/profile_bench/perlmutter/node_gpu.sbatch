#!/bin/bash
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -A m5149_g
#SBATCH -N 1
#SBATCH --gpus-per-node 4
#SBATCH -t 06:00:00
#SBATCH -J pb_node_gpu
#SBATCH -o /pscratch/sd/k/kmdalton/cxidb_33/logs/node_gpu_%j.out
# Everything per run, on one whole GPU node instead of one slurm job per rule: stack, peak finding,
# GLINT indexing + integration on the node's 4 GPUs (perlmutter/with_gpu.sh), CrystFEL integration.
# Then node_cpu.sbatch merges and refines. From experiments/profile_bench:
#   g=$(sbatch --parsable --export=ALL,CONFIG=config_61.yaml perlmutter/node_gpu.sbatch)
#   sbatch --dependency=afterok:$g --export=ALL,CONFIG=config_61.yaml perlmutter/node_cpu.sbatch
export PYTHONNOUSERSITE=1
cd /pscratch/sd/k/kmdalton/glint/experiments/profile_bench
SM="/pscratch/sd/k/kmdalton/glint/.mamba/envs/glint/bin/snakemake -s Snakefile --configfile ${CONFIG:-config.yaml}"
$SM --unlock
$SM --cores 64 --resources gpu=4 mem_mb=220000 --keep-going --rerun-incomplete --rerun-triggers mtime --until crystfel_integrate
