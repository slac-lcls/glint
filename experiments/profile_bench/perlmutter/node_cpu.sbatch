#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A m5149
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -J pb_node_cpu
#SBATCH -o /pscratch/sd/k/kmdalton/cxidb_33/logs/node_cpu_%j.out
# The rest of the workflow after node_gpu.sbatch, on one whole CPU node: the partialator merges,
# the CC1/2 cut, French-Wilson, R-free flags, torchref, anomalous peaks and the report.
export PYTHONNOUSERSITE=1
cd /pscratch/sd/k/kmdalton/glint/experiments/profile_bench
SM="/pscratch/sd/k/kmdalton/glint/.mamba/envs/glint/bin/snakemake -s Snakefile --configfile ${CONFIG:-config.yaml}"
$SM --unlock
$SM --cores 256 --resources mem_mb=460000 --keep-going --rerun-incomplete --rerun-triggers mtime
