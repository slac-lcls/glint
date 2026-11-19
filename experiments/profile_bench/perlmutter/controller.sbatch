#!/bin/bash
#SBATCH -q xfer
#SBATCH -C cron
#SBATCH -A m5149
#SBATCH -t 24:00:00
#SBATCH -J pb_controller
#SBATCH -o /pscratch/sd/k/kmdalton/cxidb_33/logs/controller_%j.out
# Snakemake controller for experiments/profile_bench: each step is its own slurm job (perlmutter profile).
# Usage, from experiments/profile_bench:
#   sbatch [--export=ALL,TARGET="--until index_integrate"] perlmutter/controller.sbatch
# The -o path and the paths below are this checkout's; edit them for another.
export PYTHONNOUSERSITE=1
cd /pscratch/sd/k/kmdalton/glint/experiments/profile_bench
/pscratch/sd/k/kmdalton/glint/.mamba/envs/glint/bin/snakemake -s Snakefile --configfile ${CONFIG:-config.yaml} --workflow-profile perlmutter ${TARGET}
