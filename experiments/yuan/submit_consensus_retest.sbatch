#!/bin/bash
#SBATCH --job-name=cbxd-consensus-v2
#SBATCH --partition=ampere
#SBATCH --account=lcls
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=0-08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Issue #12 (glint): PTS+consensus retest at N=20 with majority-scaled min_support.
# Submitted directly via sbatch (not the OnDemand GUI) so the job survives independent
# of any interactive session -- still runs under whatever QOS `--account` defaults to
# (preemptable for lcls:default on ampere; script is checkpointed/resumable either way).
cd "$SLURM_SUBMIT_DIR"
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
python run_orientation_consensus_test.py 5000000 20
