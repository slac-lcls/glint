#!/bin/bash
#SBATCH --job-name=cbxd-tan-pooling-v2
#SBATCH --partition=ampere
#SBATCH --account=lcls
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=0-04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Issue #12 (glint): scale TAN+pooling from the N=4 pilot (results_tan_pooling.jsonl) to N=20,
# full noise ladder isn't needed per the ask -- same NOISE=1e-3 as the pilot, just more crystals
# -- to check whether the crystal-3 curvature-breaks-tie finding generalizes. Independent of the
# PTS+consensus retest (submit_consensus_retest.sbatch): different mechanism, different script,
# no shared intermediate output -- safe to run in parallel.
cd "$SLURM_SUBMIT_DIR"
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
python run_tan_pooling_test.py 5000000 20
