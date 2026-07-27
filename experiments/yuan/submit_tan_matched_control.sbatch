#!/bin/bash
#SBATCH --job-name=cbxd-tan-matched-ctrl
#SBATCH --partition=ampere
#SBATCH --account=lcls
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=0-04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Issue #59 (glint): matched control isolating curvature from data-quality in the TAN-vs-PTS
# comparison. PTS_MATCHED gets the same fresh reprs TAN's tangent bonus uses, scored without
# the tangent-alignment gate -- so PTS_MATCHED vs TAN isolates curvature specifically, holding
# the extra/cleaner/independent data constant.
cd "$SLURM_SUBMIT_DIR"
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
python run_tan_matched_control.py
