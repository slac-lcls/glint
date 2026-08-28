#!/bin/bash
#SBATCH --job-name=cbxd-grid-matched-ctrl
#SBATCH --partition=ampere
#SBATCH --account=lcls:prjmaiqmag01
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Issue #59 matched control extended to deliverable 3's phase diagram (issue #11,
# run_grid_arms.py / results_grid.jsonl): adds PTS_MATCHED at all 39 already-explored
# (na, noise) grid cells x 20 crystals, to check whether "TAN pulls away from PTS as NA grows"
# is genuine curvature or (partly) the same data-quality confound issue #59 found in the flat
# four-arm comparison, and whether it grows with NA the way streak length does.
cd "$SLURM_SUBMIT_DIR"
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
python run_grid_matched_control.py
