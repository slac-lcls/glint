#!/bin/bash
# Step 1: q-frames from xtc (r0278, r0058, refined r0016), every frame's cells, whole-run cells,
# and the live arrival-order decision of the shipped StreamDriver. One A100, ~1 h expected.
#SBATCH --job-name=stop_prep
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sdfampere042,sdfampere003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/stopping_rule_live_and_scrambled/stop_env.sh"
python -u $J/stop_prep.py all
