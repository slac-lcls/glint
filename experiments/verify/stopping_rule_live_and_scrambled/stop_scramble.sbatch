#!/bin/bash
# Arm (a): 10,000 groups of 100 scrambled frames (3,333 / 3,333 / 3,334 from r0278 / r0058 / refined
# r0016) + 20 whole-run scrambled groups per run, through the live StreamDriver. 4 array tasks x one A100,
# two worker processes each; each task stops cleanly at 5.4 h and a rerun resumes from its checkpoints.
#SBATCH --job-name=stop_scramble
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sdfampere042,sdfampere003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.log
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/stopping_rule_live_and_scrambled/stop_env.sh"
python -u $J/stop_scramble.py $SLURM_ARRAY_TASK_ID 4 --G 100 --groups 3333,3333,3334 --long 20 --hours 5.4 --procs 2
