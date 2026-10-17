#!/bin/bash
# Arm (a), CPU ALTERNATIVE to j5_scramble.sbatch -- submit ONE of the two, never both.
# Same script, same groups, same seeds; index_blind_nbest runs on torch's CPU device (glint_fast.DEV =
# "cpu" on a node without a GPU). The recorded run used the GPU file (j5_scramble.sbatch).
# Measured on sdfiana027 (16 Oct, 1 thread): 353 ms at 35 peaks, 560 ms at 65, 3.2-5.4 s at
# 164-243 peaks; 100 workers per node. Part F of j5_replay.py measures how often CPU and A100 cells
# differ on the same real frames.
#SBATCH --job-name=j5_scramble_cpu
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.log
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/stopping_rule_live_and_scrambled/j5_env.sh"
python -u $J/j5_scramble.py $SLURM_ARRAY_TASK_ID 4 --G 100 --groups 3333,3333,3334 --long 20 --hours 5.4 --procs 100
