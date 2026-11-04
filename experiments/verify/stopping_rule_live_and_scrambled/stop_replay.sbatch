#!/bin/bash
# Analysis: replays on the cached cells (arrival-order starts, 400 random orders x seeds, the 480 set)
# and the scrambled-group tables. CPU only.
#SBATCH --job-name=stop_replay
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.log
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/stopping_rule_live_and_scrambled/stop_env.sh"
python -u $J/stop_replay.py --procs 32 --orders 400
