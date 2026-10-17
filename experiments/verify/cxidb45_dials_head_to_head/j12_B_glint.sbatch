#!/bin/bash
#SBATCH --job-name=j12_glint
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sdfampere042,sdfampere003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/glint_%j.out
# Step B: GLINT 95d51ce on the same lists (per-frame blind, consensus + rescue, target cell supplied),
# real and scrambled, so the GLINT rows of the block can be scored at the strict bar with DIALS's.
# Fallback account if prjdat21 is refused: -A lcls:default@ampere -q preemptable (the script skips arms already saved).
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/cxidb45_dials_head_to_head
G=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
export PYTHONPATH=$G
echo "host $(hostname)  commit $(cat $G/COMMIT)  python $(which python)"; nvidia-smi -L
sha256sum $G/experiments/prok_q.npz $O/inputs/prok_q_scrambled_s20260928.npz $J/glint_samepeaks.py
set -x
python -u $J/glint_samepeaks.py --npz $G/experiments/prok_q.npz \
  --npz-null $O/inputs/prok_q_scrambled_s20260928.npz --out $O/glint
