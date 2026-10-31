#!/bin/bash
#SBATCH --job-name=stills_score
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/score_%j.out
# Step C: every arm through GLINT's own gate (same_lattice + matched_strict at 95d51ce, ProK reference).
# Runs after A and B whatever their exit state; results.md says complete=False if any arm misses frames.
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/cxidb45_dials_head_to_head
G=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
export PYTHONPATH=$G OMP_NUM_THREADS=1
echo "host $(hostname)  commit $(cat $G/COMMIT)"
set -x
python -u $J/stills_score.py --npz $G/experiments/prok_q.npz --npz-null $O/inputs/prok_q_scrambled_s20260928.npz \
  --dials $O/dials --glint $O/glint --out $O/score
