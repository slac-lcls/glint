#!/bin/bash
# Submits the stills_process head-to-head: 4 Slurm jobs. Needs VERIFY_GLINT, VERIFY_DIR, VERIFY_OUT exported.
set -eu
export PATH=/opt/slurm/slurm-curr/bin:$PATH
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/cxidb45_dials_head_to_head
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head
mkdir -p $O/{logs,inputs,dials,glint,score}
cd $O                           # relative --output=logs/... in the sbatch files
P=$(sbatch --parsable $J/stills_0_prep.sbatch)
A=$(sbatch --parsable --dependency=afterok:$P $J/stills_A_dials.sbatch)
B=$(sbatch --parsable --dependency=afterok:$P $J/stills_B_glint.sbatch)
C=$(sbatch --parsable --dependency=afterany:$A:$B $J/stills_C_score.sbatch)
echo "submitted: prep $P  dials(array) $A  glint $B  score $C" | tee -a $O/SUBMITTED.txt
