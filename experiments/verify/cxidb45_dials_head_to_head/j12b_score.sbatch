#!/bin/bash
#SBATCH --job-name=j12b_score
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/j12b_score_%j.out
# Routes step C: every arm through the STRICT scorer, plus the regression checks against the two
# earlier rows and the paired tests against the stills_process arms (read back read-only, NOT re-run).
# ⚠ NOT `set -u` -- psconda.sh exits silently under it (verified 16 Oct).
set +u
rc=0
umask 077
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/cxidb45_dials_head_to_head
G=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head/dials_routes
REAL=$G/experiments/prok_q.npz
NULL=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head/inputs/prok_q_scrambled_s20260928.npz
mkdir -p $O/score
test "$(cat $G/COMMIT)" = 95d51ce || { echo "pinned tree is not 95d51ce"; exit 1; }

echo "host $(hostname)  job $SLURM_JOB_ID  $(date)"
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
export PYTHONPATH=$G
python -c "import numpy; print('numpy', numpy.__version__)" || { echo "GLINT env not active"; exit 1; }
sha256sum $J/score_j12b.py

set -x
J12OUT=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head
python -u $J/score_j12b.py --npz $REAL --npz-null $NULL \
    --dials $O/out \
    --j12-dials $J12OUT/dials --j12-glint $J12OUT/glint \
    --out $O/score
rc=$?
set +x
echo "done $(date)  rc=$rc"
exit $rc
