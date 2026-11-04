#!/bin/bash
#SBATCH --job-name=stills_prep
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/prep_%j.out
# Step 0: the azimuth-scrambled null copy of the 907 cxidb-45 lists (shipped scramble_qlist.scramble_frames).
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/cxidb45_dials_head_to_head
G=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
export PYTHONPATH=$G OMP_NUM_THREADS=1
echo "host $(hostname)  commit $(cat $G/COMMIT)  python $(which python)"
set -x
python -u $J/prep_null.py --tree $G --npz $G/experiments/prok_q.npz --out $O/inputs/prok_q_scrambled_s20260928.npz --seed 20260928
