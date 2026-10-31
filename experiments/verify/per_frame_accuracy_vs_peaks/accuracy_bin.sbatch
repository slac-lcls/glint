#!/bin/bash
#SBATCH --job-name=accuracy_bin
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/accuracy_bin_%j.out
# Step 3: bin the per-frame records of accuracy_index.sbatch. Submit with --dependency=afterany:<accuracy_index id>
# (afterany, so that one failed task still leaves the other runs' tables).
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/per_frame_accuracy_vs_peaks
GL=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/per_frame_accuracy_vs_peaks
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
export PYTHONPATH=$GL
umask 077
echo "bin  host $(hostname)  commit $(cat $GL/COMMIT)  $(date -Is)"
scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -E "SubmitLine|Command=|Account=|Partition=|Dependency" | sed 's/^ */slurm: /'
python -u $J/accuracy_bin.py $O
rc=$?; echo "exit $rc"; exit $rc
