#!/bin/bash
#SBATCH --job-name=j4_index
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sdfampere042,sdfampere003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-3,5%2
#SBATCH --output=logs/j4_index_%A_%a.out
# Per-frame accuracy against peaks per frame. See README.md here.
# Tasks: 0 r0016 (refined geometry), 1 r0278 (0.058 m), 2 r0058, 3 cxidb45, 5 r0278z060 (r0278 at 0.060 m,
# the distance behind the FACTS key indexed_r0278; added 16 Oct). Task 4 (ctrl17, the driver check on
# the 120 cxidb-17 frames) is NOT in the default array; it runs only with `sbatch --array=4 j4_index.sbatch`.
# One A100 per task, no --exclusive (nothing here is a timing); %2 keeps this array to two of prjdat21@ampere's five GPUs. Not `set -e`: every step reports its exit.
TAGS=(r0016 r0278 r0058 cxidb45 ctrl17 r0278z060)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
# Needs VERIFY_GLINT, VERIFY_DIR, VERIFY_OUT, VERIFY_INPUTS exported (../README.md); submit from
# $VERIFY_OUT/per_frame_accuracy_vs_peaks after `mkdir -p logs`. Account, partition, --exclude: site settings.
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/per_frame_accuracy_vs_peaks
GL=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/per_frame_accuracy_vs_peaks
rc=0
note() { echo "exit $1"; [ "$1" -eq 0 ] || rc=1; }
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
export PYTHONPATH=$GL
umask 077
echo "$TAG  host $(hostname)  commit $(cat $GL/COMMIT)  env $CONDA_DEFAULT_ENV  $(date -Is)"
scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -E "SubmitLine|Command=|WorkDir|Account=|Partition=|TRES=" | sed 's/^ */slurm: /'
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
python -c "import torch, cupy; print('torch', torch.__version__, 'cupy', cupy.__version__, 'cuda', torch.cuda.is_available())"
case $TAG in
  r0016|r0278|r0058|r0278z060) echo "=== q lists  $(date +%T)"; python -u $J/j4_qlists.py $TAG $O; note $? ;;
esac
if [ $rc -eq 0 ]; then echo "=== score  $(date +%T)"; python -u $J/j4_score.py $TAG $O; note $?; fi
echo "done $(date -Is)  rc=$rc"
exit $rc
