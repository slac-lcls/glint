#!/bin/bash
# Sustained frames/s and strict-bar yield of the shipped StreamDriver on
# mfxx49820 r0016 (all 37,710 events, refined btx geometry), one A100. README.md in this directory.
# Two identical array tasks: the second is the repeat that gives the run-to-run spread.
# Walltime 5 h (the 30-75 min estimate is unmeasured); a scoring failure does not lose the timing;
# the null step runs only if its input exists.
# Needs VERIFY_GLINT, VERIFY_DIR, VERIFY_OUT, VERIFY_INPUTS exported (../README.md). Submit from
# $VERIFY_OUT/r0016_sustained_rate after `mkdir -p logs`. Account, partition and --exclude are site settings.
#SBATCH --job-name=j1_stream
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=sdfampere042,sdfampere003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/j1_%A_%a.out
set -euo pipefail
umask 077

G=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}   # pinned GLINT tree, COMMIT = 95d51ce, read-only
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r0016_sustained_rate   # this check's scripts
OUT=${VERIFY_OUT:?set VERIFY_OUT to the output root}/r0016_sustained_rate/out
E=/sdf/data/lcls/ds/mfx/mfxx49820
CALIB=${VERIFY_INPUTS:?set VERIFY_INPUTS to the inputs root}/glint_gt_mfxx49820/calib   # 0-end.data override of the validated --geom run (read-only)
GEOM=$E/results/btx/geom/r0016.geom                   # btx-refined geometry (read-only)
CELL=79.327,79.461,38.406,90.111,89.981,90.083        # btx sample2.cell: strict-bar reference, known arm's cell
TAG=a${SLURM_ARRAY_TASK_ID:-0}

set +u   # conda activation scripts read unset variables (a job died on ADDR2LINE under set -u)
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
set -u
export PYTHONPATH=$G
export PATH=/opt/slurm/slurm-curr/bin:$PATH
mkdir -p "$OUT"

echo "== provenance"
date -Is; hostname; echo "COMMIT $(cat $G/COMMIT)"; echo "CONDA $CONDA_DEFAULT_ENV"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm --format=csv
echo "jobs sharing this node:"; squeue -h -w "$(hostname -s)" -o "%i %u %b %C %m" || true
md5sum "$GEOM" "$J/j1_stream.py" "$J/j1_summary.py" "$G/glint/stream_driver.py" "$G/experiments/record_stream_replay.py"
ls -l "$CALIB"/*/*/*

echo "== 1. pixel stream: every event of r0016 through each arm's StreamDriver"
python -u "$J/j1_stream.py" --glint "$G" --exp mfxx49820 --run 16 --det MfxEndstation.0:Epix10ka2M.0 \
  --calib-dir "$CALIB" --geom "$GEOM" --zdist 0.102973 --wavelength 1.290757 --cell "$CELL" \
  --arms defaults,events_off,wrescue,wrescue_pix,known --rescue-pixels 256 \
  --chunk 1024 --warm-events 1024 --check-calib 8 --calib gpu --window 2048 \
  --out-dir "$OUT" --tag "$TAG"

echo "== 2. null: the defaults arm's q replayed index-only, then the same q azimuth-scrambled"
echo "   (the recorder's own git provenance is null: the pinned tree has no .git; its commit is COMMIT above)"
if [ -s "$OUT/j1_${TAG}_q.npz" ] && [ -s "$OUT/j1_${TAG}_qscr.npz" ]; then
python -u "$G/experiments/record_stream_replay.py" --cupy \
  --input real="$OUT/j1_${TAG}_q.npz" --input scr="$OUT/j1_${TAG}_qscr.npz" \
  --ref real="$CELL" --ref scr="$CELL" --B 64 --min-inliers 0 \
  --out "$OUT/j1_${TAG}_null.json"
else
  echo "   SKIPPED: no q files (the defaults arm was not scored; see the WARNING lines above)"
fi

echo "== 3. summary"
python -u "$J/j1_summary.py" "$OUT"
date -Is
