# sourced by the sbatch files here: the environment GLINT's own sbatch files use, the pinned tree, the output dir.
# No defaults: VERIFY_DIR, VERIFY_GLINT, VERIFY_OUT and VERIFY_INPUTS must be exported (../README.md).
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/stopping_rule_live_and_scrambled
G=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
: "${VERIFY_INPUTS:?set VERIFY_INPUTS to the inputs root}"
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
export PYTHONPATH=$G
export GLINT_ROOT=$G
export J5_OUT=${VERIFY_OUT:?set VERIFY_OUT to the output root}/stopping_rule_live_and_scrambled
export OMP_NUM_THREADS=1          # as experiments/gen_nbest.py runs index_blind_nbest
export PYTHONDONTWRITEBYTECODE=1  # no __pycache__ written into the pinned tree
umask 077
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  array ${SLURM_ARRAY_TASK_ID:-none}  commit $(cat $G/COMMIT)  env $CONDA_DEFAULT_ENV"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$J5_OUT"
