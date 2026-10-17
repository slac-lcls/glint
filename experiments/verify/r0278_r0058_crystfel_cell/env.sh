# Environment: GLINT 95d51ce pinned tree, CrystFEL 0.12.0, ana-4.0.58-py3-minipytorch.
# No defaults: VERIFY_DIR, VERIFY_GLINT and VERIFY_OUT must be exported (../README.md).
export VDIR=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r0278_r0058_crystfel_cell   # scripts (this folder)
export CXI_TOOLS=$VERIFY_DIR/common                           # qframes_to_cxi.py (md5 in j14.sbatch)
export GLINT=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}   # GLINT 95d51ce, plain tree with a COMMIT file, never modified
export J4OUT=${VERIFY_OUT:?set VERIFY_OUT to the output root}/per_frame_accuracy_vs_peaks   # the q lists of the per-frame check
export OUT=$VERIFY_OUT/r0278_r0058_crystfel_cell    # outputs
export CF=/sdf/group/lcls/ds/tools/crystfel/0.12.0             # CrystFEL 0.12.0
export PATH=$CF/bin:/opt/slurm/slurm-curr/bin:$PATH
export SCR_SEED=20260928                                       # scramble_qlist.py default seed
export SETS="r0278_real r0278_scr r0058_real r0058_scr"
set +u
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
export PYTHONPATH=$GLINT
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset GLINT_CONSENSUS_MIN_FRAC GLINT_CONSENSUS_MIN_LEAD GLINT_CONSENSUS_STABLE GLINT_DEVICE_MERGE
umask 077
