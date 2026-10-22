#!/bin/bash
#SBATCH --job-name=glint256val
#SBATCH --partition=ampere
#SBATCH --account=lcls:prjdat21
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/sdf/home/s/smarches/glint_pr256_val/val_%j.out
# glint#256 rework validation. No `set -u`: psconda.sh reads unbound variables.
cd /sdf/home/s/smarches/glint_pr256_val
export GLINT_ROOT=/sdf/home/s/smarches/glint_pr256_val/glint
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
echo "node $(hostname); job $SLURM_JOB_ID; $(nvidia-smi -L | tr '\n' ' ')"
for ENV in ana-4.0.58-py3-minipytorch ana-4.0.59-py3-minipytorch; do
  conda activate $ENV
  export PYTHONPATH=$GLINT_ROOT
  echo; echo "################ $ENV: $(python -c 'import torch, mpi4py; print("torch", torch.__version__, "mpi4py", mpi4py.__version__)' 2>&1 | tail -1)"
  run() { echo; echo "=== $ENV :: $1"; shift; "$@" 2>&1 | grep -v -E 'OS_VER|^$'; }
  run "a. 2 tasks, one GPU each (expect RUN, different GPUs)"   srun --mpi=pmix -n 2 --gpus-per-task=1 python probe_placement.py
  run "b. 2 tasks on one GPU (expect STOP)"                      srun --mpi=pmix -n 2 --gpus=1 python probe_placement.py
  run "c. 2 tasks on one GPU, --share-gpu (expect RUN)"         srun --mpi=pmix -n 2 --gpus=1 python probe_placement.py --share-gpu
  run "d. 2 tasks, both see both GPUs (expect STOP)"            srun --mpi=pmix -n 2 --gpus=2 --gpu-bind=none python probe_placement.py
  run "e. GLINT_GPUS_PER_NODE=2 exported (expect STOP)"         env GLINT_GPUS_PER_NODE=2 srun --mpi=pmix -n 2 --gpus-per-task=1 python probe_placement.py
  run "f. timing: hide GPUs after importing mpi4py"             srun --mpi=pmix -n 1 --gpus=1 python timing_probe.py
  run "g. MPI smoke test, 2 ranks on one GPU (expect ALL PASS)" srun --mpi=pmix -n 2 --gpus=1 python $GLINT_ROOT/experiments/xtc_bridge/test_mpi_smoke.py
  conda deactivate
done
echo; echo "DONE"
