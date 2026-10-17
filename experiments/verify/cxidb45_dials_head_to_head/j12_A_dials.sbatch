#!/bin/bash
#SBATCH --job-name=j12_dials
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/dials_%A_%a.out
# Step A: dials.stills_process (DIALS 3.dev.1332-g3e144e7ea) on the 907 x 170-peak cxidb-45 lists.
# 5 arms x 4 chunks of 227 frames; one core per task, so t_proc is single-core time per frame.
# Resumable: a task re-run skips the frames already in its jsonl.
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/cxidb45_dials_head_to_head
G=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head
source /sdf/group/lcls/ds/tools/cctbx-psana2/build/setpaths.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARMS=(B0_blind_fft1d K1_kc_fft1d K2_kc_xfel K3_kc_rsgs N2_kc_xfel_scrambled)
NCH=4; N=907
T=${SLURM_ARRAY_TASK_ID:?array task}
ARM=${ARMS[$((T / NCH))]}; C=$((T % NCH))
case $ARM in
  N2_*) NPZ=$O/inputs/prok_q_scrambled_s20260928.npz; PH=K2_kc_xfel ;;
  *)    NPZ=$G/experiments/prok_q.npz;                 PH=$ARM ;;
esac
CH=$(( (N + NCH - 1) / NCH )); S=$((C * CH)); E=$((S + CH)); [ $E -gt $N ] && E=$N
echo "host $(hostname)  cpu $(lscpu | sed -n 's/^Model name:[ ]*//p')  arm $ARM  chunk $C  frames $S-$E"
echo "glint tree $(cat $G/COMMIT)  $(dials.version 2>&1 | head -1)  $(which dials.python)"
sha256sum $NPZ $J/phil/common.phil $J/phil/$PH.phil $J/dials_samepeaks.py
set -x
dials.python $J/dials_samepeaks.py --npz $NPZ --phil $J/phil/common.phil $J/phil/$PH.phil --arm $ARM \
  --start $S --stop $E --out $O/dials/$ARM/frames_$(printf %04d $S)_$(printf %04d $E).jsonl \
  --workdir $O/dials/$ARM/work_$C --rank $C
