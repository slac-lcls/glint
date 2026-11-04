#!/bin/bash
#SBATCH --job-name=routes_dials
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/routes_%A_%a.out
# Routes step A: the nine DIALS arms behind the two earlier DIALS rows (1D-FFT blind, grid with the cell),
# re-run so the STRICT scorer can grade them. One array task per arm, one core each. See README.md.
# Submit from $VERIFY_OUT/cxidb45_dials_head_to_head/dials_routes after `mkdir -p logs`.
# Dry-run timings (sdfiana027, 16 Oct): bvs 0.32 s/frame -> ~5 min; indexer 1.03 s/frame -> ~16 min.
# 2 h is ~7x the slowest arm. Caps: 1 core, 8G, no GPU, 9 tasks (one array).
# ⚠ NOT `set -u`. The DIALS setpaths.sh and psconda.sh both reference unset variables and exit
# the shell silently under it -- that is what killed two jobs (41006454, 41011737) in 2 s each,
# twice. Verified here on 16 Oct: `set -u` + setpaths = silent death, `set +u` = rc 0.
set +u
rc=0
umask 077
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/cxidb45_dials_head_to_head
G=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
O=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head/dials_routes
REAL=$G/experiments/prok_q.npz
NULL=${VERIFY_OUT:?set VERIFY_OUT to the output root}/cxidb45_dials_head_to_head/inputs/prok_q_scrambled_s20260928.npz
mkdir -p $O/out $O/logs

ARMS=(V1_bvs_fft1d_blind V2_bvs_rsgs_kc V1n_bvs_fft1d_blind_null V2n_bvs_rsgs_kc_null \
      S1_idx_fft1d_rac S2_idx_fft1d_norac S3_idx_fft1d_rac_mc200 S4_idx_rsgs_kc_norac \
      S3n_idx_fft1d_rac_mc200_null)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

# the pinned tree must be the one every other check used
test "$(cat $G/COMMIT)" = 95d51ce || { echo "pinned tree is not 95d51ce"; exit 1; }
# inputs must be the stills_process run's, byte for byte (the comparison is only fair on identical peaks)
echo "$(sha256sum $REAL | cut -c1-64)  REAL" ; echo "$(sha256sum $NULL | cut -c1-64)  NULL"
test "$(sha256sum $REAL | cut -c1-64)" = e4ac4f5486ba44005c36ab1bd2dc27c6e1527788d7c69f550ff5be5141791a75 \
  || { echo "REAL npz is not the stills_process run's"; exit 1; }
test "$(sha256sum $NULL | cut -c1-64)" = 8e690c25d474ffc10b61aeb39d02d10707bdc325732f456166f19cd2396a4df6 \
  || { echo "NULL npz is not the stills_process run's"; exit 1; }

echo "host $(hostname)  arm $ARM  task $SLURM_ARRAY_TASK_ID  job $SLURM_JOB_ID  $(date)"
grep -m1 "model name" /proc/cpuinfo
source /sdf/group/lcls/ds/tools/cctbx-psana2/build/setpaths.sh >/dev/null 2>&1
command -v dials.python >/dev/null || { echo "dials.python not on PATH after setpaths"; exit 1; }
dials.version | head -2
sha256sum $J/routes_dials.py $J/dials_samepeaks.py

export OMP_NUM_THREADS=1       # the stills_process DIALS arms were single-core; keeps t_proc comparable
set -x
# the scorer globs <root>/<ARM>/*.jsonl (routes_score.py:86), so one DIRECTORY per arm
dials.python -u $J/routes_dials.py --arm $ARM --npz $REAL --npz-null $NULL \
    --out $O/out/${ARM}/frames.jsonl
rc=$?
set +x
echo "done $(date)  rc=$rc"
exit $rc
