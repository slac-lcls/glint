#!/bin/bash
#SBATCH --job-name=merges_r0016_cxi
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
# working directory: submit.sh submits from, and passes --chdir=, $VERIFY_OUT/merges_tofile_vs_crystfel (else indexamajig.<pid>/ lands in the submitter's cwd)
#SBATCH --constraint=OS_VER:8.10   # the OS of the dry run (site setting)
# Job 0 of 5 (CPU, psana1): rebuild btx's peak-finding .cxi of mfxx49820 r0016, events 0..R16_EVENTS-1,
# with btx's own PeakFinder and r0016 settings (r0016_cxi.py). ~20 GB (2,249 hits expected x 8.65 MB).
# Dry run (16 Oct): 30 events, 1 rank, 24 s wall, 9 hits, clen 240.573 mm = btx's own r0016 stream.
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/merges_tofile_vs_crystfel/config.sh"
btx_env || exit 1
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  start $(date -u +%FT%TZ)  env $CONDA_DEFAULT_ENV"
echo "btx $(git -c safe.directory=$BTX -C $BTX log -1 --format='%h %ci' 2>/dev/null)"
D=$OUT/r0016/cxi
if [ -s "$D/r0016.lst" ]; then echo "$D/r0016.lst exists: not rebuilt"; exit 0; fi
mkdir -p "$D"
export OMPI_MCA_btl=^openib                               # the openib CPC warnings of the dry run; tcp/vader only
run mpirun -n "$R16_RANKS" python "$TOOLS_HOME/r0016_cxi.py" --outdir "$D" --max-events "$R16_EVENTS"
rc=$?
echo "exit $rc  $(date -u +%FT%TZ)"
cat "$D/peakfinding.summary" 2>/dev/null; echo
[ "$rc" = 0 ] && [ -s "$D/r0016.lst" ] || { echo "r0016 .cxi not built"; exit 1; }
du -sh "$D"
