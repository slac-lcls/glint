#!/bin/bash
#SBATCH --job-name=merges_cf
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.log
# working directory: submit.sh submits from, and passes --chdir=, $VERIFY_OUT/merges_tofile_vs_crystfel (else indexamajig.<pid>/ lands in the submitter's cwd)
#SBATCH --constraint=OS_VER:8.10   # the OS of the dry run (site setting)
# Job 2 of 5 (CPU, afterok job 0; runs beside job 1): arm CF, CrystFEL's own indexing of every frame of the
# same .cxi with the LUTE IndexCrystFEL argv of the CrystFEL functional test (xgandalf, --multi, rings-nocen 3,4,6).
# Extra flags per dataset (${D}_IDXX: --no-revalidate on r0016, as btx ran it). CFnr = CF + --no-revalidate on
# r51/r194 (sensitivity only). LUTE timing for scale: r51 round 4, 8,770 frames, 72 threads, ~8 min.
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/merges_tofile_vs_crystfel/config.sh"
glint_env || exit 1                                         # numpy for merge_tools.py only; CrystFEL by absolute path
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  start $(date -u +%FT%TZ)"
$CFBIN/indexamajig --version | head -1

for D in $DATASETS; do
    v=${D}_LIST; LIST=${!v}; v=${D}_GEOM; GEOM=${!v}; v=${D}_CELLFILE; CELLFILE=${!v}; v=${D}_IDXX; IDXX=${!v}
    [ -s "$LIST" ] || { echo "== $D: no input list $LIST; skipping"; continue; }
    mkdir -p "$OUT/$D"
    echo "== $D  $(date -u +%FT%TZ)"
    aside "$OUT/$D/CF.stream"; SECONDS=0
    run $CFBIN/indexamajig -i "$LIST" -o "$OUT/$D/CF.stream" --peaks=cxi -g "$GEOM" -p "$CELLFILE" \
        $CF_INDEX $INTEG $IDXX -j 64 --wait-for-file=0 > "$OUT/$D/idx_CF.log" 2>&1
    echo "CF exit $?  ${SECONDS} s"; grep "^Final:" "$OUT/$D/idx_CF.log"
    python "$TOOLS_HOME/merge_tools.py" keys "$OUT/$D/CF.stream" "$OUT/$D/CF.keys"
    case " $CFNR_DATASETS " in *" $D "*)                  # sensitivity arm: the same + --no-revalidate
        aside "$OUT/$D/CFnr.stream"; SECONDS=0
        run $CFBIN/indexamajig -i "$LIST" -o "$OUT/$D/CFnr.stream" --peaks=cxi -g "$GEOM" -p "$CELLFILE" \
            $CF_INDEX $INTEG --no-revalidate -j 64 --wait-for-file=0 > "$OUT/$D/idx_CFnr.log" 2>&1
        echo "CFnr exit $?  ${SECONDS} s"; grep "^Final:" "$OUT/$D/idx_CFnr.log"
        python "$TOOLS_HOME/merge_tools.py" keys "$OUT/$D/CFnr.stream" "$OUT/$D/CFnr.keys" ;;
    esac
done
echo "end $(date -u +%FT%TZ)"
