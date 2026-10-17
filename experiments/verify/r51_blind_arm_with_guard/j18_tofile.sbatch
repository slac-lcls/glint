#!/bin/bash
#SBATCH --job-name=j18_tofile
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
# working directory: submit from $VERIFY_OUT/r51_blind_arm_with_guard, or pass --chdir= to it (else indexamajig.<pid>/ lands in the submitter's cwd)
#SBATCH --constraint=OS_VER:8.10   # the OS of the dry run (site setting)
# Job 2 of 3 (CPU, afterok job 1): every arm that starts from a GLINT solution file.
#   GLb, GLk  indexamajig --indexing=file, the LUTE tofile argv (CrystFEL refines, checks cell and peaks, integrates)
#   N<s>      GLb's orientations shuffled among its frames (seed s), same argv: frame-level null
#             (how many wrong orientations CrystFEL's checks let through)
#   B0        GLb's orientations on the common frames C = CF & GLb, without peak check, refinement, retry
#   N0_<s>    the shuffled orientations on the same C frames, same flags as B0: merge-level null
# Integration settings and the dataset's extra flags (${D}_IDXX) are identical in every arm and to arm CF.
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r51_blind_arm_with_guard/config.sh"
glint_env || exit 1
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  start $(date -u +%FT%TZ)"

idx_file() {   # LIST SOL OUTSTREAM LOG [extra flags...]
    local lst=$1 sol=$2 out=$3 log=$4; shift 4
    aside "$out"; SECONDS=0
    run $CFBIN/indexamajig -i "$lst" -o "$out" --peaks=cxi -g "$GEOM" -p "$CELLFILE" \
        $FILE_INDEX --fromfile-input-file="$sol" $INTEG $IDXX "$@" -j 32 --wait-for-file=0 > "$log" 2>&1
    local rc=$?
    echo "$(basename "$out") exit $rc  ${SECONDS} s  $(grep '^Final:' "$log")"
    python "$J2_HOME/j2_tools.py" keys "$out" "${out%.stream}.keys"
}

for D in $DATASETS; do
    v=${D}_GEOM; GEOM=${!v}; v=${D}_CELLFILE; CELLFILE=${!v}; v=${D}_CELL; CELL=${!v}; v=${D}_IDXX; IDXX=${!v}
    DD=$OUT/$D
    echo "== $D  $(date -u +%FT%TZ)"
    [ -d "$DD" ] || { echo "no $DD; skipping"; continue; }
    echo "$CELL" > "$DD/cell.txt"
    for A in GLb GLk; do
        [ -s "$DD/$A.sol" ] || { echo "no $A.sol for $D: GLINT wrote no solutions"; continue; }
        python "$J2_HOME/j2_tools.py" sollist "$DD/$A.sol" "$DD/$A.lst"
        idx_file "$DD/$A.lst" "$DD/$A.sol" "$DD/$A.stream" "$DD/idx_$A.log"
    done
    [ -s "$DD/GLb.sol" ] || continue
    for s in $NULL_SEEDS; do
        python "$J2_HOME/j2_tools.py" shuffle "$DD/GLb.sol" "$DD/N$s.sol" "$s"
        idx_file "$DD/GLb.lst" "$DD/N$s.sol" "$DD/N$s.stream" "$DD/idx_N$s.log" $NOPEAKS
    done
    [ -s "$DD/CF.keys" ] || { echo "no CF.keys for $D: no common set, no B0/N0"; continue; }
    python "$J2_HOME/j2_tools.py" common "$DD/CF.keys" "$DD/GLb.keys" "$DD/C.keys"
    [ -s "$DD/GLk.keys" ] && python "$J2_HOME/j2_tools.py" common "$DD/CF.keys" "$DD/GLk.keys" "$DD/Ck.keys"
    [ -s "$DD/C.keys" ] || { echo "$D: CF and GLb share no frame; no B0/N0"; continue; }
    cp "$DD/C.keys" "$DD/C.lst"                              # "<image> //<event>" is also indexamajig's list format
    idx_file "$DD/C.lst" "$DD/GLb.sol" "$DD/B0.stream" "$DD/idx_B0.log" $NOCHECK $NOPEAKS
    for s in $NULL_SEEDS; do
        idx_file "$DD/C.lst" "$DD/N$s.sol" "$DD/N0_$s.stream" "$DD/idx_N0_$s.log" $NOCHECK $NOPEAKS
    done
done
echo "end $(date -u +%FT%TZ)"
du -sh "$OUT"
