#!/bin/bash
# One merge, in its own directory (partialator writes pr-logs/, partialator.params and pgraph files into the cwd).
#   subset of a stream (+ optional intensity permutation) -> partialator or process_hkl -> compare_hkl CC, CC*, Rsplit
#   and check_hkl, in compare_hkl's own range ("auto") and, when DRANGE="dlow dhigh" (A) is set, in that fixed range.
# usage: sweep_merge_one.sh STREAM TAG FRAC SEED PERM MERGER MDIR [NTHREADS]     (MERGER = partialator | process_hkl)
# Expects the GLINT conda env (numpy for sweep_tools.py) and CMDLOG; CrystFEL is called by absolute path.
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r51_merge_vs_frames/config.sh"
STREAM=$1 TAG=$2 FRAC=$3 SEED=$4 PERM=$5 MERGER=$6 MDIR=$7 NJ=${8:-$NJ}
mkdir -p "$MDIR" && cd "$MDIR" || exit 1
[ -s record.json ] && exit 0                                   # restartable: a finished merge is not redone
python -B "$TOOLS_HOME/sweep_tools.py" subset "$STREAM" "$FRAC" "$SEED" "$PERM" in.stream > subset.log 2>&1 \
    || { echo "subset failed: $MDIR" >&2; exit 1; }
# a failed merger stops here, before record.json exists, so a rerun of the sweep retries this merge
merge_failed() { echo "$MERGER failed: $MDIR (see $1)" >&2; exit 2; }
if [ "$MERGER" = partialator ]; then
    run "$CFBIN/partialator" -i in.stream -o merged.hkl -y "$SYM" $MERGE -j "$NJ" > partialator.log 2>&1 || merge_failed partialator.log
else
    run "$CFBIN/process_hkl" -i in.stream -o merged.hkl  -y "$SYM" $MC              > process_hkl.log  2>&1 || merge_failed process_hkl.log
    run "$CFBIN/process_hkl" -i in.stream -o merged.hkl1 -y "$SYM" $MC --odd-only  > process_hkl1.log 2>&1 || merge_failed process_hkl1.log
    run "$CFBIN/process_hkl" -i in.stream -o merged.hkl2 -y "$SYM" $MC --even-only > process_hkl2.log 2>&1 || merge_failed process_hkl2.log
fi
# A statistics command that RAN and returned non-zero (e.g. too few reflections in a small subset) is a result: it
# is listed in stats_failed.txt, which record.json carries and the collector reports. One that could not run or
# was killed (exit >= 126: not executable, not found, signal) is an error: stop before record.json, so a rerun
# retries this merge.
rm -f stats_failed.txt
stat_rc() {                                                    # stat_rc RC LOG
    [ "$1" -eq 0 ] && return 0
    [ "$1" -ge 126 ] && { echo "statistics command did not run (exit $1): $MDIR/$2" >&2; exit 3; }
    echo "$2 $1" >> stats_failed.txt
}
stats() {                                                      # stats DIR [range flags]
    local W=$1; shift; mkdir -p "$W"
    for fom in CC CCstar Rsplit; do
        run "$CFBIN/compare_hkl" merged.hkl1 merged.hkl2 -y "$SYM" -p "$CELLFILE" --fom=$fom --nshells=$NSHELLS "$@" \
            --shell-file=$W/shells_$fom.dat > $W/compare_$fom.log 2>&1
        stat_rc $? $W/compare_$fom.log
    done
    run "$CFBIN/check_hkl" merged.hkl -y "$SYM" -p "$CELLFILE" --nshells=$NSHELLS "$@" \
        --shell-file=$W/shells_check.dat > $W/check.log 2>&1
    stat_rc $? $W/check.log
}
stats auto
if [ -n "${DRANGE:-}" ]; then
    read -r DLO DHI <<< "$DRANGE"
    stats fixed --lowres=$DLO --highres=$DHI
fi
python -B "$TOOLS_HOME/sweep_tools.py" record "$MDIR" "$TAG" "$FRAC" "$SEED" "$PERM" "$MERGER" > record.log 2>&1 \
    || { echo "record failed: $MDIR (see record.log)" >&2; exit 4; }
rm -rf in.stream pr-logs pgraph-iter*.dat                       # the subset is regenerable from STREAM + FRAC + SEED + PERM
