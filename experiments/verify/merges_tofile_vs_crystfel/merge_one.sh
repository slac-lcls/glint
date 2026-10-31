#!/bin/bash
# One merge, in its own directory (partialator writes pr-logs/ and partialator.params into the cwd):
#   subset of a stream -> partialator (LUTE MergePartialator settings) -> compare_hkl CC, CC*, Rsplit -> check_hkl
# usage: merge_one.sh STREAM KEYS|- SEED FIRST MDIR SYM CELLFILE NTHREADS [extra partialator flags...]
# Expects the GLINT conda env already active (numpy for merge_tools.py); CrystFEL is called by absolute path.
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/merges_tofile_vs_crystfel/config.sh"
STREAM=$1 KEYS=$2 SEED=$3 FIRST=$4 MDIR=$5 SYM=$6 CELLFILE=$7 NJ=$8; shift 8
mkdir -p "$MDIR" && cd "$MDIR" || exit 1
# a failed earlier attempt can leave in.stream behind; remove it so a failed subset cannot be merged in its place
rm -f in.stream in.stream.missing
python "$TOOLS_HOME/merge_tools.py" subset "$STREAM" "$KEYS" "$SEED" "$FIRST" in.stream > subset.log 2>&1 \
    || { echo "subset failed: $MDIR (see subset.log)" >&2; exit 1; }
# PAIRED-SPLIT GUARD. merge_tools.py:168 writes in.stream.missing listing keys of the common
# set C that this arm does not have. For a `common*` merge that is fatal, not cosmetic: partialator
# splits into alternating halves by chunk ORDER, so one absent key shifts every later frame to the
# other half and the arms are no longer PAIRED -- the paired verdict lines in RESULTS.md would then
# compare two different splits. Fail the merge instead of producing a number nobody can trust.
# (The alternative closer to the pre-registration is a --custom-split file from the same seeded
# assignment, which 0.10.2 supports; this guard is the conservative version of the same requirement.)
# 17 Oct: the hard failure is replaced with --custom-split. The guard above was right that an
# ORDER-based split unpairs the arms when
# a key is absent; --custom-split assigns halves BY KEY from the COMMON set with the same seed for
# every arm, so a missing key costs only itself. The guard is kept as a WARNING so the shortfall
# stays visible in the log, and the paired comparison moves from merged.hkl1/.hkl2 to
# merged-halfA.hkl/merged-halfB.hkl.
SPLITARG=""
case "$(basename "$MDIR")" in
  common*)
    if [ -s in.stream.missing ]; then
        echo "PAIRING NOTE: $(wc -l < in.stream.missing) key(s) of the common set absent from this" \
             "arm; --custom-split keeps the halves paired by key. See in.stream.missing." >&2
    fi
    if [ "$KEYS" != "-" ]; then
        python "$TOOLS_HOME/merge_tools.py" splitfile "$KEYS" "$SEED" split.txt > splitfile.log 2>&1 || exit 2
        SPLITARG="--custom-split=split.txt"
    fi
    ;;
esac
grep -c "^----- Begin chunk" in.stream > n_in.txt
run "$CFBIN/partialator" -i in.stream -o merged.hkl -y "$SYM" $MERGE $SPLITARG "$@" -j "$NJ" > partialator.log 2>&1 \
    || { echo "partialator failed: $MDIR (see partialator.log)" >&2; exit 3; }
# with --custom-split the KEY-paired halves are merged-halfA/halfB.hkl; hkl1/hkl2 remain the
# order-based split and are left in place for continuity with the 16 Oct run. The order-based halves
# are used ONLY when no custom split was asked for: falling back to them would report unpaired halves
# as paired.
if [ -n "$SPLITARG" ]; then
    [ -s merged-halfA.hkl ] && [ -s merged-halfB.hkl ] \
        || { echo "custom split requested but merged-halfA/B.hkl missing or empty: $MDIR" >&2; exit 4; }
    H1=merged-halfA.hkl; H2=merged-halfB.hkl; echo "paired on: key-paired halves (--custom-split)"
else
    H1=merged.hkl1; H2=merged.hkl2; echo "paired on: order-based halves (no custom split)"
fi
# these merges are whole arms, never a deliberately tiny subset, so a failed statistics command is an error: exit
# before the cleanup, so the caller's xargs guard sees it and no split is averaged without its numbers
stats_failed() { echo "$1 failed: $MDIR (see $2)" >&2; exit 5; }
for fom in CC CCstar Rsplit; do
    run "$CFBIN/compare_hkl" "$H1" "$H2" -y "$SYM" -p "$CELLFILE" --fom=$fom --nshells=10 \
        --shell-file=shells_$fom.dat > compare_$fom.log 2>&1 || stats_failed compare_hkl compare_$fom.log
done
run "$CFBIN/check_hkl" merged.hkl -y "$SYM" -p "$CELLFILE" --nshells=10 --shell-file=shells_check.dat > check.log 2>&1 \
    || stats_failed check_hkl check.log
rm -rf in.stream pr-logs                                   # the subset is regenerable from the arm stream + keys + seed
