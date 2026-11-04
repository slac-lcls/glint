#!/bin/bash
# The cheap interactive checks run on sdfiana027 before submission (CPU only, a few frames, < 2 min per step).
# usage: dryrun.sh [OUTDIR]   (default $OUT/dryrun2); transcript: OUTDIR/dryrun.log
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/merges_tofile_vs_crystfel/config.sh"
DR=${1:-$OUT/dryrun2}; mkdir -p "$DR"; cd "$DR" || exit 1
exec > >(tee "$DR/dryrun.log") 2>&1
OUT=$DR; export DRYRUN_OUT=$DR                               # commands.log of the dry run stays in the dry-run dir

echo "== 0 r0016 .cxi rebuild, btx env, 30 events on 1 rank"
btx_env || exit 1
python -c "import numpy, h5py; print('btx env', '$BTX_ENV', 'numpy', numpy.__version__, 'h5py', h5py.__version__)" 2>&1 | tail -1
time timeout 170 python "$TOOLS_HOME/r0016_cxi.py" --outdir "$DR/r16cxi30" --max-events 30 2>&1 \
    | grep -v "CPCs\|^---\|openib\|rdmacm\|udcm\|^\s*$\|Local host\|Local device\|component\|arning\|deprecat" | tail -3
cat "$DR/r16cxi30/peakfinding.summary"; echo

glint_env || exit 1
echo "== 1 imports (GLINT env)"
python -c "import glint, torch, numpy, h5py; print('torch', torch.__version__, 'numpy', numpy.__version__)"
$CFBIN/indexamajig --version | head -1; echo "GLINT commit $(cat $GLINT_TREE/COMMIT)"

for D in r194 r0016; do
    v=${D}_GEOM; GEOM=${!v}; v=${D}_CELLFILE; CELLFILE=${!v}; v=${D}_CELL; CELL=${!v}; v=${D}_LATT; LATT=${!v}; v=${D}_IDXX; IDXX=${!v}
    if [ $D = r0016 ]; then LIST=$DR/r16cxi30/r0016.lst; NF=9; else v=${D}_LIST; LIST=${!v}; NF=8; fi
    echo "== 2 $D: GLINT on $NF frames, known cell, CPU (stored-peak reader, gate, --tofile writer, lattice $LATT)"
    time timeout 170 python -m glint.glint_cli --images "$LIST" --geom "$GEOM" --peakfinder stored --gate strict \
        --lattice "$LATT" --cell $CELL -N $NF --device cpu --tofile $D.sol -o ${D}_orient.stream | grep "FINAL\|solution file"
    python "$TOOLS_HOME/merge_tools.py" cellcheck $D.sol "$CELL" | cut -c1-160
    python "$TOOLS_HOME/merge_tools.py" shuffle $D.sol ${D}_N1.sol 1
    python "$TOOLS_HOME/merge_tools.py" sollist $D.sol ${D}.lst
    echo "== 3 $D: indexamajig on those frames: CF, tofile, N (shuffled, checks on), B0 and N0 (no peak check)"
    for A in CF GLk N1 B0 N0; do
        case $A in
            CF)  X="$CF_INDEX" ;;
            GLk) X="$FILE_INDEX --fromfile-input-file=$D.sol" ;;
            N1)  X="$FILE_INDEX --fromfile-input-file=${D}_N1.sol $NOPEAKS" ;;
            B0)  X="$FILE_INDEX --fromfile-input-file=$D.sol $NOCHECK $NOPEAKS" ;;
            N0)  X="$FILE_INDEX --fromfile-input-file=${D}_N1.sol $NOCHECK $NOPEAKS" ;;
        esac
        aside ${D}_$A.stream
        $CFBIN/indexamajig -i $D.lst -o ${D}_$A.stream --peaks=cxi -g "$GEOM" -p "$CELLFILE" $X $INTEG $IDXX -j 4 \
            --wait-for-file=0 > ${D}_$A.log 2>&1
        echo "$A exit $?  $(grep '^Final:' ${D}_$A.log)"
        python "$TOOLS_HOME/merge_tools.py" keys ${D}_$A.stream ${D}_$A.keys > /dev/null
    done
    python "$TOOLS_HOME/merge_tools.py" common ${D}_CF.keys ${D}_GLk.keys ${D}_C.keys
    echo "$CELL" > ${D}_cell.txt
    python - <<EOF
import sys; sys.path.insert(0, "$TOOLS_HOME"); import merge_tools as t
print("misorientation GLk vs CF on common frames:", t.misorientation("${D}_GLk.stream", "${D}_CF.stream",
      t.read_keys("${D}_C.keys"), open("${D}_cell.txt").read().strip()))
EOF
done

echo "== 3b r0016: CrystFEL's own indexing of the 9 rebuilt frames, peak revalidation on (LUTE default) and off (btx)"
for R in on off; do
    X=""; [ $R = off ] && X="--no-revalidate"
    aside r16reval_$R.stream
    $CFBIN/indexamajig -i "$DR/r16cxi30/r0016.lst" -o r16reval_$R.stream --peaks=cxi -g "$r0016_GEOM" -p "$r0016_CELLFILE" \
        $CF_INDEX $INTEG $X -j 9 --wait-for-file=0 > r16reval_$R.log 2>&1
    echo "revalidate $R: $(grep '^Final:' r16reval_$R.log | cut -c1-110); peaks kept per frame: $(grep '^num_peaks' r16reval_$R.stream | awk '{printf "%s ", $3}')"
done

echo "== 4 merge driver against LUTE: round-5 r194 GLINT tofile stream (LUTE's own merge: -j 72)"
V5=/sdf/data/lcls/ds/prj/prjdat21/scratch/lute_sfx_val5/sfx_mfx101343025_r194_glint
for f in CC CCstar Rsplit; do
    $CFBIN/compare_hkl $V5/mfx101343025_r194_glint.hkl1 $V5/mfx101343025_r194_glint.hkl2 -y "$r194_SYM" \
        -p "$r194_CELLFILE" --fom=$f --nshells=10 --shell-file=lute_$f.dat 2>&1 | grep "Overall\|pairs accepted"
done
time "$TOOLS_HOME/merge_one.sh" $V5/mfx101343025_r194_glint_all.stream - 0 0 "$DR/merge_val5_r194" "$r194_SYM" "$r194_CELLFILE" $MERGE_J
python - <<EOF
import sys; sys.path.insert(0, "$TOOLS_HOME"); import merge_tools as t
r = t.merge_record("$DR/merge_val5_r194")
print({k: r[k] for k in ("partialator", "CC", "CCstar", "Rsplit", "check", "n_chunks_in")})
EOF
echo "== 5 subset tool: keys, a seeded first-crystal subset"
python "$TOOLS_HOME/merge_tools.py" keys $V5/mfx101343025_r194_glint_all.stream val5.keys
head -40 val5.keys > val5_40.keys
python "$TOOLS_HOME/merge_tools.py" subset $V5/mfx101343025_r194_glint_all.stream val5_40.keys 3 1 val5_40_s3.stream
grep -c "Begin crystal" val5_40_s3.stream; rm -f val5_40_s3.stream
echo "== done $(date -u +%FT%TZ); disk $(du -sh $DR | cut -f1)"
