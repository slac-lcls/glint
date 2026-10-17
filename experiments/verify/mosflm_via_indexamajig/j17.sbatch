#!/bin/bash
#SBATCH -J j17_mosflm
#SBATCH -A lcls:prjdat21
#SBATCH -p milano
#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH -o logs/j17_%j.out
#
# MOSFLM through CrystFEL 0.12.0 indexamajig on the 120 cxidb-17 frames, the same nine arms as
# ../asdf_via_indexamajig. The earlier MOSFLM numbers (21% blind, 27% with the cell) have no surviving driver.
# CPU only. ipmosflm is CCP4 8.0's, the only CCP4 binary put on PATH. See README.md.
# Submit from $VERIFY_OUT/mosflm_via_indexamajig after `mkdir -p logs`. Account and partition: site settings.
set -o pipefail
J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/mosflm_via_indexamajig
IN=${VERIFY_INPUTS:?set VERIFY_INPUTS to the inputs root}/cxidb17_harness/inputs   # the 120-frame CXI inputs; not in this branch
SCORE=$VERIFY_DIR/common/score_arm.py                       # the scorer (md5-checked below)
P=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
C=/sdf/group/lcls/ds/tools/crystfel/0.12.0
OUT=${VERIFY_OUT:?set VERIFY_OUT to the output root}/mosflm_via_indexamajig/run_${SLURM_JOB_ID}
mkdir -p "$OUT" && chmod 700 "$OUT" && cd "$OUT" || exit 1
mkdir -p "$OUT/bin" && ln -sf /sdf/group/lcls/ds/tools/ccp4-8.0/bin/ipmosflm "$OUT/bin/ipmosflm"
export PATH=$OUT/bin:$C/bin:$PATH
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
set -u
export PYTHONPATH=$P PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
for f in frames120.txt frames120_scr.txt cf120.cxi cf120.geom cf120_events.lst cf120_qeff.txt \
         cf120scr.cxi cf120scr.geom cf120scr_events.lst cf120scr_qeff.txt; do
    [ -s "$IN/$f" ] || { echo "missing $IN/$f"; exit 1; }
done
cmp -s "$IN/frames120.txt" "$P/experiments/xgandalf/frames_cxidb_clean.txt" || { echo "frames120 differs from 95d51ce"; exit 1; }
# score_arm.py is the recorded scorer (md5 d4182cf73c292b4e4c22d907e85d260e) with its pinned-tree path read from
# VERIFY_GLINT; logic unchanged.
[ "$(md5sum < $SCORE | cut -d' ' -f1)" = 6d0e9408ecd6ebffd5ed4ddaafd64fd8 ] || { echo "score_arm.py changed"; exit 1; }
cp $J/README.md $J/j17.sbatch $J/j17_table.py "$OUT/"
{ echo "job $SLURM_JOB_ID  node $(hostname)  start $(date -Is)"; grep -m1 "model name" /proc/cpuinfo
  echo "glint commit: $(cat $P/COMMIT)"; echo "crystfel: $(indexamajig --version 2>&1 | head -1)"; echo "ipmosflm: $(readlink -f $OUT/bin/ipmosflm)"; } > PROVENANCE.txt

CELL=$P/experiments/xgandalf/lyso.cell
BASE="--peaks=cxi --indexing=mosflm --no-image-data --no-revalidate --integration=none -j 1 --profile"
OFF="--no-retry --no-refine --no-check-cell"
run() {   # run <arm> <cmd...>: one core each, all at once (the harness's pattern)
    local arm=$1; shift
    printf '%s\t%s\n' "$arm" "$*" >> ARGV.tsv
    srun --exact -n1 -c1 --cpu-bind=cores --mem=3G \
        /usr/bin/time -o "$arm.time" -f "wall_s %e user_s %U maxrss_kb %M" "$@" > "$arm.out" 2> "$arm.err" &
}
E=$IN/cf120_events.lst; G=$IN/cf120.geom; Es=$IN/cf120scr_events.lst; Gs=$IN/cf120scr.geom
run blind_on       indexamajig -i $E -g $G $BASE                                  -o $OUT/blind_on.stream       --temp-dir=$OUT
run blind_nopeaks  indexamajig -i $E -g $G $BASE --no-check-peaks                 -o $OUT/blind_nopeaks.stream  --temp-dir=$OUT
run blind_alloff   indexamajig -i $E -g $G $BASE $OFF --no-check-peaks            -o $OUT/blind_alloff.stream   --temp-dir=$OUT
run cell_on        indexamajig -i $E -g $G -p $CELL $BASE                         -o $OUT/cell_on.stream        --temp-dir=$OUT
run cell_nopeaks   indexamajig -i $E -g $G -p $CELL $BASE --no-check-peaks        -o $OUT/cell_nopeaks.stream   --temp-dir=$OUT
run cell_off       indexamajig -i $E -g $G -p $CELL $BASE $OFF                    -o $OUT/cell_off.stream       --temp-dir=$OUT
run cell_alloff    indexamajig -i $E -g $G -p $CELL $BASE $OFF --no-check-peaks   -o $OUT/cell_alloff.stream    --temp-dir=$OUT
run null_blind_alloff indexamajig -i $Es -g $Gs $BASE $OFF --no-check-peaks       -o $OUT/null_blind_alloff.stream --temp-dir=$OUT
run null_cell_alloff  indexamajig -i $Es -g $Gs -p $CELL $BASE $OFF --no-check-peaks -o $OUT/null_cell_alloff.stream --temp-dir=$OUT
wait
echo "all arms done $(date -Is)" >> PROVENANCE.txt
meta() { printf '{"argv": %s, "time": "%s", "node": "%s", "job": "%s", "glint_commit": "%s", "crystfel": "0.12.0"}' \
    "$(grep -P "^$1\t" ARGV.tsv | cut -f2- | python -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')" \
    "$(cat $1.time 2>/dev/null | tail -1)" "$(hostname)" "$SLURM_JOB_ID" "$(cat $P/COMMIT)"; }
for arm in blind_on blind_nopeaks blind_alloff cell_on cell_nopeaks cell_off cell_alloff; do
    python $SCORE --arm $arm --kind stream --result $arm.stream --profile $arm.out --events $E \
        --frames $IN/cf120_qeff.txt --out $OUT --meta "$(meta $arm)" || echo "score failed: $arm"
done
for arm in null_blind_alloff null_cell_alloff; do
    python $SCORE --arm $arm --kind stream --result $arm.stream --profile $arm.out --events $Es \
        --frames $IN/cf120scr_qeff.txt --out $OUT --meta "$(meta $arm)" || echo "score failed: $arm"
done
python $J/j17_table.py $OUT
echo "end $(date -Is)" >> PROVENANCE.txt
