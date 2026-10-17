#!/bin/bash
# Dry run on the interactive node: 30 frames per set, -j 4, a temp OUT. Nothing is submitted.
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r0278_r0058_crystfel_cell/env.sh"
set -eo pipefail
export OUT=$(mktemp -d "$VERIFY_OUT/crystfel_cell_dry.XXXX")
mkdir -p $OUT/inputs $OUT/cxi $OUT/cf && cd $OUT
cut30() { python - "$1" "$2" <<'PY'
import sys, os
sys.path.insert(0, os.environ["GLINT"]); sys.path.insert(0, os.path.join(os.environ["GLINT"], "experiments"))
from glint.glint_fast import load
from scramble_qlist import write_qlist
write_qlist(sys.argv[2], load(sys.argv[1])[:30])
PY
}
for r in r0278 r0058; do
  python $VDIR/npz2qlist_j4.py $J4OUT/${r}_qlists.npz inputs/${r}_full.txt
  cut30 inputs/${r}_full.txt inputs/${r}_real.txt
  python $GLINT/experiments/scramble_qlist.py inputs/${r}_real.txt inputs/${r}_scr.txt --seed $SCR_SEED
done
cd cxi
for s in $SETS; do python $CXI_TOOLS/qframes_to_cxi.py ../inputs/$s.txt $s | tail -1; done
for s in $SETS; do
  mkdir -p $OUT/cf/run_$s && cd $OUT/cf/run_$s && ln -sf $OUT/cxi/${s}.cxi ${s}.cxi   # the event list names it relative to the cwd
  indexamajig -i $OUT/cxi/${s}_events.lst -g $OUT/cxi/${s}.geom -o $OUT/cf/${s}.stream \
    --peaks=cxi --no-image-data --no-revalidate --indexing=xgandalf -j 4 > $OUT/cf/${s}.log 2>&1
  tail -1 $OUT/cf/${s}.log
done
cd $OUT && python $VDIR/cells.py $OUT
echo "dry run OUT=$OUT"
