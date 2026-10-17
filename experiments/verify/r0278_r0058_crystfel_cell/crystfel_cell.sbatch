#!/bin/bash
#SBATCH --job-name=crystfel_cell
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
# CrystFEL's most common cell on r0278 and r0058, blind, on the peaks behind GLINT's per-frame numbers, as a
# reference independent of GLINT's indexer. CPU only, well under 1 GB of output.
# Submit from $VERIFY_OUT/r0278_r0058_crystfel_cell after `mkdir -p logs` (README).
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r0278_r0058_crystfel_cell/env.sh"
set -eo pipefail
mkdir -p $OUT/inputs $OUT/cxi $OUT/cf
cd $OUT
cp $VDIR/README.md $VDIR/env.sh $OUT/
echo "host $(hostname)  date $(date -Is)  GLINT COMMIT $(cat $GLINT/COMMIT)  $(indexamajig --version | head -1)"
env | grep '^GLINT_' && { echo "GLINT_* variables set: the shipped defaults would not apply"; exit 1; } || true
chk() { [ "$(md5sum < "$1" | cut -d' ' -f1)" = "$2" ] || { echo "md5 of $1 changed (want $2)"; exit 1; }; }
md5sum $ACC_OUT/r0278_qlists.npz $ACC_OUT/r0058_qlists.npz $CXI_TOOLS/qframes_to_cxi.py
# qframes_to_cxi.py is the recorded run's make_cxi_pinned.py (md5 e237baae682b05988406b7acfdd70268) with its
# pinned-tree path read from VERIFY_GLINT; logic unchanged.
chk $CXI_TOOLS/qframes_to_cxi.py c23ff4debc48b5f3ba512b44a5306bcb
# q lists (same frames and peaks as the per-frame check), and azimuth-scrambled copies as the null (|q| and q_z kept, lattice
# destroyed). The null uses each run's first 400 frames: blind XGANDALF is slow on scrambled frames (dry run:
# about 20 core-s per frame with a long tail), and 400 frames are enough to show whether a dominant cell forms.
first400() { python - "$1" "$2" <<'PY'
import os, sys
sys.path.insert(0, os.environ["GLINT"]); sys.path.insert(0, os.path.join(os.environ["GLINT"], "experiments"))
from glint.glint_fast import load
from scramble_qlist import write_qlist
write_qlist(sys.argv[2], load(sys.argv[1])[:400])
PY
}
for r in r0278 r0058; do
  python $VDIR/npz2qlist.py $ACC_OUT/${r}_qlists.npz inputs/${r}_real.txt
  first400 inputs/${r}_real.txt inputs/${r}_first400.txt
  python $GLINT/experiments/scramble_qlist.py inputs/${r}_first400.txt inputs/${r}_scr.txt --seed $SCR_SEED
done
# CXI peak lists for indexamajig (make_cxi.py at 95d51ce, the copy reading the pinned tree). It writes one wavelength per set: r0278
# has one (1.26296 A); r0058's per-event wavelengths spread 0.8% (1.2615-1.2717 A), and make_cxi infers their median.
cd $OUT/cxi
for s in $SETS; do
  python $CXI_TOOLS/qframes_to_cxi.py ../inputs/$s.txt $s | tee $s.make_cxi.log
  lam=$(sed -n 's/.*; lambda \([0-9.]*\) .*/\1/p' $s.make_cxi.log)
  case $s in r0278*) want=1.26296 ;; r0058*) want=1.26434 ;; esac
  python -c "import sys; sys.exit(abs($lam - $want) > 0.003)" || { echo "$s: inferred lambda $lam, expected $want"; exit 1; }
done
# Blind XGANDALF through indexamajig (no cell, defaults otherwise), each set in its own directory
for s in $SETS; do
  mkdir -p $OUT/cf/run_$s && cd $OUT/cf/run_$s && ln -sf $OUT/cxi/${s}.cxi ${s}.cxi   # the event list names it relative to the cwd
  echo "argv: indexamajig -i $OUT/cxi/${s}_events.lst -g $OUT/cxi/${s}.geom -o $OUT/cf/${s}.stream --peaks=cxi --no-image-data --no-revalidate --indexing=xgandalf -j $SLURM_CPUS_PER_TASK"
  /usr/bin/time -v indexamajig -i $OUT/cxi/${s}_events.lst -g $OUT/cxi/${s}.geom -o $OUT/cf/${s}.stream \
    --peaks=cxi --no-image-data --no-revalidate --indexing=xgandalf -j $SLURM_CPUS_PER_TASK > $OUT/cf/${s}.log 2>&1
  tail -3 $OUT/cf/${s}.log
done
cd $OUT
python $VDIR/cells.py $OUT
md5sum inputs/*.txt cxi/*.cxi cxi/*.geom cf/*.stream > MANIFEST.md5
echo "done $(date -Is)"
