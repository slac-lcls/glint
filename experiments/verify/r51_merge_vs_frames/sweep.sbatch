#!/bin/bash
#SBATCH --job-name=r51_sweep
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --constraint=OS_VER:8.10
#SBATCH --output=logs/%x_%j.log
# Merge-redundancy sweep on mfx100848724 r51 (CPU only; one job per stage).
#   sbatch sweep.sbatch                                  stage A: CrystFEL's own indexing, LUTE round 4
#   sbatch --export=ALL,STAGE=B sweep.sbatch              stage B: the r51 CF and GLb streams of
#                                                             merges_tofile_vs_crystfel (the job stops if they are absent)
# Needs VERIFY_GLINT, VERIFY_DIR, VERIFY_OUT exported (../README.md); submit from $VERIFY_OUT/r51_merge_vs_frames.
# Per input stream: (1) the reference merge (100%, stream order, partialator) sets the fixed resolution range;
# (2) every fraction x seed, real intensities with partialator and with process_hkl, and the two nulls with
# partialator; NPAR merges at a time, NJ threads each; (3) collect -> results.json, sweep.tsv, analysis.json, summary.md.
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r51_merge_vs_frames/config.sh"
STAGE=${STAGE:-A}
case $STAGE in
    A) INPUTS=$A_INPUTS ;;
    B) INPUTS=$B_INPUTS ;;
    *) echo "STAGE must be A or B"; exit 2 ;;
esac
SOUT=$OUT/$STAGE; mkdir -p "$SOUT" "$OUT/logs"; chmod 700 "$OUT"
export CMDLOG=$SOUT/commands.log
glint_env || exit 1
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  stage $STAGE  start $(date -u +%FT%TZ)"

for IN in $INPUTS; do
    TAG=${IN%%:*}; STREAM=${IN#*:}
    if [ ! -s "$STREAM" ]; then echo "$TAG: $STREAM absent or empty; stage $STAGE stops"; exit 3; fi
    D=$SOUT/$TAG; M=$D/merges; mkdir -p "$M"
    python -B "$TOOLS_HOME/sweep_tools.py" frames "$STREAM" | tee "$D/frames.json"
    sed -n 3p "$STREAM" > "$D/input_indexamajig_argv.txt"            # line 3 of a CrystFEL stream = its full argv
    md5sum "$STREAM" > "$D/input.md5"

    # (1) reference merge: 100%, stream order -> compare_hkl's accepted range is the fixed range for this input
    DRANGE= "$TOOLS_HOME/sweep_merge_one.sh" "$STREAM" "$TAG" 1.00 0 none partialator "$D/ref" "$((NPAR * NJ))"
    DR=$(python -B "$TOOLS_HOME/sweep_tools.py" range "$D/ref") || { echo "$TAG: no reference range"; exit 4; }
    echo "$DR" > "$D/fixed_range_A.txt"; echo "$TAG fixed range (A): $DR"

    # (2) the sweep
    T=$D/tasks.txt; : > "$T"
    echo "$STREAM $TAG 1.00 0 none partialator $M/partialator_none_f1.00_s0" >> "$T"
    echo "$STREAM $TAG 1.00 0 none process_hkl $M/process_hkl_none_f1.00_s0" >> "$T"
    for P in $NULLS; do echo "$STREAM $TAG 1.00 0 $P partialator $M/partialator_${P}_f1.00_s0" >> "$T"; done
    for F in $FRACTIONS; do for S in $SEEDS; do
        echo "$STREAM $TAG $F $S none partialator $M/partialator_none_f${F}_s$S" >> "$T"
        echo "$STREAM $TAG $F $S none process_hkl $M/process_hkl_none_f${F}_s$S" >> "$T"
        for P in $NULLS; do echo "$STREAM $TAG $F $S $P partialator $M/partialator_${P}_f${F}_s$S" >> "$T"; done
    done; done
    echo "$TAG: $(wc -l < "$T") merges"
    DRANGE="$DR" xargs -P "$NPAR" -L 1 "$TOOLS_HOME/sweep_merge_one.sh" < "$T"        # NJ threads each (config.sh)
    echo "$TAG: $(ls "$M"/*/record.json 2>/dev/null | wc -l) records"
done

# reproduction check (stage A): the 100% stream-order merge against compare_hkl on LUTE's own merge of the same stream
if [ "$STAGE" = A ]; then
    R=$SOUT/repro; mkdir -p "$R"; cd "$R"
    for fom in CC CCstar Rsplit; do
        run "$CFBIN/compare_hkl" "$A_LUTE_HKL"1 "$A_LUTE_HKL"2 -y "$SYM" -p "$CELLFILE" --fom=$fom --nshells=$NSHELLS \
            --shell-file=lute_$fom.dat > lute_$fom.log 2>&1
    done
    python -B - "$R" "$SOUT/r51CF/ref" <<'EOF' > "$SOUT/repro.txt"
import os, sys
sys.path.insert(0, os.path.join(os.environ["VERIFY_DIR"], "r51_merge_vs_frames"))
from sweep_tools import parse_compare
r, ref = sys.argv[1], sys.argv[2]
print("100% stream-order merge (partialator -j 32) against LUTE's merge of the same stream (partialator -j 72), auto range")
for fom in ("CC", "CCstar", "Rsplit"):
    a = parse_compare(f"{r}/lute_{fom}.log"); b = parse_compare(f"{ref}/auto/compare_{fom}.log")
    va, vb = a.get("value"), b.get("value")
    tol = 0.1 if fom == "Rsplit" else 1e-3
    ok = va is not None and vb is not None and abs(va - vb) <= tol and a.get("pairs") == b.get("pairs")
    print(f"{fom:7s} LUTE {va} ({a.get('pairs')} pairs)   this run {vb} ({b.get('pairs')} pairs)   "
          f"{'MATCH' if ok else 'MISMATCH: stop and look before using the sweep'}")
EOF
    cat "$SOUT/repro.txt"
fi

python -B - <<EOF > "$SOUT/provenance.json"
import json, subprocess
v = subprocess.run(["$CFBIN/partialator", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
print(json.dumps({
  "job": "r51 merge statistics against frames merged (merge-redundancy sweep)", "stage": "$STAGE",
  "commit": open("$GLINT_TREE/COMMIT").read().strip() + " (GLINT tree; stage A inputs contain no GLINT output)",
  "crystfel": v + " ($CFBIN)", "env": "ana-4.0.58-py3-minipytorch (psconda conda1), numpy only",
  "inputs": "$INPUTS",
  "fractions": "$FRACTIONS", "seeds": "$SEEDS (seed 0 = stream order, 100% only)", "nulls": "$NULLS",
  "partialator_argv": "partialator -i in.stream -o merged.hkl -y $SYM $MERGE -j $NJ",
  "process_hkl_argv": "process_hkl -i in.stream -o merged.hkl[1|2] -y $SYM $MC [--odd-only|--even-only]",
  "compare_argv": "compare_hkl merged.hkl1 merged.hkl2 -y $SYM -p $CELLFILE --fom={CC,CCstar,Rsplit} --nshells=$NSHELLS [--lowres=DLO --highres=DHI]",
  "check_argv": "check_hkl merged.hkl -y $SYM -p $CELLFILE --nshells=$NSHELLS [--lowres=DLO --highres=DHI]",
  "every_call": "$CMDLOG",
}, indent=1))
EOF
python -B "$TOOLS_HOME/sweep_tools.py" collect "$SOUT"
du -sh "$SOUT"
echo "end $(date -u +%FT%TZ)"
