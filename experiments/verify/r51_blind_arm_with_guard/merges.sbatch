#!/bin/bash
#SBATCH --job-name=guard_merge
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
# working directory: submit from $VERIFY_OUT/r51_blind_arm_with_guard, or pass --chdir= to it (else indexamajig.<pid>/ lands in the submitter's cwd)
#SBATCH --constraint=OS_VER:8.10   # the OS of the dry run (site setting)
# Job 3 of 3 (CPU, afterok job 2): the merges and the results table.
#   own_<arm>          each arm's accepted crystals as they are (the LUTE-style merge; frame sets differ)
#   common_<arm>_s<k>  the frames with a crystal in BOTH CF and GLb (C.keys), first crystal per frame, in the
#                      order of seed k = 1..NSPLIT (the same order in every arm -> paired half-splits);
#                      arms CF, GLb, B0, N0_1..3  <- the comparison row and its merge-level null
#   commonk_<arm>_s<k> the same for CF and GLk (Ck.keys)
#   commonnodcc_<arm>_s<k> the common set with --no-deltacchalf (secondary protocol; CF, GLb, N0_1)
# Merges run 64 at a time, MERGE_J (= 1) thread each. -j is fixed at 1 because partialator 0.10.2's
# rejections depend on -j and on the chunk layout for -j > 1; single-threaded merges parallelise wide instead.
set -uo pipefail
source "${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/r51_blind_arm_with_guard/config.sh"
glint_env || exit 1
# merge_one.sh re-sources merges_tofile_vs_crystfel/config.sh, which would reset OUT to that check's
# directory and append this job's argv to ITS commands.log. DRYRUN_OUT is that config's OUT override,
# so exporting it keeps every merge of this job, and its provenance, under this job's OUT.
export DRYRUN_OUT=$OUT
echo "host $(hostname)  job ${SLURM_JOB_ID:-}  start $(date -u +%FT%TZ)"
TASKS=$OUT/merge_tasks.txt; aside "$TASKS"; : > "$TASKS"

for D in $DATASETS; do
    v=${D}_SYM; SYM=${!v}; v=${D}_CELLFILE; CELLFILE=${!v}
    DD=$OUT/$D; M=$DD/merges; mkdir -p "$M"
    [ -s "$DD/CF.keys" ] || { echo "no CF.keys for $D; skipping"; continue; }
    for A in CF CFnr GLb GLk; do
        [ -s "$DD/$A.stream" ] && echo "$DD/$A.stream - 0 0 $M/own_$A $SYM $CELLFILE $MERGE_J" >> "$TASKS"
    done
    for s in $NULL_SEEDS; do                       # the frame-level null is merged only if CrystFEL kept >= 20
        n=$(wc -l < "$DD/N$s.keys" 2>/dev/null || echo 0)
        if [ "$n" -ge 20 ]; then echo "$DD/N$s.stream - 0 0 $M/own_N$s $SYM $CELLFILE $MERGE_J" >> "$TASKS"
        else echo "$D N$s: CrystFEL accepted $n shuffled solutions; not merged" | tee -a "$DD/null_frames.txt"; fi
    done
    for k in $(seq 1 $NSPLIT); do
        if [ -s "$DD/C.keys" ]; then
            for A in CF GLb B0 N0_1 N0_2 N0_3; do
                [ -s "$DD/$A.stream" ] && echo "$DD/$A.stream $DD/C.keys $k 1 $M/common_${A}_s$k $SYM $CELLFILE $MERGE_J" >> "$TASKS"
            done
            for A in CF GLb N0_1; do               # secondary protocol: no Delta-CC1/2 rejection
                [ -s "$DD/$A.stream" ] && echo "$DD/$A.stream $DD/C.keys $k 1 $M/commonnodcc_${A}_s$k $SYM $CELLFILE $MERGE_J $NODCC" >> "$TASKS"
            done
        fi
        if [ -s "$DD/Ck.keys" ]; then
            for A in CF GLk; do
                [ -s "$DD/$A.stream" ] && echo "$DD/$A.stream $DD/Ck.keys $k 1 $M/commonk_${A}_s$k $SYM $CELLFILE $MERGE_J" >> "$TASKS"
            done
        fi
    done
done
echo "$(wc -l < $TASKS) merges"

# ===== REPRODUCTION GATE, before any comparison merge ==========================================
# Merge LUTE's own r51 GLINT stream with this harness and require the published FOM. If this does
# not reproduce, the harness is not the one the comparison numbers would be compared against, and
# every merge below would be unreadable -- so stop here rather than produce 197 numbers nobody can
# trust. Target measured today at -j 1, 8 and 72 (it is LUTE's hkl1/hkl2): CC 0.1751131, 2686 pairs.
REPRO_STREAM=$R51_VAL7/mfx100848724_r51_glint_all.stream    # R51_VAL7: config.sh
REPRO_DIR=$M/repro_r51_lute
REPRO_CC_WANT=0.1751131
REPRO_PAIRS_WANT=2686
if [ ! -s "$REPRO_STREAM" ]; then
    echo "REPRO_FAILED: $REPRO_STREAM missing or empty" | tee "$M/REPRO_FAILED"
    exit 1
fi
"$TOOLS_HOME/merge_one.sh" "$REPRO_STREAM" - 0 0 "$REPRO_DIR" "$r51_SYM" "$r51_CELLFILE" 1 \
    > "$M/repro_r51_lute.log" 2>&1
# Parse compare_hkl's real output. Verified against a compare_hkl log on disk, which prints exactly:
#   "3 reflection pairs accepted."   and   "Overall CC = 0.9569087"
# (the phrase is "reflection pairs accepted", and the CC is only in the log, not in the shell file)
REPRO_CC=$(grep -oE 'Overall CC = [0-9.]+' "$REPRO_DIR/compare_CC.log" 2>/dev/null | tail -1 | awk '{print $4}')
REPRO_PAIRS=$(grep -oE '[0-9]+ reflection pairs accepted' "$REPRO_DIR/compare_CC.log" 2>/dev/null | tail -1 | awk '{print $1}')
echo "repro gate: CC=${REPRO_CC:-none} (want $REPRO_CC_WANT)  pairs=${REPRO_PAIRS:-none} (want $REPRO_PAIRS_WANT)"
OK=$(python - "$REPRO_CC" "$REPRO_CC_WANT" "$REPRO_PAIRS" "$REPRO_PAIRS_WANT" <<'PYEOF'
import sys
try:
    cc, want, pr, pwant = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    # 1e-6 absolute on CC: the target is quoted to 7 figures and was identical at -j 1, 8 and 72,
    # so anything beyond rounding is a real difference in the harness, not in the arithmetic.
    print("1" if (abs(float(cc) - want) <= 1e-6 and int(pr) == pwant) else "0")
except Exception:
    print("0")
PYEOF
)
if [ "$OK" != "1" ]; then
    echo "REPRO_FAILED: r51 LUTE merge gave CC=${REPRO_CC:-none}/${REPRO_PAIRS:-none} pairs," \
         "want $REPRO_CC_WANT/$REPRO_PAIRS_WANT. Not running any comparison merge." | tee "$M/REPRO_FAILED"
    exit 1
fi
echo "repro gate PASSED"
# ==============================================================================================

SECONDS=0
xargs -P 64 -L 1 "$TOOLS_HOME/merge_one.sh" < "$TASKS"
echo "merges done in ${SECONDS} s"

python - <<EOF > "$OUT/provenance.json"
import json, os, subprocess
cf = subprocess.run(["$CFBIN/indexamajig", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
r16 = "$OUT/r0016/cxi/r0016_cxi.json"
print(json.dumps({
  "commit": open("$GLINT_TREE/COMMIT").read().strip(),
  "glint_tree": "$GLINT_TREE",
  "crystfel": cf + " ($CFBIN)",
  "glint_env": "ana-4.0.58-py3-minipytorch (psconda conda1)",
  "glint_argv_blind": "python -m glint.glint_cli --images LIST --geom GEOM --peakfinder stored --gate strict --lattice LATT --tofile GLb.sol -o GLb_orient.stream",
  "glint_argv_known": "the same + --cell CELL (GLk)",
  "cf_index_argv": "indexamajig -i LIST -o CF.stream --peaks=cxi -g GEOM -p CELLFILE $CF_INDEX $INTEG IDXX -j 64 --wait-for-file=0",
  "idxx_per_dataset": {"r51": "$r51_IDXX", "r194": "$r194_IDXX", "r0016": "$r0016_IDXX"},
  "cfnr": "CF + --no-revalidate on $CFNR_DATASETS (own merge only)",
  "file_index_argv": "indexamajig -i A.lst -o A.stream --peaks=cxi -g GEOM -p CELLFILE $FILE_INDEX --fromfile-input-file=SOL $INTEG IDXX -j 32 --wait-for-file=0",
  "nocheck_extra (B0, N0)": "$NOCHECK $NOPEAKS",
  "frame_null_extra (N)": "$NOPEAKS",
  "merge_argv": "partialator -i in.stream -o merged.hkl -y SYM $MERGE -j $MERGE_J", "secondary_merge_extra": "$NODCC",
  "compare_argv": "compare_hkl merged.hkl1 merged.hkl2 -y SYM -p CELLFILE --fom={CC,CCstar,Rsplit} --nshells=10",
  "check_argv": "check_hkl merged.hkl -y SYM -p CELLFILE --nshells=10",
  "nsplit": $NSPLIT, "null_seeds": "$NULL_SEEDS", "datasets": "$DATASETS",
  "r0016_cxi": json.load(open(r16)) if os.path.exists(r16) else None,
  "btx": "$BTX (env $BTX_ENV)",
}, indent=1))
EOF
python "$TOOLS_HOME/merge_tools.py" collect "$OUT"
du -sh "$OUT"
echo "end $(date -u +%FT%TZ)"
