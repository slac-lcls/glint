#!/bin/bash
# Fast XGANDALF, 32 concurrent instances on 32 physical cores of an
# S3DF ampere node (2 x AMD EPYC 7542), the host type of blind_ms. CPU only; no GPU is requested.
# README.md in this folder. Needs VERIFY_GLINT, VERIFY_DIR, VERIFY_OUT, VERIFY_INPUTS exported and bin/xg_driver
# built (README); submit from $VERIFY_OUT/xgandalf_fast_cores after `mkdir -p logs`.
#SBATCH --job-name=j6_xg32
#SBATCH --partition=ampere
#SBATCH --account=lcls:prjdat21
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=sdfampere042,sdfampere003
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# --cpus-per-task=64: ampere nodes are 2 sockets x 32 cores x 2 threads and Slurm allocates whole cores
# (CR_CORE_MEMORY), so 64 logical CPUs = 32 physical cores. One instance per core, pinned to the core's
# first thread; the sibling thread is ours and stays idle.
# Fallback (EPYC 7702, the node type of job 37187585; the host label then changes):
#   sbatch -p roma -c 32 $VERIFY_DIR/xgandalf_fast_cores/j6_xg32.sbatch

set -uo pipefail
umask 077
export PATH=/opt/slurm/slurm-curr/bin:$PATH
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1   # write no new .pyc files into the pinned tree

J=${VERIFY_DIR:?set VERIFY_DIR to experiments/verify}/xgandalf_fast_cores
G=${VERIFY_GLINT:?set VERIFY_GLINT to the pinned GLINT tree}
S=${VERIFY_OUT:?set VERIFY_OUT to the output root}/xgandalf_fast_cores
C=/sdf/group/lcls/ds/tools/crystfel/0.12.0
XG=$J/bin/xg_driver
F120=$G/experiments/frames_cxidb_clean.txt
F480=${VERIFY_INPUTS:?set VERIFY_INPUTS to the inputs root}/q480_fix.txt
NINST=${NINST:-32}
PASSES120=${PASSES120:-3}
QREPS=${QREPS:-3}
SEED=20260928
ARMS=${ARMS:-"S1_120 D1_24 S1_480 P32_120 Q32_120 N32_120 P32_480 Q32_480 S1_120b"}
DRYRUN=${DRYRUN:-0}

RUN=${RUN:-$S/run_${SLURM_JOB_ID:-manual}}
mkdir -p "$RUN"/{raw,in,score}
RAW=$RUN/raw
LOCK=${TMPDIR:-/tmp}/j6_${SLURM_JOB_ID:-manual}_$$
mkdir -p "$LOCK"
CMDLOG=$RUN/commands.log
note() { echo "[$(date +%T)] $*"; }

set +u   # the conda activation scripts read unset variables
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh
conda activate ana-4.0.58-py3-minipytorch
set -u
export PYTHONPATH=$G

# ------------------------------------------------------------------ dry run: tiny inputs, 2 instances
if [ "$DRYRUN" = "1" ]; then
  awk '/^FRAME/{n++} n<=4' "$F120" > "$RUN/in/dry120.txt"
  awk '/^FRAME/{n++} n<=4' "$F480" > "$RUN/in/dry480.txt"
  F120=$RUN/in/dry120.txt; F480=$RUN/in/dry480.txt
  NINST=2; PASSES120=1; QREPS=1
  ARMS=${DRYARMS:-"S1_120 S1_480 P32_120 Q32_120 N32_120 P32_480 Q32_480 S1_120b"}
fi

# ------------------------------------------------------------------ provenance
{
  echo "run $RUN"
  echo "date: $(date -Is)"
  echo "host: $(hostname)  slurm job: ${SLURM_JOB_ID:-none}  partition: ${SLURM_JOB_PARTITION:-none}  account: ${SLURM_JOB_ACCOUNT:-none}  qos: ${SLURM_JOB_QOS:-none}"
  echo "glint tree: $G  COMMIT: $(cat $G/COMMIT)"
  echo "CrystFEL: $($C/bin/indexamajig --version 2>&1 | head -1)  libxgandalf: $C/lib64/libxgandalf.so"
  echo "driver: $XG built from $J/xg_driver.cpp (byte copy of the driver of job 37187585)"
  echo "g++ used for the build (on sdfiana027): g++ (GCC) 8.5.0; -O2 -std=c++14 -I$C/include -I$C/include/eigen3 -L$C/lib64 -lxgandalf -Wl,-rpath,$C/lib64"
  echo "arms: $ARMS  NINST=$NINST PASSES120=$PASSES120 QREPS=$QREPS DRYRUN=$DRYRUN"
  echo "--- sha256"
  sha256sum "$XG" "$J/xg_driver.cpp" "$C/lib64/libxgandalf.so" "$F120" "$F480" "$J/j6_prep.py" "$J/j6_analyze.py" "$J/j6_xg32.sbatch" \
            "$G/experiments/xgandalf/score_xg_gate.py" "$G/experiments/scramble_qlist.py" "$G/experiments/check_numbers.py"
  echo "--- os"; cat /etc/redhat-release; uname -r
  echo "--- lscpu"; lscpu
  echo "--- affinity of this shell"; taskset -pc $$
  echo "--- load at start"; cat /proc/loadavg
  echo "--- jobs on this node at start: $(squeue -h -w "$(hostname -s)" 2>/dev/null | wc -l)"
  echo "--- python"; which python3; python3 --version
} > "$RUN/provenance.txt" 2>&1
cp "$J/README.md" "$RUN/README_at_start.md" 2>/dev/null   # the README as the job starts

# ------------------------------------------------------------------ one logical CPU per physical core
mapfile -t TOPO < <(python3 - <<'EOF'
import os, subprocess
allowed = sorted(os.sched_getaffinity(0))
first = {}
for line in subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET"], stdout=subprocess.PIPE,
                           universal_newlines=True).stdout.splitlines():
    if line.startswith("#"):
        continue
    cpu, core, sock = map(int, line.split(","))
    if cpu in allowed and (sock, core) not in first:
        first[(sock, core)] = cpu
for (sock, core), cpu in sorted(first.items()):
    print(cpu, sock, core)
EOF
)
PIN=(); SOCK=()
for row in "${TOPO[@]}"; do set -- $row; PIN+=("$1"); SOCK+=("$2"); done
NCORE=${#PIN[@]}
{
  echo "--- physical cores in the allocation: $NCORE"
  printf 'pinned logical CPUs: %s\n' "${PIN[*]}"
  echo "cores per socket: $(printf '%s\n' "${SOCK[@]:0:$NINST}" | awk '{c[$1]++} END{for (k in c) printf "socket %s: %d   ", k, c[k]}') (of the $NINST pinned)"
} | tee -a "$RUN/provenance.txt"
if [ "$NCORE" -lt "$NINST" ]; then
  note "ABORT: $NCORE physical cores in the allocation, $NINST needed"; exit 2
fi
CPULIST=$(IFS=,; echo "${PIN[*]:0:$NINST}")

# ------------------------------------------------------------------ inputs (scratch, this run only)
note "prep"
{
  echo "prep: python3 $J/j6_prep.py $F120 $RUN/in/f120 --ninst $NINST --subset 24"
  echo "prep: python3 $J/j6_prep.py $F480 $RUN/in/f480 --ninst $NINST"
  echo "prep: python3 $G/experiments/scramble_qlist.py $F120 $RUN/in/scr120.txt --seed $SEED"
  echo "prep: python3 $J/j6_prep.py $RUN/in/scr120.txt $RUN/in/scr120 --ninst $NINST"
} >> "$CMDLOG"
python3 "$J/j6_prep.py" "$F120" "$RUN/in/f120" --ninst "$NINST" --subset 24 || { note "ABORT prep 120"; exit 3; }
python3 "$J/j6_prep.py" "$F480" "$RUN/in/f480" --ninst "$NINST" || { note "ABORT prep 480"; exit 3; }
python3 "$G/experiments/scramble_qlist.py" "$F120" "$RUN/in/scr120.txt" --seed "$SEED" || { note "ABORT scramble"; exit 3; }
python3 "$J/j6_prep.py" "$RUN/in/scr120.txt" "$RUN/in/scr120" --ninst "$NINST" || { note "ABORT prep scr"; exit 3; }
SUB24=$RUN/in/f120/sub24.txt
N120=$(grep -c '^FRAME' "$F120"); N480=$(grep -c '^FRAME' "$F480")

# ------------------------------------------------------------------ CPU clock trace (every 20 s)
echo "setup" > "$RAW/current_arm"
( while :; do
    m=$(awk -v L="$CPULIST" 'BEGIN{n=split(L,a,",");for(i=1;i<=n;i++)w[a[i]]=1}
         /^processor/{p=$3} /^cpu MHz/{if(p in w){s+=$4;k++}} END{if(k)printf "%.0f",s/k}' /proc/cpuinfo)
    echo "$(date +%s) $(cat "$RAW/current_arm") $m" >> "$RAW/mhz.tsv"; sleep 20
  done ) &
MHZPID=$!
disown "$MHZPID"
trap 'kill $MHZPID 2>/dev/null; rm -rf "$LOCK"' EXIT

# ------------------------------------------------------------------ smoke test
# One fast frame on the first pinned core before an hour of 32 cores is spent: a driver or library
# that does not run on this node aborts here. Not analysed (the analysis globs arm tags only).
echo "smoke" > "$RAW/current_arm"
echo "smoke: taskset -c ${PIN[0]} $XG $RUN/in/f120/one/f_0.txt blind 2 3" >> "$CMDLOG"
taskset -c "${PIN[0]}" "$XG" "$RUN/in/f120/one/f_0.txt" blind 2 3 > "$RAW/smoke.out" 2> "$RAW/smoke.err"; rc=$?
nf=$(awk 'NF==11 || (NF==3 && $2=="NONE")' "$RAW/smoke.out" | wc -l)
if [ "$rc" -ne 0 ] || [ "$nf" -ne 1 ]; then
  note "ABORT: smoke test rc=$rc result lines=$nf (raw/smoke.out, raw/smoke.err)"; exit 4
fi
note "smoke ok: $(cat "$RAW/smoke.out")"

# ------------------------------------------------------------------ arm runners
# timestamp line: TAG a b cpu t0 t1 rc   (single/parallel: a=instance b=pass; queue: a=slot b=position k)
run_single() {  # TAG FRAMES PITCH GDIT
  local tag=$1 fr=$2 pitch=$3 gdit=$4 cpu=${PIN[0]}
  echo "$tag" > "$RAW/current_arm"
  echo "$tag: taskset -c $cpu $XG $fr blind $pitch $gdit" >> "$CMDLOG"
  local t0 t1 rc
  t0=$(date +%s.%N)
  taskset -c "$cpu" "$XG" "$fr" blind "$pitch" "$gdit" > "$RAW/$tag.out" 2> "$RAW/$tag.err"; rc=$?
  t1=$(date +%s.%N)
  echo "$tag 0 1 $cpu $t0 $t1 $rc" > "$RAW/$tag.times"
  note "$tag rc=$rc frames=$(wc -l < "$RAW/$tag.out") wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}') s"
}

run_parallel() {  # TAG ROTDIR PASSES PITCH GDIT NI
  local tag=$1 rot=$2 passes=$3 pitch=$4 gdit=$5 ni=$6 pids=() i
  echo "$tag" > "$RAW/current_arm"
  echo "$tag: for i in 0..$((ni-1)): taskset -c PIN[i] $XG $rot/rot_\$i.txt blind $pitch $gdit, $passes pass(es), all at once; PIN=${PIN[*]:0:$ni}" >> "$CMDLOG"
  for ((i=0; i<ni; i++)); do
    ( cpu=${PIN[$i]}
      for ((p=1; p<=passes; p++)); do
        t0=$(date +%s.%N)
        taskset -c "$cpu" "$XG" "$rot/rot_$i.txt" blind "$pitch" "$gdit" > "$RAW/$tag.i$i.p$p.out" 2> "$RAW/$tag.i$i.p$p.err"; rc=$?
        t1=$(date +%s.%N)
        echo "$tag $i $p $cpu $t0 $t1 $rc" >> "$RAW/$tag.i$i.times"
      done ) &
    pids+=($!)
  done
  wait "${pids[@]}"
  note "$tag done: $(cat "$RAW"/$tag.i*.p*.out | wc -l) solves"
}

run_queue() {  # TAG ONEDIR N PITCH GDIT NW : frames 0..N-1 dealt one at a time to NW pinned workers
  local tag=$1 one=$2 n=$3 pitch=$4 gdit=$5 nw=$6 pids=() s
  echo "$tag" > "$RAW/current_arm"
  echo "$tag: queue of $n single-frame files $one/f_<k>.txt over $nw workers; each: taskset -c PIN[slot] $XG $one/f_<k>.txt blind $pitch $gdit; PIN=${PIN[*]:0:$nw}" >> "$CMDLOG"
  echo 0 > "$LOCK/$tag.next"
  for ((s=0; s<nw; s++)); do
    ( cpu=${PIN[$s]}
      while :; do
        k=$(flock "$LOCK/$tag.lock" bash -c "k=\$(cat '$LOCK/$tag.next'); echo \$((k+1)) > '$LOCK/$tag.next'; echo \$k")
        [[ "$k" =~ ^[0-9]+$ ]] || break
        [ "$k" -ge "$n" ] && break
        t0=$(date +%s.%N)
        taskset -c "$cpu" "$XG" "$one/f_$k.txt" blind "$pitch" "$gdit" >> "$RAW/$tag.s$s.out" 2>> "$RAW/$tag.s$s.err"; rc=$?
        t1=$(date +%s.%N)
        echo "$tag $s $k $cpu $t0 $t1 $rc" >> "$RAW/$tag.s$s.times"
      done ) &
    pids+=($!)
  done
  wait "${pids[@]}"
  note "$tag done: $(cat "$RAW"/$tag.s*.out 2>/dev/null | wc -l) of $n frames"
}

# ------------------------------------------------------------------ the arms, in order
for arm in $ARMS; do
  case $arm in
    S1_120)  run_single S1_120  "$F120" 2 3 ;;                 # alone: latency + yield (expect 88/96)
    D1_24)   run_single D1_24   "$SUB24" 6 4 ;;                # alone, CrystFEL defaults: host calibration
    S1_480)  run_single S1_480  "$F480" 2 3 ;;                 # alone at n=480
    P32_120) run_parallel P32_120 "$RUN/in/f120/rot" "$PASSES120" 2 3 "$NINST" ;;   # sustained throughput
    Q32_120) for r in $(seq 1 "$QREPS"); do run_queue "Q32_120_r$r" "$RUN/in/f120/one" "$N120" 2 3 "$NINST"; done ;;
    N32_120) run_queue N32_120 "$RUN/in/scr120/one" "$N120" 2 3 "$NINST" ;;          # null: no lattice
    P32_480) run_parallel P32_480 "$RUN/in/f480/rot" 1 2 3 "$NINST" ;;
    Q32_480) run_queue Q32_480 "$RUN/in/f480/one" "$N480" 2 3 "$NINST" ;;
    S1_120b) run_single S1_120b "$F120" 2 3 ;;                 # drift check
    P64_120) # optional, off by default: 64 instances on both hardware threads of the 32 cores (SMT)
             mapfile -t ALL < <(python3 -c "import os; print('\n'.join(map(str, sorted(os.sched_getaffinity(0)))))")
             PIN=("${ALL[@]}"); python3 "$J/j6_prep.py" "$F120" "$RUN/in/f120_64" --ninst 64 >/dev/null
             run_parallel P64_120 "$RUN/in/f120_64/rot" 1 2 3 64 ;;
    *) note "unknown arm $arm" ;;
  esac
done
echo "analysis" > "$RAW/current_arm"
{ echo "--- load at end"; cat /proc/loadavg; echo "--- jobs on this node at end: $(squeue -h -w "$(hostname -s)" 2>/dev/null | wc -l)"; } >> "$RUN/provenance.txt"

# ------------------------------------------------------------------ analysis (FACTS read from the pinned tree)
echo "analysis: python3 $J/j6_analyze.py $RUN --glint $G --f120 $F120 --f480 $F480 --sub24 $SUB24 --scr120 $RUN/in/scr120.txt" >> "$CMDLOG"
python3 "$J/j6_analyze.py" "$RUN" --glint "$G" --f120 "$F120" --f480 "$F480" --sub24 "$SUB24" --scr120 "$RUN/in/scr120.txt"
note "finished: $RUN/summary.md"
