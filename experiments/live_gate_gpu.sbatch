#!/bin/bash
#SBATCH --job-name=glint_livegate_gpu
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --exclude=sdfampere042,sdfampere003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --output=/sdf/home/s/smarches/glint_esc_outputs/livegate/livegate_gpu_%j.out
# fix/live-gate-null (StreamDriver null_floor=): the CPU measurement (RESULTS_live_gate_null.md) on the A100's fused path.
# Submit from the worktree's HEAD: sbatch --export=ALL,HEAD_PIN=$(git rev-parse --short HEAD),WT=$PWD experiments/live_gate_gpu.sbatch
# 1) test_live_gate_floor.py on the node (its real-frame half then runs index_fused on the GPU) + the three pins;
# 2) the published 480 arm, default gate, gated 333/480 / 10 / 1 (default path unchanged on this branch);
# 3) the same arm with --null-floor cxidb17 (reported, not gated);
# 4) the scrambled arm (480 real, then 480 scrambled as "scr"), default gate and --null-floor cxidb17
#    (CPU: scr accepted 234 -> 5, lyso strict 331 -> 332, watchdog rescues 9 -> 33);
# 5) live_gate_null.py measure on the fused path, 8 fit copies (~2 min), then fit: do the GPU counts give the same floor?
# Every step prints its own exit status and the job exits non-zero if any step failed (rc below). Not
# `set -e`: that stops at the first failure and hides the steps after it, and psconda.sh/conda activate
# return non-zero on their own (Copilot review of #214).
rc=0
note() { echo "exit $1"; [ "$1" -eq 0 ] || rc=1; }
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
export PYTHONPATH=.
O=$HOME/glint_esc_outputs/livegate
mkdir -p $O
cd "${WT:?set WT to the worktree}"
: "${HEAD_PIN:?set HEAD_PIN to the commit this job must run}"
test "$(git rev-parse --short HEAD)" = "$HEAD_PIN" || { echo "wrong HEAD $(git rev-parse --short HEAD), want $HEAD_PIN"; exit 1; }
echo "host $(hostname)  branch $(git rev-parse --short HEAD)  $(date +%T)"
nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv,noheader
python -c "import cupy, torch; print('cupy', cupy.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
for t in test_live_gate_floor test_cell_registry test_lock_gate_wiring test_per_lattice; do
  echo "=== $t"; python experiments/$t.py | tail -3; note ${PIPESTATUS[0]}
done
COMMON="--input lyso=$HOME/q480_fix.txt --B 20 --dmin 2.0 --warmup-rescue --adaptive-relock --min-inliers 10"
echo "=== 480 arm, default gate, gated  $(date +%T)"
python experiments/record_stream_replay.py $COMMON --out $O/replay_480_off.json --expect 333/480 --expect-wresc 10 --expect-relock 1 \
    2>&1 | tail -6; note ${PIPESTATUS[0]}
echo "=== 480 arm, --null-floor cxidb17  $(date +%T)"
python experiments/record_stream_replay.py $COMMON --null-floor cxidb17 --out $O/replay_480_floor.json 2>&1 | tail -6; note ${PIPESTATUS[0]}
SCRF=$HOME/glint_esc_outputs/effort/q480_scrambled.txt
test "$(md5sum $SCRF | cut -d' ' -f1)" = "bd132bd01f202178aa6b3274d8393c64" || { echo "scrambled list md5 mismatch"; exit 1; }
SCR="--input lyso=$HOME/q480_fix.txt --input scr=$SCRF --ref scr=79.02,79.02,37.98,90,90,90 --B 20 --dmin 2.0 --warmup-rescue --adaptive-relock --min-inliers 10"
echo "=== scrambled arm, default gate  $(date +%T)"
python experiments/record_stream_replay.py $SCR --out $O/replay_scr_off.json 2>&1 | tail -6; note ${PIPESTATUS[0]}
echo "=== scrambled arm, --null-floor cxidb17  $(date +%T)"
python experiments/record_stream_replay.py $SCR --null-floor cxidb17 --out $O/replay_scr_floor.json 2>&1 | tail -6; note ${PIPESTATUS[0]}
echo "=== live_gate_null measure, fused path, 8 fit copies  $(date +%T)"
python experiments/live_gate_null.py measure --input $HOME/q480_fix.txt --null-input $SCRF --k-fit 8 --out $O/live_gate_null_480_gpu.npz \
    2>&1 | tail -3; note ${PIPESTATUS[0]}
python experiments/live_gate_null.py fit $O/live_gate_null_480_gpu.npz --input $HOME/q480_fix.txt --xgandalf $HOME/xgd480_fix.txt \
    --json $O/live_gate_null_480_gpu_fit.json > $O/live_gate_null_480_gpu_fit.txt 2>&1
note $?
head -16 $O/live_gate_null_480_gpu_fit.txt
python - <<'PY'
import numpy as np, os
O = os.path.expanduser("~/glint_esc_outputs/livegate")
g = np.load(f"{O}/live_gate_null_480_gpu.npz"); c = np.load("experiments/live_gate_null_480.npz")
n = c["n"]; fl = 0.0224 * n + 5.32 + 1.211 * np.sqrt(n)
for k in ("inl_real", "inl_null"):
    print(k, "GPU vs committed CPU: identical", int((g[k] == c[k]).sum()), "/ 480, |diff|<=2", int((abs(g[k] - c[k]) <= 2).sum()))
live = lambda m: (m >= 10) & (m >= 0.15 * n)
print("GPU live null", int(live(g["inl_null"]).sum()), " floor null", int((live(g["inl_null"]) & (g["inl_null"] >= fl)).sum()),
      " fit copies floor", round(float((live(g["inl_fit"].T).T & (g["inl_fit"] >= fl[:, None])).mean()), 4))
PY
note $?
echo "=== done $(date +%T), rc $rc"
exit $rc
