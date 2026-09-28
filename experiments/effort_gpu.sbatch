#!/bin/bash
#SBATCH --job-name=glint_effort_gpu
#SBATCH --account=lcls:prjdat21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --exclusive
#SBATCH --exclude=sdfampere042,sdfampere003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
# feat/adaptive-effort (StreamDriver effort=: the known-cell depth follows the hit rate, deep search on the misses).
# Results of job 39344280 (28 Sep 2026, sdfampere013): experiments/RESULTS_stream_effort_a100.md. Submit from the directory the
# .out should land in; needs ~/q480_fix.txt (the published 480 q list) and a checkout of the branch at $GLINT_WT.
# 1) the new suite and the hit-ring suite (device path: the fast-path signature changed) on the node;
# 2) the published 480 streaming arm, effort OFF, gated 333/480, 10 watchdog rescues, 1 relock (default path unchanged);
# 3) effort ON at three budgets: rate_hz 35000 (0.03 ms/hit -> tier 0, explicit shipped depth: expected identical),
#    2000 (0.5 ms/hit -> tier 2), 120 (8.3 ms/hit -> tier 3 + deep search on misses). Reported, not gated. The replay runs
#    at B=20, so these check the MECHANISM (tier choice, logging, escalation accounting), not the B=120 cost table.
# 4) the scrambled-null arms: the real 480 first (the driver locks on them), then the same 480 with every peak rotated by a
#    random azimuth (glint.multilattice.scramble_azimuth, seeded per frame) as a second species "scr". Its `indexed` count is
#    the fast path's chance-accept rate under the live gate at that tier, its `escalated` count the deep search's. OFF,
#    35000, 2000, and 120 with n_gpu=2 (so the deep search stays on through the all-miss tail: budget 16.7 ms).
# Rerun note: job 39344280 generated q480_scrambled.txt inline (same code and seed); from the next commit the arm calls the
# committed experiments/scramble_qlist.py, so the whole job reruns from the repository.
# Exclusive: job 39294566 on shared sdfampere003 got cudaErrorDevicesUnavailable on every GPU step.
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh >/dev/null 2>&1
conda activate ana-4.0.58-py3-minipytorch >/dev/null 2>&1
export PYTHONPATH=.
O=$HOME/glint_esc_outputs/effort
mkdir -p $O
cd ${GLINT_WT:-$HOME/glint_effort_wt}
# EXPECT_HEAD=<short sha> pins the checkout the job must run on (job 39344280 ran on 70f5c61); unset = run whatever is checked out
test -z "$EXPECT_HEAD" || test "$(git rev-parse --short HEAD)" = "$EXPECT_HEAD" || { echo "wrong HEAD $(git rev-parse --short HEAD), expected $EXPECT_HEAD"; exit 1; }
echo "host $(hostname)  branch $(git rev-parse --short HEAD)  $(date +%T)"
nvidia-smi --query-gpu=index,name,compute_mode,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import cupy, torch; print('cupy', cupy.__version__, 'torch', torch.__version__)"
echo "=== test_stream_effort (numpy path)"; python experiments/test_stream_effort.py; echo "exit $?"
echo "=== test_stream_hit_ring, device path"; GLINT_TEST_GPU=1 python experiments/test_stream_hit_ring.py | tail -1; echo "exit ${PIPESTATUS[0]}"
echo "=== test_stream_hit_ring, numpy path"; python experiments/test_stream_hit_ring.py | tail -1; echo "exit ${PIPESTATUS[0]}"
COMMON="--input lyso=$HOME/q480_fix.txt --B 20 --dmin 2.0 --warmup-rescue --adaptive-relock --min-inliers 10"
echo "=== 480 arm, effort OFF, gated  $(date +%T)"
python experiments/record_stream_replay.py $COMMON --out $O/replay_480_off.json --expect 333/480 --expect-wresc 10 --expect-relock 1 \
    2>&1 | tail -6; echo "exit ${PIPESTATUS[0]}"
for R in 35000 2000 120; do
  echo "=== 480 arm, effort ON rate_hz=$R n_gpu=1  $(date +%T)"
  python experiments/record_stream_replay.py $COMMON --effort "{\"rate_hz\": $R, \"n_gpu\": 1}" --out $O/replay_480_effort_$R.json \
      2>&1 | tail -6; echo "exit ${PIPESTATUS[0]}"
done
echo "=== scrambled 480 for the null arms  $(date +%T)"
python experiments/scramble_qlist.py $HOME/q480_fix.txt $O/q480_scrambled.txt --seed 20260928; echo "exit $?"
SCR="--input lyso=$HOME/q480_fix.txt --input scr=$O/q480_scrambled.txt --ref scr=79.02,79.02,37.98,90,90,90 --B 20 --dmin 2.0 --warmup-rescue --adaptive-relock --min-inliers 10"
echo "=== scrambled arm, effort OFF  $(date +%T)"
python experiments/record_stream_replay.py $SCR --out $O/replay_scr_off.json 2>&1 | tail -6; echo "exit ${PIPESTATUS[0]}"
for R in 35000 2000; do
  echo "=== scrambled arm, effort ON rate_hz=$R n_gpu=1  $(date +%T)"
  python experiments/record_stream_replay.py $SCR --effort "{\"rate_hz\": $R, \"n_gpu\": 1}" --out $O/replay_scr_effort_$R.json \
      2>&1 | tail -6; echo "exit ${PIPESTATUS[0]}"
done
echo "=== scrambled arm, effort ON rate_hz=120 n_gpu=2  $(date +%T)"
python experiments/record_stream_replay.py $SCR --effort '{"rate_hz": 120, "n_gpu": 2}' --out $O/replay_scr_effort_120.json \
    2>&1 | tail -6; echo "exit ${PIPESTATUS[0]}"
echo "=== SUMMARY"
python - <<'PY'
import json, glob, os
O = os.path.expanduser("~/glint_esc_outputs/effort")
for f in sorted(glob.glob(f"{O}/replay_*.json")):
    h = json.load(open(f))["header"]; c = h.get("counters", {}); e = c.get("effort"); t = h.get("totals", {})
    line = (f"{os.path.basename(f):30s} strict {t.get('strict_ok')}/{t.get('n')} indexed {c.get('indexed')} wresc {c.get('n_watchdog_rescued')} "
            f"relock {c.get('n_relock')} gate_rej {c.get('gate_rejected')} elapsed {h.get('elapsed_s')}s")
    for sp, v in (h.get("by_species") or {}).items():
        line += f"\n    {sp:5s} n {v['n']} indexed {v['indexed']} escalated {v.get('escalated')} strict {v['ok']} miss {v['miss']} warmup {v['warmup']} rescued {v['rescued']}"
    if e:
        line += (f"\n    effort: tier {e['tier']} deep {e['deep']} budget {e['budget_ms']:.3f} ms hit_est {e['hit_est']:.3f} miss {e['miss_frac']:.3f}"
                 f" | fast_by_tier {e['fast_by_tier']} deep_tried {e['n_deep_tried']} escalated {e['n_escalated']} searches {e['deep_searches']} changes {e['n_changes']}")
        line += "\n    log: " + "; ".join(f"at {l['at']}: T{l['tier']}{'+deep' if l['deep'] else ''} ({l['budget_ms']} ms, hit {l['hit_est']}, miss {l['miss_frac']})" for l in e["log"])
    print(line)
PY
echo "done $(date +%T)"
