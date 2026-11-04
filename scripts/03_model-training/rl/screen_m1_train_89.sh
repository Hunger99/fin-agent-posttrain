#!/bin/bash
# M1 on the new dataset's 2,350 training tasks, twice each (research/log/2026-11-01-0010.md, 20:0x UTC), to pick the
# GRPO tasks by how often M1 solves them: a server on each of the GPUs in SCREEN_GPUS (default 8 9) (free and unlocked), run.py under H1 with 30
# turns, resuming into runs/screen_m1_train; then the servers stop and the locks go.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN; P=$FP_DATA_D2/harness_v2/pilot
MK=$FP_DATA_D2/market_v1; V3=$ROOT/src/02_harness/code/v3
R=$FP_RUNS/qwen3.5-4b-base-en; PY=$FP_VENVS/vllm-0.12/bin/python
export PYTHONDONTWRITEBYTECODE=1; LOG=$MK/runs/screen_m1_train.status; OUT=$MK/runs/screen_m1_train; K=24
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
free_gpu() { U=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $U)" = 0 ]; }
slog() { echo $EN/logs/serve_4b_screenm1_gpu$1.log; }
G=""; urls=""
for x in ${SCREEN_GPUS:-8 9}; do
  free_gpu $x && mkdir $P/locks/gpu$x 2>/dev/null || { say "GPU $x busy or locked: skipped"; continue; }
  ( MAXSEQS=$K setsid sh $ROOT/scripts/serve_4b_keep_thinking.sh $x 1816$x $R/M1 M1 $(slog $x) < /dev/null > /dev/null 2>&1 & )
  G="$G $x"
done
n=0
for x in $G; do
  for i in $(seq 90); do curl -s -m 5 http://127.0.0.1:1816$x/v1/models | grep -q '"M1"' && break; sleep 10; done
  if curl -s -m 5 http://127.0.0.1:1816$x/v1/models | grep -q '"M1"'; then urls="$urls --base-url http://127.0.0.1:1816$x/v1"; n=$((n + 1))
  else say "server on GPU $x did not start"; fi
done
say "screen_m1_train starts on $n servers (GPUs$G)"
[ $n -gt 0 ] && $PY -B $V3/run.py --tasks $MK/hardset_v1/train_market.jsonl --db $MK/market_v1.sqlite --config H1 --runs 2 \
  --max-steps 30 --concurrency $((K * n)) --provider qwen --model M1 $urls --out $OUT >> $OUT.log 2>&1
say "run.py exit $?: $(wc -l < $OUT/results.jsonl 2>/dev/null) rows"
ps=""; for x in $G; do p=$(head -n 1 $(slog $x) | grep -o "pid [0-9]*" | cut -d' ' -f2); [ -n "$p" ] && { kill -- -$p 2>/dev/null; ps="$ps $p"; }; done
sleep 20; for p in $ps; do kill -9 -- -$p 2>/dev/null; done
for x in $G; do rmdir $P/locks/gpu$x 2>/dev/null; done
say "screen_m1_train done"
