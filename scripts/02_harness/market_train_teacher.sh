#!/bin/sh
# The teacher on the market train set (3,400 tasks in seeded shuffled order, v3, H1, one run each)
# (research/log/2026-10-31-0150.md). Resumable: run it again and it skips the episodes in results.jsonl.
ROOT=$(cd "$(dirname "$0")/../.." && pwd); . "$ROOT/scripts/paths.sh"
M=$FP_DATA_D2/market_v1
V3=$ROOT/src/02_harness/code/v3
PY=$FP_VENVS/vllm-0.12/bin/python
ST=$M/runs/market_train_teacher.status
say() { echo "$* $(date -u '+%F %T')" >> $ST; }
curl -s -m 10 http://127.0.0.1:18011/v1/models | grep -q qwen3.8 || { say "teacher 18011 not up: not started"; exit 1; }
say "teacher market train starts"
$PY -B $V3/run.py --tasks $M/tasks/market_tasks_train_shuffled.jsonl --db $M/market_v1.sqlite --config H1 --runs 1 \
  --concurrency 32 --provider qwen --model qwen3.8-27b --base-url http://127.0.0.1:18011/v1 \
  --out $M/runs/train_q38_H1 >> $M/runs/train_q38_H1.log 2>&1
say "teacher market train exit $?"
say "market_train_teacher done"
