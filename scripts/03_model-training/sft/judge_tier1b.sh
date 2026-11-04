#!/bin/bash
# judge_tier1.sh over both teacher servers (research/log/2026-10-31-0150.md, 15:3x UTC): on 18012 alone the judge's 32
# requests filled that server while 18011 sat nearly idle. The tier-1 collection has ended, so one pass judges what
# is left (judge_sft.py skips the episodes it has judged).
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
MK=$FP_DATA_D2/market_v1; LOG=$MK/runs/judge_tier1.status
PY=$FP_VENVS/vllm-0.12/bin/python; J=$ROOT/src/03_model-training/code/judge_sft.py
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
say "judging the rest over 18011 and 18012"
$PY -B $J --source market:$MK/runs/train_q38_H1:$MK/tasks/market_tasks_train_shuffled.jsonl \
  --out $MK/runs/judge_market_train.jsonl --base-url http://127.0.0.1:18011/v1 --base-url http://127.0.0.1:18012/v1 \
  --model qwen3.8-27b --concurrency 32 --only-passed >> $MK/runs/judge_tier1.log 2>&1
say "judge exit $?: $(wc -l < $MK/runs/judge_market_train.jsonl) rows"
say "judge_tier1 done"
