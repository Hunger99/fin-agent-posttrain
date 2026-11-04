#!/bin/bash
# teacher_hard.sh's hard-tier collection, restarted (research/log/2026-10-31-0150.md, 19:3x UTC) with harness/run.py
# version 4, which hands each episode to the teacher server with the fewest in flight, at 40 in flight instead of 48:
# both runs of the 7 templates other than div_event_study first, then div_event_study's second run (neither
# composition uses that template), into the same results; then the method judge on the passed episodes as before.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
MK=$FP_DATA_D2/market_v1; LOG=$MK/runs/teacher_hard.status
V3=$ROOT/src/02_harness/code/v3; PY=$FP_VENVS/vllm-0.12/bin/python
J=$ROOT/src/03_model-training/code/judge_sft.py; export PYTHONDONTWRITEBYTECODE=1
U1=http://127.0.0.1:18011/v1; U2=http://127.0.0.1:18012/v1; OUT=$MK/runs/train_hard_q38_H1
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
grep -v '"template": "div_event_study"' $MK/tasks/market_tasks_hard_train.jsonl > $MK/runs/hard_train_7.jsonl
grep '"template": "div_event_study"' $MK/tasks/market_tasks_hard_train.jsonl > $MK/runs/hard_train_des.jsonl
say "teacher_hard2 starts: $(wc -l < $MK/runs/hard_train_7.jsonl) tasks of 7 templates, then $(wc -l < $MK/runs/hard_train_des.jsonl) div_event_study"
for T in $MK/runs/hard_train_7.jsonl $MK/runs/hard_train_des.jsonl; do
  $PY -B $V3/run.py --tasks $T --db $MK/market_v1.sqlite --config H1 --runs 2 --concurrency 40 \
    --provider qwen --model qwen3.8-27b --base-url $U1 --base-url $U2 --out $OUT >> $MK/runs/train_hard_q38_H1.log 2>&1
  say "hard train ($(basename $T)) exit $?: $(wc -l < $OUT/results.jsonl) rows"
done
$PY -B $J --source hard:$OUT:$MK/tasks/market_tasks_hard_train.jsonl --out $MK/runs/judge_hard_train.jsonl \
  --base-url $U1 --base-url $U2 --model qwen3.8-27b --concurrency 64 --only-passed >> $MK/runs/judge_hard.log 2>&1
say "judge exit $?: $(wc -l < $MK/runs/judge_hard_train.jsonl) rows"
say "teacher_hard done"
