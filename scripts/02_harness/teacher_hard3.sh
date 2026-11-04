#!/bin/bash
# Composition 甲 (research/log/2026-10-31-0150.md, 23:4x UTC): once teacher_hard2.sh has finished the 7 templates, stop
# it before div_event_study's second run, then the teacher on the 700 new hard-tier train tasks (2 runs each, same
# settings and output as teacher_hard2.sh), then the method judge on every passed hard-tier episode.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
MK=$FP_DATA_D2/market_v1; LOG=$MK/runs/teacher_hard.status
V3=$ROOT/src/02_harness/code/v3; PY=$FP_VENVS/vllm-0.12/bin/python
J=$ROOT/src/03_model-training/code/judge_sft.py; export PYTHONDONTWRITEBYTECODE=1
U1=http://127.0.0.1:18011/v1; U2=http://127.0.0.1:18012/v1; OUT=$MK/runs/train_hard_q38_H1
X=$MK/tasks/market_tasks_hard_train_extra.jsonl; T2=3234265
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
[ -f $X ] || cp $MK/tasks/_hard_extra_candidate/market_tasks_hard_train_extra.jsonl $X
until grep -q "hard train (hard_train_7.jsonl) exit" $LOG || ! kill -0 $T2 2>/dev/null; do sleep 20; done
kill -- -$T2 2>/dev/null; sleep 10; kill -9 -- -$T2 2>/dev/null
say "teacher_hard3: teacher_hard2 stopped before div_event_study run 2; $(wc -l < $X) new tasks, sha256 $(sha256sum $X | cut -c1-12)"
$PY -B $V3/run.py --tasks $X --db $MK/market_v1.sqlite --config H1 --runs 2 --concurrency 40 \
  --provider qwen --model qwen3.8-27b --base-url $U1 --base-url $U2 --out $OUT >> $MK/runs/train_hard_q38_H1.log 2>&1
say "hard train (extra) exit $?: $(wc -l < $OUT/results.jsonl) rows"
$PY -B $J --source hard:$OUT:$MK/tasks/market_tasks_hard_train.jsonl+$X --out $MK/runs/judge_hard_train.jsonl \
  --base-url $U1 --base-url $U2 --model qwen3.8-27b --concurrency 64 --only-passed >> $MK/runs/judge_hard.log 2>&1
say "judge exit $?: $(wc -l < $MK/runs/judge_hard_train.jsonl) rows"
say "teacher_hard done"
