#!/bin/bash
# Qwen3.5-4B-Base on the wealth and ML test set again, with <|im_end|> as an end token (research/log/2026-10-31-0150.md,
# 16:3x UTC): chain_base_test.sh's run() with the server reading base_stop/generation_config.json (GENCFG) and a new
# arm name, qwen3.5-4b-base_H1_stop; the first rows stay where they are. It takes GPU 4 as soon as the official
# Qwen3.5-4B's dev runs in screen_chain3c.sh free it (that chain looks for a free GPU only every 5 minutes and takes
# GPU 4 back after this ends), then the paired comparisons of hm3.py.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
EN=$FP_DATA_EN
O=$FP_DATA_D2/harness_v2; H=$EN/harness; P=$O/pilot
MK=$FP_DATA_D2/market_v1/runs
PY=$FP_VENVS/vllm-0.12/bin/python; LOG=$P/base_test_stop.status; export PYTHONDONTWRITEBYTECODE=1
export GENCFG=$ROOT/src/02_harness/code/v3/templates/base_stop
BASE=$FP_MODELS_D2/Qwen3.5-4B-Base; M=qwen3.5-4b-base; A=${M}_H1_stop
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
free_gpu() { U=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $U)" = 0 ]; }
up() { for i in $(seq 90); do curl -s -m 5 http://127.0.0.1:$1/v1/models | grep -q "\"$2\"" && return 0; sleep 10; done; return 1; }
stop_srv() { p=$(head -n 1 $1 | grep -o "pid [0-9]*" | cut -d' ' -f2); [ -n "$p" ] && { kill -- -$p 2>/dev/null; sleep 20; kill -9 -- -$p 2>/dev/null; }; }
run() {  # <part> <concurrency> <port>: as chain_base_test.sh, two passes
  local part=$1 k=$2 pass T
  if [ $part = wealth ]; then T="--tasks $H/tasks_test.jsonl --db $H/C.sqlite"; else T="--tasks $H/tasks_ml_test.jsonl --db none"; fi
  for pass in 1 2; do
    $PY -B $O/code_collect/run.py $T --concurrency $k --runs 1 --config H1 --max-steps 100 --keep-results 10 --summary-chars 1000 \
      --provider qwen --model $M --base-url http://127.0.0.1:$3/v1 --out $O/runs_q4b_test/${A}_$part >> $O/logs/q4bt_${A}_$part.log 2>&1
    echo "pass $pass exit $?" >> $O/logs/q4bt_${A}_$part.log
  done
}
say "waiting for the official model's dev runs to free GPU 4"
until grep -q "official docfinqa dev exit" $MK/screen_chain3.status; do sleep 60; done
until free_gpu 4 && mkdir $P/locks/gpu4 2>/dev/null; do sleep 5; done
SL=$EN/logs/serve_4b_basetest_${A}_gpu4.log
( MAXSEQS=96 setsid sh $ROOT/scripts/serve_4b.sh 4 18104 $BASE $M $SL < /dev/null > /dev/null 2>&1 & )
if up 18104 $M; then
  say "base model H1 test (end token fixed) starts on GPU 4"
  run wealth 64 18104 & w=$!; run ml 12 18104; wait $w
  say "base model H1 test done: $(wc -l < $O/runs_q4b_test/${A}_wealth/results.jsonl) wealth rows, $(wc -l < $O/runs_q4b_test/${A}_ml/results.jsonl) ml rows"
else say "base model server did not start"; fi
stop_srv $SL; rmdir $P/locks/gpu4
$PY -B $ROOT/src/04_evaluation/utils/hm3.py $O/runs_q4b_test $H/tasks_test.jsonl $H/tasks_ml_test.jsonl ${A}:A1_H1 ${A}:A1-base_H1 ${A}:r3q_H1 ${A}:qwen3.5-4b_H1 > $O/hm3_four_arms_stop.json 2>&1
say "comparisons computed: $O/hm3_four_arms_stop.json; base_test_stop done"
