#!/bin/sh
# A 4B Qwen3.5 model in the harness (research/log/2026-10-30-0410.md): one A6000, Qwen3.5's own chat template (the one
# Qwen3.5-4B-Base and Qwen3.5-4B ship, with tools, XML tool calls and thinking), the teacher's tool-call and reasoning
# parsers. GENCFG, when set, is a folder whose generation_config.json vLLM reads instead of the model's
# (src/02_harness/code/v3/templates/base_stop/ for Qwen3.5-4B-Base). Usage: serve_4b.sh <gpu> <port> <model dir> <served name> <log>
GPU=$1; PORT=$2; MODEL=$3; NAME=$4; LOG=$5
ROOT=$(cd "$(dirname "$0")/.." && pwd); . "$ROOT/scripts/paths.sh"
V=$FP_VENVS/vllm-0.32
echo "start $(date -u) gpu $GPU port $PORT model $MODEL pid $$" > $LOG
PATH=$V/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$GPU VLLM_CACHE_ROOT=$HOME/.cache/vllm HF_HOME=$HOME/.cache/huggingface \
  exec $V/bin/vllm serve $MODEL --served-model-name $NAME --host 127.0.0.1 --port $PORT \
  --max-model-len 65536 --max-num-seqs ${MAXSEQS:-32} --gpu-memory-utilization 0.85 --chat-template $ROOT/src/02_harness/code/v3/templates/q35_native.jinja ${GENCFG:+--generation-config $GENCFG} \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 >> $LOG 2>&1
