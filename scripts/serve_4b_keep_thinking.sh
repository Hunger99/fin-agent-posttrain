#!/bin/sh
# A 4B Qwen3.5 model for harness v3: serve_4b.sh (research/log/2026-10-30-0410.md) with every flag the same except the
# chat template, which is qwen35_keep_thinking.jinja next to this script: Qwen3.5's own template, except that every
# assistant turn after the first user message keeps its thinking, so v3's nudge (a user message after a reply without a
# tool call) no longer drops the earlier turns' thinking and the token prefix holds (docs/03_harness.md, 9).
# GENCFG, when set, is a folder whose generation_config.json vLLM reads instead of the model's: for
# Qwen3.5-4B-Base, base_stop/ next to this script adds <|im_end|> to its end tokens, as the SFT checkpoints have
# (research/log/2026-10-31-0150.md).
# Usage: serve_4b_keep_thinking.sh <gpu> <port> <model dir> <served name> <log>
GPU=$1; PORT=$2; MODEL=$3; NAME=$4; LOG=$5
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); . "$ROOT/scripts/paths.sh"
V=$FP_VENVS/vllm-0.32; HERE=$ROOT/src/02_harness/code/v3/templates
echo "start $(date -u) gpu $GPU port $PORT model $MODEL pid $$ template $HERE/qwen35_keep_thinking.jinja" > $LOG
PATH=$V/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$GPU VLLM_CACHE_ROOT=$HOME/.cache/vllm HF_HOME=$HOME/.cache/huggingface \
  exec $V/bin/vllm serve $MODEL --served-model-name $NAME --host 127.0.0.1 --port $PORT \
  --max-model-len 65536 --max-num-seqs ${MAXSEQS:-32} --gpu-memory-utilization 0.85 --chat-template $HERE/qwen35_keep_thinking.jinja ${GENCFG:+--generation-config $GENCFG} \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 >> $LOG 2>&1
