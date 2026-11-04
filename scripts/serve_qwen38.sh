#!/bin/sh
# Qwen3.8-27B teacher (research/log/2026-10-29-1815.md), tensor-parallel over two A6000s with vLLM 0.32; the vision
# encoder gets no inputs. Tool calls in the chat template's XML form (qwen3_coder parser), thinking split off by the qwen3
# reasoning parser; the FlashInfer sampler is off, since it compiles at start-up and this host has no build tools on PATH. P2P and custom all-reduce are off as for the gpt-oss teachers (serve_teacher.sh).
# Usage: [EXTRA='<more vllm flags>'] serve_qwen38.sh <gpu,gpu> <port> <log>
GPUS=$1; PORT=$2; LOG=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd); . "$ROOT/scripts/paths.sh"
V=$FP_VENVS/vllm-0.32
echo "start $(date -u) gpus $GPUS port $PORT pid $$" > $LOG
PATH=$V/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$GPUS NCCL_P2P_DISABLE=1 VLLM_CACHE_ROOT=$HOME/.cache/vllm HF_HOME=$HOME/.cache/huggingface \
  exec $V/bin/vllm serve $FP_MODELS_D2/Qwen3.8-27B --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port $PORT --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len ${MAXLEN:-65536} --max-num-seqs ${MAXSEQS:-32} --gpu-memory-utilization ${GPU_UTIL:-0.90} \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 $EXTRA >> $LOG 2>&1
