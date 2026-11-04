#!/bin/bash
# flash-attn 2.8.3.post2 from source for venvs/verl (torch 2.13.0+cu129) with the host's CUDA 12.4, sm80 only
# (research/log/2026-11-01-0010.md, 20:0x UTC).
export CUDA_HOME=/usr/local/cuda-12.4 PATH=/usr/local/cuda-12.4/bin:$FP_VENVS/verl/bin:$PATH
export FLASH_ATTN_CUDA_ARCHS=80 TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=32 NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE
export TMPDIR=${TMPDIR:-/tmp}/fa_build UV_LINK_MODE=hardlink
echo "start $(date -u '+%F %T')"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
$HOME/.local/bin/uv pip install --python $FP_VENVS/verl/bin/python --no-build-isolation -v \
  -c $ROOT/environment_installation/requirements/verl_constraints.txt "flash-attn==2.8.3.post2"
echo "exit $? $(date -u '+%F %T')"
