#!/bin/bash
# New inference environment for Qwen3.5/Qwen3.8 (research/log/2026-10-29-1815.md): vLLM 0.32.0 built for CUDA 12.9, which
# the 550 driver (CUDA 12.4) runs through CUDA 12 minor-version compatibility, as vLLM 0.12 with CUDA 12.8 already does.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
UV=$HOME/.local/bin/uv; V=$FP_VENVS/vllm-0.32
date -u '+start %F %T'
$UV venv $V --python 3.12 &&
$UV pip install --python $V/bin/python \
  "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
echo "exit $?"
$V/bin/python -c "import torch, vllm; print('torch', torch.__version__, torch.version.cuda, 'cuda ok', torch.cuda.is_available(), 'vllm', vllm.__version__)"
date -u '+end %F %T'
