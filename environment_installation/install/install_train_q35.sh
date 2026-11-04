#!/bin/bash
# Training environment for Qwen3.5 (research/log/2026-10-29-1815.md): LLaMA-Factory 0.9.5 as in venvs/llamafactory, but
# transformers 5.20.2, whose Qwen3.5 linear-attention layers take packed-sequence boundaries (cu_seq_lens_q), with the
# kernels that honour them (flash-linear-attention, causal-conv1d); torch 2.8 (CUDA 12.8) because fla needs torch >= 2.7.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); . "$ROOT/scripts/paths.sh"
UV=$HOME/.local/bin/uv; V=$FP_VENVS/train-q35; P="--python $V/bin/python"
date -u '+start %F %T'
set -e
$UV venv $V --python 3.11
$UV pip install $P torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
$UV pip install $P -r $ROOT/environment_installation/requirements/train_q35_base_reqs.txt
$UV pip install $P transformers==5.20.2 deepspeed==0.16.9 fla-core==0.6.0 flash-linear-attention==0.6.0 \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
$UV pip install $P --no-deps llamafactory==0.9.5
$V/bin/python -c "import torch, transformers, flash_attn, fla, causal_conv1d, deepspeed, llamafactory; print('torch', torch.__version__, torch.version.cuda, 'transformers', transformers.__version__, 'fla', fla.__version__, 'llamafactory', llamafactory.__version__, 'cuda', torch.cuda.is_available())"
date -u '+end %F %T'
