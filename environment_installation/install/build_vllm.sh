#!/bin/sh
# vLLM 0.12.0 environment for the gpt-oss-120b teacher, same versions as host 1's working vLLM venvs.
ROOT=$(cd "$(dirname "$0")/../.." && pwd); . "$ROOT/scripts/paths.sh"
export UV_CACHE_DIR=$HOME/.cache/uv
UV=$HOME/.local/bin/uv
V=$FP_VENVS/vllm-0.12
L=$FP_DATA_ZH/logs/build_vllm.log
echo "start $(date -u)" > $L
$UV venv --python 3.12 $V >> $L 2>&1
$UV pip install --python $V/bin/python vllm==0.12.0 >> $L 2>&1
echo "install exit $? $(date -u)" >> $L
$V/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" >> $L 2>&1
du -sh $V >> $L
echo DONE >> $L
