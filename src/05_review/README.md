# 05_review：跨阶段的回归测试

- **职责：** 放不属于单个阶段的回归测试。
- **`tests/test_strict_grade.py`：** 检查 `src/04_evaluation/code/strict_grade.py`，它经 harness v3 的判分（02 阶段）读已有的行情运行。运行命令见下面的“跑全部回归”（只读数据，约 5 分钟）。
- **`tests/rl_chain/`：** 检查 `scripts/` 下的链路和取卡库：`test_conf.sh <工作目录>`（链路的配置读取和阶段判断，不碰 GPU）；`test_handover.sh`、`test_stop.sh`（用桩 nvidia-smi 和临时锁目录检查取卡、交接和停止；会调用 GPU-claim 工具，需 `local.env` 的 `GPU_CLAIM_PY`、`GPU_CLAIM_TESTS`，临时目录在 `${TEST_TMP:-${TMPDIR:-/tmp}}`）。
- **输入、输出：** 只读仓库代码和外部数据，临时文件写到临时目录，不写仓库。
- **不在这里的测试：** harness v3 自己的测试留在 `src/02_harness/code/v3/tests/`（它们以 `tests` 包 import，移出会断）；03 阶段只测本阶段工具的测试在 `src/03_model-training/tests/`。

## 跑全部回归

在仓库根执行；只用 CPU，临时文件写到 `$TMPDIR`。各测试用哪个 venv 与迁移前相同。

```bash
. environment_installation/local.env
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=${TMPDIR:-/tmp}/pycache
( cd src/02_harness/code/v3 && $FP_VENVS/vllm-0.12/bin/python -B -m unittest discover -v -s tests -t . )
$FP_VENVS/train-q35/bin/python -B src/03_model-training/tests/test_builders.py
$FP_VENVS/verl/bin/python -B src/03_model-training/tests/test_dyn_filter.py
$FP_VENVS/verl/bin/python -B src/03_model-training/tests/test_form_errors.py
$FP_VENVS/verl/bin/python -B src/03_model-training/tests/test_subsample.py "$(mktemp -d)"
$FP_VENVS/vllm-0.12/bin/python -B src/05_review/tests/test_strict_grade.py
bash src/05_review/tests/rl_chain/test_conf.sh "$(mktemp -d)"
# Calls the GPU-claim tool with a stub nvidia-smi and a temporary lock directory:
TEST_TMP=$(mktemp -d) bash src/05_review/tests/rl_chain/test_handover.sh
TEST_TMP=$(mktemp -d) bash src/05_review/tests/rl_chain/test_stop.sh
```

harness v3 的 `test_template.py` 在迁移前就有 3 项报错（`rl_data.py` 的 `norm` 被单独取出时缺 `call_args`），其余应全部通过。`src/02_harness/code/v3/tests/regress_golden.py` 不在这组里：它要先用 harness v2 的标准解生成对照行，再在完整题集上跑，用法见它的 docstring。
