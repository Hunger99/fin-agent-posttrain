# configs：实验配置

按用途和阶段分目录，阶段编号与 `src/` 相同。机器路径不放这里：数据、检查点、模型、缓存和 venv 的外部路径集中在 [`environment_installation/local.env`](../environment_installation/local.env.example)（模板入库，实际值不入库），由 `scripts/paths.sh` 读入，变量表见 [environment_installation/README.md](../environment_installation/README.md#本机外部路径localenv)。

| 目录 | 放什么 | 谁读 |
| --- | --- | --- |
| `01_data-processing/prompts/` | 教师打分、文档问答出题与检查、种子题推理的 14 个提示词模板 | `src/01_data-processing/code/doc_qa*.py`、`utils/score_docs.py`、`utils/seed_reasoning.py`、`utils/doc_qa_scope.py`（经各文件的 `PROMPTS`） |
| `03_model-training/sft/` | LLaMA-Factory 的 SFT 配置：第一轮 Qwen3-4B-Base 的 `sft_4b_*`、`sft_G`，第二轮 Qwen3.5-4B-Base 的 `sft_q35_4b_*`（M1 为 `sft_q35_4b_M1_en.yaml`），吞吐测试 `speed_4b_*` | `src/03_model-training/code/train_q35.py`（Qwen3.5，`train-q35` venv）；第一轮由 `llamafactory` venv 里的 LLaMA-Factory 读 |
| `03_model-training/deepspeed/` | ZeRO-1、ZeRO-2、ZeRO-2 加优化器卸载的 DeepSpeed 配置 | SFT yaml 的 `deepspeed:` |
| `03_model-training/verl/agent_loop.yaml` | verl 的 agent loop 注册（`fin_agent_loop.FinHarnessAgentLoop`） | `scripts/03_model-training/rl/run_grpo*.sh`、`run_ppo.sh` |
| `03_model-training/grpo/` | 链路的可调参数：`rl_chain_c.conf`（DYN、LR、MINI 等；不存在时用默认值）和说明格式的 `rl_chain_c.conf.example` | `scripts/rl_chain_c.sh` |

**SFT yaml 里的绝对路径：** LLaMA-Factory 只读字面值，所以 `model_name_or_path`、`tokenized_path`、`dataset_dir`、`output_dir` 写成 `${FP_MODELS_D1}`、`${FP_DATA_EN}`、`${FP_RUNS}` 等 `local.env` 变量，`deepspeed:` 写 `${FP_REPO}/configs/03_model-training/deepspeed/<名>.json`。LLaMA-Factory 不展开变量，所以运行前先 `. environment_installation/local.env && export FP_REPO=$(pwd)`，再用 `envsubst < <配置>.yaml > <临时目录>/<配置>.yaml` 渲染出实际路径，把渲染后的文件交给训练脚本。

**运行时实际用的配置：** GRPO 每次运行由 hydra 记下组合后的完整配置，归档在 `outputs/archive/runs/<运行>/provenance/hydra/`；`DRY=1` 空跑时打印到 `outputs/archive/runs/rl-chain-c-dry--no-data--*/<EXP>_DYN<n>.yaml`。
