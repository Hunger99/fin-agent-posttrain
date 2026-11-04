# 03_model-training：agent SFT 数据、SFT 与 GRPO

- **职责：** 把教师轨迹过滤、打包成 agent SFT 数据，训练 SFT 模型（M1 等），再在 M1 上做 GRPO。
- **输入：** 混合数据和教师轨迹（`FP_DATA_D2/market_v1/runs/train_*_q38_H1` 等）、`configs/03_model-training/` 的 SFT yaml 和 verl 配置、GRPO 的 parquet 数据（`FP_DATA_D2/grpo/`）。
- **输出（都在仓库外）：** agent SFT 数据（`FP_DATA_D2/agent_sft/`）、SFT 检查点（`FP_RUNS/qwen3.5-4b-base-en/M1` 等）、GRPO 检查点与 `metrics.jsonl`（`FP_RUNS/rl/<运行名>/`）。
- **`code/`：** `filter_sft.py`（轨迹的多重过滤）、`build_agent_sft_v3.py`（按 v3 模板编码和打包）、`form_errors.py`、`judge_sft.py`、`judge_cal_set.py`、`train_q35.py`（Qwen3.5 的 SFT 训练，用 `train-q35` venv）、`verl_grpo/`（GRPO 的 agent loop `fin_agent_loop.py` 和丢组用的采样器 `dyn_filter.py`，verl 按路径加载）。
- **`utils/`：** `make_grpo_data.py`（筛题结果转成 GRPO parquet）、`subsample_grpo_data.py`（按模板分层抽到 939 题）、`rl_data.py`、`build_agent_sft.py`（v2 轨迹）、`build_dpo_pairs.py`、`judge_calibration.py`、`fetch_models.py`、`rl_health.py`、`mfu_offline.py`。
- **`tests/`：** `test_builders.py`（`train-q35` venv）、`test_dyn_filter.py`（`verl` venv）、`test_form_errors.py`（只用标准库）、`test_subsample.py <工作目录>`（`verl` venv）。
- **入口：** 启动器在 `scripts/03_model-training/`：SFT 为 `sft/chain_m1b.sh`、`chain_m1c.sh`（M1 时期，GPU 号写死）；GRPO 为 `rl/grpo2.sh`，它调用 `rl/run_grpo2.sh`，由 `scripts/rl_chain_c.sh` 按阶段调用。
