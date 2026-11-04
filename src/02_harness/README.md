# 02_harness：agent 循环与任务环境

- **职责：** 让模型在任务环境里多步调用工具完成任务，并按程序判分。现行版本是 v3；旧版 v1、v2 留在 `code/legacy/`，因为项目日志和 `src/04_evaluation/utils/harness_report.py` 还引用它们的运行。
- **输入：** 任务文件（jsonl）、环境数据库（财富管理的 `A/B/C.sqlite`、行情的 `market_v1.sqlite`）、一个或多个 vLLM 推理服务的地址。
- **输出：** 每个任务的 `results.jsonl`（判分）和 `trajectories.jsonl`（轨迹），写到 `--out` 给的目录（评测和教师采集都在 `FP_DATA_D2/market_v1/runs/` 下）。
- **`code/v3/`：** 整体移动的单元，内部相对路径不变：`harness/` 包（循环、环境、判分），入口 `run.py`（跑任务）、`golden.py`、`collect.py`、`envtool.py`（生成和检查任务与数据库）；`templates/` 是推理服务用的对话模板（`qwen35_keep_thinking.jinja`、`q35_native.jinja`）和起点模型的 `base_stop/generation_config.json`；`tests/` 是单元测试。
- **被谁 import：** GRPO 的 agent loop（`src/03_model-training/code/verl_grpo/fin_agent_loop.py`，经 `FIN_HARNESS_V3`）、`src/04_evaluation/code/strict_grade.py` 和 `src/03_model-training/code/judge_cal_set.py`。
- **入口：** `python -B src/02_harness/code/v3/run.py -h`；测试 `python -B -m unittest discover -s src/02_harness/code/v3/tests -t src/02_harness/code/v3`（用 `vllm-0.12` 的 venv）。启动推理服务的脚本在 `scripts/serve_*.sh`。
