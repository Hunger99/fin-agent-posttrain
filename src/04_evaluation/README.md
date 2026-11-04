# 04_evaluation：领域分、通用分、harness 成功率与配对比较

- **职责：** 给检查点打分并做模型间的配对比较。
- **输入：** 检查点（HF 格式；verl 检查点由 `scripts/04_evaluation/eval_ckpt_c.sh` 先合并）、测试与开发文件、harness 的运行目录。
- **输出（都在仓库外）：** 领域分与通用分的逐题结果，harness 运行的比较 JSON（`FP_DATA_D2/harness_v2/hm3_*.json`、`FP_RUNS/rl/eval_<名>.compare.*`），严格重判结果（仓库内 `outputs/archive/runs/strict-regrade--hardset_v1--20261102T2246Z/`）。
- **`code/`：** `num_score.py`（数值答案的判对规则）、`strict_grade.py`（严格判分，经 harness v3 的判分）、`finsearchcomp/`（FinSearchComp 外部测试集的构建）。
- **`utils/`：** `num_eval.py`、`mc_eval.py`、`mc_generate.py`（领域分和 MMLU）、`lmeval_items.py`（lm-evaluation-harness 输出转逐题）、`paired_compare.py`、`paired_runs.py`、`hm3.py`、`hm5.py`、`did_reworded.py`（配对比较）、`regrade_strict.py`（重判已有运行）、`harness_report.py`，以及中文阶段的 `model_v4.py`、`tokenizer_v4.py`、`train_chat_template.py`、`select_start_model.py`；`rescore_domain_en.py` 是改写过领域分结果的一次性步骤。
- **入口：** 启动器在 `scripts/04_evaluation/`：`eval_ckpt_c.sh <检查点> <名>`（行情、FinSearchComp、DocFinQA、自建环境）、`regrade_run.sh <名>`、`general_c.sh`（通用能力）、`judge_c.sh`、`reworded_c.sh`。
