# scripts/04_evaluation：评测启动器

| 脚本 | 做什么 | 状态 |
| --- | --- | --- |
| `eval_ckpt_c.sh <检查点> <名>` | 合并 verl 检查点，检查配置，在行情、FinSearchComp、DocFinQA 测试集和开发集（每题 4 次）、自建环境测试集上评测，写配对比较 | 现行（链路 5a） |
| `regrade_run.sh <名>` | 用原判分和严格判分重判一个模型的行情运行，写到 `STRICT_DIR/<名>/` | 现行（链路 5a） |
| `general_c.sh`、`eval_general.sh` | 通用能力：MMLU、GSM8K、IFEval；`eval_general.sh` 生成每个模型的 `jobs.sh` | 现行（链路 5b） |
| `judge_c.sh` | 用 Qwen3.8-27B 做 judge 校准，状态写到 `CHAIN_DIR/judge_cal/`；实现 `judge_calibration.py` 在 `src/03_model-training/utils/`，因为被校准的是 03 阶段筛 SFT 轨迹用的 judge（`judge_sft.py`），启动器放在这里是因为它是链路评测步的一部分 | 现行（链路 5d） |
| `reworded_c.sh`、`compare_reworded.sh`、`compare_reworded_rl.sh` | 换措辞测试集上的评测和比较 | 现行（链路 5c） |
| `after_m1.sh`、`base_test_stop.sh` | M1 训完后四个模型在新数据集测试文件上的评测；起点模型以 `<\|im_end\|>` 为结束符重跑自建环境测试集 | 一次性，GPU 号写死 |

`DRY=1` 时现行脚本只打印命令，输出写到 `DRY_DIR`。
