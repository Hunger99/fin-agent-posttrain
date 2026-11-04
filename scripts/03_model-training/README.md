# scripts/03_model-training：SFT 与 RL 的启动器

## `rl/`：GRPO 与 PPO

| 脚本 | 做什么 | 状态 |
| --- | --- | --- |
| `run_grpo2.sh` | 一次 verl GRPO 运行（带可选的丢组采样，`DYN=1`）；`HERE` 指向 `src/03_model-training/code/verl_grpo` | 现行（grpo2–grpo4） |
| `grpo2.sh` | grpo2/3/4 的启动器：取卡、余量检查、调用 `run_grpo2.sh`、失败时为重启保留 GPU、`stop` 子命令；`DRY=1` 时把 `run_grpo2.sh` 复制到空跑目录，用 `py_cfg_job.sh` 只打印配置 | 现行 |
| `screen_c.sh` | M1 在 RL 新题池上的 8 次筛题，再用 `make_grpo_data.py` 出 GRPO parquet | 现行（链路第 2 步） |
| `py_cfg_job.sh` | 空跑时代替 python：记下参数，让 hydra 打印组合后的配置就退出 | 现行 |
| `run_grpo.sh` | run1、run1b 用的 GRPO 运行（不带丢组） | run1、run1b |
| `grpo_run1_resume7.sh`、`gpu_guard_run1b.sh` | run1b 的启动器与崩溃守护 | run1b |
| `run_ppo.sh`、`ppo2.sh` | PPO 的运行与启动器 | 备用，本轮不跑 |
| `grpo_run1.sh`、`pause_eval_step4.sh`、`screen_m1_train_89.sh` | run1 的启动、暂停评测和 run1 前的筛题 | 一次性，GPU 号写死 |

## `sft/`：M1 的 SFT

`chain_m1b.sh`、`chain_m1c.sh`（过滤轨迹 → 打包 → `train_q35.py` 训练 M1，用 `configs/03_model-training/sft/sft_q35_4b_M1_en.yaml`）和 `judge_tier1b.sh`（判教师轨迹）是 M1 时期的一次性启动器，GPU 号写死。

