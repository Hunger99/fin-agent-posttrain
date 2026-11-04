# scripts：启动器

所有脚本从自己的位置推出仓库根 `ROOT`，再读 [`paths.sh`](paths.sh)；`paths.sh` 读 `environment_installation/local.env`（本机外部路径，没有就报错退出），并定义仓库内的三个输出位置：严格重判 `STRICT_DIR`、链路的日志与 judge 状态 `CHAIN_DIR`、空跑输出 `DRY_DIR`，都在 `outputs/archive/runs/` 下。

## 现行链路：GRPO 与评测

[`rl_chain_c.sh`](rl_chain_c.sh) 按顺序跑：等 run1b（grpo1）结束 → 8 次筛题（`03_model-training/rl/screen_c.sh`）→ 抽 939 题 → grpo2、grpo3、grpo4（都经 `03_model-training/rl/grpo2.sh`）→ 四个检查点的评测与严格重判（`04_evaluation/eval_ckpt_c.sh`、`regrade_run.sh`）→ 通用能力与 judge 校准（`general_c.sh`、`judge_c.sh`）→ 换措辞测试集（`reworded_c.sh`）。已完成的阶段在重启时跳过，各阶段的设计写在脚本头注释里。

- **空跑：** `cd /tmp && DRY=1 bash <仓库>/scripts/rl_chain_c.sh`。不等待、不取卡、不启动进程，只写 `DRY_DIR`，打印每个阶段会执行的命令和 verl 的完整配置。
- **启动：** `cd /tmp && [KEPT="<GPU>"] [ROOT_MIN_GB=20] setsid nohup bash <仓库>/scripts/rl_chain_c.sh > <仓库>/outputs/archive/runs/rl-chain-c--market_v1--20261103T0726Z/logs/rl_chain_c.out 2>&1 < /dev/null &`。cwd 用 `/tmp`，因为 verl（hydra）把运行记录写在启动时的当前目录下。
- **停止：** `bash <仓库>/scripts/rl_chain_c.sh stop`（向链路发 TERM，各阶段按自己的方式停下并释放 GPU，最多等 20 分钟）。
- **链路的可调参数：** `configs/03_model-training/grpo/rl_chain_c.conf`（DYN、LR、MINI 等，格式见同目录的 `.example`；没有这个文件时用默认值）。
- **状态文件：** 在仓库外 `FP_RUNS/rl/`（`rl_chain_c.status`、`rl_chain_c.pid`、各阶段的 `.status`）。

## 顶层的共用脚本

| 脚本 | 作用 |
| --- | --- |
| `paths.sh` | 读 `local.env`，定义仓库内的输出位置；被所有脚本 source |
| `gpuclaim_lib.sh`、`gpuclaim_c_lib.sh` | 经 GPU-claim 工具取卡、交接、为重启保留和释放 GPU 的函数库；被各阶段 source |
| `free8.sh` | 更早的取卡规则（任意 N 张空卡，NFS 锁目录），run1b 及其守护进程在用 |
| `serve_4b.sh`、`serve_4b_keep_thinking.sh` | 用 vLLM 0.32 服务 4B 模型（Qwen3.5 自带模板；保留每轮思考的 v3 模板）；以 `sh` 启动 |
| `serve_qwen38.sh` | 服务教师 Qwen3.8-27B（judge 校准用） |

## 各阶段的启动脚本

- [`01_data-processing/`](01_data-processing/README.md)：没有启动脚本。
- [`02_harness/`](02_harness/README.md)：采集教师轨迹和起点模型的筛题，M1 时期的一次性启动器。
- [`03_model-training/`](03_model-training/README.md)：`sft/`（M1 的 SFT，一次性）和 `rl/`（GRPO、PPO 的启动器，现行链路的阶段脚本也在这里）。
- [`04_evaluation/`](04_evaluation/README.md)：现行链路的评测阶段，以及 M1 时期的一次性评测启动器。

“一次性启动器”指产出 M1 和 grpo1 结果时用过的脚本：GPU 号、运行名写死，留作结果的记录，复用前要先改这些值。
