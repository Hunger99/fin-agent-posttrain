# 安装与本机环境

本项目用 9 个 Python 虚拟环境（venv），都放在本机系统盘的 `FP_VENVS`（如 `/home/<user>/venvs`），不放仓库：仓库在多人共享的 NFS 盘上，venv 的海量小文件会拖慢所有人。本目录放每个 venv 的建法、锁文件，以及本机外部路径的设置 `local.env`。

## 本机外部路径：`local.env`

所有脚本经 `scripts/paths.sh` 读 `local.env`，没有这个文件就报错退出。新机器上先 `cp local.env.example local.env` 再填值；`local.env` 不入库（`.gitignore` 已排除），`local.env.example` 入库。

| 变量 | 示例值 | 放什么 |
| --- | --- | --- |
| `FP_DATA_EN` | `/home/<user>/datasets/fin-posttrain-en` | 英文阶段的原始数据、处理后的数据池、混合数据、自建环境的数据库和任务 |
| `FP_DATA_ZH` | `/home/<user>/datasets/fin-posttrain` | 中文阶段的数据（历史） |
| `FP_DATA_D2` | `/mnt/<data disk>/<user>/datasets/fin-posttrain-en` | agent SFT 数据、GRPO parquet、行情数据集 `market_v1`（含评测运行 `runs/`）、`harness_v2`、`judge_cal` |
| `FP_RUNS` | `/mnt/<data disk>/<user>/runs/fin-posttrain` | SFT 检查点和 RL 运行 `rl/` |
| `FP_MODELS_D1` | `/mnt/<model disk 1>/<user>/models` | Qwen3-4B-Base、Qwen3-14B、Nemotron 向量模型、gpt-oss-120b |
| `FP_MODELS_D2` | `/mnt/<model disk 2>/<user>/models` | Qwen3.5-4B-Base、Qwen3.5-4B、Qwen3.8-27B |
| `FP_CACHE` | `/mnt/<data disk>/<user>/cache/verl` | verl 运行的 HF、Ray 溢出和 vLLM 缓存 |
| `FP_VENVS` | `/home/<user>/venvs` | 下表的 venv |
| `MARKET_SANDBOX_PYTHON` | `/home/<user>/venvs/mkt/bin/python` | 行情环境里 Python 工具的沙箱解释器（`market.py` 读这个变量） |
| `GPU_CLAIM_PY` | 本机 GPU-claim 工具的 `gpu_claim.py` | 取卡；`scripts/gpuclaim_lib.sh` 的 `GC` 默认值 |
| `GPU_CLAIM_TESTS` | 该工具的测试文件 | 只给 `src/05_review/tests/rl_chain/` 的两个测试用 |
| `FP_HF_DATASETS_CACHE` | `/mnt/<data disk>/<user>/cache/hf_datasets` | SFT 的 Hugging Face datasets 缓存（`chain_m1b.sh`、`chain_m1c.sh`；不设时用 `$HOME/.cache/huggingface/datasets`） |
| `FP_TMP` | `/tmp/<user>` | 本机盘上的 Ray 临时目录和 Triton、Inductor 缓存（RL 启动器；不设时用 `/tmp`） |

LLaMA-Factory 的 SFT yaml 不展开变量，`model_name_or_path`、`dataset_dir`、`output_dir` 和 `deepspeed:` 写的是字面绝对路径（见 `configs/README.md`）。

## 机器前提

- NVIDIA 550 驱动（CUDA 12.4），`CUDA_HOME=/usr/local/cuda-12.4`；CUDA 12.8/12.9 的 wheel 靠 CUDA 12 的小版本兼容运行，verl 主分支官方用的 CUDA 13 跑不了。
- 包管理用 uv（`$HOME/.local/bin/uv`，0.13.5）；所有 venv 都由它建。uv 缓存已在 11-03 清过，重建任何 venv 都要重新下载。

## venv 一览

锁文件是各 venv 现装的完整包列表（`requirements/freeze/<venv>.txt`，2026-11-03 导出），以它为准。

| venv | Python | 用途 | 关键包 | 建法 |
| --- | --- | --- | --- | --- |
| `fin-posttrain` | 3.10.12 | 数据处理：转换、去污染、去重、bge-m3 向量 | torch 2.6.0+cu124、transformers 5.20.2、pyarrow 25.0.1 | 没有脚本；按锁文件重建 |
| `embed-tf5` | 3.10.12 | Nemotron-3-Embed 向量去污染（要 transformers ≥5.2） | torch 2.9.0+cu128、transformers 5.20.2、sentence-transformers 6.2.0 | 没有脚本；按锁文件重建 |
| `vllm-0.12` | 3.12.15 | gpt-oss-120b 推理服务；harness v2/v3 的客户端 | vllm 0.12.0、torch 2.9.0、transformers 4.57.6 | `install/build_vllm.sh` |
| `lmeval` | 3.12.15 | 第一轮的 lm-evaluation-harness 评测 | lm-eval 0.4.13、vllm 0.12.0 | 没有脚本；与 `vllm-0.12` 的 dill、fsspec 冲突而另建 |
| `llamafactory` | 3.11.17 | Qwen3-4B-Base 的 SFT（G、R 系列） | llamafactory 0.9.5、torch 2.6.0+cu124、transformers 5.5.4、deepspeed 0.16.9、flash-attn 2.7.4.post1 | 没有脚本；按锁文件重建 |
| `train-q35` | 3.11.17 | Qwen3.5-4B 的 SFT（R3′、A1、A1-base、M1） | llamafactory 0.9.5、torch 2.8.0+cu128、transformers 5.20.2、flash-attn 2.8.3.post2、fla 0.5.2、causal-conv1d 1.7.0 | `install/install_train_q35.sh`（读 `requirements/train_q35_base_reqs.txt`） |
| `vllm-0.32` | 3.12.15 | Qwen3.5、Qwen3.8 推理服务（教师、4B 模型评测、筛题）；含 lm-eval | vllm 0.32.0+cu129、torch 2.13.0+cu129、transformers 5.19.0 | `install/install_vllm032.sh` |
| `verl` | 3.12.15 | GRPO（以及备用的 PPO）训练 | verl（git 8df5baeed0）、vllm 0.32.0+cu129、torch 2.13.0+cu129、transformers 5.16.1、flash-attn 2.8.3.post2（源码编译）、ray 2.59.0 | 没有建 venv 的脚本；flash-attn 用 `install/build_flash_attn.sh` 编译（读 `requirements/verl_constraints.txt`） |
| `mkt` | 3.12.15 | 行情环境的沙箱解释器；`market_tasks/generate.py` | numpy 2.5.4、pandas 3.1.0、lxml、requests | 没有脚本；按锁文件重建 |

## 建法

有脚本的 venv 直接运行脚本（脚本里的 venv 位置和日志位置是本机路径，换机器先改）：

```bash
sh   environment_installation/install/build_vllm.sh          # vllm-0.12
bash environment_installation/install/install_vllm032.sh     # vllm-0.32
bash environment_installation/install/install_train_q35.sh   # train-q35
bash environment_installation/install/build_flash_attn.sh    # verl 里的 flash-attn（verl venv 建好之后）
```

没有脚本的 venv 按锁文件重建，以下命令没有实际运行过：

```bash
uv venv --python <上表的 Python> $FP_VENVS/<venv>
uv pip install --python $FP_VENVS/<venv>/bin/python -r environment_installation/requirements/freeze/<venv>.txt
```

`verl` 要先装 torch 和 vLLM 的 wheel，再用 `--no-build-isolation` 编 flash-attn，顺序见项目日志 `research/log/2026-11-01-0010.md` 的 11-01 20:0x 和 20:4x。

## 记录与现装不一致的地方

- `install_train_q35.sh` 写的是 fla 0.6.0，现装 0.5.2：fla 0.6.0 要求更新的 triton（日志 2026-10-29-1815）。LLaMA-Factory 限定 transformers ≤5.6，训练时用 `DISABLE_VERSION_CHECK=1` 跳过。
- 这个脚本当时的安装记录结尾是 flash_attn 的 `ImportError: undefined symbol`；现装的是 cxx11abiTRUE 的 wheel，修法没有写进脚本。
- `llamafactory`、`embed-tf5` 的 transformers 版本与日志记载不同（日志记 5.6.0、5.2.0，现装 5.5.4、5.20.2）。
