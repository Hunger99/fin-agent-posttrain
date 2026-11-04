# 01_data-processing：取数、去污染、去重、配比，以及行情快照和行情题

- **职责：** 把公开数据集变成训练用的数据池和混合数据，并为 harness 准备离线行情环境和行情题。
- **输入：** Hugging Face 上的金融与通用数据集、评测集全文（`FP_DATA_EN/raw`）；行情、FRED、SEC 的原始数据；`configs/01_data-processing/prompts/` 的提示词模板。
- **输出（都在仓库外）：** 去污染、去重后的金融池和通用池，混合数据 G/R1/R3/R3_nosyn/R4（`FP_DATA_EN/mixtures`）；行情快照 `market_v1.sqlite`、两档行情题 `hardset_v1/`、RL 新题池 `rl_pool_v2/`、换措辞题集（`FP_DATA_D2/market_v1`）。
- **`code/`：** 被别的文件 import 的模块：`corpus_window_decontam.py`、`doc_qa.py`、`doc_qa_en.py`，以及整体移动的 `market_tasks/`（行情题的模板、生成、校验，入口 `market_tasks/generate.py`）。
- **`utils/`：** 各步的命令行工具，每个文件头的 docstring 写明用法；文件名带 `_en` 的是英文阶段（现行），不带的是中文阶段（历史）。`market_snapshot/` 是行情快照的构建和校验（说明见其 README）。`docqa_recheck_en.py`、`select_sets_en_v2.py` 是改过数据的一次性步骤。
- **入口：** 这一阶段没有启动脚本，直接运行 `utils/` 下的命令。
