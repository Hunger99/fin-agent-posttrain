# scripts/02_harness：采集教师轨迹与筛题

M1 时期的一次性启动器，GPU 号、端口和运行名写死，留作结果的记录。

| 脚本 | 做什么 | 读入 | 产出（`FP_DATA_D2/market_v1/runs/`） |
| --- | --- | --- | --- |
| `market_train_teacher.sh` | 教师在 3,400 道行情训练题上各跑一次（需已在 18011 端口起好教师服务） | `market_v1/tasks/market_tasks_train_shuffled.jsonl` | `train_q38_H1` |
| `teacher_hard2.sh`、`teacher_hard3.sh` | 教师在难题档训练题上各跑两次（`teacher_hard3.sh` 加上 700 道新难题），再由 `judge_sft.py` 判通过的轨迹 | 难题档训练任务 | `train_hard_q38_H1`、`judge_hard_train.jsonl` |
| `screen_chain3d.sh` | 起点模型 Qwen3.5-4B-Base 依次在行情一档、财富管理与机器学习、难题档的训练任务上筛题 | 行情任务和 agent SFT 快照里的任务 | `screen_base_*` 等筛题运行 |
