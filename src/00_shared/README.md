# 00_shared：跨阶段共用的工具

- **职责：** 放不属于单个阶段、被多个阶段的启动器共用的实现。现在只有 `utils/gpu_holder.py`：取到一张 GPU 后先占住一块显存，等我们自己的进程出现在这张卡上就释放并退出，免得取卡和真正启动之间被别人占走。
- **输入：** 环境变量 `CUDA_VISIBLE_DEVICES`（一张卡）、`HOLD_GB`（占多少显存）、`MAX_S`（最长占多久），参数是物理 GPU 号。
- **输出：** 没有文件输出，只有占卡进程本身。
- **入口：** 不直接运行，由 `scripts/free8.sh` 启动（`HOLDER=$ROOT/src/00_shared/utils/gpu_holder.py`）；GPU-claim 工具有自己的占卡进程。
