#!/bin/bash
# The first GRPO run on M1 (research/log/2026-11-01-0010.md, 21:xx UTC): once screen_m1_train has both runs of every task,
# make_grpo_data.py keeps the tasks some run failed plus a seeded 25% of the tasks both runs passed, the dev file is the
# validation set, and run_grpo.sh trains on GPUs 2-9 (the author's 8 GPUs) with MemReward's batch: 128 tasks a step x 8,
# one update a step, lr 1e-6, 3 epochs; validation and a checkpoint every 4 steps, the last 2 checkpoints kept.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
P=$FP_DATA_D2/harness_v2/pilot; G=$FP_DATA_D2/grpo
MK=$FP_DATA_D2/market_v1; S=$ROOT/src/03_model-training/code/verl_grpo
RL=$FP_RUNS/rl; LOG=$RL/run1.status; PY=$FP_VENVS/verl/bin/python
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
free_gpu() { U=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $U)" = 0 ]; }
say "run1 waits for both screening runs of the 2,350 tasks"
until [ "$(grep -c '"run": 2' $MK/runs/screen_m1_train/results.jsonl)" -ge 2350 ] && ! pgrep -u "$(id -un)" -f 'screen_m1_train' > /dev/null; do sleep 300; done
$PY -B $ROOT/src/03_model-training/utils/make_grpo_data.py --tasks $MK/hardset_v1/train_market.jsonl --screen $MK/runs/screen_m1_train/results.jsonl \
  --solved-frac 0.25 --out $G/run1_train.parquet >> $LOG 2>&1 || { say "train data failed"; exit 1; }
$PY -B $ROOT/src/03_model-training/utils/make_grpo_data.py --tasks $MK/hardset_v1/dev_market.jsonl --out $G/run1_val.parquet >> $LOG 2>&1 || { say "val data failed"; exit 1; }
for x in 2 3 4 5 6 7 8 9; do until free_gpu $x && mkdir $P/locks/gpu$x 2>/dev/null; do sleep 60; done; done
say "run1 starts on GPUs 2-9"
export NCCL_P2P_DISABLE=1 RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"'"$FP_CACHE"'/ray_spill"}}'
GPUS=2,3,4,5,6,7,8,9 EXP=run1 TRAIN=$G/run1_train.parquet VAL=$G/run1_val.parquet BATCH=128 MINI=128 STEPS=null EPOCHS=3 \
  SAVE_FREQ=4 TEST_FREQ=4 VAL_BEFORE=True VAL_BATCH=200 RESP=${RESP:-28672} TOKLEN=${TOKLEN:-32768} \
  bash $ROOT/scripts/03_model-training/rl/run_grpo.sh ray_kwargs.ray_init.num_cpus=64 +ray_kwargs.ray_init._temp_dir=${FP_TMP:-/tmp}/ray \
  trainer.max_actor_ckpt_to_keep=2 ${EXTRA} > $RL/run1.log 2>&1
say "run1 exit $?"
for x in 2 3 4 5 6 7 8 9; do rmdir $P/locks/gpu$x 2>/dev/null; done
say "run1 done"
