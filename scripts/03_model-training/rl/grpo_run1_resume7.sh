#!/bin/bash
# run1b: GRPO continued from run1's step-4 weights on 6 GPUs (research/log/2026-11-01-0010.md, 11-02 08:2x UTC; the author:
# "你用六张，把 learning rate 和 batch size 改一下，马上启动"). Takes the first 6 free GPUs of 0-9 (free8.sh's rule, checked every
# second; one gpu_holder.py per GPU until our processes appear), then runs run_grpo.sh from rl/run1/hf_step4 with 96 tasks x 8 a
# step, lr 8.7e-7, 24 steps, a fresh optimizer. Writes "run1 resume exit <rc> ..." to run1.status so rl_chain_b.sh can follow.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
RL=$FP_RUNS/rl
G=$FP_DATA_D2/grpo; S=$ROOT/src/03_model-training/code/verl_grpo
LOG=$RL/run1.status; source $ROOT/scripts/free8.sh; N=6
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
grabn() {  # free8.sh's grab8 for N GPUs
  local pick got x ok
  while true; do
    set -- $(free_list)
    if [ $# -ge $N ]; then
      pick=$(echo "$@" | tr ' ' '\n' | head -n $N | tr '\n' ' '); got=""; ok=1
      for x in $pick; do if mkdir $LOCKS/gpu$x 2>/dev/null; then got="$got $x"; else ok=0; break; fi; done
      if [ $ok = 1 ] && [ "$(rmdir_free_check $pick)" = $N ]; then
        for x in $pick; do ( CUDA_VISIBLE_DEVICES=$x HOLD_GB=36 MAX_S=600 setsid $HPY -B $HOLDER $x < /dev/null > /dev/null 2>&1 & ); done
        echo $pick; return 0; fi
      for x in $got; do rmdir $LOCKS/gpu$x 2>/dev/null; done
    fi
    sleep 1
  done
}
for attempt in 1 2 3; do
  say "resume7 (run1b) attempt $attempt waits for any $N free GPUs of 0-9 (checked every 1 s)"
  PICK=$(grabn); GL=$(echo $PICK | tr ' ' ',')
  say "run1b starts (resume7 attempt $attempt) on GPUs $GL: from run1/hf_step4, 96 x 8 a step, lr 8.7e-7, 24 steps"
  t0=$(date +%s)
  NCCL_P2P_DISABLE=1 RAY_object_spilling_config='{"type":"filesystem","params":{"directory_path":"'"$FP_CACHE"'/ray_spill"}}' \
  MODEL=$RL/run1/hf_step4 GPUS=$GL EXP=run1b TRAIN=$G/run1_train.parquet VAL=$G/run1_val.parquet BATCH=96 MINI=96 LR=8.7e-7 STEPS=24 EPOCHS=3 \
    SAVE_FREQ=4 TEST_FREQ=4 VAL_BEFORE=False VAL_BATCH=200 RESP=28672 TOKLEN=32768 \
    bash $ROOT/scripts/03_model-training/rl/run_grpo.sh ray_kwargs.ray_init.num_cpus=64 +ray_kwargs.ray_init._temp_dir=${FP_TMP:-/tmp}/ray \
    trainer.max_actor_ckpt_to_keep=2 trainer.resume_mode=disable > $RL/run1b_attempt$attempt.log 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  say "run1 resume exit $rc after ${dt}s (run1b)"
  for x in $PICK; do rmdir $LOCKS/gpu$x 2>/dev/null; done
  if [ $rc != 0 ] && [ $dt -lt 1200 ] && others_on $PICK; then say "another user's process is on GPUs $GL: waiting again"; continue; fi
  break
done
say "run1 resume done (run1b)"
