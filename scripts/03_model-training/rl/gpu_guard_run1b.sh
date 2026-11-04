#!/bin/bash
# Crash guard for run1b (research/log/2026-11-01-0010.md, 11-02 08:3x UTC; the author: if our run crashes, hold its GPUs at once and keep
# them until the fixed code runs again). Checks every second that run1b's training driver (main_ppo, pid TP) is alive. When it exits,
# waits up to 60 s for grpo_run1_resume7.sh's exit line in run1.status: "exit 0" (a normal end) -> nothing held, the guard exits so
# rl_chain_b.sh can go on; otherwise it starts gpu_holder.py on each of run1b's GPUs (36 GB, at most 6 h; a holder skips a GPU that has
# another user's process and frees its GPU as soon as one of our processes appears there, i.e. the relaunch) and keeps their locks.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
RL=$FP_RUNS/rl
source $ROOT/scripts/free8.sh; GPUS="3 4 5 6 7 8"; TP=4005525; WRAP=4005387; LOG=$RL/run1b_guard.status
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
say "guard watches run1b's driver $TP on GPUs $GPUS"
while kill -0 $TP 2>/dev/null; do sleep 1; done
say "run1b's driver exited"
for i in $(seq 60); do
  R=$(grep '^run1 resume exit ' $RL/run1.status | tail -n 1)
  case "$R" in "run1 resume exit 0 "*"(run1b)"*) say "run1b ended normally ($R): nothing held"; exit 0;;
               "run1 resume exit "*"(run1b)"*) break;; esac
  kill -0 $WRAP 2>/dev/null || break
  sleep 1
done
for x in $GPUS; do mkdir $LOCKS/gpu$x 2>/dev/null
  ( CUDA_VISIBLE_DEVICES=$x HOLD_GB=36 MAX_S=21600 setsid $HPY -B $HOLDER $x < /dev/null > /dev/null 2>&1 & ); done
say "crash hold: holders started on GPUs $GPUS (last exit line: ${R:-none}); they free a GPU when one of our processes appears there"
