#!/bin/bash
# Pause run1 at its step-4 checkpoint and evaluate it with harness v3's test settings (research/log/2026-11-01-0010.md,
# 11-02 00:1x UTC): once ckpt/latest_checkpointed_iteration.txt reads 4, stop the training (grpo_run1.sh then frees the
# GPU locks), merge the checkpoint into a Hugging Face model, and at once (a) run dev_market x4 on GPUs 2-7 as M1's
# dev_hs_market_M1_H1_x4 ran (serve_4b_keep_thinking.sh, 24 per server, H1, 100 steps) and (b) a 1-step GRPO smoke test
# with Ulysses sequence parallel 2 and a 61,440-token response budget on GPUs 8, 9 (verl #6284). Then the locks go;
# resuming run1 is decided from the results by the rule in the log.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); . "$ROOT/scripts/paths.sh"
RL=$FP_RUNS/rl; MK=$FP_DATA_D2/market_v1
P=$FP_DATA_D2/harness_v2/pilot; EN=$FP_DATA_EN
G=$FP_DATA_D2/grpo; S=$ROOT/src/03_model-training/code/verl_grpo
V3=$ROOT/src/02_harness/code/v3; M1=$FP_RUNS/qwen3.5-4b-base-en/M1
VPY=$FP_VENVS/verl/bin/python; EPY=$FP_VENVS/vllm-0.12/bin/python; WRAP=1947128
HF=$RL/run1/hf_step4; OUT=dev_hs_market_run1s4_H1_x4; LOG=$RL/pause_step4.status; K=24
export PYTHONDONTWRITEBYTECODE=1
say() { echo "$1 $(date -u '+%F %T')" >> $LOG; }
free_gpu() { U=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $U)" = 0 ]; }
slog() { echo $EN/logs/serve_4b_run1s4_gpu$1.log; }
say "pause_step4 waits for the step-4 checkpoint"
until [ "$(cat $RL/run1/ckpt/latest_checkpointed_iteration.txt 2>/dev/null)" = 4 ]; do
  kill -0 $WRAP 2>/dev/null || { say "run1 ended before step 4: nothing done"; exit 1; }; sleep 60
done
say "step-4 checkpoint written; stopping run1"
RG=$(pgrep -P $WRAP | head -n 1); TP=$(pgrep -P "$RG" | head -n 1)
[ -n "$TP" ] && kill -TERM $TP
for i in $(seq 36); do kill -0 $WRAP 2>/dev/null || break; sleep 5; done
pgrep -g $WRAP > /dev/null && { kill -9 -- -$WRAP 2>/dev/null; say "killed what was left of run1's process group"; }
for i in $(seq 60); do a=1; for x in 2 3 4 5 6 7 8 9; do free_gpu $x || a=0; done; [ $a = 1 ] && break; sleep 5; done
if [ $a = 0 ]; then  # leftovers of run1 (Ray workers in their own sessions): only our user's processes on GPUs 2-9
  for u in $(for x in 2 3 4 5 6 7 8 9; do nvidia-smi -i $x --query-gpu=uuid --format=csv,noheader; done); do
    for q in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep $u | cut -d, -f2); do
      [ "$(ps -o user= -p $q)" = "$(id -un)" ] && { kill -9 $q; say "killed leftover pid $q"; }; done; done
  for i in $(seq 24); do a=1; for x in 2 3 4 5 6 7 8 9; do free_gpu $x || a=0; done; [ $a = 1 ] && break; sleep 5; done
fi
[ $a = 1 ] || { say "GPUs 2-9 not free 5 minutes after the stop: stopped"; exit 1; }
for x in 2 3 4 5 6 7 8 9; do rmdir $P/locks/gpu$x 2>/dev/null; done
say "run1 stopped; GPUs 2-9 free"
for x in 2 3 4 5 6 7 8 9; do mkdir $P/locks/gpu$x 2>/dev/null || { say "GPU $x locked by another job: stopped"; exit 1; }; done
# (b) the sequence-parallel smoke test, in the background
( export NCCL_P2P_DISABLE=1
  GPUS=8,9 EXP=smoke_sp2 TRAIN=$G/smoke_sp2_train.parquet VAL=$G/smoke_val.parquet BATCH=2 MINI=2 STEPS=1 RESP=61440 \
    TOKLEN=32768 bash $ROOT/scripts/03_model-training/rl/run_grpo.sh actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=2 ray_kwargs.ray_init.num_cpus=24 \
    +ray_kwargs.ray_init._temp_dir=${FP_TMP:-/tmp}/ray_sp2 > $RL/smoke_sp2.log 2>&1
  say "smoke_sp2 exit $?" ) &
SMOKE=$!
# (a) merge the checkpoint and evaluate it
$VPY -m verl.model_merger merge --backend fsdp --local_dir $RL/run1/ckpt/global_step_4/actor --target_dir $HF > $RL/merge_step4.log 2>&1 \
  || { say "merge failed (see merge_step4.log)"; wait $SMOKE; for x in 2 3 4 5 6 7 8 9; do rmdir $P/locks/gpu$x; done; exit 1; }
for f in chat_template.jinja tokenizer.json tokenizer_config.json processor_config.json generation_config.json preprocessor_config.json video_preprocessor_config.json; do
  [ -f $HF/$f ] || { [ -f $M1/$f ] && cp $M1/$f $HF/; }
done
say "merged: $(python3 -c "import json; c=json.load(open('$HF/config.json')); t=c.get('text_config',c); print(c.get('architectures'), (t.get('rope_parameters') or {}).get('rope_theta'))")"
G6="2 3 4 5 6 7"; urls=""; n=0
for x in $G6; do ( MAXSEQS=$K setsid sh $ROOT/scripts/serve_4b_keep_thinking.sh $x 1816$x $HF run1s4 $(slog $x) < /dev/null > /dev/null 2>&1 & ); done
for x in $G6; do
  for i in $(seq 90); do curl -s -m 5 http://127.0.0.1:1816$x/v1/models | grep -q '"run1s4"' && break; sleep 10; done
  curl -s -m 5 http://127.0.0.1:1816$x/v1/models | grep -q '"run1s4"' && { urls="$urls --base-url http://127.0.0.1:1816$x/v1"; n=$((n + 1)); } || say "server on GPU $x did not start"
done
say "dev x4 on $n servers"
[ $n -gt 0 ] && $EPY -B $V3/run.py --tasks $MK/hardset_v1/dev_market.jsonl --db $MK/market_v1.sqlite --config H1 --runs 4 --max-steps 100 \
  --concurrency $((K * n)) --provider qwen --model run1s4 $urls --out $MK/runs/$OUT >> $MK/runs/$OUT.log 2>&1
say "dev x4 exit $?: $(wc -l < $MK/runs/$OUT/results.jsonl) rows, $(grep -c '"passed": true' $MK/runs/$OUT/results.jsonl) passed"
ps=""; for x in $G6; do p=$(head -n 1 $(slog $x) | grep -o "pid [0-9]*" | cut -d' ' -f2); [ -n "$p" ] && { kill -- -$p 2>/dev/null; ps="$ps $p"; }; done
sleep 20; for p in $ps; do kill -9 -- -$p 2>/dev/null; done
for x in $G6; do rmdir $P/locks/gpu$x 2>/dev/null; done
wait $SMOKE
for x in 8 9; do rmdir $P/locks/gpu$x 2>/dev/null; done
say "pause_step4 done"
