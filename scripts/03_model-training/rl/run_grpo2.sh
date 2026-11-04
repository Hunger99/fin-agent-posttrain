#!/bin/bash
# run_grpo.sh plus dynamic sampling (research/log/2026-11-01-0010.md, 11-02 22:3x UTC, item 3): groups whose 8 rewards
# are all equal, zero advantage under GRPO, are dropped during training and replaced by fresh prompts until the batch
# holds BATCH mixed groups, through verl V1's own group filter (algorithm.filter_groups, which also makes the
# dataloader fetch one prompt at a time) with dyn_filter.py's CappedDapoReplayBuffer as the sampler. Everything else is
# run_grpo.sh's, line for line. New settings: REFILL (prompts dispatched per dropped group, 2.0 as in verl), CAP (prompts
# a step may dispatch before filtering stops for that step, the first batch included; 3 x BATCH by default), INFLIGHT
# (batches of prompts pending or running at once, 1: as many concurrent episodes as run1b had) and DYN=0 (run_grpo.sh's
# sampling, for an A/B run with the same launcher).
# A step now draws more than BATCH prompts (about BATCH / 0.72 at run1b's 72% mixed groups), while verl still counts
# an epoch as len(TRAIN) // BATCH steps and restarts the dataloader whenever it runs out, so give STEPS explicitly.
# Usage: GPUS=2,3,4,5,6,7 EXP=<name> TRAIN=<parquet> VAL=<parquet> BATCH=<prompts> MINI=<prompts> STEPS=<n> \
#        [REFILL=2.0] [CAP=<prompts>] [INFLIGHT=1] [DYN=1] [EPOCHS=<n>] [SAVE_FREQ=-1] [TEST_FREQ=-1] [VAL_BEFORE=False] \
#        bash run_grpo2.sh [extra hydra overrides]
set -eo pipefail
HERE=$(cd "$(dirname "$0")/../../../src/03_model-training/code/verl_grpo" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd); . "$ROOT/scripts/paths.sh"
PY=$FP_VENVS/verl/bin/python
M1=${MODEL:-$FP_RUNS/qwen3.5-4b-base-en/M1}
OUTD=$FP_RUNS/rl/$EXP; mkdir -p $OUTD
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$HERE PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export FIN_HARNESS_V3=$ROOT/src/02_harness/code/v3 FIN_MAX_STEPS=${FIN_MAX_STEPS:-30}
export FIN_DB=$FP_DATA_D2/market_v1/market_v1.sqlite FIN_EPISODE_LOG_DIR=$OUTD/episodes
export VLLM_CACHE_ROOT=$FP_CACHE/vllm HF_HOME=$FP_CACHE/hf
export TRITON_CACHE_DIR=${FP_TMP:-/tmp}/verl_triton TORCHINDUCTOR_CACHE_DIR=${FP_TMP:-/tmp}/verl_inductor
export VERL_FILE_LOGGER_PATH=$OUTD/metrics.jsonl WANDB_MODE=disabled RAY_DEDUP_LOGS=0 RAY_local_fs_capacity_threshold=0.995
DYNARGS=()
if [ "${DYN:-1}" = 1 ]; then
  DYNARGS=(algorithm.filter_groups.enable=True algorithm.filter_groups.metric=reward
    trainer.v1.sampler.custom_sampler.path=$HERE/dyn_filter.py trainer.v1.sampler.custom_sampler.name=CappedDapoReplayBuffer
    +trainer.v1.sampler.sampler_kwargs.train_batch_size=$BATCH +trainer.v1.sampler.sampler_kwargs.metric=reward
    +trainer.v1.sampler.sampler_kwargs.refill_factor=${REFILL:-2.0}
    +trainer.v1.sampler.sampler_kwargs.max_prompts_per_step=${CAP:-$((3 * BATCH))}
    +trainer.v1.sampler.sampler_kwargs.max_inflight_gen_batches=${INFLIGHT:-1})
fi
$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False algorithm.norm_adv_by_std_in_grpo=True \
  data.train_files=$TRAIN data.val_files=$VAL data.train_batch_size=$BATCH data.val_batch_size=${VAL_BATCH:-64} \
  data.max_prompt_length=4096 data.max_response_length=${RESP:-24576} data.filter_overlong_prompts=${FILTER:-False} \
  data.truncation=error data.return_raw_chat=True data.shuffle=True +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path=$M1 actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${TOKLEN:-32768} \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=${KL:-0.001} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=${AGG:-seq-mean-token-mean} actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD:-False} actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD:-False} \
  actor_rollout_ref.ref.strategy=fsdp2 actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True actor_rollout_ref.ref.fsdp_config.entropy_checkpointing=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.n=${N:-8} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=${MEM:-0.65} \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.enforce_eager=${EAGER:-False} actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_prefix_caching=${PREFIX:-True} actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.max_model_len=$((4096 + ${RESP:-24576})) \
  actor_rollout_ref.rollout.multi_turn.enable=True actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
  actor_rollout_ref.rollout.agent.num_workers=${WORKERS:-8} actor_rollout_ref.rollout.agent.default_agent_loop=fin_harness \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=$ROOT/configs/03_model-training/verl/agent_loop.yaml \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  trainer.critic_warmup=0 trainer.logger=['console','file'] trainer.project_name=fin_grpo trainer.experiment_name=$EXP \
  trainer.n_gpus_per_node=$NG trainer.nnodes=1 trainer.default_local_dir=$OUTD/ckpt \
  trainer.save_freq=${SAVE_FREQ:--1} trainer.test_freq=${TEST_FREQ:--1} trainer.val_before_train=${VAL_BEFORE:-False} \
  trainer.total_epochs=${EPOCHS:-1} trainer.total_training_steps=$STEPS trainer.balance_batch=True \
  "${DYNARGS[@]}" "$@"
