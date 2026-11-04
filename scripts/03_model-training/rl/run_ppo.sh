#!/bin/bash
# PPO (actor and critic, GAE) on M1 with harness v3's market episodes (fin_agent_loop.py; research/log/
# 2026-11-01-0010.md, 11-02 05:0x UTC), to compare with GRPO on the algorithm alone: the data, actor, reference, rollout
# and trainer settings are run_grpo.sh's; GAE with a learned critic replaces the group-normalised advantage.
# - Six GPUs (the same log, 11-02 08:3x UTC): 96 tasks a step (BATCH=96 MINI=96, 16 a GPU as run1's 128 on 8), and the
#   default learning rates are the 128-task ones scaled by sqrt(96/128), run1b's rule: actor 1e-6 -> 8.7e-7 (LR),
#   critic 1e-5 -> 8.7e-6 (CRITIC_LR).
# - GAE gamma 1, lambda 1 (verl's default and ReTool's multi-turn PPO recipe): the only reward is the 0/1 grade on the
#   last of up to RESP response tokens, which any discount would erase. verl's GAE skips tool-result tokens.
# - Critic: M1 as Qwen3_5ForTokenClassification (verl's value-model loader; M1's backbone, a new 2560 -> 1 head),
#   lr constant, 1e-5 at 128 tasks a step (verl's PPO example; the APO pitfalls notes give 5e-6 to 1e-5), CRITIC_MINI
#   tasks per critic update (24 at 96 tasks a step: 4 updates a step of 192 trajectories, as ReTool's PPO and as 32 of
#   128 gave), WARMUP critic-only steps first (verl's trainer.critic_warmup is WARMUP + 1, since steps count from 1),
#   value clip VCLIP (10: with 0/1 returns it in effect never binds, where verl's 0.5 would limit how far one step moves
#   a value).
# - The new head starts with std 0.001 (initializer_range on the critic's top-level config; every other weight is M1's):
#   M1's values on two H1 prompts then have std 0.14 (-0.49 to 0.58), against std 2.8 (-9.8 to 11.6) at the default
#   0.02, while the returns are 0 or 1.
# - One sequence per critic micro-batch: verl passes packed sequences' boundaries (cu_seqlens) only to a forward that
#   takes them, which the actor's patched forward does and Qwen3_5ForTokenClassification's does not, so a packed critic
#   micro-batch would run the Gated DeltaNet layers across sequences. verl main's critic worker takes its
#   micro-batching from critic.fsdp, not from critic.use_dynamic_bsz / ppo_max_token_len_per_gpu.
# - Critic parameters and optimizer stay on the CPU outside its two phases (values, update), so the rollout and the
#   actor update keep run_grpo.sh's GPU memory; global batch and learning rates do not depend on the micro-batching.
# Usage: GPUS=3,4,5,6,7,8 EXP=<name> TRAIN=<parquet> VAL=<parquet> BATCH=<prompts> MINI=<prompts> STEPS=<n> \
#        [EPOCHS=<n>] [WARMUP=3] [LR=8.7e-7] [CRITIC_LR=8.7e-6] [CRITIC_MINI=24] [VCLIP=10] [SAVE_FREQ=-1] [TEST_FREQ=-1] \
#        [VAL_BEFORE=False] [RESP=24576] [OFFLOAD=False] [OUTD=<run dir>] [PY=<python>] bash run_ppo.sh [extra hydra overrides]
set -eo pipefail
HERE=$(cd "$(dirname "$0")/../../../src/03_model-training/code/verl_grpo" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd); . "$ROOT/scripts/paths.sh"
PY=${PY:-$FP_VENVS/verl/bin/python}
M1=$FP_RUNS/qwen3.5-4b-base-en/M1
OUTD=${OUTD:-$FP_RUNS/rl/$EXP}; mkdir -p $OUTD
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$HERE PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export FIN_HARNESS_V3=$ROOT/src/02_harness/code/v3 FIN_MAX_STEPS=${FIN_MAX_STEPS:-30}
export FIN_DB=$FP_DATA_D2/market_v1/market_v1.sqlite FIN_EPISODE_LOG_DIR=$OUTD/episodes
export VLLM_CACHE_ROOT=$FP_CACHE/vllm HF_HOME=$FP_CACHE/hf
export TRITON_CACHE_DIR=${FP_TMP:-/tmp}/verl_triton TORCHINDUCTOR_CACHE_DIR=${FP_TMP:-/tmp}/verl_inductor
export VERL_FILE_LOGGER_PATH=$OUTD/metrics.jsonl WANDB_MODE=disabled RAY_DEDUP_LOGS=0 RAY_local_fs_capacity_threshold=0.995
$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae algorithm.gamma=1.0 algorithm.lam=1.0 algorithm.use_kl_in_reward=False \
  data.train_files=$TRAIN data.val_files=$VAL data.train_batch_size=$BATCH data.val_batch_size=${VAL_BATCH:-64} \
  data.max_prompt_length=4096 data.max_response_length=${RESP:-24576} data.filter_overlong_prompts=${FILTER:-False} \
  data.truncation=error data.return_raw_chat=True data.shuffle=True +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path=$M1 actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=${LR:-8.7e-7} \
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
  critic.strategy=fsdp2 critic.model.path=$M1 critic.model.use_remove_padding=True \
  +critic.model.override_config.initializer_range=0.001 \
  critic.model.enable_gradient_checkpointing=True critic.optim.lr=${CRITIC_LR:-8.7e-6} \
  critic.ppo_mini_batch_size=${CRITIC_MINI:-24} critic.ppo_epochs=1 critic.cliprange_value=${VCLIP:-10} \
  critic.use_dynamic_bsz=False critic.ppo_micro_batch_size_per_gpu=1 +critic.fsdp.use_dynamic_bsz=False \
  +critic.fsdp.micro_batch_size_per_gpu=1 +critic.fsdp.infer_micro_batch_size_per_gpu=1 \
  critic.fsdp.param_offload=True critic.fsdp.optimizer_offload=True critic.fsdp.use_torch_compile=False \
  trainer.critic_warmup=$((${WARMUP:-3} + 1)) trainer.logger=['console','file'] trainer.project_name=fin_ppo \
  trainer.experiment_name=$EXP trainer.n_gpus_per_node=$NG trainer.nnodes=1 trainer.default_local_dir=$OUTD/ckpt \
  trainer.save_freq=${SAVE_FREQ:--1} trainer.test_freq=${TEST_FREQ:--1} trainer.val_before_train=${VAL_BEFORE:-False} \
  trainer.total_epochs=${EPOCHS:-1} trainer.total_training_steps=$STEPS trainer.balance_batch=True \
  "$@"
