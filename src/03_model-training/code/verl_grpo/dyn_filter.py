"""DAPO-style dynamic sampling with a per-step generation cap, as a custom replay buffer for verl's V1 sync trainer
(research/log/2026-11-01-0010.md, 11-02 22:3x UTC, item 3).

run_grpo.sh launches verl.trainer.main_ppo, whose default trainer.use_v1=true runs PPOTrainerSync (TaskRunnerV1 in
run1b_attempt1.log), not ray_trainer.RayPPOTrainer. That trainer's ReplayBuffer already filters groups whose reward is
the same in all n rollouts when algorithm.filter_groups.enable=true: each evicted group earns 2 refill prompts, at
most max_inflight_gen_batches * train_batch_size prompts are pending or running at once, and once train_batch_size
groups are sampleable the requests in flight are drained and the surplus discarded. It has no cap on how many prompts
a step may generate: algorithm.filter_groups.max_num_gen_batches is ignored with a warning. In run1 (steps 1-4) and
run1b (steps 1-11) 72% of the 1,568 groups were mixed, 61-80% a step, but a policy that keeps improving pushes that
share down, and a run1b step already took 66-80 minutes, 35-45 of them generating.

CappedDapoReplayBuffer is that ReplayBuffer with three changes, plugged in through trainer.v1.sampler.custom_sampler
so verl stays as installed:
- max_prompts_per_step caps the prompts a step dispatches, the first batch from prepare_step included. Once reached,
  filtering stops for the rest of the step: groups finishing from then on are kept whatever their rewards, and if
  the groups kept plus those in flight cannot fill the batch, exactly the missing number is dispatched, so a step
  generates at most max_prompts_per_step + train_batch_size prompts. DAPO's recipe raises an error at its cap; a
  zero-variance group only costs its share of the loss denominator (GRPO gives it advantage 0), so the run goes on
  and the step's metrics say it happened.
- refill_factor sets the refill prompts per evicted group (verl's 2 by default). Replaying run1b's groups through this
  loop (each group's slowest episode as its duration, 200 steps of 96), 2 drew 181 prompts a step, 35 of them surplus,
  and generation took 2.0x as long as without filtering (2.2x at the 90th percentile); 1 drew 132 with no surplus but
  took 2.3x (3.0x), since each refill waits for the group it replaced to finish first.
- the batch takes mixed groups first, then the zero-variance groups kept after the cap, each oldest first as verl
  orders them.
Validation and every other path are verl's own (super().sample). Settings come from trainer.v1.sampler.sampler_kwargs,
since verl passes a custom sampler only that dict: train_batch_size (required, data.train_batch_size), metric
("reward": the reward_score fin_agent_loop.py sets, which verl stores as rm_scores at rollout time), max_inflight_gen_
batches (1, the concurrency run1b had; 2 shortened the replayed generation by 3%), refill_factor (2.0) and
max_prompts_per_step (3 * train_batch_size: at most 226 were drawn in the replay, so the cap binds only once markedly
fewer groups are mixed than in run1b). algorithm.filter_groups.enable=true must also be set: it is what makes verl's
dataloader fetch one prompt at a time, which refills of any size need.

Per-step metrics, besides verl's training/filter_groups/{evicted,discarded_surplus}_samples:
training/filter_groups/{dispatched_prompts, cap_reached, kept_zero_variance, mixed_in_batch} and
training/filter_groups/evicted_at_reward_<r> (verl's own count by shared reward goes to wandb only).
"""
import logging
import math
import os
import time

from verl.trainer.ppo.v1.replay_buffer import (DAPO_FILTERED_REWARD_COUNTS_KEY, ReplayBuffer,
                                              _accumulate_eviction_metrics)
from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class CappedDapoReplayBuffer(ReplayBuffer):
    def __init__(self, trainer_mode, trainer_config, max_off_policy_threshold, max_off_policy_strategy, sampler_kwargs,
                 refill_fn=None, **kwargs):
        kw = dict(sampler_kwargs or {})
        assert trainer_mode == "sync", f"CappedDapoReplayBuffer is for trainer.v1.trainer_mode=sync, not {trainer_mode}"
        batch = int(kw["train_batch_size"])
        self.refill_factor = float(kw.get("refill_factor", 2.0))
        self.max_prompts_per_step = int(kw.get("max_prompts_per_step", 3 * batch))
        assert self.refill_factor > 0 and self.max_prompts_per_step >= batch, kw
        super().__init__(trainer_mode, trainer_config, max_off_policy_threshold, max_off_policy_strategy,
                         sampler_kwargs, refill_fn=refill_fn, filter_groups_metric=str(kw.get("metric", "reward")),
                         train_batch_size=batch, gen_batch_size=1,
                         max_inflight_gen_batches=int(kw.get("max_inflight_gen_batches", 1)),
                         sync_refill_failed_groups=bool(kw.get("sync_refill_failed_groups", False)), **kwargs)

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps, partition_id, batch_size):
        if partition_id == "val":
            return super().sample(global_steps, partition_id, batch_size)
        assert batch_size == self.train_batch_size, (batch_size, self.train_batch_size)
        last_debug_time, metrics = time.time(), {}
        dispatched = batch_size  # prepare_step submitted one batch before sampling
        credit, draining, capped = 0, False, False
        max_inflight = self.max_inflight_gen_batches * self.train_batch_size
        while True:
            self._sync_metadata_from_transfer_queue()
            flat, flat_counts = self._dapo_filtered_keys(partition_id)
            failed = set()
            if self.sync_refill_failed_groups:
                failed = self.failure_keys[partition_id] - {k.split("_")[0] for k in self.partitions[partition_id]}
            reasons = (set(), set() if capped else flat, failed, flat_counts if not capped else {})
            evicted, _, n_flat, ev_metrics = self._evict_terminal_groups(global_steps, partition_id, reasons)
            if evicted:
                _accumulate_eviction_metrics(metrics, ev_metrics, 0)
            sampleable = (self.finished_keys[partition_id] | self.failure_keys[partition_id]) - evicted
            inflight = len(self.pending_keys[partition_id]) + len(self.running_keys[partition_id])
            enough = len(sampleable) >= batch_size
            if enough:
                draining, credit = True, 0
            if not draining:
                if not capped:
                    credit += math.ceil(self.refill_factor * n_flat) + len(failed)
                    capped = dispatched >= self.max_prompts_per_step
                if capped:
                    n = max(0, batch_size - len(sampleable) - inflight)
                else:
                    n = min(credit, max(0, max_inflight - inflight), self.max_prompts_per_step - dispatched)
                if n > 0:
                    self.refill_fn(n)
                    dispatched += n
                    credit = 0 if capped else credit - n
                    continue
            if enough and inflight == 0:
                break
            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)

        order = {k: self.prompt_global_steps[partition_id].get(k, 0) for k in sampleable}
        kept_flat = sampleable & flat
        ranked = sorted(sampleable - kept_flat, key=order.get) + sorted(kept_flat, key=order.get)
        selected, surplus = ranked[:batch_size], set(ranked[batch_size:])
        snapshot = dict(self.partitions[partition_id])
        if surplus:
            self._clear_groups(partition_id, surplus)
            metrics["training/filter_groups/discarded_surplus_samples"] = len(surplus)
        if not any(k.split("_")[0] in set(selected) for k in snapshot):
            raise RuntimeError("Sync replay buffer selected terminal groups with no materializable trajectories.")
        metrics.update({"training/filter_groups/dispatched_prompts": dispatched,
                        "training/filter_groups/cap_reached": int(capped),
                        "training/filter_groups/kept_zero_variance": len(kept_flat & set(selected)),
                        "training/filter_groups/mixed_in_batch": len(set(selected) - kept_flat)})
        # verl logs its {shared reward: groups} dict to wandb only
        for value, count in metrics.get(DAPO_FILTERED_REWARD_COUNTS_KEY, {}).items():
            metrics[f"training/filter_groups/evicted_at_reward_{value:g}"] = count
        if capped:
            logger.warning(f"step {global_steps}: max_prompts_per_step={self.max_prompts_per_step} reached, "
                           f"{len(kept_flat & set(selected))} zero-variance groups kept in the batch")
        return self._materialize_batch(partition_id, selected, snapshot), metrics
