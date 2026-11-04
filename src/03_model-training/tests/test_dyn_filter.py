"""CPU tests of src/03_model-training/code/verl_grpo/dyn_filter.py's CappedDapoReplayBuffer on synthetic groups (research/log/2026-11-01-0010.md,
11-02 22:3x UTC, item 3).

A fake TransferQueue stands in for the store verl's replay buffer polls (kv_list, kv_batch_get, kv_clear), and a fake
rollout finishes each dispatched prompt's group of n rewards after a set number of polls, so the real sample loop runs,
verl's own eviction and selection helpers included: the zero-variance test, eviction and refill, the cap and what
follows it, mixed-first selection with the surplus discarded, the batch handed to the trainer, and, with the cap out of
reach and verl's refill factor 2, the same dispatch, eviction and discard counts as verl's ReplayBuffer.sample.

Run: $FP_VENVS/verl/bin/python -B src/03_model-training/tests/test_dyn_filter.py   (verl's venv, CPU only)
"""
import os
import random
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "verl_grpo"))
from verl.trainer.ppo.v1 import replay_buffer as rb  # noqa: E402

import dyn_filter  # noqa: E402

N = 8
MIXED, PASS, FAIL = [1.0] * 5 + [0.0] * 3, [1.0] * N, [0.0] * N


class Sim:
    """Fake TransferQueue plus a rollout: prompt i gets groups[i]'s rewards after durations[i] polls (1 by default)."""

    def __init__(self, groups, durations=None, step=1):
        self.groups, self.durations, self.step = list(groups), list(durations or []), step
        self.store, self.scores, self.pending, self.now, self.dispatched = {"train": {}, "val": {}}, {}, [], 0, 0

    def kv_list(self):
        return {p: {k: dict(t) for k, t in v.items()} for p, v in self.store.items()}

    def kv_batch_get(self, keys, partition_id, select_fields):
        assert select_fields == ["rm_scores"], select_fields
        return {"rm_scores": [self.scores[k] for k in keys]}

    def kv_clear(self, partition_id, keys):
        for k in keys:
            self.store[partition_id].pop(k, None)
            self.scores.pop(k, None)

    def refill(self, n, partition="train"):
        for _ in range(n):
            i = self.dispatched
            uid = f"u{i:04d}"  # no "_": verl splits trajectory keys on it
            self.store[partition][uid] = {"is_prompt": True, "status": "running", "global_steps": self.step}
            d = self.durations[i] if i < len(self.durations) else 1
            self.pending.append((self.now + d, partition, uid, self.groups[i]))
            self.dispatched += 1
        return n

    def tick(self, partition_id=None, last=None):
        self.now += 1
        assert self.now < 1000, "sample() did not terminate"
        keep = []
        for t, p, uid, rewards in self.pending:
            if t > self.now:
                keep.append((t, p, uid, rewards))
            elif uid in self.store[p]:
                self.store[p][uid]["status"] = "finished"
                for j, r in enumerate(rewards):
                    key = f"{uid}_{j}_0"
                    self.store[p][key] = {"status": "finished"}
                    self.scores[key] = torch.tensor([0.0, 0.0, r])  # rm_scores: the reward on the last token
        self.pending = keep
        return last


def make(sim, cls=dyn_filter.CappedDapoReplayBuffer, batch=4, **kw):
    if cls is dyn_filter.CappedDapoReplayBuffer:
        buf = cls("sync", {}, 8, "drop", {"train_batch_size": batch, **kw}, refill_fn=sim.refill)
    else:  # verl's own, as trainer_base._build_replay_buffer builds it with algorithm.filter_groups enabled
        buf = cls("sync", {}, 8, "drop", {}, refill_fn=sim.refill, filter_groups_metric="reward",
                  train_batch_size=batch, gen_batch_size=1,
                  max_inflight_gen_batches=kw.get("max_inflight_gen_batches", 1))
    buf._wait_for_next_poll = sim.tick
    return buf


def run(sim, buf, batch=4, partition="train"):
    sim.refill(batch, partition)  # prepare_step's first batch
    with mock.patch.object(rb, "tq", sim):
        meta, metrics = buf.sample(global_steps=1, partition_id=partition, batch_size=batch)
    uids = sorted({k.split("_")[0] for k in meta.keys})
    return meta, metrics, uids


def rewards_of(sim, uid):
    return next(g for i, g in enumerate(sim.groups) if f"u{i:04d}" == uid)


class TestCappedDapo(unittest.TestCase):
    def test_zero_variance_groups(self):
        sim = Sim([PASS, FAIL, MIXED, [1.0] + [0.0] * 7])
        buf = make(sim)
        sim.refill(4)
        sim.tick()
        with mock.patch.object(rb, "tq", sim):
            buf._sync_metadata_from_transfer_queue()
            flat, counts = buf._dapo_filtered_keys("train")
        self.assertEqual(flat, {"u0000", "u0001"})
        self.assertEqual(dict(counts), {1.0: 1, 0.0: 1})

    def test_all_mixed_needs_no_refill(self):
        sim = Sim([MIXED] * 4)
        meta, m, uids = run(sim, make(sim))
        self.assertEqual((sim.dispatched, uids), (4, ["u0000", "u0001", "u0002", "u0003"]))
        self.assertEqual(len(meta.keys), 4 * N)
        self.assertEqual((m["training/filter_groups/cap_reached"], m["training/filter_groups/mixed_in_batch"]), (0, 4))
        self.assertNotIn("training/filter_groups/evicted_samples", m)

    def test_refill_two_per_evicted_and_surplus_discarded(self):
        sim = Sim([PASS, MIXED, FAIL, MIXED] + [MIXED] * 4)
        meta, m, uids = run(sim, make(sim))
        self.assertEqual(sim.dispatched, 8)  # 2 evicted -> 4 refills
        self.assertEqual(m["training/filter_groups/evicted_samples"], 2)
        self.assertEqual(m["training/filter_groups/discarded_surplus_samples"], 2)
        self.assertEqual([m[f"training/filter_groups/evicted_at_reward_{r}"] for r in (1, 0)], [1, 1])
        self.assertTrue(all(len(set(rewards_of(sim, u))) > 1 for u in uids))
        self.assertEqual(len(meta.keys), 4 * N)
        # evicted and surplus groups and the selected prompts are cleared; the batch's trajectories stay for the trainer
        self.assertEqual(set(sim.store["train"]), set(meta.keys))

    def test_refill_factor_one(self):
        sim = Sim([PASS, MIXED, FAIL, MIXED] + [MIXED] * 4)
        _, m, uids = run(sim, make(sim, refill_factor=1.0))
        self.assertEqual(sim.dispatched, 6)
        self.assertNotIn("training/filter_groups/discarded_surplus_samples", m)
        self.assertEqual(len(uids), 4)

    def test_cap_stops_filtering_and_fills_the_batch(self):
        for cap, dispatched in ((8, 8), (6, 8)):  # at 6: 2 refills reach the cap, then the 2 the batch still lacks
            sim = Sim([PASS] * 40)
            meta, m, uids = run(sim, make(sim, max_prompts_per_step=cap))
            self.assertEqual(sim.dispatched, dispatched, cap)
            self.assertEqual(m["training/filter_groups/evicted_samples"], 4)
            self.assertEqual([m[f"training/filter_groups/{k}"] for k in ("cap_reached", "kept_zero_variance")], [1, 4])
            self.assertEqual(len(meta.keys), 4 * N)

    def test_mixed_first_after_cap(self):
        # A flat, then C flat and D mixed refilled; at the cap B (mixed, slow) is still running, so B, C, D compete
        sim = Sim([PASS, MIXED, PASS, MIXED], durations=[1, 5, 1, 2])
        _, m, uids = run(sim, make(sim, batch=2, max_prompts_per_step=4, max_inflight_gen_batches=2), batch=2)
        self.assertEqual(uids, ["u0001", "u0003"])
        self.assertEqual(m["training/filter_groups/discarded_surplus_samples"], 1)
        self.assertEqual([m[f"training/filter_groups/{k}"] for k in ("cap_reached", "kept_zero_variance")], [1, 0])

    def test_validation_is_not_filtered(self):
        sim = Sim([PASS, FAIL, MIXED, PASS])
        meta, m, uids = run(sim, make(sim), partition="val")
        self.assertEqual((sim.dispatched, len(uids)), (4, 4))

    def test_same_counts_as_verl_without_cap(self):
        for seed in range(20):
            rng = random.Random(seed)
            groups = [rng.choices([PASS, FAIL, MIXED], weights=[0.27, 0.01, 0.72])[0] for _ in range(400)]
            durations = [rng.randint(1, 6) for _ in range(400)]
            got = []
            for cls in (dyn_filter.CappedDapoReplayBuffer, rb.ReplayBuffer):
                sim = Sim(groups, durations)
                kw = {"max_prompts_per_step": 10**6} if cls is dyn_filter.CappedDapoReplayBuffer else {}
                _, m, uids = run(sim, make(sim, cls=cls, batch=16, **kw), batch=16)
                got.append((sim.dispatched, m.get("training/filter_groups/evicted_samples", 0),
                            m.get("training/filter_groups/discarded_surplus_samples", 0), len(uids),
                            all(len(set(rewards_of(sim, u))) > 1 for u in uids)))
            self.assertEqual(got[0], got[1], seed)
            self.assertTrue(got[0][4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
