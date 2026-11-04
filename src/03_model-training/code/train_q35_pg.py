"""train_q35.py with a per-token policy-gradient loss, for the OPD arm of the RL/OPD pilot (research/log/2026-10-30-1830.md).

A packed dataset from rl_data.py pack --mode opd carries two columns beside the labels: token_lp_old, the sampling
model's log-probability of each labelled token, and token_adv, its advantage (for OPD the teacher's minus the sampling
model's log-probability, the per-token reverse-KL term). The loss is PPO's clipped surrogate,
    -sum_t min(r_t A_t, clip(r_t, 1 - EPS, 1 + EPS) A_t),  r_t = exp(log p(x_t) - token_lp_old_t),
over the labelled tokens, divided as train_q35.py divides the cross-entropy (by num_items_in_batch when the trainer
passes it); the log-probabilities are taken in the same recomputed chunks. Batches without these columns get
train_q35.py's cross-entropy, so the entry point also trains a plain SFT set. Needs remove_unused_columns: false in the
config, so the trainer hands the two columns to the model.

Usage: torchrun --nproc_per_node <gpus> train_q35_pg.py <config.yaml>   (in venvs/train-q35)
"""
import os

import torch
import torch.nn.functional as F
from llamafactory.data import collator
from llamafactory.train.tuner import run_exp
from torch.utils.checkpoint import checkpoint

import train_q35  # noqa: F401  (packing boundaries for the linear-attention layers, the chunked SFT loss)
from train_q35 import CHUNK, q35

EPS = 0.2
CHUNK = train_q35.CHUNK = int(os.environ.get("Q35_LOSS_CHUNK", CHUNK))  # smaller only for smoke tests on a shared GPU
_sft_forward = q35.Qwen3_5ForConditionalGeneration.forward
_calls = [0]
_call = collator.SFTDataCollatorWith4DAttentionMask.__call__
COLS = ("token_adv", "token_lp_old")


def call(self, features):
    """The two columns go around LLaMA-Factory's collator: for a model with a vision tower it adds a dummy image's
    tokens to the first sequence (at the padding side), so the columns are placed against the real tokens and the
    rest is zero; those tokens carry no label."""
    extra = {k: [f.pop(k) for f in features] for k in COLS if k in features[0]}
    batch = _call(self, features)
    n = batch["input_ids"].shape[1]
    for k, rows in extra.items():
        t = torch.zeros(len(rows), n)
        for i, v in enumerate(rows):
            v = torch.tensor(v, dtype=torch.float32)
            if self.tokenizer.padding_side == "left":
                t[i, n - len(v):] = v
            else:
                t[i, :len(v)] = v
        batch[k] = t
    return batch


collator.SFTDataCollatorWith4DAttentionMask.__call__ = call


def _chunk_lp(hidden, target, weight):
    return -F.cross_entropy(F.linear(hidden, weight).float(), target, reduction="none")


def forward(self, input_ids=None, labels=None, token_adv=None, token_lp_old=None, logits_to_keep=0, **kwargs):
    if labels is None or token_adv is None:
        return _sft_forward(self, input_ids=input_ids, labels=labels, logits_to_keep=logits_to_keep, **kwargs)
    outputs = self.model(input_ids=input_ids, **kwargs)

    def shift(x, v):  # the value for the token each position predicts, as the labels are shifted
        return F.pad(x, (0, 1), value=v)[..., 1:]

    keep = shift(labels, -100) != -100
    hidden, target = outputs[0][keep], shift(labels, -100)[keep]
    adv, old = shift(token_adv.float(), 0.0)[keep], shift(token_lp_old.float(), 0.0)[keep]
    total = outputs[0].new_zeros((), dtype=torch.float32)
    clipped = 0
    for i in range(0, target.numel(), CHUNK):
        lp = checkpoint(_chunk_lp, hidden[i:i + CHUNK], target[i:i + CHUNK], self.lm_head.weight, use_reentrant=False)
        ratio, a = torch.exp(lp - old[i:i + CHUNK]), adv[i:i + CHUNK]
        total = total - torch.minimum(ratio * a, ratio.clamp(1 - EPS, 1 + EPS) * a).sum()
        clipped += int(((ratio - 1).abs() > EPS).sum())
    n = kwargs.get("num_items_in_batch")
    loss = total / (n if n is not None else max(target.numel(), 1))
    _calls[0] += 1
    if _calls[0] <= 3 or _calls[0] % 50 == 0:  # evidence the PG loss is active, and how far the policy has moved
        print(f"train_q35_pg: call {_calls[0]}, tokens {target.numel()}, mean adv {adv.mean().item():.3f}, "
              f"clipped {clipped / max(target.numel(), 1):.3f}, loss {loss.item():.4f}", flush=True)
    return q35.Qwen3_5CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=outputs.past_key_values,
                                             hidden_states=outputs.hidden_states, attentions=outputs.attentions)


q35.Qwen3_5ForConditionalGeneration.forward = forward

if __name__ == "__main__":
    run_exp()
