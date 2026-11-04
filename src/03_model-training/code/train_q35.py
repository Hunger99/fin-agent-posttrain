"""LLaMA-Factory SFT for Qwen3.5 with packed samples kept apart (research/log/2026-10-29-1815.md).

Three quarters of Qwen3.5's layers are gated delta-net (linear attention) layers, each with a short causal convolution
in front. With neat packing, LLaMA-Factory 0.9.5 gives the model only position ids that restart at each sample; the
full-attention layers read the sample boundaries from them, but the linear-attention layers do not, so a sample's
recurrent state and its first three convolution inputs come from the sample packed before it. In a check on
Qwen3.5-4B-Base this moved a packed sample's per-token loss by up to 6.3 nats. transformers 5.20 keeps the samples apart
when the batch also carries the boundaries in the two forms its kernels take: cu_seq_lens_q/k (and max_length_q/k) for
flash-linear-attention's chunk_gated_delta_rule, and seq_idx, each token's sample number, for causal-conv1d; with both,
the largest per-token difference fell to 0.13 nats, bf16 noise. This entry point adds them to every packed batch and
otherwise runs LLaMA-Factory as its own launcher does. LLaMA-Factory's own fix for the same leak (patch_qwen3_5_forward,
from axolotl) replaces the decoder and delta-net forwards written for transformers <= 5.6 and fails on 5.20 (the decoder
layer has no layer_type), so it is switched off here.

Agent SFT packs sequences of 16,384 tokens (research/log/2026-10-30-0855.md), where the stock loss materializes the
logits of the whole 248k-word vocabulary for every position and upcasts them to fp32, 15 GiB for one sequence, which
ran out of memory on the A6000s. The loss here takes the vocabulary projection and the cross-entropy only at the
positions that carry a label, in chunks recomputed during the backward pass; it is the stock causal-LM loss (labels
shifted by one, summed and divided by num_items_in_batch when the trainer passes it, else averaged), so the value and
the gradients are unchanged.

Usage: torchrun --nproc_per_node <gpus> train_q35.py <config.yaml>   (in venvs/train-q35)
"""
import torch
import torch.nn.functional as F
from llamafactory.data import collator
from llamafactory.model import patcher
from llamafactory.train.tuner import run_exp
from torch.utils.checkpoint import checkpoint
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

CHUNK = 2048  # labelled positions per vocabulary-projection chunk: 2,048 x 248,320 fp32 logits is about 2 GB

_call = collator.SFTDataCollatorWith4DAttentionMask.__call__
_shown = []


def call(self, features):
    batch = _call(self, features)
    pos = batch.get("position_ids")
    if self.neat_packing and pos is not None:
        pos = pos[0] if pos.dim() == 3 else pos  # multimodal rope: the three text rows are the same
        (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(pos)
        seq_idx = (torch.cumsum((pos == 0).int(), dim=-1) - 1).to(torch.int32)
        batch.update(cu_seq_lens_q=cu_q, cu_seq_lens_k=cu_k, max_length_q=max_q, max_length_k=max_k, seq_idx=seq_idx)
    if not _shown:  # the first batch's boundaries go to the log, as evidence the patch is active
        _shown.append(1)
        print(f"train_q35: first batch {tuple(batch['input_ids'].shape)}, samples packed: "
              f"{0 if pos is None else int((pos == 0).sum())}, keys: {sorted(batch)}", flush=True)
    return batch


collator.SFTDataCollatorWith4DAttentionMask.__call__ = call
patcher.patch_qwen3_5_forward = lambda model: print("train_q35: LLaMA-Factory's Qwen3.5 forward patch skipped", flush=True)

_forward = q35.Qwen3_5ForConditionalGeneration.forward


def _chunk_loss(hidden, target, weight):
    return F.cross_entropy(F.linear(hidden, weight).float(), target, reduction="sum")


def forward(self, input_ids=None, labels=None, logits_to_keep=0, **kwargs):
    if labels is None:
        return _forward(self, input_ids=input_ids, labels=None, logits_to_keep=logits_to_keep, **kwargs)
    outputs = self.model(input_ids=input_ids, **kwargs)
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
    keep = shifted != -100
    hidden, target = outputs[0][keep], shifted[keep]
    total = outputs[0].new_zeros((), dtype=torch.float32)
    for i in range(0, target.numel(), CHUNK):
        total = total + checkpoint(_chunk_loss, hidden[i:i + CHUNK], target[i:i + CHUNK], self.lm_head.weight,
                                   use_reentrant=False)
    n = kwargs.get("num_items_in_batch")
    loss = total / (n if n is not None else max(target.numel(), 1))
    return q35.Qwen3_5CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=outputs.past_key_values,
                                             hidden_states=outputs.hidden_states, attentions=outputs.attentions)


q35.Qwen3_5ForConditionalGeneration.forward = forward

if __name__ == "__main__":
    run_exp()
