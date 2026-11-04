"""Build the agent-SFT training set from collected trajectories as LLaMA-Factory's packed, tokenized dataset
(research/log/2026-10-30-0855.md).

Each kept trajectory is rendered with Qwen3.5's own chat template, the one Qwen3.5-4B-Base and Qwen3.5-4B ship and the
student is served with: the tools in the system turn, tool calls in its XML form, and every turn's thinking. LLaMA-Factory's
sharegpt format has no place for the thinking that precedes a tool call, so the rendering is done here and handed to
LLaMA-Factory through tokenized_path. The loss covers each assistant turn (thinking, text and tool calls, through its
<|im_end|>) except one with a call that got a form error back (version 3 below), which stays in the context; system,
user and tool turns are masked. A trajectory with a call whose arguments are not a JSON object cannot be rendered by
the template, so it is dropped (counted as "arguments not a JSON object"). Samples are packed as LLaMA-Factory's neat
packing packs them (greedy knapsack to the cutoff, per-sample attention-mask numbers, restarting position ids), so
train_q35.py can give the linear-attention layers the sample boundaries.

Kept: the last recorded row of each (task, run) that passed grading, its trajectory with one user turn (a second one is
the harness's nudge after a reply without tool calls, which would also make the template drop the earlier thinking),
and at most --max-per-task runs per task; trajectories longer than --cutoff tokens are counted, not cut. An optional
general slice (--extra, sharegpt messages such as the R3 mixture) is rendered with the same template and no tools.
Output: <out>/ (a DatasetDict for tokenized_path) and <out>.manifest.json with the counts per source and template.

Version 2 (research/log/2026-10-31-0150.md) adds two options, both off by default so the default output is version 1's:
--keep takes filter_sft.py's keep list (sources named <env>_<sampled|golden>) and keeps only the episodes on it;
--mask-rejected leaves out of the loss the assistant turns the trajectory marks as wrong, as build_agent_sft_v3.py does
(a reply that drew a nudge, a turn with a call the harness rejected), keeping them in the context.

Version 3 (research/log/2026-11-01-0010.md, 20:2x UTC) leaves out of the loss, by default, each assistant turn with a
call that got a form error back (form_errors.py: a tool name the episode does not offer, or arguments that do not fit
the tool's parameters, a write's schema rejection for a missing or unexpected key included), as build_agent_sft_v3.py
version 2 does, and keeps the turn in the context. M1_wa, built before it, trained on 125 such turns in its 5,684
trajectories: 121 called a tool the episode does not offer, mostly a variant of set_checklist or redeem (checklist,
update_checklist, redemption, redem), and 4 passed an argument the tool does not take or left out one it needs; none
was a schema rejection. Errors from running a well-formed call keep their loss, and --mask-rejected alone still decides
whether nudged turns and the other rejected ones do (a write that breaks a rule or has a value it cannot take, such
as a non-positive amount, and a finish that verification rejected). The manifest counts the turns masked for a form
error (assistant_turns, "masked for a form error"), and --train-form-errors gives them the loss again, which rebuilds
A1's and M1_wa's data as they were.

Usage: python -B build_agent_sft.py --collect <collect dir> --code <harness code snapshot> --tokenizer <dir> --out <dir>
       [--cutoff 16384] [--max-per-task 2] [--extra <mix.jsonl> --extra-n 2000] [--seed 20261030]
       [--keep <keep.jsonl>] [--mask-rejected] [--train-form-errors]
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

from datasets import Dataset, DatasetDict
from llamafactory.data.processor.processor_utils import greedy_knapsack
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))
from form_errors import call_args, mask_form_errors  # noqa: E402

IGNORE = -100
ap = argparse.ArgumentParser()
ap.add_argument("--collect", required=True)
ap.add_argument("--code", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cutoff", type=int, default=16384)
ap.add_argument("--max-per-task", type=int, default=2)
ap.add_argument("--extra", default=None)
ap.add_argument("--extra-n", type=int, default=0)
ap.add_argument("--seed", type=int, default=20261030)
ap.add_argument("--keep", default=None)
ap.add_argument("--mask-rejected", action="store_true")
ap.add_argument("--train-form-errors", action="store_true",
                help="give the turns with a form error a loss, as version 2 did (to rebuild A1's and M1_wa's data)")
a = ap.parse_args()

sys.path.insert(0, a.code)
import agent  # noqa: E402
import ml_tasks  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.tokenizer)
IM_START, IM_END = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
ASSISTANT = tok("assistant\n", add_special_tokens=False).input_ids
TOOLS = {"wealth": agent.TOOLS + [agent.CHECKLIST_TOOL], "ml": ml_tasks.TOOLS + [agent.CHECKLIST_TOOL]}  # H1's tools


def encode(messages, tools, train=None):
    """Token ids, labels and the number of assistant turns rendered: only the assistant turns, through their
    <|im_end|>, carry a loss; train[k] (default: all) says whether the k-th assistant turn does."""
    ids = tok(tok.apply_chat_template(messages, tools=tools, tokenize=False), add_special_tokens=False).input_ids
    labels, i, n, k = [IGNORE] * len(ids), 0, len(ASSISTANT), 0
    while i < len(ids):
        if ids[i] == IM_START and ids[i + 1:i + 1 + n] == ASSISTANT:
            on = train is None or (k < len(train) and train[k])
            k += 1
            j = i + 1 + n
            while j < len(ids) and ids[j] != IM_END:
                if on:
                    labels[j] = ids[j]
                j += 1
            if j < len(ids) and on:
                labels[j] = ids[j]
            i = j
        i += 1
    return ids, labels, k


def trained_turns(msgs):
    """As build_agent_sft_v3.trained_turns (--mask-rejected): False for each assistant message that drew a nudge or
    had a call rejected; form errors are taken out separately, by form_errors.mask_form_errors."""
    out = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        j, ok = i + 1, True
        if not m.get("tool_calls"):
            ok = not (j < len(msgs) and msgs[j]["role"] == "user")
        while j < len(msgs) and msgs[j]["role"] == "tool":
            ok = ok and not (msgs[j].get("content") or "").lstrip().startswith('{"status": "rejected"')
            j += 1
        out.append(ok)
    return out


def norm(m):
    m = {k: v for k, v in m.items() if not k.startswith("_") and k != "name"}
    if m.get("tool_calls"):
        m["tool_calls"] = [dict(c, function=dict(c["function"], arguments=call_args(c))) for c in m["tool_calls"]]
    return m


def last_rows(path):
    d = {}
    if os.path.exists(path):
        for r in map(json.loads, open(path, encoding="utf-8")):
            d[(r["task_id"], r["run"])] = r
    return d


samples, counts, dropped, turns = [], collections.Counter(), collections.Counter(), collections.Counter()
keep = None
if a.keep:
    keep = {(j["source"], j["task_id"], j["run"]) for j in map(json.loads, open(a.keep, encoding="utf-8"))}
for env in ("wealth", "ml"):
    tmpl = {t["task_id"]: t["template"] for t in map(json.loads, open(f"{a.collect}/{env}/round1_tasks.jsonl"))}
    for source in ("sampled", "golden"):
        d = f"{a.collect}/{env}/{source}"
        res = last_rows(f"{d}/results.jsonl")
        traj = {}
        if os.path.exists(f"{d}/trajectories.jsonl"):
            for t in map(json.loads, open(f"{d}/trajectories.jsonl", encoding="utf-8")):
                traj[(t["task_id"], t["run"])] = t
        per_task = collections.Counter()
        for k in sorted(res):
            r = res[k]
            if not r["passed"] or k not in traj:
                continue
            if keep is not None and (f"{env}_{source}",) + k not in keep:
                dropped["not in the keep list"] += 1
                continue
            msgs = traj[k]["messages"]
            if sum(m["role"] == "user" for m in msgs) != 1:
                dropped["second user turn"] += 1
                continue
            if per_task[k[0]] >= a.max_per_task:
                dropped["over max per task"] += 1
                continue
            if any(call_args(c) is None for m in msgs for c in m.get("tool_calls") or []):
                dropped["arguments not a JSON object"] += 1  # the template renders a call from its arguments' items
                continue
            train, n_form = (trained_turns(msgs) if a.mask_rejected else None), 0
            if not a.train_form_errors:
                masked, n_form = mask_form_errors(msgs, train)
                train = masked if n_form else train  # None (every turn) where none is masked, as before
            ids, labels, n_asst = encode([norm(m) for m in msgs], TOOLS[env], train)
            if train is not None and n_asst != len(train):
                raise SystemExit(f"{d} {k}: {n_asst} assistant turns rendered, {len(train)} in the messages")
            if len(ids) > a.cutoff:
                dropped[f"longer than {a.cutoff}"] += 1
                continue
            per_task[k[0]] += 1
            turns["masked for a form error"] += n_form
            counts[(env, source, tmpl[k[0]])] += 1
            samples.append((ids, labels))
n_traj = len(samples)
if a.extra and a.extra_n:
    rows = [json.loads(line) for line in open(a.extra, encoding="utf-8")]
    for r in random.Random(a.seed).sample(rows, a.extra_n):
        ids, labels, _ = encode(r["messages"], None)
        if len(ids) <= a.cutoff:
            samples.append((ids, labels))
            counts[("general", "extra", "-")] += 1
        else:
            dropped["general longer than cutoff"] += 1

# neat packing as LLaMA-Factory 0.9.5's PackedSupervisedDatasetProcessor does it
lengths = [len(s[0]) for s in samples]
by_len = collections.defaultdict(list)
for i, n in enumerate(lengths):
    by_len[n].append(i)
cols = collections.defaultdict(list)
for knapsack in greedy_knapsack(list(lengths), a.cutoff):  # it empties the list it is given
    ids, mask, pos, lab, bounds = [], [], [], [], [0]
    for j, n in enumerate(knapsack):
        s = samples[by_len[n].pop()]
        ids += s[0]
        lab += s[1]
        pos += list(range(n))
        mask += [j + 1] * n
        bounds.append(bounds[-1] + n)
    pad = a.cutoff - len(ids) + 1
    ids += [tok.pad_token_id] * pad
    pos += [0] * pad
    lab += [IGNORE] * pad
    mask += [0] * pad
    bounds.append(bounds[-1] + pad)
    cols["input_ids"].append(ids)
    cols["attention_mask"].append(mask)
    cols["position_ids"].append(pos)
    cols["labels"].append(lab)
    cols["packing_params"].append({"sequence_boundaries": bounds, "image_subseq_ids": [2 ** 31 - 1],
                                   "video_subseq_ids": [2 ** 31 - 1], "audio_subseq_ids": [2 ** 31 - 1],
                                   "right_padding_length": pad})
    for k in ("images", "videos", "audios"):
        cols[k].append(None)
DatasetDict({"train": Dataset.from_dict(dict(cols))}).save_to_disk(a.out)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]  # noqa: E731
manifest = {"trajectories": n_traj, "general": len(samples) - n_traj, "packed_sequences": len(cols["input_ids"]),
            "tokens": sum(lengths), "loss_tokens": sum(sum(x != IGNORE for x in s[1]) for s in samples),
            "assistant_turns": dict(turns), "dropped": dict(dropped),
            "by_source": {"/".join(k): v for k, v in sorted(counts.items())},
            "args": vars(a), "inputs": {f"{e}/{s}": sha(f"{a.collect}/{e}/{s}/results.jsonl")
                                         for e in ("wealth", "ml") for s in ("sampled", "golden")
                                         if os.path.exists(f"{a.collect}/{e}/{s}/results.jsonl")}}
json.dump(manifest, open(a.out.rstrip("/") + ".manifest.json", "w"), indent=1)
print(json.dumps({k: manifest[k] for k in ("trajectories", "general", "packed_sequences", "tokens", "loss_tokens",
                                           "assistant_turns", "dropped")}))
