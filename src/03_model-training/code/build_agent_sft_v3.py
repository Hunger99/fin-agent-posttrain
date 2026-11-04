"""Build an agent-SFT training set from harness v3 runs as LLaMA-Factory's packed, tokenized dataset
(research/log/2026-10-31-0150.md); build_agent_sft.py does the same for version 2's collections.

Each kept trajectory is rendered from its messages exactly as they were sent and the tools it was offered (both are in
v3's trajectory line), with the chat template the v3 student is served with (--template, by default
src/02_harness/code/v3/templates/qwen35_keep_thinking.jinja): Qwen3.5's own template, except that every assistant turn after
the first user message keeps its thinking. A trajectory with a nudge (a user message after a reply without a tool call)
is therefore kept: under that template the nudge no longer drops the earlier turns' thinking, so the rendering is the
context the model saw. The loss covers each assistant turn (thinking, text and tool calls, through its <|im_end|>) that
the harness accepted, and leaves out three kinds the trajectory itself marks as wrong: a reply that drew a nudge, a
turn with a call the harness rejected (a finish that verification rejected, a write that broke a rule), and a turn with
a call that got a form error back (form_errors.py: a tool name the episode does not offer, or arguments that do not
fit the tool's parameters). These turns stay in the context, so the turns after them, the recovery, are still
learned. System, user and tool turns are masked. A call whose arguments are not a JSON object, the third form error,
cannot be rendered by the template, so its trajectory is dropped (counted as "arguments not a JSON object").

Kept: the last recorded row of each (task, run) that passed grading, at most --max-per-task runs per task (in run
order); trajectories longer than --cutoff tokens are counted, not cut. An optional general slice (--extra, sharegpt
messages such as the R3 mixture) is rendered with the same template and no tools. --add-packed takes the samples of
datasets this script or build_agent_sft.py already packed (such as A1's), split back at their sequence boundaries, so
they enter token for token unchanged and are packed again together with the new ones. Packing is LLaMA-Factory's neat
packing, as in build_agent_sft.py. Output: <out>/ (a DatasetDict for tokenized_path) and <out>.manifest.json with the
counts per run directory and template, and of the assistant turns with and without a loss.

Version 2 (research/log/2026-11-01-0010.md, 20:2x UTC) adds the third kind. M1, built before it, trained on 30 such
turns in its 4,791 market and hard-tier trajectories: 8 called a tool the episode does not offer, and 22 passed an
argument the tool does not take or left out one it needs. Errors from running a well-formed call, such as
run_python's traceback of the model's code and a ToolError, keep their loss as before. The manifest counts the turns
masked for a form error alone (assistant_turns, "masked for a form error"), and --train-form-errors gives them the
loss again, which rebuilds M1's data as it was. Samples taken with --add-packed keep the labels they were packed with,
so a rebuilt M1 takes its wealth and ML part from build_agent_sft.py version 3, which masks the same turns, rather
than from M1_wa.

Version 3 (research/log/2026-11-01-0010.md, 21:4x UTC) checks each --add-packed set before anything is built, since a
set packed before that masking, such as M1_wa, would otherwise bring back the loss on its 125 form-error turns with no
trace in the output. A set is taken only when its <dir>.manifest.json shows it was built with the masking on
(unmasked_reason); otherwise the build stops, unless --allow-unmasked-packed is given. --train-form-errors skips the
check, since its own trajectories keep that loss too, so M1 can still be rebuilt from M1_wa bit for bit. The manifest
records for each added set whether it was masked (add_packed_form_errors_masked), which the check reads when the
output is itself added later.

Usage: python -B build_agent_sft_v3.py --runs <run dir>[,<run dir>...] --tokenizer <dir> --out <dir>
       [--template <jinja>] [--cutoff 32768] [--max-per-task 2] [--extra <mix.jsonl> --extra-n 2000] [--seed 20261030]
       [--add-packed <packed dir>[,<packed dir>...] [--allow-unmasked-packed]] [--keep <keep.jsonl>]
       [--train-form-errors]
       (--runs entries may be NAME=DIR)
"""
import argparse
import collections
import hashlib
import json
import os
import random

from datasets import Dataset, DatasetDict
from llamafactory.data.processor.processor_utils import greedy_knapsack
from transformers import AutoTokenizer

from form_errors import call_args, mask_form_errors

IGNORE = -100
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "..", "..", "02_harness", "code", "v3", "templates", "qwen35_keep_thinking.jinja")


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="v3 run directories (results.jsonl, trajectories.jsonl), comma-separated, "
                                                  "each optionally NAME=DIR (NAME: its source name in a --keep list)")
    ap.add_argument("--keep", default=None, help="filter_sft.py's keep list: only these (source, task, run) episodes")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=os.path.normpath(TEMPLATE))
    ap.add_argument("--cutoff", type=int, default=32768)
    ap.add_argument("--max-per-task", type=int, default=2)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--extra-n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--add-packed", default=None, help="packed datasets whose samples are added unchanged, comma-separated")
    ap.add_argument("--allow-unmasked-packed", action="store_true",
                    help="add --add-packed sets whose manifest does not show form-error masking, which stop the build "
                         "otherwise")
    ap.add_argument("--train-form-errors", action="store_true",
                    help="give the turns with a form error a loss, as version 1 did (to rebuild M1's data)")
    return ap.parse_args(argv)


def unmasked_reason(path, _seen=None):
    """Why the packed set at path may give form-error turns a loss, or None when its manifest shows they were masked.

    Masked means that <path>.manifest.json has train_form_errors false in its args, which this script (version 2 on),
    build_agent_sft.py (version 3 on) and rl_data.py pack (version 2 on) record from the flag that turns the masking
    off, and that every set it took in with --add-packed was masked too: as its add_packed_form_errors_masked says, or,
    in a manifest written before that record (M1_fix's), by this test on each path of its args.add_packed. rl_data.py
    pack --mode opd also records train_form_errors false but keeps those turns' labels, so it does not count.
    """
    mf = path.rstrip("/") + ".manifest.json"
    seen = (_seen or set()) | {os.path.realpath(mf)}
    if not os.path.exists(mf):
        return f"{mf} does not exist"
    try:  # an unreadable manifest is a reason like any other, so the two flags still let the set in
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        args = m.get("args") or {}
        assert isinstance(args, dict)
    except (OSError, ValueError, AttributeError, AssertionError) as e:
        return f"{mf} cannot be read as a manifest ({type(e).__name__})"
    if "train_form_errors" not in args:
        return "its manifest's args have no train_form_errors, so it was built before form-error masking"
    if args["train_form_errors"]:
        return "it was built with --train-form-errors"
    if args.get("mode") == "opd":
        return "rl_data.py pack --mode opd keeps the form-error turns' labels"
    record = m.get("add_packed_form_errors_masked")
    if record is not None:
        bad = [p for p, ok in record.items() if not ok]
        return f"it took in {bad[0]} unmasked (--allow-unmasked-packed)" if bad else None
    for p in str(args.get("add_packed") or "").split(","):
        if p and os.path.realpath(p.rstrip("/") + ".manifest.json") in seen:
            return f"its args.add_packed names {p}, a set already on this chain"
        why = unmasked_reason(p, seen) if p else None
        if why:
            return f"it took in {p}, where {why}"
    return None


def unpack(path):
    """The samples of a packed DatasetDict (as pack() writes it), split at its sequence boundaries, padding dropped."""
    from datasets import load_from_disk
    out = []
    for row in load_from_disk(path)["train"]:
        b = row["packing_params"]["sequence_boundaries"][:-1]  # the last segment is the right padding
        out += [(row["input_ids"][s:e], row["labels"][s:e]) for s, e in zip(b, b[1:])]
    return out


def norm(m):
    """A message as the template takes it: tool-call arguments as objects (each a JSON object, which main checks),
    harness-private keys dropped."""
    m = {k: v for k, v in m.items() if not k.startswith("_") and k != "name"}
    if m.get("tool_calls"):
        m["tool_calls"] = [dict(c, function=dict(c["function"], arguments=call_args(c))) for c in m["tool_calls"]]
    return m


def rejected(text):
    """A tool result the harness returned as a rejection (v3's {"status": "rejected", ...}, a checklist may follow)."""
    return text.lstrip().startswith('{"status": "rejected"')


def trained_turns(msgs):
    """For each assistant message in order: False when it drew a nudge or a call of it was rejected, True otherwise;
    form errors are taken out separately, by form_errors.mask_form_errors."""
    out = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        j, ok = i + 1, True
        if not m.get("tool_calls"):
            ok = not (j < len(msgs) and msgs[j]["role"] == "user")  # a reply without a call draws a nudge
        while j < len(msgs) and msgs[j]["role"] == "tool":
            ok = ok and not rejected(msgs[j].get("content") or "")
            j += 1
        out.append(ok)
    return out


class Encoder:
    def __init__(self, tokenizer, template):
        self.tok = AutoTokenizer.from_pretrained(tokenizer)
        self.template = open(template, encoding="utf-8").read()
        self.im_start, self.im_end = self.tok.convert_tokens_to_ids("<|im_start|>"), self.tok.convert_tokens_to_ids("<|im_end|>")
        self.assistant = self.tok("assistant\n", add_special_tokens=False).input_ids

    def __call__(self, messages, tools, train=None):
        """Token ids and labels; train[k] says whether the k-th assistant turn carries a loss (default: all)."""
        text = self.tok.apply_chat_template(messages, tools=tools, chat_template=self.template, tokenize=False)
        ids = self.tok(text, add_special_tokens=False).input_ids
        labels, i, n, k = [IGNORE] * len(ids), 0, len(self.assistant), 0
        while i < len(ids):
            if ids[i] == self.im_start and ids[i + 1:i + 1 + n] == self.assistant:
                on = train is None or (k < len(train) and train[k])
                k += 1
                j = i + 1 + n
                while j < len(ids) and ids[j] != self.im_end:
                    if on:
                        labels[j] = ids[j]
                    j += 1
                if j < len(ids) and on:
                    labels[j] = ids[j]
                i = j
            i += 1
        return ids, labels, k


def last_rows(path):
    d = {}
    if os.path.exists(path):
        for r in map(json.loads, open(path, encoding="utf-8")):
            d[(r["task_id"], r["run"])] = r
    return d


def pack(samples, cutoff, pad_id):
    """Neat packing as LLaMA-Factory 0.9.5's PackedSupervisedDatasetProcessor does it (as build_agent_sft.py)."""
    lengths = [len(s[0]) for s in samples]
    by_len = collections.defaultdict(list)
    for i, n in enumerate(lengths):
        by_len[n].append(i)
    cols = collections.defaultdict(list)
    for knapsack in greedy_knapsack(list(lengths), cutoff):  # it empties the list it is given
        ids, mask, pos, lab, bounds = [], [], [], [], [0]
        for j, n in enumerate(knapsack):
            s = samples[by_len[n].pop()]
            ids += s[0]
            lab += s[1]
            pos += list(range(n))
            mask += [j + 1] * n
            bounds.append(bounds[-1] + n)
        pad = cutoff - len(ids) + 1
        ids += [pad_id] * pad
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
    return cols


def main(argv=None):
    a = parse(argv)
    packed = [p for p in (a.add_packed or "").split(",") if p]
    why_unmasked = {p: unmasked_reason(p) for p in packed}
    for p, why in why_unmasked.items():
        if why and not (a.train_form_errors or a.allow_unmasked_packed):
            raise SystemExit(f"--add-packed {p}: {why}; its labels enter unchanged, so its form-error turns would keep "
                             "their loss. Rebuild it with form-error masking on, or pass --allow-unmasked-packed to "
                             "add it as it is")
    enc = Encoder(a.tokenizer, a.template)
    samples, counts, dropped, turns = [], collections.Counter(), collections.Counter(), collections.Counter()
    named = [x.split("=", 1) if "=" in x else (os.path.basename(x.rstrip("/")), x) for x in a.runs.split(",") if x]
    run_dirs = [d for _, d in named]
    keep = None
    if a.keep:
        keep = {(j["source"], j["task_id"], j["run"]) for j in map(json.loads, open(a.keep, encoding="utf-8"))}
    for name, d in named:
        res = last_rows(f"{d}/results.jsonl")
        if any(r.get("harness") != "v3" for r in res.values()):
            raise SystemExit(f"{d}/results.jsonl holds rows that are not harness v3's; use build_agent_sft.py")
        traj = {}
        for t in map(json.loads, open(f"{d}/trajectories.jsonl", encoding="utf-8")):
            traj[(t["task_id"], t["run"])] = t  # the last line of a (task, run) is the one its last row belongs to
        per_task = collections.Counter()
        for k in sorted(res):
            r = res[k]
            if not r["passed"]:
                continue
            if keep is not None and (name,) + k not in keep:
                dropped["not in the keep list"] += 1
                continue
            if k not in traj:
                dropped["no trajectory"] += 1
                continue
            if per_task[k[0]] >= a.max_per_task:
                dropped["over max per task"] += 1
                continue
            msgs = traj[k]["messages"]
            if any(call_args(c) is None for m in msgs for c in m.get("tool_calls") or []):
                dropped["arguments not a JSON object"] += 1  # the template renders a call from its arguments' items
                continue
            train, n_form = trained_turns(msgs), 0
            if not a.train_form_errors:
                train, n_form = mask_form_errors(msgs, train)
            ids, labels, n_asst = enc([norm(m) for m in msgs], traj[k]["tools"], train)
            if n_asst != len(train):
                raise SystemExit(f"{d} {k}: {n_asst} assistant turns rendered, {len(train)} in the messages")
            if len(ids) > a.cutoff:
                dropped[f"longer than {a.cutoff}"] += 1
                continue
            if not any(train):
                dropped["no turn with a loss"] += 1
                continue
            per_task[k[0]] += 1
            counts[(name, r.get("template") or "-")] += 1
            turns["with a loss"] += sum(train)
            turns["masked"] += len(train) - sum(train)
            turns["masked for a form error"] += n_form
            turns["trajectories with a nudge"] += any(m["role"] == "user" for m in msgs[2:])
            samples.append((ids, labels))
    n_traj = len(samples)
    if a.extra and a.extra_n:
        rows = [json.loads(line) for line in open(a.extra, encoding="utf-8")]
        for r in random.Random(a.seed).sample(rows, a.extra_n):
            ids, labels, _ = enc(r["messages"], None)
            if len(ids) <= a.cutoff:
                samples.append((ids, labels))
                counts[("general", "-")] += 1
            else:
                dropped["general longer than cutoff"] += 1
    for p in packed:
        added = unpack(p)
        long = sum(len(s[0]) > a.cutoff for s in added)
        if long:
            raise SystemExit(f"{p}: {long} samples longer than --cutoff {a.cutoff}")
        samples += added
        counts[("packed:" + os.path.basename(p.rstrip("/")), "-")] += len(added)
    if not samples:
        raise SystemExit("nothing to train on")
    cols = pack(samples, a.cutoff, enc.tok.pad_token_id)
    DatasetDict({"train": Dataset.from_dict(dict(cols))}).save_to_disk(a.out)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]  # noqa: E731
    n_added = sum(v for k, v in counts.items() if k[0].startswith("packed:"))
    manifest = {"trajectories": n_traj, "general": len(samples) - n_traj - n_added, "added_from_packed": n_added,
                "packed_sequences": len(cols["input_ids"]),
                "tokens": sum(len(s[0]) for s in samples),
                "loss_tokens": sum(sum(x != IGNORE for x in s[1]) for s in samples),
                "assistant_turns": dict(turns), "dropped": dict(dropped),
                "add_packed_form_errors_masked": {p: why is None for p, why in why_unmasked.items()},
                "by_source": {"/".join(k): v for k, v in sorted(counts.items())}, "args": vars(a),
                "template_sha": sha(a.template), "inputs": {d: sha(f"{d}/results.jsonl") for d in run_dirs},
                "keep_sha": sha(a.keep) if a.keep else None}
    json.dump(manifest, open(a.out.rstrip("/") + ".manifest.json", "w"), indent=1)
    print(json.dumps({k: manifest[k] for k in ("trajectories", "general", "added_from_packed", "packed_sequences", "tokens", "loss_tokens",
                                               "assistant_turns", "dropped", "add_packed_form_errors_masked")}))


if __name__ == "__main__":
    main()
