"""Training sets for the RL/OPD pilot from harness rollouts (research/log/2026-10-30-1830.md).

Three steps, each a subcommand:
  render  each (task, run)'s last recorded row and its trajectory, rendered as build_agent_sft.py renders them (Qwen3.5's
          own chat template, the episode's H1 tools), one JSON line per trajectory with its token ids, labels on every
          assistant turn through its <|im_end|>, its grade, and form_error_spans, the [start, end) token spans of the
          turns that got a form error back (form_errors.py); a trajectory with a call whose arguments are not a JSON
          object cannot be rendered by the template and is dropped;
  score   adds one model's log-probability of every labelled token, from a vLLM server's prompt_logprobs (the teacher's
          for OPD, the sampling model's own as the old policy); teacher and student share one tokenizer, so the same ids
          are scored by both;
  pack    the LLaMA-Factory packed dataset, as build_agent_sft.py packs it: --mode sft keeps passed trajectories (at most
          --max-per-task per task) with the plain loss, less the form_error_spans, as build_agent_sft.py version 3
          leaves those turns out (--train-form-errors keeps them); --mode opd keeps every trajectory and adds two
          per-token columns aligned with labels, token_lp_old (the sampling model's log-probability) and token_adv
          (teacher minus old log-probability, the per-token reverse-KL advantage, clipped to +-adv_clip), which
          train_q35_pg.py reads.

Version 2 (research/log/2026-11-01-0010.md, 20:2x UTC) adds form_error_spans and the drop to render and the masking to
pack --mode sft. A file rendered before has no form_error_spans, so pack --mode sft stops on it unless
--train-form-errors is given, which packs the labels as rendered. --mode opd is unchanged and keeps the form-error
turns: its per-token loss follows the teacher-minus-old advantage instead of imitating the sampled token, so the tokens
of a call the teacher finds less likely than the sampling model did get a negative advantage and are made less likely,
not learned.

Usage: python -B rl_data.py render --rollouts <dir>[,<dir>...] --tasks <tasks.jsonl>[,...] --code <harness snapshot>
           --tokenizer <dir> --out <rendered.jsonl> [--cutoff 16384]
       python -B rl_data.py score --rendered <in.jsonl> --out <out.jsonl> --base-url <url> --model <name> --field <name>
           [--concurrency 4]
       python -B rl_data.py pack --rendered <in.jsonl> --mode sft|opd --out <dir> [--max-per-task 2] [--adv-clip 5]
           [--train-form-errors]
"""
import argparse
import asyncio
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))
from form_errors import call_args, mask_form_errors  # noqa: E402

IGNORE = -100
ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("render")
r.add_argument("--rollouts", required=True)
r.add_argument("--tasks", required=True)
r.add_argument("--code", required=True)
r.add_argument("--tokenizer", required=True)
r.add_argument("--out", required=True)
r.add_argument("--cutoff", type=int, default=16384)
s = sub.add_parser("score")
s.add_argument("--rendered", required=True)
s.add_argument("--out", required=True)
s.add_argument("--base-url", required=True)
s.add_argument("--model", required=True)
s.add_argument("--field", required=True)
s.add_argument("--concurrency", type=int, default=4)
p = sub.add_parser("pack")
p.add_argument("--rendered", required=True)
p.add_argument("--mode", choices=["sft", "opd"], required=True)
p.add_argument("--out", required=True)
p.add_argument("--cutoff", type=int, default=16384)
p.add_argument("--max-per-task", type=int, default=2)
p.add_argument("--adv-clip", type=float, default=5.0)
p.add_argument("--tokenizer", default=os.environ.get("FP_MODELS_D2", "<FP_MODELS_D2 unset>") + "/Qwen3.5-4B-Base")
p.add_argument("--train-form-errors", action="store_true",
               help="--mode sft: give the turns with a form error a loss, as version 1 did")
a = ap.parse_args()


def render():
    from transformers import AutoTokenizer
    sys.path.insert(0, a.code)
    import agent
    import ml_tasks
    import report_gate
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    head = tok("assistant\n", add_special_tokens=False).input_ids
    # the tools an H1 episode offers: its class's tools and the checklist (agent.Episode.__init__)
    kinds = {"wealth": agent.Episode, "ml": ml_tasks.MLEpisode, "report_gate": report_gate.GateEpisode}
    tools = {k: c.TOOLS + [agent.CHECKLIST_TOOL] for k, c in kinds.items()}

    def encode(messages, tl):  # as build_agent_sft.py; also each assistant turn's labelled [start, end)
        ids = tok(tok.apply_chat_template(messages, tools=tl, tokenize=False), add_special_tokens=False).input_ids
        labels, spans, i, n = [IGNORE] * len(ids), [], 0, len(head)
        while i < len(ids):
            if ids[i] == im_start and ids[i + 1:i + 1 + n] == head:
                j = i + 1 + n
                while j < len(ids) and ids[j] != im_end:
                    labels[j] = ids[j]
                    j += 1
                if j < len(ids):
                    labels[j] = ids[j]
                spans.append([i + 1 + n, min(j + 1, len(ids))])
                i = j
            i += 1
        return ids, labels, spans

    def norm(m):  # as build_agent_sft.py
        m = {k: v for k, v in m.items() if not k.startswith("_") and k != "name"}
        if m.get("tool_calls"):
            m["tool_calls"] = [dict(c, function=dict(c["function"], arguments=call_args(c))) for c in m["tool_calls"]]
        return m

    def lines(path):  # a run still writing may have a partial last line, which is skipped
        for line in open(path, encoding="utf-8"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                dropped["unreadable line"] += 1

    task = {}
    for f in a.tasks.split(","):
        for t in map(json.loads, open(f, encoding="utf-8")):
            task[t["task_id"]] = t
    kept, form_turns, dropped = 0, 0, collections.Counter()
    with open(a.out, "w", encoding="utf-8") as out:
        for d in a.rollouts.split(","):
            rows = {}
            for row in lines(f"{d}/results.jsonl"):
                rows[(row["task_id"], row["run"])] = row
            for tr in lines(f"{d}/trajectories.jsonl"):
                k = (tr["task_id"], tr["run"])
                row = rows.get(k)
                if tr["task_id"] not in task:
                    continue
                if row is None or row["end"] in ("api_error", "budget"):
                    dropped["no graded row"] += 1
                    continue
                msgs = tr["messages"]
                if sum(m["role"] == "user" for m in msgs) != 1:
                    dropped["second user turn"] += 1
                    continue
                if any(call_args(c) is None for m in msgs for c in m.get("tool_calls") or []):
                    dropped["arguments not a JSON object"] += 1  # the template renders a call from its arguments' items
                    continue
                t = task[tr["task_id"]]
                env = t.get("env", "wealth")
                ids, labels, spans = encode([norm(m) for m in msgs], tools[env])
                if len(ids) > a.cutoff:
                    dropped[f"longer than {a.cutoff}"] += 1
                    continue
                train, n_form = mask_form_errors(msgs)
                if n_form and len(spans) != len(train):
                    raise SystemExit(f"{d} {k}: {len(spans)} assistant turns rendered, {len(train)} in the messages")
                out.write(json.dumps({"task_id": k[0], "run": k[1], "source": os.path.basename(d.rstrip("/")),
                                      "template": t["template"], "env": env, "passed": bool(row["passed"]),
                                      "ids": ids, "labels": labels,
                                      "form_error_spans": [sp for sp, on in zip(spans, train) if not on]}) + "\n")
                kept += 1
                form_turns += n_form
    print(json.dumps({"kept": kept, "turns with a form error": form_turns, "dropped": dict(dropped)}))


def score():
    from openai import AsyncOpenAI
    rows = [json.loads(line) for line in open(a.rendered, encoding="utf-8")]
    client = AsyncOpenAI(base_url=a.base_url, api_key="none", timeout=3600, max_retries=3)
    sem = asyncio.Semaphore(a.concurrency)

    async def one(x):
        async with sem:
            resp = await client.completions.create(model=a.model, prompt=x["ids"], max_tokens=1, temperature=0,
                                                   extra_body={"prompt_logprobs": 0})
        plp = resp.choices[0].prompt_logprobs  # one entry per prompt token (None for the first), keyed by token id
        lp = []
        for i, (tid, lab) in enumerate(zip(x["ids"], x["labels"])):
            e = plp[i] if i < len(plp) else None
            v = (e or {}).get(str(tid)) if lab != IGNORE else None
            lp.append(None if v is None else (v["logprob"] if isinstance(v, dict) else v.logprob))
        miss = sum(v is None for v, lab in zip(lp, x["labels"]) if lab != IGNORE)
        x[a.field] = [0.0 if v is None else v for v in lp]
        x[a.field + "_missing"] = miss
        return x

    async def main():
        done = await asyncio.gather(*(one(x) for x in rows))
        with open(a.out, "w", encoding="utf-8") as out:
            for x in done:
                out.write(json.dumps(x) + "\n")
        print(json.dumps({"scored": len(done), "labelled tokens missing a log-prob": sum(x[a.field + "_missing"] for x in done)}))

    asyncio.run(main())


def pack():
    from datasets import Dataset, DatasetDict
    from llamafactory.data.processor.processor_utils import greedy_knapsack
    from transformers import AutoTokenizer
    pad_id = AutoTokenizer.from_pretrained(a.tokenizer).pad_token_id
    rows = [json.loads(line) for line in open(a.rendered, encoding="utf-8")]
    samples, counts, per_task, n_form = [], collections.Counter(), collections.Counter(), 0
    for x in sorted(rows, key=lambda x: (x["task_id"], x["source"], x["run"])):
        labels = x["labels"]
        if a.mode == "sft":
            if not x["passed"] or per_task[x["task_id"]] >= a.max_per_task:
                continue
            per_task[x["task_id"]] += 1
            extra = None
            if not a.train_form_errors:
                if "form_error_spans" not in x:
                    raise SystemExit(f"{a.rendered} has no form_error_spans (rendered before version 2): render it "
                                     "again, or pass --train-form-errors to pack its labels as they are")
                labels = list(labels)
                for st, e in x["form_error_spans"]:
                    labels[st:e] = [IGNORE] * (e - st)
                n_form += len(x["form_error_spans"])
        else:
            lab = x["labels"]
            old, tea = x["lp_old"], x["lp_teacher"]
            adv = [max(-a.adv_clip, min(a.adv_clip, t - o)) if l != IGNORE else 0.0 for l, o, t in zip(lab, old, tea)]
            extra = (adv, [o if l != IGNORE else 0.0 for l, o in zip(lab, old)])
        counts[(x["source"], x["template"], "passed" if x["passed"] else "failed")] += 1
        samples.append((x["ids"], labels, extra))
    lengths = [len(t[0]) for t in samples]
    by_len = collections.defaultdict(list)
    for i, n in enumerate(lengths):
        by_len[n].append(i)
    cols = collections.defaultdict(list)
    for knapsack in greedy_knapsack(list(lengths), a.cutoff):  # as build_agent_sft.py (neat packing)
        ids, mask, pos, lab, adv, old, bounds = [], [], [], [], [], [], [0]
        for j, n in enumerate(knapsack):
            t = samples[by_len[n].pop()]
            ids += t[0]
            lab += t[1]
            pos += list(range(n))
            mask += [j + 1] * n
            bounds.append(bounds[-1] + n)
            if a.mode == "opd":
                adv += t[2][0]
                old += t[2][1]
        pad = a.cutoff - len(ids) + 1
        ids += [pad_id] * pad
        pos += [0] * pad
        lab += [IGNORE] * pad
        mask += [0] * pad
        bounds.append(bounds[-1] + pad)
        cols["input_ids"].append(ids)
        cols["attention_mask"].append(mask)
        cols["position_ids"].append(pos)
        cols["labels"].append(lab)
        if a.mode == "opd":
            cols["token_adv"].append(adv + [0.0] * pad)
            cols["token_lp_old"].append(old + [0.0] * pad)
        cols["packing_params"].append({"sequence_boundaries": bounds, "image_subseq_ids": [2 ** 31 - 1],
                                       "video_subseq_ids": [2 ** 31 - 1], "audio_subseq_ids": [2 ** 31 - 1],
                                       "right_padding_length": pad})
        for k in ("images", "videos", "audios"):
            cols[k].append(None)
    DatasetDict({"train": Dataset.from_dict(dict(cols))}).save_to_disk(a.out)
    labelled = [v for x in samples for v, l in zip(x[2][0], x[1]) if l != IGNORE] if a.mode == "opd" else []
    manifest = {"mode": a.mode, "trajectories": len(samples), "packed_sequences": len(cols["input_ids"]),
                "tokens": sum(lengths), "loss_tokens": sum(sum(v != IGNORE for v in x[1]) for x in samples),
                "by_source": {"/".join(k): v for k, v in sorted(counts.items())}, "args": vars(a),
                "rendered_sha256": hashlib.sha256(open(a.rendered, "rb").read()).hexdigest()[:16]}
    if a.mode == "sft":
        manifest["assistant_turns_masked_for_a_form_error"] = n_form
    if labelled:
        labelled.sort()
        manifest["adv"] = {"mean": sum(labelled) / len(labelled), "p05": labelled[len(labelled) // 20],
                           "median": labelled[len(labelled) // 2], "clipped_low": sum(v <= -a.adv_clip for v in labelled)}
    json.dump(manifest, open(a.out.rstrip("/") + ".manifest.json", "w"), indent=1)
    print(json.dumps({k: manifest[k] for k in ("trajectories", "packed_sequences", "tokens", "loss_tokens")} | {"adv": manifest.get("adv")}))


{"render": render, "score": score, "pack": pack}[a.cmd]()
