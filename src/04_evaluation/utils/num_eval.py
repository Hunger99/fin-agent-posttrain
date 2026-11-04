"""Score numeric questions over a report excerpt (the English domain score) by generating the answer.

research/log/2026-10-26-2010.md: no chat template, the same prompt for base and instruct models, as the
multiple-choice protocol of docs/history/04_evaluation.md section 2 does. One instruction line, three examples of the
item's own dataset (report excerpt, question, the gold calculation, "Answer: X"), then the item, ending with
"Calculation:"; greedy decoding. The examples are three of the 50 shortest-context train items that have a
written-out calculation, picked by a salted hash of their ids, so every model sees the same prompt.

The answer is read and scored by num_score.py; an answer with no number counts as wrong and is counted apart.

Output: one line per item (id, source, gold, pred, correct, generated text) and a summary on stdout.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python num_eval.py --model <dir> --items <jsonl>[,<jsonl>] --shots <jsonl>[,<jsonl>]
       --out <jsonl> [--tokenizer <dir>] [--salt shots-20261026] [--max-new 256] [--gpu-util 0.88] [--chat-template]
"""
import argparse
import hashlib
import json
import os
import sys

from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))
from num_score import correct, parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--items", required=True, help="comma-separated item files")
ap.add_argument("--shots", required=True, help="comma-separated train files; examples come from the same source")
ap.add_argument("--out", required=True)
ap.add_argument("--tokenizer", default=None)
ap.add_argument("--salt", default="shots-20261026")
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--gpu-util", type=float, default=0.88, help="vLLM memory share; lower it on a GPU shared with others")
ap.add_argument("--chat-template", action="store_true",
                help="diagnostic only (research/log/2026-10-27-1633.md): wrap the prompt in the tokenizer's chat template")
args = ap.parse_args()
INSTRUCTION = ("Answer the question from the excerpt of a company's annual report. Write the calculation, then the "
               "final answer as a number on the last line, after \"Answer:\".")


def load(paths):
    return [r for p in paths.split(",") for r in map(json.loads, open(p, encoding="utf-8"))]


def block(r, with_answer):
    text = f"Report:\n{r['context']}\n\nQuestion: {r['question']}\nCalculation:"
    if with_answer:
        ans = r["answer"] if isinstance(r["answer"], str) else f"{r['answer']:g}"
        text += f" {r['calc']}\nAnswer: {ans}"
    return text


items = load(args.items)
shots = {}
for r in load(args.shots):
    if r["calc"]:
        shots.setdefault(r["source"], []).append(r)
for src, rows in shots.items():
    rows.sort(key=lambda r: len(r["context"]))
    pool = rows[:50]
    pool.sort(key=lambda r: hashlib.sha256((args.salt + r["id"]).encode()).hexdigest())
    shots[src] = pool[:3]
prompts = ["\n\n".join([INSTRUCTION] + [block(s, True) for s in shots[it["source"]]] + [block(it, False)])
           for it in items]

# vLLM starts its engine process by spawn once CUDA is initialized, as Qwen3.5's multimodal warmup does under
# vLLM 0.32 (research/log/2026-10-29-1815.md); the spawned process imports this script again, so the model runs
# only in the main one
if __name__ == "__main__":
    llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=args.gpu_util)
    params = SamplingParams(temperature=0, max_tokens=args.max_new, stop=["\n\nReport:", "\nReport:", "\n\n\n"])
    if args.chat_template:
        # the same prompt as one user turn of the tokenizer's chat template, generation from the assistant turn; passed as
        # token ids so that the template's special tokens stay single tokens
        tok = llm.get_tokenizer()
        prompts = [{"prompt_token_ids": tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                                                add_generation_prompt=True)} for p in prompts]
    outs = llm.generate(prompts, params)


    n = ok = unparsed = 0
    by_src = {}
    with open(args.out, "w", encoding="utf-8") as f:
        for it, o in zip(items, outs):
            text = o.outputs[0].text
            pred = parse(text, it["answer"])
            c = correct(pred, it["answer"])
            n += 1
            ok += c
            unparsed += pred is None
            s = by_src.setdefault(it["source"], [0, 0])
            s[0] += 1
            s[1] += c
            f.write(json.dumps({"id": it["id"], "source": it["source"], "gold": it["answer"], "pred": pred, "correct": c,
                                "text": text}, ensure_ascii=False) + "\n")
    print(json.dumps({"model": args.model, "items": n, "accuracy": round(ok / n, 4), "unparsed": unparsed,
                      "by_source": {k: {"items": v[0], "accuracy": round(v[1] / v[0], 4)} for k, v in by_src.items()},
                      "shots": {k: [r["id"] for r in v] for k, v in shots.items()}}))
