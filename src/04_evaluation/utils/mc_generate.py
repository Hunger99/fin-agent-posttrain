"""Answer multiple-choice items by generating in the model's chat format, and score the letter it writes.

docs/history/04_evaluation.md section 2's supplementary score: the chat template, greedy decoding, the letter after
the last "答案：" or "答案:". Here it is zero-shot with thinking off (enable_thinking=False for Qwen3), as
an instruct model is used; an answer without a parsable letter counts as wrong and is counted apart
(research/log/2026-10-26-0409.md).

Output: one line per item (id, exam, gold, pred, correct, the generated text) and a summary on stdout.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python mc_generate.py --model <dir> --items <jsonl> --out <jsonl>
       [--task 单项选择题] [--max-new 256] [--tokenizer <dir>]
"""
import argparse
import json
import re

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--items", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--task", default="单项选择题")
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--tokenizer", default=None, help="a trained run's tokenizer copy from tokenizer_v4.py, which carries "
                                                    "the training chat template")
args = ap.parse_args()
ANSWER = re.compile(r"答案\s*[:：]\s*\**\s*([A-Z])")

with open(args.items, encoding="utf-8") as f:
    items = [r for r in map(json.loads, f) if r.get("task", args.task) == args.task]
llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
tok = llm.get_tokenizer()


def prompt(it):
    options = "".join(f"{k}. {str(v).strip()}\n" for k, v in sorted(it["choices"].items()))
    user = (f"以下是关于{it['exam']}的单项选择题，请选出正确答案。\n\n{it['question'].strip()}\n{options}\n"
            "请直接作答，最后一行写\"答案：X\"，X 为选项字母。")
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


outs = llm.generate([prompt(it) for it in items], SamplingParams(temperature=0.0, max_tokens=args.max_new))
correct = unparsed = 0
with open(args.out, "w", encoding="utf-8") as f:
    for it, o in zip(items, outs):
        text = o.outputs[0].text
        found = ANSWER.findall(text)
        pred = found[-1] if found else None
        unparsed += pred is None
        ok = pred == it["answer"]
        correct += ok
        f.write(json.dumps({"id": it["id"], "exam": it["exam"], "gold": it["answer"], "pred": pred, "correct": ok,
                            "text": text}, ensure_ascii=False) + "\n")
print(json.dumps({"model": args.model, "items": len(items), "accuracy": round(correct / max(len(items), 1), 4),
                  "unparsed": unparsed}, ensure_ascii=False))
