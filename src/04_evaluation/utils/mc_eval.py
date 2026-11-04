"""Score multiple-choice items by the log-likelihood of each option letter after "答案：".

docs/history/04_evaluation.md section 2: FinanceIQ, CFLUE and CMMLU use this one code path. The input has no chat
template: one instruction line, N examples (question, options, "答案：X"), then the item's question and
options, ending with "答案："; the prediction is the letter whose continuation the model scores highest.
Whether a letter carries a leading space follows the tokenizer: when the tokens of prompt + letter are the
prompt's tokens plus one token, one forward pass reads that token's log-probability for every letter;
otherwise each letter's continuation is scored as a whole ("fallback" in the output).

Examples come from --shots, matched by the item's "exam" (FinanceIQ's dev split by category, CFLUE's own
validation split by exam, CMMLU's dev split by subject), taken in the order of a salted hash of their ids;
when --shots is the items file itself, the examples are removed from scoring, as section 2 says for CFLUE.

Output: one line per scored item (id, exam, gold, pred, correct, letter log-probabilities) and a summary
line on stdout.

--lang en prints the English form of the same prompt for MMLU (research/log/2026-10-26-2006.md): the
instruction "The following are multiple choice questions (with answers) about <subject>.", "Answer:" in
place of "答案：", and each letter scored with the space before it, " A", as the examples write it.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python mc_eval.py --model <dir> --items <jsonl> --shots <jsonl> --out <jsonl>
       [--task 单项选择题] [--n-shots 5] [--salt shots-20261025] [--max-tokens 16384] [--exclude <exam>,...]
       [--lang en] [--padding right]

--padding right pads each batch on the right instead of the left, for Qwen3.5 (research/log/2026-10-29-1815.md): its
linear-attention layers and convolutions let left padding change the letter log-probabilities, while on the right the
padding comes after every real token and so cannot reach it.
"""
import argparse
import collections
import hashlib
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--items", required=True)
ap.add_argument("--shots", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--task", default="单项选择题")
ap.add_argument("--n-shots", type=int, default=5)
ap.add_argument("--salt", default="shots-20261025")
ap.add_argument("--max-tokens", type=int, default=16384, help="tokens per forward batch")
ap.add_argument("--exclude", default="", help="comma-separated exams left out of scoring, such as CMMLU's 3 "
                                               "subjects that overlap finance exams (docs/history/04_evaluation.md section 1)")
ap.add_argument("--halfwidth", action="store_true",
                help="print the questions, options and example answers with half-width Chinese punctuation "
                     "(docs/history/decisions.md item 8 (3)); the closing 答案： stays as it is")
ap.add_argument("--lang", default="zh", choices=["zh", "en"])
ap.add_argument("--padding", default="left", choices=["left", "right"])
args = ap.parse_args()
ANSWER, SEP = ("Answer:", " ") if args.lang == "en" else ("答案：", "")


def read(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in map(json.loads, f) if r.get("task", args.task) == args.task]


# The half-width forms IndustryCorpus2 prints Chinese punctuation in (docs/history/decisions.md item 8)
HALF = str.maketrans({"，": ",", "。": ".", "、": ",", "；": ";", "：": ":", "？": "?", "！": "!", "（": "(", "）": ")",
                      "“": '"', "”": '"', "‘": "'", "’": "'", "《": '"', "》": '"', "【": "[", "】": "]"})


def show(text):
    return str(text).strip().translate(HALF) if args.halfwidth else str(text).strip()


def block(it, answer=None):
    text = show(it["question"]) + "\n" + "".join(f"{k}. {show(v)}\n" for k, v in sorted(it["choices"].items()))
    return text + ANSWER + (f"{SEP}{answer}\n\n" if answer else "")


items, pool = read(args.items), read(args.shots)
items = [it for it in items if it["exam"] not in set(filter(None, args.exclude.split(",")))]
by_exam = collections.defaultdict(list)
for it in sorted(pool, key=lambda it: hashlib.md5((args.salt + it["id"]).encode()).hexdigest()):
    by_exam[it["exam"]].append(it)
shots = {e: v[:args.n_shots] for e, v in by_exam.items()}
shot_ids = {it["id"] for v in shots.values() for it in v}
n_items = len(items)
items = [it for it in items if it["id"] not in shot_ids]
task_name = "单项选择题" if args.task == "单项选择题" else args.task


def prompt(it):
    if args.lang == "en":
        head = f"The following are multiple choice questions (with answers) about {it['exam'].replace('_', ' ')}.\n\n"
    else:
        head = f"以下是关于{it['exam']}的{task_name}，请选出正确答案。\n\n"
    return head + "".join(block(s, s["answer"]) for s in shots.get(it["exam"], [])) + block(it)


tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

# Per item: prompt tokens and, per letter, the single continuation token when the split allows it
work = []
for it in items:
    p = prompt(it)
    ids = tok(p, add_special_tokens=False).input_ids
    letters = sorted(it["choices"])
    single = {}
    for L in letters:
        full = tok(p + SEP + L, add_special_tokens=False).input_ids
        if len(full) == len(ids) + 1 and full[:-1] == ids:
            single[L] = full[-1]
    work.append((it, p, ids, letters, single if len(single) == len(letters) else None))


@torch.inference_mode()
def last_logprobs(seqs):
    """Log-softmax over the vocabulary at the last position of each sequence.

    Left padding puts every last position at -1, so only that position's logits are computed; position
    ids are counted over real tokens, so padding does not shift them. Right padding computes the logits at each
    sequence's own last position, and its position ids already count from the first token.
    """
    n = max(len(s) for s in seqs)
    x = torch.full((len(seqs), n), tok.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    if args.padding == "right":
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        keep = sorted({len(s) - 1 for s in seqs})
        out = model(input_ids=x.cuda(), attention_mask=mask.cuda(), logits_to_keep=torch.tensor(keep).cuda()).logits
        out = out[torch.arange(len(seqs)), torch.tensor([keep.index(len(s) - 1) for s in seqs])]
        return torch.log_softmax(out.float(), dim=-1)
    for i, s in enumerate(seqs):
        x[i, n - len(s):] = torch.tensor(s)
        mask[i, n - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=x.cuda(), attention_mask=mask.cuda(), position_ids=pos.cuda(), logits_to_keep=1).logits
    return torch.log_softmax(out[:, -1].float(), dim=-1)


@torch.inference_mode()
def continuation_logprob(ctx, cont):
    x = torch.tensor([ctx + cont]).cuda()
    lp = torch.log_softmax(model(input_ids=x).logits[0, len(ctx) - 1:-1].float(), dim=-1)
    return float(lp[torch.arange(len(cont)), torch.tensor(cont).cuda()].sum())


results = {}
single = sorted((i for i, w in enumerate(work) if w[4] is not None), key=lambda i: len(work[i][2]))
batch = []
for i in single + [None]:
    if i is not None and (not batch or (len(batch) + 1) * len(work[i][2]) <= args.max_tokens):
        batch.append(i)
        continue
    for j, lp in zip(batch, last_logprobs([work[j][2] for j in batch])):
        results[j] = ({L: float(lp[t]) for L, t in work[j][4].items()}, False)
    batch = [i]
for i, (it, p, ids, letters, one) in enumerate(work):
    if one is None:  # the tokenizer merges a letter with what precedes it
        results[i] = ({L: continuation_logprob(ids, tok(p + SEP + L, add_special_tokens=False).input_ids[len(ids):])
                       for L in letters}, True)

correct = 0
by = collections.Counter()
with open(args.out, "w", encoding="utf-8") as f:
    for i, (it, *_rest) in enumerate(work):
        lp, fallback = results[i]
        pred = max(lp, key=lp.get)
        ok = pred == it["answer"]
        correct += ok
        by[(it["exam"], ok)] += 1
        f.write(json.dumps({"id": it["id"], "exam": it["exam"], "gold": it["answer"], "pred": pred, "correct": ok,
                            "logprobs": {k: round(v, 4) for k, v in lp.items()}, "fallback": fallback}, ensure_ascii=False) + "\n")
print(json.dumps({"model": args.model, "items": len(work), "shots_removed_from_scoring": n_items - len(work),
                  "accuracy": round(correct / max(len(work), 1), 4), "fallback": sum(r[1] for r in results.values())},
                 ensure_ascii=False))
