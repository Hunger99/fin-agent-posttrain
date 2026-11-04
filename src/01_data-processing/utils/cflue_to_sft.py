"""Turn the CFLUE train 80% part into non-synthetic SFT samples from each item's own answer and analysis.

docs/history/01_data-processing.md section 5: the 80% part becomes SFT samples; docs/history/decisions.md item 1 makes
these the non-synthetic CFLUE samples of the domain pool. Section 4's format: the user turn is the task
type, the question and one choice per line; the assistant turn is the item's analysis followed by the
answer line "答案：X", or the answer line alone when the item has no analysis. Ids stay the CFLUE ids, so
the decontamination flags of interim/decontam/train.*.flags.jsonl apply to these samples unchanged.
Samples over 4,096 tokens are dropped, as convert.py drops them for the other SFT sources.

Usage: python cflue_to_sft.py $FP_DATA_ZH
"""
import json
import os
import sys

from tokenizers import Tokenizer

ROOT = sys.argv[1]
MAX_TOKENS = 4096  # docs/history/02_model-training.md section 2
tok = Tokenizer.from_file(os.path.join(ROOT, "raw", "qwen2.5-1.5b-tokenizer", "tokenizer.json"))

with open(os.path.join(ROOT, "interim", "cflue", "train80.jsonl"), encoding="utf-8") as f:
    items = [json.loads(line) for line in f]
rows, too_long, no_analysis = [], 0, 0
for it in items:
    user = f"{it['task']}：{it['question']}\n" + "\n".join(f"{k}. {v}" for k, v in sorted(it["choices"].items()))
    analysis = (it.get("analysis") or "").strip()
    no_analysis += not analysis
    assistant = (analysis + "\n" if analysis else "") + f"答案：{it['answer']}"
    n = len(tok.encode(user + assistant, add_special_tokens=False).ids)
    if n > MAX_TOKENS:
        too_long += 1
        continue
    rows.append({"id": it["id"], "source": "cflue/train80",
                 "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                 "meta": {"license": "CC BY-NC-SA 4.0", "task": it["task"], "has_reasoning": False,
                          "has_analysis": bool(analysis), "n_tokens": n}})
with open(os.path.join(ROOT, "processed", "sft", "cflue_train80.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"items {len(items):,}, written {len(rows):,}, dropped over {MAX_TOKENS} tokens {too_long:,}, "
      f"without analysis {no_analysis:,}, mean tokens {sum(r['meta']['n_tokens'] for r in rows) / len(rows):.0f}")
