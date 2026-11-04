"""Decontamination step: drop domain samples whose answer uses material the prompt does not give.

research/log/2026-10-27-2138.md, after the author's lesson (research/log/2026-10-27-2040.md): a sample whose answer
draws on material outside what the prompt provides teaches the model to answer without looking things up. The check
found such samples only in IndustryInstruction, whose questions were generated from articles that the prompts leave
out; in the other domain sources the flags read were false ones, so the rules apply to IndustryInstruction alone.
A sample is dropped when either holds:
- its prompt, under 600 characters, refers to material it does not contain ("according to the text", "mentioned in
  the article", "the provided data", "the author", "X's argument" and the like);
- its answer has two or more specific numbers (above 12 and not a year) that the prompt does not contain.

Output: interim/decontam/ungrounded_ids.json ({stem: [ids]}), which lock_domain_pool_en.py drops with the other
decontamination removals, and one JSON line of counts on stdout.

Usage: python grounding_filter_en.py <English data root>
"""
import json
import os
import re
import sys

R = sys.argv[1]
SOURCES = ["industryinstruction_finance_en"]
REFERS = re.compile(r"\b(according to|based on|in|from|mentioned in|described in) the (text|passage|article|report|document|"
                    r"data|information|paragraph|context|content|material|survey|study|announcement|statement)\b|"
                    r"\bthe (provided|given|above|following) (text|passage|article|report|document|data|information|context|"
                    r"content|material)\b|\bthe author\b|'s (argument|article|report|view)\b", re.I)
NUM = re.compile(r"(?<![\w.])-?\$?\d[\d,]*(?:\.\d+)?%?")


def values(text):
    out = set()
    for m in NUM.finditer(text):
        s = re.sub(r"[^\d.]", "", m.group(0)).strip(".")
        try:
            out.add(float(s))
        except ValueError:
            pass
    return out


flagged, counts = {}, {}
for stem in SOURCES:
    out, n, by_ref, by_num = [], 0, 0, 0
    with open(os.path.join(R, "processed", "sft", f"{stem}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            prompt = "\n".join(m["content"] for m in r["messages"] if m["role"] in ("system", "user"))
            reply = "\n".join(m["content"] for m in r["messages"] if m["role"] == "assistant")
            ref = len(prompt) < 600 and bool(REFERS.search(prompt))
            known = values(prompt)
            new = [v for v in values(reply) if v not in known and v > 12 and not (1900 <= v <= 2035 and v == int(v))]
            num = len(new) >= 2
            by_ref += ref
            by_num += num
            if ref or num:
                out.append(r["id"])
    flagged[stem] = out
    counts[stem] = {"samples": n, "refers_to_missing_material": by_ref, "outside_numbers": by_num, "dropped": len(out)}
with open(os.path.join(R, "interim", "decontam", "ungrounded_ids.json"), "w", encoding="utf-8") as f:
    json.dump(flagged, f)
print(json.dumps(counts))
