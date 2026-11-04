"""Generate seed-question reasoning with the local teacher: up to four tries per CFLUE question, keep the first correct one.

docs/history/01_data-processing.md section 6: the teacher answers each seed question with reasoning and an answer;
only an answer matching the gold answer is kept, one per question, and a question with four wrong answers
does not enter training. Tries run one after another and stop at the first correct answer, which keeps
section 6's outcome (one correct sample per question, at most four tries) at a fraction of four
generations per question. Generation follows section 6: reasoning effort medium, temperature 1.0, top-p 1.0,
at most 4,096 new tokens; each try's sampling seed derives from the question id and the try number, so a
rerun reproduces it. The teacher's reasoning channel is saved as ``reasoning`` and its final message as
``content``.

Input: question ids, one per line. Output (appended, resumable): one line per question with the kept
sample or the failed tries.

Usage: python seed_reasoning.py <data root> <ids.txt> <output.jsonl> <concurrency> <base_url>...
"""
import asyncio
import hashlib
import itertools
import json
import os
import re
import sys

from openai import AsyncOpenAI

ROOT, IDS, OUT, CONC, URLS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:]
TRIES = 4  # section 6
TEMPLATE = "seed_reasoning_v1"
PROMPTS = os.path.join(os.pardir, os.pardir, os.pardir, "configs", "01_data-processing", "prompts")
PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), PROMPTS, TEMPLATE + ".txt"), encoding="utf-8").read()
ANSWER = re.compile(r"答案\s*[:：]\s*([A-Fa-f](?:[\s,，、]*[A-Fa-f])*)")


def letters(s):
    return "".join(sorted(set(re.sub(r"[^A-F]", "", s.upper()))))


def seed(qid, t):
    return int(hashlib.md5(f"{qid}-{t}".encode()).hexdigest()[:8], 16)


async def main():
    with open(IDS, encoding="utf-8") as f:
        want = {line.strip() for line in f if line.strip()}
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    with open(os.path.join(ROOT, "interim", "cflue", "train80.jsonl"), encoding="utf-8") as f:
        items = [it for it in map(json.loads, f) if it["id"] in want and it["id"] not in done]
    clients = itertools.cycle([AsyncOpenAI(base_url=u, api_key="local", timeout=900) for u in URLS])
    sem = asyncio.Semaphore(CONC)
    out = open(OUT, "a", encoding="utf-8")
    n_ok = n_done = 0

    async def one(it):
        nonlocal n_ok, n_done
        question = f"{it['task']}：{it['question']}\n" + "\n".join(f"{k}. {v}" for k, v in sorted(it["choices"].items()))
        prompt = PROMPT.replace("{question}", question)
        tries = []
        for t in range(TRIES):
            async with sem:
                try:
                    r = await next(clients).chat.completions.create(
                        model="gpt-oss-120b", temperature=1.0, top_p=1.0, max_tokens=4096, seed=seed(it["id"], t),
                        messages=[{"role": "user", "content": prompt}], extra_body={"reasoning_effort": "medium"})
                except Exception as e:  # a failed call counts as a wrong try, recorded with its error
                    tries.append({"error": repr(e)[:200]})
                    continue
            m = r.choices[0].message
            found = ANSWER.findall(m.content or "")
            got = letters(found[-1]) if found else ""
            tries.append({"answer": got, "finish": r.choices[0].finish_reason, "tokens": r.usage.completion_tokens})
            if got and got == letters(it["answer"]):
                row = {"id": it["id"], "ok": True, "tries": tries, "reasoning": getattr(m, "reasoning_content", None) or "",
                       "content": m.content, "meta": {"template": TEMPLATE, "temperature": 1.0, "top_p": 1.0,
                                                      "reasoning_effort": "medium", "seed": seed(it["id"], t)}}
                break
        else:
            row = {"id": it["id"], "ok": False, "tries": tries}
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        n_done += 1
        n_ok += row["ok"]
        if n_done % 200 == 0:
            print(f"{n_done}/{len(items)} questions, {n_ok} correct within {TRIES} tries", flush=True)

    await asyncio.gather(*(one(it) for it in items))
    out.close()
    print(f"finished {n_done} questions, {n_ok} correct within {TRIES} tries")


asyncio.run(main())
