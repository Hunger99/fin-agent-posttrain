"""Ask the teacher whether each kept document-QA question is knowledge a Chinese finance qualification exam tests.

research/log/2026-10-25-1210.md read 100 questions kept under prompt v3 and found about half of them outside
FinanceIQ's exam subjects: industry or technical knowledge, trading lore and slogans, and rules generalized
from one company. This check (prompts/doc_qa_scope_v1.txt) runs after doc_qa.py on its kept questions only,
with doc_qa.py's generation settings, and adds "in_scope" to each of them; the rows are written again to a
new file, so the doc_qa.py output is left as it is.

Usage: python doc_qa_scope.py <doc_qa output.jsonl> <output.jsonl> <concurrency> <base_url>...
"""
import asyncio
import hashlib
import itertools
import json
import os
import re
import sys

from openai import AsyncOpenAI

SRC, OUT, CONC, URLS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
VERSION = "doc_qa_scope_v1"
PROMPTS = os.path.join(os.pardir, os.pardir, os.pardir, "configs", "01_data-processing", "prompts")
PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), PROMPTS, VERSION + ".txt"), encoding="utf-8").read()


async def main():
    with open(SRC, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    clients = itertools.cycle([AsyncOpenAI(base_url=u, api_key="local", timeout=900) for u in URLS])
    sem = asyncio.Semaphore(CONC)

    async def judge(cid, j, q):
        key = f"{cid}-scope{j}"
        async with sem:
            r = await next(clients).chat.completions.create(
                model="gpt-oss-120b", temperature=1.0, top_p=1.0, max_tokens=2048,
                seed=int(hashlib.md5(key.encode()).hexdigest()[:8], 16),
                messages=[{"role": "user", "content": PROMPT.replace("{question}", q["question"]).replace("{answer}", q["answer"])}],
                extra_body={"reasoning_effort": "medium"})
        text = r.choices[0].message.content or ""
        q["in_scope"] = bool(re.search(r"考试知识\s*[:：]\s*是", text))

    await asyncio.gather(*(judge(r["chunk_id"], j, q) for r in rows for j, q in enumerate(r["questions"]) if q["kept"]))
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            r["meta"]["templates"]["scope"] = VERSION
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = [q for r in rows for q in r["questions"] if q["kept"]]
    print(f"kept {len(kept)}, in scope {sum(q['in_scope'] for q in kept)}")


asyncio.run(main())
