"""Run again the consistency check of English doc QA questions whose check call failed with a connection error.

research/log/2026-10-26-2247.md. The teachers were stopped at about 00:21 UTC on 2026-10-27 while some chunks had
their questions written but not yet checked; those questions carry "error" instead of "consistent" and were dropped,
and since their chunk line has no error the resume did not retry them. This sends each one the same check call as
src/01_data-processing/code/doc_qa_en.py (same prompt, settings and seed key), sets "consistent" and "kept" by the same
rule and rewrites the file in place, after copying it to <file>.before_recheck_<time>. Other errors stay as they are.

Usage: python docqa_recheck_en.py <docqa_en_v2.jsonl> <concurrency> <base_url>
"""
import asyncio
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from openai import AsyncOpenAI  # noqa: E402

import doc_qa_en as dq  # noqa: E402

F, CONC, URL = sys.argv[1], int(sys.argv[2]), sys.argv[3]


async def main():
    rows = [json.loads(line) for line in open(F, encoding="utf-8")]
    todo = [(r, j, q) for r in rows for j, q in enumerate(r["questions"]) if "Connection error" in q.get("error", "")]
    client, sem = AsyncOpenAI(base_url=URL, api_key="local", timeout=900), asyncio.Semaphore(CONC)
    stats = {"rechecked": 0, "kept": 0, "error": 0}

    async def one(r, j, q):
        prompt = dq.P["check"].replace("{chunk}", r["chunk"]).replace("{question}", q["question"])
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model="gpt-oss-120b", temperature=1.0, top_p=1.0, max_tokens=4096,
                    seed=dq.seed(f"{r['chunk_id']}-chk{j}"), messages=[{"role": "user", "content": prompt}],
                    extra_body={"reasoning_effort": "medium"})
        except Exception as e:
            q["error"] = repr(e)[:200]
            stats["error"] += 1
            return
        again = resp.choices[0].message.content or ""
        del q["error"]
        q["answer_again"] = again
        tail = again.rsplit("Answer:", 1)[-1] if "Answer:" in again else ""
        q["consistent"] = dq.agree(q["answer_value"], dq.first_number(tail))
        q["kept"] = q["consistent"]
        stats["rechecked"] += 1
        stats["kept"] += q["kept"]

    await asyncio.gather(*(one(*t) for t in todo))
    shutil.copyfile(F, F + time.strftime(".before_recheck_%Y%m%d%H%M"))
    with open(F + ".tmp", "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(F + ".tmp", F)
    print(json.dumps({"to_recheck": len(todo), **stats}))


asyncio.run(main())
