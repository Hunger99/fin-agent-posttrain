"""Score corpus documents 0-5 for their value for learning finance, with the local gpt-oss-120b teacher.

docs/history/01_data-processing.md section 2: the teacher scores a sample of FinCorpus documents, a classifier on
bge-m3 vectors learns the scores, and documents scoring >= 3 are kept. The prompt is
prompts/fin_edu_score_v1.txt, adapted from FineWeb-Edu's additive 5-point prompt. Generation follows
section 6: reasoning effort medium, temperature 1.0, top-p 1.0, at most 4,096 new tokens.

Each input line is {"id", "source", "text"}; each output line adds "score" (None when no "分数：N" line is
found), the answer text and token counts. Output is appended and ids already in it are skipped, so an
interrupted run resumes. Several endpoints (one per vLLM instance) are used round-robin.

Usage: python score_docs.py <input.jsonl> <output.jsonl> <concurrency> <base_url>...

SCORE_EFFORT=low in the environment replaces the medium reasoning effort; it exists to measure whether
the cheaper setting gives the same scores (research/log/2026-10-25-0851.md).
"""
import asyncio
import itertools
import json
import os
import re
import sys
import time

from openai import AsyncOpenAI

SRC, OUT, CONC, URLS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
MAX_CHARS = 3000  # the opening of each document; long announcements would otherwise crowd the context
PROMPTS = os.path.join(os.pardir, os.pardir, os.pardir, "configs", "01_data-processing", "prompts")
PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), PROMPTS, "fin_edu_score_v1.txt"),
              encoding="utf-8").read()
SCORE = re.compile(r"分数\s*[:：]\s*([0-5])")
EFFORT = os.environ.get("SCORE_EFFORT", "medium")


async def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    with open(SRC, encoding="utf-8") as f:
        docs = [d for d in map(json.loads, f) if d["id"] not in done]
    clients = itertools.cycle([AsyncOpenAI(base_url=u, api_key="local") for u in URLS])
    sem = asyncio.Semaphore(CONC)
    out = open(OUT, "a", encoding="utf-8")
    t0, n_done, n_tok = time.time(), 0, 0

    async def one(d):
        nonlocal n_done, n_tok
        async with sem:
            client = next(clients)
            try:
                r = await client.chat.completions.create(
                    model="gpt-oss-120b", temperature=1.0, top_p=1.0, max_tokens=4096,
                    messages=[{"role": "user", "content": PROMPT.replace("{text}", d["text"][:MAX_CHARS])}],
                    extra_body={"reasoning_effort": EFFORT})
                answer = r.choices[0].message.content or ""
                m = SCORE.findall(answer)
                row = {"id": d["id"], "source": d["source"], "score": int(m[-1]) if m else None,
                       "answer": answer[-400:], "completion_tokens": r.usage.completion_tokens,
                       "prompt_tokens": r.usage.prompt_tokens, "finish": r.choices[0].finish_reason}
            except Exception as e:  # a failed call is recorded and retried on the next run, not fatal
                row = {"id": d["id"], "source": d["source"], "score": None, "error": repr(e)[:300]}
            if "error" not in row:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            n_done += 1
            n_tok += row.get("completion_tokens", 0)
            if n_done % 100 == 0:
                el = time.time() - t0
                print(f"{n_done}/{len(docs)} docs, {n_done / el:.2f} docs/s, {n_tok / el:.0f} completion tok/s", flush=True)

    await asyncio.gather(*(one(d) for d in docs))
    out.close()
    el = time.time() - t0
    print(f"finished {n_done} docs in {el:.0f} s, {n_done / max(el, 1):.2f} docs/s, {n_tok / max(el, 1):.0f} completion tok/s")


asyncio.run(main())
