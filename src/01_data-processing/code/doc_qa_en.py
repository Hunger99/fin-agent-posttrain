"""Synthesize English document QA, numeric questions computed from a document, with the local teacher.

research/log/2026-10-26-2222.md. The author chose (2026-10-26) that English document QA ask for a number to be
computed from a given financial document, the same task as the English domain score, so a training sample is
"document + question -> calculation and answer". The structure follows doc_qa.py: chunks of 1,000-2,000
characters of consecutive lines, at most MAX_CHUNKS per document, the teacher at reasoning effort medium,
temperature 1.0, top-p 1.0 and a seed derived from the chunk id; output is appended and resumable by chunk id.

A chunk is used only when it holds at least MIN_NUMBERS numbers. The teacher writes 1-2 questions per chunk
(prompts/doc_qa_en_gen_v2.txt by default; DOC_QA_EN_GEN=v1 for the first pilot's prompt), each with the excerpt numbers it uses, a calculation and a numeric answer. A
question is kept when:
- every number it lists occurs in the chunk (commas, currency signs and percent signs ignored);
- the question is English (no Han character);
- it is a financial calculation: not every listed number is a year, and the question does not ask for a time span
  (SPAN); the v1 pilot kept such questions (research/log/2026-10-26-2222.md), so v2 also forbids them in the prompt;
- its answer parses as one number;
- the teacher, shown only the chunk and the question (prompts/doc_qa_en_check_v1.txt), answers again with a
  number that agrees by the English domain score's rule: within 1%, or after multiplying or dividing by 100
  (src/04_evaluation/utils/num_eval.py).

Input lines: {"id", "source", "text"}. Output: one line per chunk with its questions and each question's checks.

Usage: python doc_qa_en.py <input.jsonl> <output.jsonl> <max chunks> <concurrency> <base_url>...
"""
import asyncio
import hashlib
import itertools
import json
import os
import re
import sys

from openai import AsyncOpenAI

MIN_CHUNK, MAX_CHUNK, MAX_CHUNKS, MIN_NUMBERS = 1000, 2000, 2, 4
BATCH = 1000  # chunks in flight at once
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(os.pardir, os.pardir, os.pardir, "configs", "01_data-processing", "prompts")
VERSION = {"gen": os.environ.get("DOC_QA_EN_GEN", "v2"), "check": "v1"}
P = {k: open(os.path.join(HERE, PROMPTS, f"doc_qa_en_{k}_{v}.txt"), encoding="utf-8").read() for k, v in VERSION.items()}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
HAN = re.compile(r"[一-鿿]")
YEAR = re.compile(r"^(19|20)\d\d$")
# Questions about time spans: the v1 pilot kept them, though they are not financial calculations
SPAN = re.compile(r"how many (years|months|days|weeks)|number of (years|months|days)|in years|time gap|years (after|before|between)", re.I)


def chunks(text):
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > MAX_CHUNK:
            if len(cur) >= MIN_CHUNK:
                out.append(cur)
            cur = ""
            if len(line) > MAX_CHUNK:  # an overlong line is not cut mid-sentence; it is skipped
                continue
        cur += line
        if len(cur) >= MIN_CHUNK and cur.endswith("\n"):
            out.append(cur)
            cur = ""
        if len(out) >= MAX_CHUNKS:
            break
    if len(out) < MAX_CHUNKS and len(cur) >= MIN_CHUNK:
        out.append(cur)
    return [c for c in out[:MAX_CHUNKS] if len(NUM.findall(c)) >= MIN_NUMBERS]


def seed(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def norm_num(s):
    return re.sub(r"[,$%\s]", "", str(s)).rstrip(".")


def first_number(text):
    m = NUM.search(str(text).replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "").rstrip("."))
    except ValueError:
        return None


def agree(a, b):
    if a is None or b is None:
        return False
    tol = 0.005 if a == 0 else 0.01 * abs(a)
    return any(abs(x - a) <= tol for x in (b, b * 100, b / 100))


def parse(text):
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [q for q in items if isinstance(q, dict) and isinstance(q.get("question"), str)
            and isinstance(q.get("calculation"), str) and isinstance(q.get("numbers"), list) and "answer" in q]


async def main():
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(line)["chunk_id"] for line in f}
    todo = []
    with open(SRC, encoding="utf-8") as f:
        for d in map(json.loads, f):
            for k, c in enumerate(chunks(d["text"])):
                cid = f"{d['id']}#c{k}"
                if cid not in done:
                    todo.append((cid, d["source"], c))
            if len(todo) + len(done) >= LIMIT:
                break
    todo = todo[:max(LIMIT - len(done), 0)]
    clients = itertools.cycle([AsyncOpenAI(base_url=u, api_key="local", timeout=900) for u in URLS])
    sem = asyncio.Semaphore(CONC)
    out = open(OUT, "a", encoding="utf-8")
    # "cut" counts replies that stopped at max_tokens, so a truncated reasoning is not mistaken for an empty answer
    stats = {"chunks": 0, "questions": 0, "numbers_ok": 0, "consistent": 0, "kept": 0, "tokens": 0, "cut": 0}

    async def call(prompt, key, max_tokens=4096):
        async with sem:
            r = await next(clients).chat.completions.create(
                model="gpt-oss-120b", temperature=1.0, top_p=1.0, max_tokens=max_tokens, seed=seed(key),
                messages=[{"role": "user", "content": prompt}], extra_body={"reasoning_effort": "medium"})
        stats["tokens"] += r.usage.completion_tokens
        stats["cut"] += r.choices[0].finish_reason == "length"
        return r.choices[0].message.content or ""

    async def one(cid, source, chunk):
        n = 2 if len(chunk) >= 1500 else 1
        try:
            items = parse(await call(P["gen"].replace("{n}", str(n)).replace("{chunk}", chunk), cid + "-gen"))
        except Exception as e:  # recorded; the chunk is retried on the next run only if its line is removed
            items, err = [], repr(e)[:200]
        else:
            err = None
        flat = norm_num(chunk)
        for j, q in enumerate(items):
            nums = [norm_num(x) for x in q["numbers"] if norm_num(x)]
            q["numbers_ok"] = bool(nums) and all(x in flat for x in nums)
            q["english"] = not HAN.search(q["question"])
            q["financial"] = not (all(YEAR.match(x) for x in nums) or SPAN.search(q["question"]))
            q["answer_value"] = first_number(q["answer"])
            q["kept"] = False
            if not (q["numbers_ok"] and q["english"] and q["financial"] and q["answer_value"] is not None):
                continue
            try:
                again = await call(P["check"].replace("{chunk}", chunk).replace("{question}", q["question"]), f"{cid}-chk{j}")
            except Exception as e:
                q["error"] = repr(e)[:200]
                continue
            q["answer_again"] = again
            tail = again.rsplit("Answer:", 1)[-1] if "Answer:" in again else ""
            q["consistent"] = agree(q["answer_value"], first_number(tail))
            q["kept"] = q["consistent"]
        row = {"chunk_id": cid, "source": source, "chunk": chunk, "questions": items,
               "meta": {"templates": {k: f"doc_qa_en_{k}_{v}" for k, v in VERSION.items()}, "temperature": 1.0,
                        "top_p": 1.0, "reasoning_effort": "medium", "seed_key": cid}}
        if err:
            row["error"] = err
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        stats["chunks"] += 1
        stats["questions"] += len(items)
        stats["numbers_ok"] += sum(bool(q.get("numbers_ok")) for q in items)
        stats["consistent"] += sum(bool(q.get("consistent")) for q in items)
        stats["kept"] += sum(q["kept"] for q in items)
        if stats["chunks"] % 50 == 0:
            print(json.dumps(stats), flush=True)

    # Chunks run in batches: with every chunk's generation queued at once, a chunk's check calls waited behind all
    # remaining generations, so questions were written only at the very end (research/log/2026-10-26-2247.md).
    for i in range(0, len(todo), BATCH):
        await asyncio.gather(*(one(*t) for t in todo[i:i + BATCH]))
    out.close()
    print("finished", json.dumps(stats))


if __name__ == "__main__":
    SRC, OUT, LIMIT, CONC, URLS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
    asyncio.run(main())
