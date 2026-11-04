"""Synthesize document QA with the local teacher and filter it, per docs/history/01_data-processing.md section 6.

Section 6's document-QA line: cleaned news, articles and announcements are cut into 1,000-2,000 character
chunks; the teacher writes 2-3 questions per chunk of three types (fact extraction, numeric calculation,
concept explanation) with an answer and an evidence sentence; a question is kept when its evidence occurs
verbatim in the chunk and the teacher, shown only the chunk and the question, answers it again
consistently. Consistency is exact after whitespace and punctuation are dropped, or else the teacher's own
"一致/不一致" judgment (prompts/doc_qa_judge_v1.txt), since free-text answers rarely match word for word.
Generation follows section 6: reasoning effort medium, temperature 1.0, top-p 1.0, at most 4,096 new tokens,
with per-call seeds derived from the chunk id.

A chunk is consecutive lines of one document, from 1,000 up to 2,000 characters; a document gives at most
MAX_CHUNKS chunks, so long documents do not dominate. Every kept question carries its chunk, so the E4 check
can read it against the source and the training format can still choose to include it or not.

Training is closed-book, the question alone (docs/history/decisions.md item 3), so a question must stand without its
chunk. Before the consistency check, a question passes a self-containment filter: no deictic word that
points at the chunk (DEICTIC), and the teacher's "自含：能" to prompts/doc_qa_self_v2.txt, which also labels
the question general knowledge or a specific entity's fact, for the share the pilot reports. The generation prompt
version is GEN (v2 asks for the subject's full name and date; v1 did not, and most of its smoke-test
questions needed the chunk).

The v2 pilot (research/log/2026-10-25-1143.md) kept mostly single entities' facts, and a fifth of its kept
questions carried a year their chunk does not contain, so v3 asks only for transferable knowledge and three
rule checks now run before the self check: every year in the question occurs in the chunk (year_ok); the
question is mostly Chinese (chinese); and evidence is matched with punctuation as well as whitespace dropped,
since half the pilot's evidence failures differed only in punctuation. Under v3 a question is kept only when
the self check also labels it 通用知识 (GENERAL_ONLY), and the self check (doc_qa_self_v3) reads the answer
too, since from the question alone it labelled arithmetic on one company's figures as general knowledge.
Set DOC_QA_SCOPE=1 to add the exam-scope check of docs/history/decisions.md item 7 (prompts/doc_qa_scope_v1.txt):
after the self check and before the consistency check, a question the teacher does not judge to be
knowledge a Chinese finance qualification exam tests is dropped ("in_scope").

Input lines: {"id", "source", "text"}. Output (appended, resumable by chunk id): one line per chunk with its
questions and each question's filter results.

Usage: python doc_qa.py <input.jsonl> <output.jsonl> <max chunks> <concurrency> <base_url>...
"""
import asyncio
import hashlib
import itertools
import json
import os
import re
import sys

from openai import AsyncOpenAI

MIN_CHUNK, MAX_CHUNK, MAX_CHUNKS = 1000, 2000, 2  # section 6's chunk size; chunks per document chosen by the model
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(os.pardir, os.pardir, os.pardir, "configs", "01_data-processing", "prompts")
GEN = os.environ.get("DOC_QA_GEN", "v2")
GENERAL_ONLY = GEN >= "v3"
YEAR = re.compile(r"(?:19|20)\d\d")
VERSION = {"gen": GEN, "self": "v3" if GENERAL_ONLY else "v2", "check": "v1", "judge": "v1"}  # self v3 also reads the answer
if os.environ.get("DOC_QA_SCOPE") == "1":
    VERSION["scope"] = "v1"
P = {k: open(os.path.join(HERE, PROMPTS, f"doc_qa_{k}_{v}.txt"), encoding="utf-8").read() for k, v in VERSION.items()}
PUNCT = re.compile(r"[\s，。、；：！？,.;:!?（）()《》“”\"'【】\[\]]")
# Words that point back at the chunk; "公司" alone counts only when no company name precedes it in the question
DEICTIC = re.compile(r"本次|此次|该公司|该行|该基金|该产品|上述|上文|材料|本文|文中|以上|如下|下表|本公告|本报告|^公司|[，。；]公司")


def chunks(text):
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > MAX_CHUNK:
            if len(cur) >= MIN_CHUNK:
                out.append(cur)
            cur = ""
            if len(line) > MAX_CHUNK:  # a single overlong line is not cut mid-sentence; it is skipped
                continue
        cur += line
        if len(cur) >= MIN_CHUNK and len(out) < MAX_CHUNKS and cur.endswith("\n"):
            out.append(cur)
            cur = ""
        if len(out) >= MAX_CHUNKS:
            break
    if len(out) < MAX_CHUNKS and len(cur) >= MIN_CHUNK:
        out.append(cur)
    return out[:MAX_CHUNKS]


def seed(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def parse(text):
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [q for q in items if isinstance(q, dict) and all(isinstance(q.get(k), str) for k in ("type", "question", "answer", "evidence"))]


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
    stats = {"chunks": 0, "questions": 0, "evidence_ok": 0, "self_contained": 0, "kept": 0, "tokens": 0}

    async def call(prompt, key, max_tokens=4096):
        async with sem:
            r = await next(clients).chat.completions.create(
                model="gpt-oss-120b", temperature=1.0, top_p=1.0, max_tokens=max_tokens, seed=seed(key),
                messages=[{"role": "user", "content": prompt}], extra_body={"reasoning_effort": "medium"})
        stats["tokens"] += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    async def one(cid, source, chunk):
        n = 3 if len(chunk) >= 1500 else 2
        try:
            items = parse(await call(P["gen"].replace("{n}", str(n)).replace("{chunk}", chunk), cid + "-gen"))
        except Exception as e:  # recorded; the chunk is retried on the next run only if its line is removed
            items, err = [], repr(e)[:200]
        else:
            err = None
        flat = PUNCT.sub("", chunk)
        for j, q in enumerate(items):
            q["evidence_ok"] = bool(PUNCT.sub("", q["evidence"])) and PUNCT.sub("", q["evidence"]) in flat
            q["kept"] = False
            q["deictic"] = bool(DEICTIC.search(q["question"]))
            q["year_ok"] = all(y in chunk for y in YEAR.findall(q["question"]))
            han, latin = len(re.findall(r"[\u4e00-\u9fff]", q["question"])), len(re.findall(r"[A-Za-z]", q["question"]))
            q["chinese"] = han >= latin
            if not q["evidence_ok"] or q["deictic"] or not q["year_ok"] or not q["chinese"]:
                continue
            try:
                # v1 of this check asked whether the question could be answered with knowledge alone, and the
                # teacher then rejected self-contained company questions too; v2 separates the two judgments
                selfc = await call(P["self"].replace("{question}", q["question"]).replace("{answer}", q["answer"]),
                                   f"{cid}-self{j}", max_tokens=2048)
                q["self_contained"] = bool(re.search(r"自含\s*[:：]\s*能", selfc))
                q["category"] = "具体事实" if "具体事实" in selfc else ("通用知识" if "通用知识" in selfc else None)
                if not q["self_contained"] or (GENERAL_ONLY and q["category"] != "通用知识"):
                    continue
                if "scope" in P:
                    scope = await call(P["scope"].replace("{question}", q["question"]).replace("{answer}", q["answer"]),
                                       f"{cid}-scope{j}", max_tokens=2048)
                    q["in_scope"] = bool(re.search(r"考试知识\s*[:：]\s*是", scope))
                    if not q["in_scope"]:
                        continue
                again = await call(P["check"].replace("{chunk}", chunk).replace("{question}", q["question"]), f"{cid}-chk{j}")
                q["answer_again"] = again
                if PUNCT.sub("", again) == PUNCT.sub("", q["answer"]):
                    q["consistent"] = "exact"
                else:
                    verdict = await call(P["judge"].replace("{question}", q["question"]).replace("{a}", q["answer"])
                                         .replace("{b}", again), f"{cid}-jdg{j}", max_tokens=2048)
                    q["consistent"] = "judged" if verdict.strip().startswith("一致") else "no"
                q["kept"] = q["consistent"] != "no"
            except Exception as e:
                q["error"] = repr(e)[:200]
        row = {"chunk_id": cid, "source": source, "chunk": chunk, "questions": items,
               "meta": {"templates": {k: f"doc_qa_{k}_{v}" for k, v in VERSION.items()}, "temperature": 1.0, "top_p": 1.0,
                        "reasoning_effort": "medium", "seed_key": cid}}
        if err:
            row["error"] = err
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        stats["chunks"] += 1
        stats["questions"] += len(items)
        stats["evidence_ok"] += sum(q["evidence_ok"] for q in items)
        stats["self_contained"] += sum(bool(q.get("self_contained")) for q in items)
        stats["kept"] += sum(q["kept"] for q in items)
        if stats["chunks"] % 50 == 0:
            print(json.dumps(stats), flush=True)

    await asyncio.gather(*(one(*t) for t in todo))
    out.close()
    print("finished", json.dumps(stats))


if __name__ == "__main__":  # sample_doc_qa.py imports chunks()
    SRC, OUT, LIMIT, CONC, URLS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
    asyncio.run(main())
