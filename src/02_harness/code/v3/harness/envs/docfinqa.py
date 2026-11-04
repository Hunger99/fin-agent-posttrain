"""DocFinQA in the harness ("docfinqa"): a question about one 10-K filing, read only through document tools.

The filing is too long for the prompt (165k to 1.5M characters), so the model sees it through retrieval:
search_document (BM25 over chunks of about CHUNK characters, with OVERLAP between neighbours), read_chunk and the
calculator, and ends with finish. There are no business rules, so the constraint part has nothing to check; the
checklist, verification (every number of the answer traced to a tool result) and correction work as for the wealth
tasks. A question is correct by the English domain score's rule (src/04_evaluation/utils/num_eval.py): the answer's last
number, itself or multiplied or divided by 100, within 1% of the gold answer (0.005 when the gold answer is 0); a
yes/no question needs the same word. A gold answer with units ("$ 13.4 million") is its first number; the few
free-text gold answers are left out.

Version 2: a snippet is the SNIPPET characters of the passage that hold the most of the query's words (by BM25 idf),
read_chunk can return the neighbouring passages with it, and the search tool says that BM25 matches words, not quoted
phrases. Version 2.1: the filing is read with the literal "\\t" and "\\n" as spaces, an empty search comes with a
note, and the system prompt asks for all arithmetic on the calculator. Version 3 (research/log/2026-10-31-0150.md):
the same environment and grading behind the EnvKind interface; the documents directory is the task's "docs_dir" when
it has one, and run.py's --db otherwise, as in version 2. read_chunk with 2 neighbours on each side returns about
7,100 characters, under the harness's cap on a tool result.

Usage: python -B envtool.py prepare-docfinqa <raw dev.json or test.json> <split> <out dir>
       (writes <out dir>/tasks_docfinqa_<split>.jsonl and one text file per filing under <out dir>/docfinqa_<split>/)
"""
import hashlib
import json
import os
import re

from harness.envs import base
from harness.envs.base import S, ToolError, schema
from harness.envs.wealth import core

CHUNK, OVERLAP, SNIPPET, MAX_NEIGHBORS = 1500, 200, 240, 2
# an exponent is part of the number (research/log/2026-10-29-0442.md)
NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")

SYSTEM = """You answer questions about a company's annual report (Form 10-K). The report is too long to show; read it with the tools: search_document finds passages, read_chunk shows a passage in full, and calculator does arithmetic: do all arithmetic with it, not in your reasoning. Act without asking questions. When done, call finish with the answer as a single number (a percentage as its number, e.g. 12.5 for 12.5%; no units), or with yes or no when the question asks for one."""
DOC_TOOLS = [
    schema("search_document", "BM25 search over the report's passages: it matches words, not quoted phrases. Returns "
           "chunk ids, each with the part of the passage that best matches the query.",
           {"query": S, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
    schema("read_chunk", "The full text of one passage of the report, optionally with up to 2 neighbouring passages "
           "on each side (a table or sentence may continue across passages).",
           {"chunk_id": {"type": "integer", "minimum": 0},
            "neighbors": {"type": "integer", "minimum": 0, "maximum": MAX_NEIGHBORS}}, ["chunk_id"]),
]
TOOLS = DOC_TOOLS + [
    schema("calculator", "Evaluate one or more arithmetic expressions in one call, e.g. [\"(5829-5735)/5735*100\"].",
           {"expressions": {"type": "array", "items": S, "minItems": 1}}, ["expressions"]),
    schema("finish", "End with the answer, a single number.", {"answer": S}, ["answer"]),
]
EXAMPLES = {"search_document": {"query": "total revenue 2015", "top_k": 5}, "read_chunk": {"chunk_id": 12, "neighbors": 1},
            "calculator": {"expressions": ["(5829-5735)/5735*100"]}, "finish": {"answer": "1.64"}}


class DocEnv(base.Env):
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.text = text = f.read().replace("\\t", " ").replace("\\n", " ")
        self.starts = list(range(0, max(1, len(text) - OVERLAP), CHUNK - OVERLAP))
        self.chunks = [text[i:i + CHUNK] for i in self.starts]
        self.bm25 = core.BM25([(i, "", c) for i, c in enumerate(self.chunks)])

    def snippet(self, chunk, query):
        """The SNIPPET characters of the chunk, whitespace collapsed, holding the most query words by idf; a window
        starts at the chunk's start or at the word boundary up to 20 characters before a query word."""
        flat = " ".join(chunk.split())
        words = set(core.TOKEN.findall(query.lower()))
        hits = [(m.start(), m.group()) for m in core.TOKEN.finditer(flat.lower()) if m.group() in words]
        best, best_score = 0, -1.0
        for start in [0] + [flat.rfind(" ", 0, max(0, p - 20)) + 1 for p, _ in hits]:
            score = sum(self.bm25.idf.get(w, 0) for w in {w for p, w in hits if start <= p < start + SNIPPET - len(w)})
            if score > best_score:
                best, best_score = start, score
        return ("…" if best else "") + flat[best:best + SNIPPET] + ("…" if best + SNIPPET < len(flat) else "")

    def search_document(self, query, top_k=5):
        hits = self.bm25.search(str(query), max(1, min(int(top_k), 10)))
        if not hits:
            return {"results": [], "note": "No passage contains any word of the query; search again with other words."}
        return {"results": [{"chunk_id": i, "snippet": self.snippet(c, str(query))} for i, _, c in hits]}

    def read_chunk(self, chunk_id, neighbors=0):
        i, n = int(chunk_id), int(neighbors)
        if not 0 <= i < len(self.chunks):
            raise ToolError(f"chunk_id must be between 0 and {len(self.chunks) - 1}")
        if not 0 <= n <= MAX_NEIGHBORS:
            raise ToolError(f"neighbors must be between 0 and {MAX_NEIGHBORS}")
        if n == 0:
            return {"chunk_id": i, "text": self.chunks[i]}
        lo, hi = max(0, i - n), min(len(self.chunks) - 1, i + n)
        return {"chunk_ids": list(range(lo, hi + 1)), "text": self.text[self.starts[lo]:self.starts[hi] + CHUNK]}

    def calculator(self, expressions):
        return core.calculate(expressions)


def parse(text):
    nums = NUM.findall((text or "").replace("$", ""))
    try:
        return float(nums[-1].replace(",", "").rstrip(".")) if nums else None
    except ValueError:
        return None


def agree(pred, gold):
    if pred is None:
        return False
    tol = 0.005 if gold == 0 else abs(gold) * 0.01
    return any(abs(p - gold) <= tol for p in (pred, pred * 100, pred / 100))


def grade_doc(task, answer):
    gold = task["answer"]
    if isinstance(gold, str):
        words = re.findall(r"\b(yes|no)\b", (answer or "").lower())
        return {"passed": bool(words) and words[-1] == gold, "pred": words[-1] if words else None, "gold": gold}
    pred = parse(answer)
    return {"passed": answer is not None and agree(pred, gold), "pred": pred, "gold": gold}


class DocKind(base.EnvKind):
    name = "docfinqa"
    SYSTEM, TOOLS, EXAMPLES = SYSTEM, TOOLS, EXAMPLES

    def make_env(self, task, db_path):
        return DocEnv(os.path.join(task.get("docs_dir") or db_path, task["doc"] + ".txt"))

    def grade(self, task, env):
        return grade_doc(task, env.answer)

    def categorize(self, ep, g):
        if g["passed"]:
            return None
        if ep.end in ("context_overflow", "step_limit", "hard_cap"):
            return ep.end
        if max(ep.calls.values(), default=0) >= 3:
            return "loop"
        return "wrong_answer" if g["pred"] is not None else "no_number"


def prepare(raw, split, out):
    items = json.load(open(raw, encoding="utf-8"))
    docs = os.path.join(out, f"docfinqa_{split}")
    os.makedirs(docs, exist_ok=True)
    tasks, skipped = [], 0
    for i, it in enumerate(items):
        raw_ans = str(it["Answer"]).strip()
        if raw_ans.lower() in ("yes", "no"):  # as the domain score: a yes/no question needs the same word
            gold = raw_ans.lower()
        else:
            nums = NUM.findall(raw_ans.replace("$", ""))
            if not nums or not re.match(r"^\W*-?\d", raw_ans.replace("$", "").strip()):  # free text, not a number
                skipped += 1
                continue
            gold = float(nums[0].replace(",", "").rstrip("."))
        text = it["Context"].replace("\t", " ")
        key = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        path = os.path.join(docs, key + ".txt")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        tasks.append({"task_id": f"docfinqa-{split}-{i:04d}", "env": "docfinqa", "instance": "docfinqa", "split": split,
                      "type": "docfinqa", "template": "docfinqa", "feasible": True, "doc": key,
                      "prompt": f"Question about the annual report: {it['Question'].strip()}", "answer": gold})
    with open(os.path.join(out, f"tasks_docfinqa_{split}.jsonl"), "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(json.dumps({"split": split, "items": len(items), "tasks": len(tasks), "non_numeric_answers": skipped,
                      "filings": len(os.listdir(docs))}))
