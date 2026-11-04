"""DocFinQA in the harness: a question about one 10-K filing, read only through document tools (docs/results.md).

Candidates 20 and 21 run the fixed model under H0 and H1 on DocFinQA's 922 test questions; the result is only
reported. The filing is too long for the prompt (165k to 1.5M characters), so the model sees it through retrieval:
search_document (BM25 over chunks of about CHUNK characters, with OVERLAP between neighbours), read_chunk and the
calculator, and ends with finish. There are no business rules, so the constraint part has nothing to check; context
management, verification (every number of the answer traced to a tool result) and correction work as in agent.py.
A question is correct by the English domain score's rule (src/04_evaluation/utils/num_eval.py): the answer's last number,
itself or multiplied or divided by 100, within 1% of the gold answer (0.005 when the gold answer is 0); a yes/no
question needs the same word. A gold answer with units ("$ 13.4 million") is its first number; the few free-text
gold answers are left out.

Usage: python docfinqa.py prepare <raw dev.json or test.json> <split> <out dir>
       (writes <out dir>/tasks_docfinqa_<split>.jsonl and one text file per filing under <out dir>/docfinqa_<split>/)
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))
import agent  # noqa: E402
from env import core  # noqa: E402

CHUNK, OVERLAP = 1500, 200
# an exponent is part of the number (research/log/2026-10-29-0442.md)
NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")

SYSTEM = """You answer questions about a company's annual report (Form 10-K). The report is too long to show; read it with the tools: search_document finds passages, read_chunk shows a passage in full, and calculator does arithmetic. Act without asking questions. When done, call finish with the answer as a single number (a percentage as its number, e.g. 12.5 for 12.5%; no units), or with yes or no when the question asks for one."""
TOOLS = [
    agent.schema("search_document", "BM25 search over the report's passages; returns chunk ids and snippets.",
                 {"query": agent.S, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
    agent.schema("read_chunk", "The full text of one passage of the report.", {"chunk_id": {"type": "integer", "minimum": 0}},
                 ["chunk_id"]),
    agent.schema("calculator", "Evaluate one or more arithmetic expressions in one call, e.g. [\"(5829-5735)/5735*100\"].",
                 {"expressions": {"type": "array", "items": agent.S, "minItems": 1}}, ["expressions"]),
    agent.schema("finish", "End with the answer, a single number.", {"answer": agent.S}, ["answer"]),
]


class DocEnv:
    def __init__(self, path):
        text = open(path, encoding="utf-8").read()
        self.chunks = [text[i:i + CHUNK] for i in range(0, max(1, len(text) - OVERLAP), CHUNK - OVERLAP)]
        self.bm25 = core.BM25([(i, "", c) for i, c in enumerate(self.chunks)])
        self.finished, self.answer = False, None

    def search_document(self, query, top_k=5):
        hits = self.bm25.search(str(query), max(1, min(int(top_k), 10)))
        return {"results": [{"chunk_id": i, "snippet": " ".join(c.split())[:240]} for i, _, c in hits]}

    def read_chunk(self, chunk_id):
        i = int(chunk_id)
        if not 0 <= i < len(self.chunks):
            raise core.ToolError(f"chunk_id must be between 0 and {len(self.chunks) - 1}")
        return {"chunk_id": i, "text": self.chunks[i]}

    def calculator(self, expressions):
        return core.calculate(expressions)

    def finish(self, answer):
        self.finished, self.answer = True, str(answer)
        return {"status": "finished"}


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


class DocEpisode(agent.Episode):
    SYSTEM, TOOLS = SYSTEM, TOOLS

    def make_env(self, docs_dir):
        return DocEnv(os.path.join(docs_dir, self.task["doc"] + ".txt"))

    def grade(self):
        gold, answer = self.task["answer"], self.env.answer
        if isinstance(gold, str):
            words = re.findall(r"\b(yes|no)\b", (answer or "").lower())
            return {"passed": bool(words) and words[-1] == gold, "pred": words[-1] if words else None, "gold": gold}
        pred = parse(answer)
        ok = answer is not None and agree(pred, gold)
        return {"passed": ok, "pred": pred, "gold": self.task["answer"]}

    def categorize(self, g):
        if g["passed"]:
            return None
        if self.end in ("context_overflow", "step_limit"):
            return self.end
        if max(self.calls.values(), default=0) >= 3:
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


if __name__ == "__main__" and sys.argv[1] == "prepare":
    prepare(*sys.argv[2:5])
