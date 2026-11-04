"""Flag English training samples that copy a FinQA or TAT-QA reference item in ways the earlier checks miss.

research/log/2026-10-28-0040.md. The full-chain audit of that entry found copies of the excerpt reference items
(FinQA and TAT-QA test and dev, and the train 20% held for preference pairs) that no 8-gram, number or vector check
flags, because the sample carries one table or row of the excerpt, carries the table as JSON, or asks the same
question of another page or table of the same report. Three checks, each with its own flags:
- qa_match, three rules. (1) The sample's question (after the last "Question:", else the last paragraph of the
  prompt, a trailing "Answer:" removed; BizBench and Agentar put it after the table), lowercased and cut to letters
  and digits, equals a reference question, its final answer (the \\boxed{} value, else what follows the last
  "Answer:", else the reply's last 300 characters) equals the reference answer, and every number of the gold
  calculation occurs in the sample. The last condition drops the two coincidences the audit read, a generic
  question with the same answer over another company's table; an item without such numbers needs one shared win5
  window instead. (2) The sample carries every distinctive operand of the gold calculation (at least 2) and the gold
  answer, anywhere, and its question shares at least 0.6 of its words (Jaccard) with the reference's. A first run
  that looked for the answer only in the reply missed 23 pool samples, 13 of which hold the operands in one row of
  another table of the same report, so that the reference question can be answered from the sample.
  (3) The reply carries every distinctive operand and the final answer equals the gold answer, whatever the
  question: the audit's pairs of the same calculation on another page share 0.25-0.53 of their question words
  ("what portion of the aggregate carrying value of long-term debt should be reported as a current liability"
  against "what percentage of outstanding amounts ... is current"). The operands must have at least 8 significant
  digits together, the fewest of the audit's pairs (50.33 * 4999): at 6, five news articles in doc_qa_en matched
  15.7 - 15.1 = 0.6 and (16.7 + 16.8) / 2, and the audit's coincidences 300 * 6.5 and (0.3 + 0.2) / 2 have 3 and 2.
  They must also be in the reply: a first run that took them from anywhere matched BizBench's bare "Answer: 220"
  to 108 + 114 = 222, both in the SEC filing of its prompt.
- win5: windows of 5 consecutive distinctive numbers of the reference excerpt that occur in the same order in one
  message of the sample (adjacent repeats merged on both sides, at least 4 distinct values in a window; a window
  found in more than 20 different excerpts is dropped), flagged when the questions share at least 0.8 of their
  words, the sample carries every distinctive operand of the gold calculation, or 3 or more windows are shared, or
  2 that are half of the reference's: an excerpt with 7 distinctive numbers has only 3 windows.
- table_hash: the sample contains a TAT-QA reference table whole, compared after lowercasing, dropping thousands
  separators and turning cell bars and other symbols into spaces, so another bar layout still matches.
  convert_en.py split TAT-QA 80/20 by the whole context, and a table that comes with two different paragraphs
  fell on both sides. The raw table uid is unique per context, so it cannot join them; tables with fewer than 4
  numbers (other than years) are skipped as too short to identify a table.
Answers match within 1%, or at 100 times for a percent written as a ratio, and unsigned, as "decreased by 3.69%"
answers -3.69. A distinctive number is not a whole number up to 100 or in 1900-2035: 100 starts every stock-index
row, and in a first run the S&P 500 row of one five-year period gave two companies' performance tables 4 shared
windows. FinQA's escaped quote marks ("201c", "201d") are not numbers. Gold-calculation numbers are those left of
"=" in each step that are not an earlier step's result, without 100, 1000 and 1,000,000 (unit changes); rule (1)
also leaves out integers 0-12 (the divisors of averages and month counts) unless nothing else is left, and the
operands are the distinctive ones.

Output: interim/decontam/<source>.{qa_match,win5,table_hash}.flags.jsonl and reports/decontam_en_excerpt_copy.json.

Usage: python excerpt_copy_decontam_en.py <English data root> [source ...]
"""
import collections
import json
import math
import os
import re
import sys
import unicodedata

EN = sys.argv[1]
REF_SETS = [("FinQA test", "processed/eval/finqa_test.jsonl"), ("FinQA dev", "processed/eval/finqa_dev.jsonl"),
            ("TAT-QA test", "processed/eval/tatqa_test.jsonl"), ("TAT-QA dev", "processed/eval/tatqa_dev.jsonl"),
            ("FinQA train 20%", "interim/seed/finqa_train_pref.jsonl"),
            ("TAT-QA train 20%", "interim/seed/tatqa_train_pref.jsonl")]
TATQA_RAW = {"train": "tatqa_dataset_train.json", "dev": "tatqa_dataset_dev.json",
             "test": "tatqa_dataset_test_gold.json"}
SOURCES = ["finqa_train80", "tatqa_train80", "dianjin_r1_finqa", "bizbench", "agentar_deepfinance_en", "doc_qa_en",
           "finance_alpaca"]
if sys.argv[2:]:
    SOURCES = [s for s in SOURCES if s in sys.argv[2:]]
TOL, CALC_JACCARD, CALC_DIGITS = 0.01, 0.6, 8
K, MIN_DISTINCT, COMMON, WIN_JACCARD, MIN_WINDOWS = 5, 4, 20, 0.8, 3
MIN_TABLE_NUMBERS, TOP = 4, 10
CONSTANTS = {100.0, 1000.0, 1000000.0}
NUM = re.compile(r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
QUOTE = re.compile(r"(?<![\d.])201[cd]")  # FinQA's escaped quote marks, as in "the 201c2012 credit facility 201d"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def values(text):
    out = set()
    for m in NUM.finditer(QUOTE.sub(" ", text)):
        v = float(m.group(0).replace(",", ""))
        if math.isfinite(v):  # a digit string too long for a float
            out.add(round(v, 6))
    return out


def distinctive(v):
    return not (v == int(v) and (v <= 100 or 1900 <= v <= 2035))


def seq(text):
    out = []
    for m in NUM.finditer(QUOTE.sub(" ", text)):
        v = float(m.group(0).replace(",", ""))
        if math.isfinite(v) and distinctive(v) and (not out or out[-1] != round(v, 6)):
            out.append(round(v, 6))  # "-17.1 ( 17.1 )" and a JSON cell 17.1 give one number
    return out


def windows(xs):
    # windows of fewer distinct values, such as (100, 100, 200, 200, 300), matched arithmetic in math replies
    return {w for w in (tuple(xs[i:i + K]) for i in range(len(xs) - K + 1)) if len(set(w)) >= MIN_DISTINCT}


def calc_numbers(calc):
    """(rule (1) numbers, distinctive operands) of a gold calculation such as "959.2 - 991.1 = -31.9; -31.9 / 991.1"."""
    ops, results = set(), set()
    for step in str(calc or "").split(";"):
        lhs, _, rhs = step.rpartition("=") if "=" in step else (step, "", "")
        ops |= values(lhs) - results
        results |= values(rhs)
    ops -= CONSTANTS
    guard = {v for v in ops if not (v == int(v) and v <= 12)}
    return guard or ops, {v for v in ops if distinctive(v)}


def digits(v):
    return len(f"{v:.6f}".rstrip("0").replace(".", "").strip("0"))


def qnorm(text):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", text).lower()).split())


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def question(user):
    u = re.sub(r"\s*Answer:\s*$", "", user.strip())
    if "Question:" in u:
        u = u.rsplit("Question:", 1)[1]
        paras = [p for p in re.split(r"\n\s*\n", u) if p.strip()]
        return paras[0] if paras else ""  # DianJin follows the question with an instruction paragraph
    paras = [p for p in re.split(r"\n\s*\n", u) if p.strip()]
    return paras[-1] if paras else u


def final_answer(reply):
    boxed = re.findall(r"\\boxed\{+([^{}]*)\}+", reply)
    if boxed:
        return boxed[-1]
    marks = list(re.finditer(r"(?i)answer\W{0,4}[:：]", reply))  # "Answer:", "**Answer**:", "Final Answer:**"
    return reply[marks[-1].end():][:300] if marks else reply[-300:]


def same_value(x, a):
    return any(abs(x * k - a) <= TOL * abs(a) + 1e-9 for k in (1, 100, 0.01))


def answer_in(ref_answer, nums, text):
    if isinstance(ref_answer, str):
        return f" {qnorm(ref_answer)} " in f" {qnorm(text)} "
    a = abs(float(ref_answer))
    return any(same_value(x, a) for x in nums)


def flat(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", unicodedata.normalize("NFKC", text).lower())
    return " " + " ".join(re.sub(r"[^a-z0-9.]+|(?<!\d)\.|\.(?!\d)", " ", text).split()) + " "


refs, by_q, by_op, by_win, docs_of = [], collections.defaultdict(list), collections.defaultdict(list), \
    collections.defaultdict(set), collections.defaultdict(set)
for set_name, rel in REF_SETS:
    for it in read_jsonl(os.path.join(EN, rel)):
        i = len(refs)
        guard, ops = calc_numbers(it.get("calc"))
        q = qnorm(it["question"])
        refs.append({"set": set_name, "id": it["id"], "words": set(q.split()), "answer": it["answer"], "guard": guard,
                     "ops": ops, "digits": sum(map(digits, ops)), "windows": 0})
        by_q[q].append(i)
        if len(ops) >= 2:
            for v in ops:
                by_op[v].append(i)
        for w in windows(seq(it.get("context", ""))):
            by_win[w].add(i)
            refs[i]["windows"] += 1
            docs_of[w].add(it.get("context", ""))
common = [w for w, d in docs_of.items() if len(d) > COMMON]
for w in common:
    del by_win[w]
del docs_of

tables = {}  # TAT-QA question uid -> flattened table of its context
for split, fname in TATQA_RAW.items():
    with open(os.path.join(EN, "raw", "tat-qa", fname), encoding="utf-8") as f:
        for c in json.load(f):
            t = flat("\n".join(" | ".join(str(x).strip() for x in row) for row in c["table"]["table"]))
            for q in c["questions"]:
                tables[f"tatqa-{split}-{q['uid']}"] = t
table_refs = collections.defaultdict(list)
for i, r in enumerate(refs):
    t = tables.get(r["id"])
    if t and len({v for v in values(t) if v != int(v) or not 1900 <= v <= 2035}) >= MIN_TABLE_NUMBERS:
        table_refs[t].append(i)
df = collections.Counter(tok for t in table_refs for tok in set(t.split()))
by_key = collections.defaultdict(list)  # a table can only be contained in a text that has its rarest token
for t in table_refs:
    toks = set(t.split())
    by_key[min(toks, key=lambda x: (df[x], not x[0].isdigit(), -len(x)))].append(t)


def matches(hits):
    return [[refs[i]["set"], refs[i]["id"], s] for i, s in sorted(hits.items(), key=lambda x: -x[1])[:TOP]]


summary = []
for name in SOURCES:
    out = {c: [] for c in ("qa_match", "win5", "table_hash")}
    rules, per_set = collections.Counter(), {c: collections.Counter() for c in out}
    rows = read_jsonl(os.path.join(EN, "processed", "sft", f"{name}.jsonl"))
    for r in rows:
        user = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        reply = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "assistant"), "")
        text = "\n".join((m.get("content") or "") + "\n" + (m.get("reasoning") or "") for m in r["messages"])
        said = "\n".join((m.get("content") or "") + "\n" + (m.get("reasoning") or "") for m in r["messages"]
                         if m["role"] == "assistant")
        nums, said_nums, q = values(text), values(said), qnorm(question(user))
        words, final = set(q.split()), final_answer(reply)
        shared = collections.defaultdict(set)
        for m in r["messages"]:  # per message, so a window does not span two turns
            for part in ("content", "reasoning"):
                for w in windows(seq(m.get(part) or "")):
                    for i in by_win.get(w, ()):
                        shared[i].add(w)

        qa, fired = {}, set()
        for i in by_q.get(q, ()):
            ref = refs[i]
            if (answer_in(ref["answer"], values(final), final) and ref["guard"] <= nums
                    and (ref["guard"] or shared.get(i))):
                qa[i] = 1.0
                fired.add("qa_match:same_question")
        have = collections.Counter(i for v in nums for i in by_op.get(v, ()))
        for i, c in have.items():
            ref = refs[i]
            if i in qa or c < len(ref["ops"]) or isinstance(ref["answer"], str):
                continue
            jac = jaccard(words, ref["words"])
            if jac >= CALC_JACCARD and answer_in(ref["answer"], nums, text):
                qa[i] = round(jac, 3)
                fired.add("qa_match:operands")
            elif (ref["digits"] >= CALC_DIGITS and ref["ops"] <= said_nums
                  and answer_in(ref["answer"], values(final), final)):
                qa[i] = round(jac, 3)
                fired.add("qa_match:same_calculation")

        win = {}
        for i, ws in shared.items():
            ref = refs[i]
            why = [k for k, ok in (("question", jaccard(words, ref["words"]) >= WIN_JACCARD),
                                   ("operands", len(ref["ops"]) >= 2 and ref["ops"] <= nums),
                                   ("windows", len(ws) >= min(MIN_WINDOWS, max(2, ref["windows"] / 2)))) if ok]
            if why:
                win[i] = len(ws)
                fired.update(f"win5:{k}" for k in why)

        tab, body = {}, flat(text)
        for tok in set(body.split()):
            for t in by_key.get(tok, ()):
                if t in body:
                    tab.update((i, 1.0) for i in table_refs[t])

        rules.update(fired)
        for check, hits in (("qa_match", qa), ("win5", win), ("table_hash", tab)):
            if hits:
                out[check].append({"id": r["id"], "matches": matches(hits)})
                per_set[check].update({refs[i]["set"] for i in hits})
    for check, flagged in out.items():
        with open(os.path.join(EN, "interim", "decontam", f"{name}.{check}.flags.jsonl"), "w", encoding="utf-8") as f:
            for x in flagged:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary.append({"source": name, "samples": len(rows), **{c: len(v) for c, v in out.items()},
                    "samples_by_rule": dict(rules), "by_set": {c: dict(v) for c, v in per_set.items()}})
    print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
with open(os.path.join(EN, "reports", "decontam_en_excerpt_copy.json"), "w", encoding="utf-8") as f:
    json.dump({"reference_items": len(refs), "windows": len(by_win), "common_windows_dropped": len(common),
               "tatqa_tables": len(table_refs), "sources": summary}, f, ensure_ascii=False, indent=1)
