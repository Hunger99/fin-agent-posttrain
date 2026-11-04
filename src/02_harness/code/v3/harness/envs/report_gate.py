"""Composite template (i), t6_report_gate ("report_gate", research/log/2026-10-30-1235.md): a question about a
company's annual report decides whether a customer subscribes to a fund that holds the company.

The question is a DocFinQA one whose gold answer g is a percentage (the question says "percent" or "%", 1 <= |g| <= 1000),
asked "as a percentage", so that comparing it with the threshold has no 12.5-or-0.125 ambiguity. An answer greater than
T% opens the gate: the customer invests $A, or the largest whole $100 that the rules allow when they do not allow $A, or
nothing when no amount is allowed, naming the rule (t5_review_switch's sizing, its three cases drawn evenly among those
possible); otherwise no trade. The gate is open for half the tasks, with T = g - m, and closed for the other half, with
T = g + m, m = max(25% of |g|, 1), so an answer within DocFinQA's 1% tolerance takes the gold decision.

The model has the wealth-management tools and docfinqa's search_document and read_chunk over the one report. The
wealth part is graded by grade.grade as it stands; the report's answer is found when a number of the final answer is
within 1% of g, or, written as a fraction, within 1% after multiplying by 100 (DocFinQA's rule without dividing by 100,
which here would let an amount such as $1,250 match 12.5%).

Version 3 (research/log/2026-10-31-0150.md): the environment is one GateEnv class over the package's one core module,
so a missing document or chunk raises the ToolError the harness catches (in version 2, report_gate.py loaded env/core.py
a second time, and such an error ended the whole run). Tasks and grading are version 2's.

Usage: python -B envtool.py make-gate <instance.sqlite> <split> <tasks_docfinqa.jsonl> <docs dir> <n> <seed> <out.jsonl>
       python -B envtool.py verify-gate <instance.sqlite> <tasks.jsonl>   (exits 1 when a check fails)
"""
import collections
import json
import os
import random
import re

from harness.envs import docfinqa
from harness.envs.wealth import core, grade, tasks
from harness.envs.wealth.kind import TOOLS as WEALTH_TOOLS, SYSTEM as WEALTH_SYSTEM, WealthKind

PERCENT_Q = re.compile(r"percent|%", re.I)
SYSTEM = WEALTH_SYSTEM + """

Some requests also concern a company's annual report (Form 10-K), which is too long to show: search_document finds passages of it and read_chunk shows a passage in full."""


class GateEnv(core.Env):
    """The wealth-management environment with the document tools over one annual report."""

    def __init__(self, db_path, doc_path):
        super().__init__(db_path)
        self.doc = docfinqa.DocEnv(doc_path)

    def search_document(self, query, top_k=5):
        return self.doc.search_document(query, top_k)

    def read_chunk(self, chunk_id, neighbors=0):
        return self.doc.read_chunk(chunk_id, neighbors)


def doc_found(gold, nums):
    return any(abs(n - gold) <= 0.01 * abs(gold) or abs(100 * n - gold) <= 0.01 * abs(gold) for n in nums)


def grade_gate(task, final_state, answer):
    """grade.grade on the wealth part, with the report's answer as one more key (kind "doc")."""
    g = grade.grade(dict(task, keys=[k for k in task["keys"] if k["kind"] != "doc"]), final_state, answer)
    nums = [] if answer is None else grade.numbers(grade.normalize(answer))
    miss = [k for k in task["keys"] if k["kind"] == "doc" and not doc_found(k["value"], nums)]
    g["missing_keys"] += miss
    g["passed"] = g["passed"] and not miss
    return g


class GateKind(WealthKind):
    name = "report_gate"
    SYSTEM, TOOLS = SYSTEM, WEALTH_TOOLS[:-1] + docfinqa.DOC_TOOLS + WEALTH_TOOLS[-1:]
    EXAMPLES = dict(WealthKind.EXAMPLES, **{k: docfinqa.EXAMPLES[k] for k in ("search_document", "read_chunk")})

    def make_env(self, task, db_path):
        return GateEnv(task.get("db") or db_path, os.path.join(task["docs_dir"], task["doc"] + ".txt"))

    def grade(self, task, env):
        return grade_gate(task, env.state(task["customers"]), env.answer)


def t6_report_gate(env, rng, d, opened):
    c = rng.choice(tasks.all_customers(env))
    cu = tasks.customer(env, c)
    ok = [dict(r) for r in env.db.execute("SELECT * FROM products WHERE risk_level <= ? AND NOT subscription_suspended",
                                          (cu["risk_level"],))]
    if not ok:
        return None
    p = rng.choice(ok)
    b, cash = p["product_id"], cu["cash"]
    room, low = tasks.subscribe_room(env, c, b), tasks.minimum(env, c, b)
    cases = []
    if room >= max(500, low):
        cases.append("full")
        if room + 200 <= cash:
            cases.append("capped")
    elif cash >= 600:
        cases.append("blocked")
    if not cases:
        return None
    case = rng.choice(cases)
    if case == "full":
        a = amount = rng.randrange(max(500, int(low)) // 100, room // 100 + 1) * 100
    elif case == "capped":
        a, amount = rng.randrange(room // 100 + 2, int(cash) // 100 + 1) * 100, room
    else:
        a, amount = rng.randrange(5, int(cash) // 100 + 1) * 100, 0
    g = d["answer"]
    m = max(0.25 * abs(g), 1.0)
    t = round(g - m if opened else g + m, 1)
    question = d["prompt"].split(": ", 1)[1].strip()
    question += "" if question[-1] in "?." else "?"
    prompt = (f"Customer {c} is considering fund {b}, which holds a large position in the company whose annual report "
              f"(Form 10-K) the document tools read. First answer this question about the report, as a percentage (for "
              f"12.5% write 12.5): {question} If the answer is greater than {t}%, invest ${a:,} in {b} for the customer, or, "
              f"if the business rules do not allow ${a:,}, the largest whole multiple of $100 that they allow; if the rules "
              f"allow no amount, make no subscription and say which rule prevents it. If the answer is {t}% or less, make "
              f"no trade. Report the answer to the question and, if the customer invests, the amount invested and the "
              f"units bought.")
    keys, broken, solve = [{"kind": "doc", "value": g}], [], None
    if opened and amount:
        fee = round(amount * p["subscription_fee"], 2)
        keys += [tasks.money(amount), tasks.units(round((amount - fee) / p["nav"], 4))]
        solve = lambda e: e.subscribe(c, b, amount)  # noqa: E731
    elif opened:
        broken = [r for r, _ in core.check(env.db, "subscribe", {"customer_id": c, "product_id": b, "amount": max(room, 100)})]
        if not broken:
            return None
    info = {"threshold": t, "gate": "open" if opened else "closed", "case": case if opened else "closed",
            "requested": a, "doc": d["doc"], "doc_task_id": d["task_id"]}
    return (prompt, [c], keys, broken, solve), info


def make(db_path, split, doc_tasks, docs_dir, n, seed):
    env = core.Env(db_path)
    instance = env.db.execute("SELECT substr(customer_id, 1, 1) FROM customers LIMIT 1").fetchone()[0]
    rng = random.Random(seed)
    pool = [d for d in map(json.loads, open(doc_tasks, encoding="utf-8"))
            if isinstance(d["answer"], float) and PERCENT_Q.search(d["prompt"]) and 1 <= abs(d["answer"]) <= 1000]
    if n > len(pool):
        raise SystemExit(f"only {len(pool)} percentage questions for {n} tasks")
    rng.shuffle(pool)
    out = []
    for j, d in enumerate(pool[:n]):  # each report question once; the gate open for every other one
        for _ in range(200):
            got = t6_report_gate(env, rng, d, j % 2 == 0)
            if got is not None:
                break
        else:
            raise SystemExit(f"could not draw a task for {d['task_id']}")
        rec = tasks.record(env, f"{instance}-{split}-t6_report_gate-{j:04d}", instance, split, "T6", t6_report_gate, True, got[0])
        rec.update(got[1], writes=True, env="report_gate", docs_dir=os.path.abspath(docs_dir))
        out.append(rec)
    rng.shuffle(out)
    return out


def verify(db_path, tasks_path, per_task=None):
    """The checks of version 2's verify; with per_task, a list that receives each task's check outcomes."""
    base = core.Env(db_path)

    def state(task, calls):
        e = core.Env.__new__(core.Env)
        e.db = core.sqlite3.connect(":memory:")
        base.db.backup(e.db)
        e.db.row_factory = core.sqlite3.Row
        broke = []
        for tool, args in calls:
            broke += core.check(e.db, tool, args)
            getattr(e, tool)(**args)
        return e.state(task["customers"]), broke

    bad, counts = [], collections.Counter()
    for task in map(json.loads, open(tasks_path, encoding="utf-8")):
        g = task["keys"][0]["value"]
        rest = " ".join(str(k["value"]) for k in task["keys"][1:]) + "".join(f" rule {r}" for r in task["violated_rules"][:1])
        answer, wrong = f"The answer is {g}%. {rest}", f"The answer is {round(1.5 * g, 2)}%. {rest}"
        final, broke = state(task, task["reference_calls"])
        ok_ref = grade_gate(task, final, answer)["passed"] and not broke
        ok_wrong_fails = not grade_gate(task, final, wrong)["passed"]
        counts[task["case"]] += 1
        never = grade_gate(task, state(task, [])[0], answer)["passed"]
        always = grade_gate(task, state(task, [["subscribe", {"customer_id": task["customers"][0], "product_id":
                            re.search(r"fund (\S+), which", task["prompt"]).group(1), "amount": task["requested"]}]])[0], answer)["passed"]
        counts[f"never-trade passes ({task['case']})"] += never
        counts[f"always-invest-as-asked passes ({task['case']})"] += always
        should_fail_never = task["case"] in ("full", "capped")
        should_fail_always = task["case"] != "full"
        if per_task is not None:
            per_task.append({"task_id": task["task_id"], "ref": ok_ref, "wrong_fails": ok_wrong_fails, "never": never,
                             "always": always})
        if not ok_ref or not ok_wrong_fails or (should_fail_never and never) or (should_fail_always and always):
            bad.append((task["task_id"], task["case"], ok_ref, ok_wrong_fails, never, always))
    print(json.dumps({"tasks": sum(counts[k] for k in ("full", "capped", "blocked", "closed")), "counts": counts,
                      "failed_checks": len(bad)}))
    for x in bad[:10]:
        print(x)
    return not bad
