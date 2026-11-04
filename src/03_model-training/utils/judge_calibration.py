"""Runs the stage-8 method judge on the calibration set of judge_cal_set.py and scores it
(research/log/2026-11-01-0010.md, 11-02 06:0x UTC).

run    sends each item's two rendered messages with the request judge_sft.py makes for M1's data: the same model and
       server, max_tokens from judge_sft.parse()'s default (8,192), temperature 0, thinking on with reasoning effort
       "low", one retry when the reply has no {"verdict": "ok" | "flawed"} object, then "unparsed". It refuses to start
       if judge_sft.py's request no longer has these literals, so the two cannot drift apart. Verdicts are appended
       to --out one JSON line per item (id, verdict, reason, attempts, tokens, seconds) and the run resumes past the
       ids already there; an item whose request fails is left out and retried on the next start.
score  reports, from the verdicts, the ok rate on C0 (a lower bound on specificity, since C0's true label is only
       "presumably ok"), the recall on flawed items (C1 by differing parameter groups, C2 by rule, by run, passed
       or not), unparsed counts, and every category per template, each rate with a 95% Wilson interval. Stage 8 keeps
       only "ok", so the report also gives the share stage 8 would drop (flawed or unparsed). With --earlier it adds
       how C0 and C2's teacher items were judged in M1's own stage 8 (the same prompt, so this checks
       reproducibility at temperature 0).
check  re-renders every item from its source episode and question task with judge_sft.transcript() and compares
       the hash of the user message, so the set holds exactly the prompts stage 8 would send.

Usage: python -B judge_calibration.py run --set <set.jsonl> --out <verdicts.jsonl> --base-url <url> [--base-url ...]
           [--model qwen3.8-27b] [--concurrency 32] [--limit N]
       python -B judge_calibration.py score --set <set.jsonl> --verdicts <verdicts.jsonl> [--earlier <judge.jsonl>,...]
           [--out <report.json>]
       python -B judge_calibration.py check --set <set.jsonl>
"""
import argparse
import asyncio
import collections
import hashlib
import inspect
import itertools
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))
import judge_sft  # noqa: E402

MK = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1"
# stage 8's own judge files name the teacher runs by these sources (judge_tier1b.sh, teacher_hard3.sh)
EARLIER_SOURCE = {"train_q38_H1": "market", "train_hard_q38_H1": "hard"}
REQUEST = ("temperature=0.0",
           'extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}}')


def parse(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--base-url", action="append", required=True)
    r.add_argument("--model", default="qwen3.8-27b")
    r.add_argument("--concurrency", type=int, default=32)  # stage 8 sent 32 requests per TP-2 teacher server
    r.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("score")
    s.add_argument("--set", required=True)
    s.add_argument("--verdicts", required=True)
    s.add_argument("--earlier", default="")
    s.add_argument("--out", default=None)
    c = sub.add_parser("check")
    c.add_argument("--set", required=True)
    return ap.parse_args(argv)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def judge_defaults():
    return judge_sft.parse(["--source", "x:y:z", "--out", "x", "--base-url", "x", "--model", "x"])


# -- run ------------------------------------------------------------------------------------------------------------

async def run(a):
    src = inspect.getsource(judge_sft.main_async)
    missing = [x for x in REQUEST if x not in src]
    if missing:
        raise SystemExit(f"judge_sft.py's request changed ({missing} not found); update judge_calibration.py first")
    from openai import AsyncOpenAI
    max_tokens = judge_defaults().max_tokens
    clients = itertools.cycle([AsyncOpenAI(base_url=u, api_key="none", timeout=1800, max_retries=2)
                               for u in a.base_url])  # judge_sft.py's client settings
    items = list(jsonl(a.set))
    done = {j["id"] for j in jsonl(a.out)} if os.path.exists(a.out) else set()
    todo = [it for it in items if it["id"] not in done][:a.limit]
    sem, out = asyncio.Semaphore(a.concurrency), open(a.out, "a", encoding="utf-8")
    n = collections.Counter()
    t_start = time.time()

    async def one(it):
        async with sem:
            client, t0, toks = next(clients), time.time(), {"prompt": 0, "completion": 0}
            j, attempts, text = None, 0, ""
            try:
                for _ in range(2):  # judge_sft.py: one retry on an unparsed reply
                    attempts += 1
                    r = await client.chat.completions.create(
                        model=a.model, max_tokens=max_tokens, temperature=0.0, messages=it["messages"],
                        extra_body={"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"}})
                    text = r.choices[0].message.content or ""
                    if r.usage:
                        toks["prompt"] += r.usage.prompt_tokens
                        toks["completion"] += r.usage.completion_tokens
                    j = judge_sft.verdict_of(text)
                    if j:
                        break
            except Exception as e:  # noqa: BLE001  left out of the file, so the next start retries it
                n["request_error"] += 1
                print(f"{it['id']}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                return
            row = {"id": it["id"], "verdict": j["verdict"] if j else "unparsed", "reason": (j or {}).get("reason", ""),
                   "attempts": attempts, "tokens": toks, "seconds": round(time.time() - t0, 1)}
            if not j:
                row["reply"] = text[:2000]
            n[row["verdict"]] += 1
            out.write(json.dumps(row) + "\n")
            out.flush()

    await asyncio.gather(*(one(it) for it in todo))
    print(json.dumps({"items": len(items), "already": len(done), "judged": len(todo) - n["request_error"], **n,
                      "seconds": round(time.time() - t_start)}))


# -- score ----------------------------------------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, 100 * (c - h)), 1), round(min(100.0, 100 * (c + h)), 1)]


def cell(rows, target):
    """Counts of one group of items and the rate of `target` (ok for C0, flawed for C1/C2) among the judged ones."""
    c = collections.Counter(r["verdict"] for r in rows)
    judged = c["ok"] + c["flawed"] + c["unparsed"]
    hit = c[target]
    drop = c["flawed"] + c["unparsed"]
    return {"n": len(rows), "judged": judged, "ok": c["ok"], "flawed": c["flawed"], "unparsed": c["unparsed"],
            f"{target}_rate": round(100 * hit / judged, 1) if judged else None, "ci95": wilson(hit, judged),
            "stage8_drop_rate": round(100 * drop / judged, 1) if judged else None}


def score(a):
    items = list(jsonl(a.set))
    ver = {j["id"]: j for j in jsonl(a.verdicts)} if os.path.exists(a.verdicts) else {}
    rows = [dict(it, verdict=ver.get(it["id"], {}).get("verdict", "missing")) for it in items]
    by = lambda rs, f: {k: cell([r for r in rs if f(r) == k], rs[0]["target"])  # noqa: E731
                        for k in sorted({f(r) for r in rs}, key=str)} if rs else {}
    rep = {}
    for cat, target in (("C0", "ok"), ("C1", "flawed"), ("C2", "flawed")):
        rs = [dict(r, target=target) for r in rows if r["category"] == cat]
        rep[cat] = {"all": cell(rs, target), "by_template": by(rs, lambda r: r["template"]),
                    "by_tier": by(rs, lambda r: r["tier"])}
        if cat == "C1":
            rep[cat]["by_n_differing"] = by(rs, lambda r: r["n_differing"])
            rep[cat]["by_differing"] = by(rs, lambda r: "+".join(r["differing"]))
        if cat == "C2":
            rep[cat]["by_rule"] = by(rs, lambda r: r["rule"])
            rep[cat]["by_run"] = by(rs, lambda r: r["source"]["run_dir"])
            rep[cat]["by_passed"] = by(rs, lambda r: r["source"]["passed"])
            rep[cat]["by_rule_passed"] = by(rs, lambda r: f"{r['rule']} passed={r['source']['passed']}")
    flawed = [dict(r, target="flawed") for r in rows if r["category"] in ("C1", "C2")]
    rep["C1+C2"] = cell(flawed, "flawed")
    rep["missing"] = sum(r["verdict"] == "missing" for r in rows)
    if a.earlier:
        early = {}
        for p in a.earlier.split(","):
            for j in jsonl(p):
                early[(j["source"], j["task_id"], j["run"])] = j["verdict"]
        pairs = collections.Counter()
        for r in rows:
            s = r["source"]
            k = (EARLIER_SOURCE.get(s["run_dir"]), s["task_id"], s["run"])
            if r["category"] in ("C0", "C2") and k in early:
                pairs[(r["category"], early[k], r["verdict"])] += 1
        rep["earlier_vs_now"] = {f"{c} earlier={e} now={v}": n for (c, e, v), n in sorted(pairs.items())}
    if a.out:
        with open(a.out, "w") as f:
            f.write(json.dumps(rep, indent=1) + "\n")
    print("| group | n | judged | ok | flawed | unparsed | rate (95% CI) | stage-8 drop |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")

    def line(name, c):
        rate = c.get("ok_rate", c.get("flawed_rate"))
        ci = f" [{c['ci95'][0]}, {c['ci95'][1]}]" if c["ci95"] else ""
        print(f"| {name} | {c['n']} | {c['judged']} | {c['ok']} | {c['flawed']} | {c['unparsed']} | "
              f"{rate}{ci} | {c['stage8_drop_rate']} |")

    for cat in ("C0", "C1", "C2"):
        line(f"{cat} ({'ok rate' if cat == 'C0' else 'recall'})", rep[cat]["all"])
        for part, cells in rep[cat].items():
            if part not in ("all", "by_template"):
                for k, c in cells.items():
                    line(f"{cat} {part[3:]} {k}", c)
    line("C1+C2 (recall)", rep["C1+C2"])
    for cat in ("C0", "C1", "C2"):
        for k, c in rep[cat]["by_template"].items():
            line(f"{cat} template {k}", c)
    print(json.dumps({"missing": rep["missing"], **({"earlier_vs_now": rep["earlier_vs_now"]} if a.earlier else {})}))


# -- check ----------------------------------------------------------------------------------------------------------

def check(a):
    items = list(jsonl(a.set))
    cut = judge_defaults().result_chars
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))
    import judge_cal_set
    tasks, _ = judge_cal_set.load_tasks()
    want = collections.defaultdict(set)
    for it in items:
        want[it["source"]["run_dir"]].add((it["source"]["task_id"], it["source"]["run"]))
    msgs = judge_cal_set.traj_of(want)
    bad = []
    for it in items:
        m = msgs[(it["source"]["run_dir"], (it["source"]["task_id"], it["source"]["run"]))]
        user = judge_sft.transcript(tasks[it["question_task"]], m, cut)
        if hashlib.sha256(user.encode()).hexdigest() != it["user_sha256"] or it["messages"][1]["content"] != user or \
                it["messages"][0]["content"] != judge_sft.SYSTEM:
            bad.append(it["id"])
    print(json.dumps({"items": len(items), "mismatched": bad}))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    args = parse()
    if args.cmd == "run":
        asyncio.run(run(args))
    elif args.cmd == "score":
        score(args)
    else:
        check(args)
