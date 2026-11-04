"""Calibration set with known labels for the stage-8 method judge (judge_sft.py), built without human labels
(research/log/2026-11-01-0010.md, 11-02 06:0x UTC).

The judge reads a market question, the generator's reference method, the agent's tool calls and results, and its final
answer, and is told the answer already matched; it returns "ok" or "flawed". This script builds judge inputs whose
label is known by construction, rendered with judge_sft.py's own SYSTEM and transcript() so that each prompt is the
one stage 8 would have sent:
  C0 control     real passed teacher episodes (train_q38_H1, train_hard_q38_H1), a seeded random draw per template that
                 never reads a judge verdict; label "ok", true label only "presumably ok", so the ok rate measured on
                 them is a lower bound on the judge's specificity. Episodes that a C2 rule flags are not drawn.
  C1 transplant  task T's question and reference method with the calls, results and final answer of a passed teacher
                 episode E of another task of the same template; E's answer fails T's keys and at least one reference
                 value of the two tasks differs by more than its tolerance; label "flawed". Pairs whose tasks differ
                 in one parameter group only (e.g. only the window) are taken first; the differing groups are
                 recorded, and beside them, as a note that plays no part in the selection, the parameters left out
                 of the comparison (NOT_COMPARED, e.g. vol_scale) that also differ.
  C2 rule error  real episodes of any of the five runs (passed or failed: the judge's prompt says the answer matched
                 either way) that ended with an accepted finish and whose data calls break a rule checked in code
                 (RULES below: price type, date window, date points, rolling look-back, FRED vintage, instrument);
                 label "flawed". The two convention rules (price type, vintage) also test on the snapshot that the
                 two conventions give different values on the dates and fields the reference reads (a volume is the
                 same split-adjusted and total-return), since the judge is told that
                 differences which cannot change the value are fine; a broken rule changes at least one value the
                 reference computes, not necessarily the final answer.

Data calls are read from the get_prices / get_fred / get_corporate_actions tool calls (their arguments; a call counts
only when its result returned rows) and from the data.* calls inside run_python code (ast, with string constants and
simple loops over literal lists resolved). The window of a call is the one it asked for, never the rows of the page
it read, because the transcript cuts results to 800 characters and a rule may rest only on what the judge sees. An
episode with a data call this cannot resolve (a computed symbol or date, a helper function, a call through another
name) is left out of the rules that need that call kind, rather than guessed. C2 also needs every call's arguments
within the 4,000 characters the transcript shows.

Output: <out>/judge_cal_set.jsonl (one item per line: id, category, template, label, rule or differing parameters,
evidence, source episode, question task, and the two judge messages) and <out>/judge_cal_manifest.json (counts,
seeds, inputs and their sha256, and the number of candidates per rule).

Usage: python -B judge_cal_set.py --out <dir> [--cache <dir>] [--seed 20261102] [--c0-per-template 4:5]
       [--c1-per-template 4:5] [--c2-total 100]
"""
import argparse
import ast
import collections
import datetime
import hashlib
import json
import os
import random
import sqlite3
import sys
from bisect import bisect_left, bisect_right

sys.dont_write_bytecode = True  # importing src/02_harness must leave no __pycache__ there
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_sft  # noqa: E402  SYSTEM and transcript() are the judge's own rendering

V3 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "02_harness", "code", "v3")
MK = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1"
DB = f"{MK}/market_v1.sqlite"
TASK_FILES = {"tier1": f"{MK}/tasks/market_tasks_train_shuffled.jsonl",
              "hard": f"{MK}/tasks/market_tasks_hard_train.jsonl",
              "hard_extra": f"{MK}/tasks/market_tasks_hard_train_extra.jsonl",
              "hardset_train": f"{MK}/hardset_v1/train_market.jsonl"}
# run directory under market_v1/runs -> whose episodes; the teacher's are the ones stage 8 judged for M1
RUNS = {"train_q38_H1": "teacher", "train_hard_q38_H1": "teacher", "screen_m1_train": "M1",
        "screen_base_hard": "base", "screen_base_train": "base"}
CUT = 4000  # judge_sft.transcript() shows the first 4,000 characters of a call's arguments
ADJ = {"raw": "raw", "unadjusted": "raw", "none": "raw", "split": "split", "split_adjusted": "split",
       "total": "total", "total_return": "total"}  # market_query.ADJUSTMENTS, the aliases the tool accepts
FIELDS = ("open", "high", "low", "close", "volume")  # market_query.FIELDS
LO, HI = "0000-00-00", "9999-99-99"  # an open start or end of a data.* call (the whole history)
UNKNOWN = object()


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None, help="directory for the per-episode call summaries (rebuilt if absent)")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--c0-per-template", default="4:5", help="items per tier-1 template : per hard template")
    ap.add_argument("--c1-per-template", default="4:5")
    ap.add_argument("--c2-total", type=int, default=100)
    return ap.parse_args(argv)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# -- tasks ----------------------------------------------------------------------------------------------------------

def load_tasks():
    """(task_id -> task over the four task files, a task in several files being the same dict (checked);
    task_id -> "tier1" or "hard" for the tasks of M1's market templates: tier 1's 17 and the hard set's 7, without
    div_event_study, which M1 does not use)."""
    tasks, where = {}, collections.defaultdict(set)
    for name, p in TASK_FILES.items():
        for t in jsonl(p):
            if t["task_id"] in tasks and tasks[t["task_id"]] != t:
                raise SystemExit(f"task {t['task_id']} differs between task files")
            tasks[t["task_id"]] = t
            where[t["task_id"]].add(name)
    tier = {}
    for i, w in where.items():
        if "tier1" in w:
            tier[i] = "tier1"
        elif "hardset_train" in w:
            tier[i] = "hard"
    return tasks, tier


# -- call summaries -------------------------------------------------------------------------------------------------

class Resolver(ast.NodeVisitor):
    """Possible values of the names in run_python code: a name bound only to string/number constants, to lists of
    them, or as the loop variable over such a list, resolves to the set of those values; any other binding makes it
    unresolvable. A set over-approximates what one call saw, which keeps the rules conservative: a rule fires only
    when the required value is absent from every possibility."""

    def __init__(self):
        self.vals = collections.defaultdict(set)
        self.lists = collections.defaultdict(list)
        self.bad = set()

    def bind(self, target, values):
        if isinstance(target, ast.Name):
            if values is UNKNOWN:
                self.bad.add(target.id)
            else:
                self.vals[target.id] |= values
        else:  # unpacking binds each stored name to an unknown value; df['x'] = ... binds no name
            for n in ast.walk(target):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    self.bad.add(n.id)

    def literal_list(self, node):
        if isinstance(node, (ast.List, ast.Tuple)) and all(isinstance(e, ast.Constant) for e in node.elts):
            return [e.value for e in node.elts]
        return None

    def visit_Assign(self, node):
        for t in node.targets:
            lst = self.literal_list(node.value)
            if isinstance(t, ast.Name) and lst is not None:
                self.lists[t.id].append(lst)
                self.vals[t.id].add(("list", tuple(lst)))
            elif isinstance(node.value, ast.Constant):
                self.bind(t, {node.value.value})
            else:
                self.bind(t, UNKNOWN)
        self.generic_visit(node)

    def iter_values(self, it):
        lst = self.literal_list(it)
        if lst is not None:
            return set(lst)
        if isinstance(it, ast.Name) and it.id in self.lists and it.id not in self.bad:
            if all(isinstance(v, tuple) and v[:1] == ("list",) for v in self.vals[it.id]):
                return {x for lst in self.lists[it.id] for x in lst}
        return UNKNOWN

    def visit_For(self, node):
        self.bind(node.target, self.iter_values(node.iter))
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self.bind(node.target, self.iter_values(node.iter))
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.bind(node.target, UNKNOWN)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.bind(node.target, {node.value.value} if isinstance(node.value, ast.Constant) else UNKNOWN)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.bad.add(a.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for a in node.args.args + node.args.kwonlyargs:
            self.bad.add(a.arg)
        self.generic_visit(node)

    def visit_With(self, node):
        for it in node.items:
            if it.optional_vars is not None:
                self.bind(it.optional_vars, UNKNOWN)
        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        self.bind(node.target, UNKNOWN)
        self.generic_visit(node)

    def visit_Call(self, node):  # syms.append('X') would add a value the literal list does not show
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and \
                f.attr in ("append", "extend", "insert", "add", "update", "remove", "pop", "clear"):
            self.bad.add(f.value.id)
        self.generic_visit(node)

    def resolve(self, node):
        """Set of scalar values, or UNKNOWN."""
        if node is None:
            return {None}
        if isinstance(node, ast.Constant):
            return {node.value}
        if isinstance(node, ast.Name) and node.id not in self.bad and node.id in self.vals:
            vs = self.vals[node.id]
            if any(isinstance(v, tuple) and v[:1] == ("list",) for v in vs):
                return UNKNOWN
            return set(vs)
        return UNKNOWN


SIGS = {"get_prices": ("symbol", "start", "end", "adjustment", "fields", "interval"),
        "get_fred": ("series_id", "start", "end", "vintage"),
        "get_corporate_actions": ("symbol", "start", "end", "kind")}


def code_calls(code):
    """(calls, unresolved kinds) of the data.get_prices / get_fred / get_corporate_actions calls in run_python code.
    Each call is a dict of its resolved arguments (a value or a set of values). Code that does not parse never ran,
    so it has no calls. A kind is unresolved when an argument the rules read cannot be resolved, or when the code text
    names the function more often than the calls found (a call through another name or getattr)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return [], set()
    r = Resolver()
    r.visit(tree)
    calls, bad, found = [], set(), collections.Counter()
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SIGS):
            continue
        fn = n.func.attr
        found[fn] += 1
        if not (isinstance(n.func.value, ast.Name) and n.func.value.id == "data") or \
                any(isinstance(a, ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords):
            bad.add(fn)
            continue
        args = dict(zip(SIGS[fn], n.args))
        args.update({k.arg: k.value for k in n.keywords})
        need = {"get_prices": ("symbol", "start", "end", "adjustment"),
                "get_fred": ("series_id", "start", "end", "vintage"), "get_corporate_actions": ("symbol",)}[fn]
        out = {"fn": fn}
        for k in need:
            v = r.resolve(args.get(k))
            if v is UNKNOWN or any(x is not None and not isinstance(x, str) for x in v):
                bad.add(fn)
                break
            out[k] = sorted(v, key=lambda x: (x is None, x or ""))
        else:
            calls.append(out)
    for fn in SIGS:
        if code.count(fn) > found[fn]:
            bad.add(fn)
    return calls, bad


def first_json(text):
    try:
        return json.JSONDecoder().raw_decode(text or "")[0]
    except ValueError:
        return None


def canon_symbol(s, secs):
    """market_query.symbol(): upper case, then '.' -> '-' or '/' removed; None when the snapshot has no such symbol."""
    s = str(s or "").strip().upper()
    for c in (s, s.replace(".", "-"), s.replace("/", "")):
        if c in secs:
            return c
    return None


def summarize(traj, secs):
    """Compact record of an episode's data calls: price, fred and actions fetches with the window each asked for
    (a tool call counts only when its result returned rows), the unresolved call kinds, the longest call arguments,
    and the number of finish calls."""
    results = {m.get("tool_call_id"): m.get("content") or "" for m in traj["messages"] if m["role"] == "tool"}
    prices, fred, actions, bad, longest, finishes = [], [], [], set(), 0, 0
    for m in traj["messages"]:
        if m["role"] != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            name, raw = c["function"]["name"], c["function"]["arguments"] or ""
            longest = max(longest, len(raw))
            try:
                args = json.loads(raw)
            except ValueError:
                args = None
            if not isinstance(args, dict):
                if name in ("get_prices", "get_fred", "get_corporate_actions", "run_python"):
                    bad.add(name)
                continue
            res = first_json(results.get(c.get("id"), ""))
            if name == "finish":
                finishes += 1
            elif name == "get_prices":
                # the window a call asked for, not the rows of the page it read: the transcript cuts results to 800
                # characters, so a rule may only rest on what the judge can see, the call's arguments
                if isinstance(res, dict) and res.get("rows"):
                    prices.append({"via": "tool", "symbol": res.get("symbol"), "adj": res.get("adjustment"),
                                   "cover": [str(args.get("start")), str(args.get("end"))]})
            elif name == "get_fred":
                if isinstance(res, dict) and res.get("rows"):
                    fred.append({"via": "tool", "series": res.get("series_id"), "vintage": res.get("vintage"),
                                 "cover": [str(args.get("start")), str(args.get("end"))]})
            elif name == "get_corporate_actions":
                actions.append({"via": "tool", "symbol": canon_symbol(args.get("symbol"), secs)})
            elif name == "run_python":
                calls, b = code_calls(str(args.get("code") or ""))
                bad |= {f"code_{x}" for x in b}
                for k in calls:
                    if k["fn"] == "get_prices":
                        for s in k["symbol"]:
                            for adj in k["adjustment"]:
                                a = ADJ.get(str(adj or "raw").strip().lower())
                                if a is None:
                                    continue  # an adjustment the sandbox rejects returns no data
                                for st in k["start"]:
                                    for en in k["end"]:
                                        prices.append({"via": "code", "symbol": canon_symbol(s, secs), "adj": a,
                                                       "cover": [st or LO, en or HI]})
                    elif k["fn"] == "get_fred":
                        for s in k["series_id"]:
                            for v in k["vintage"]:
                                v = str(v or "as_of_cutoff").strip().lower()
                                for st in k["start"]:
                                    for en in k["end"]:
                                        fred.append({"via": "code", "series": str(s).strip().upper(), "vintage": v,
                                                     "cover": [st or LO, en or HI]})
                    else:
                        for s in k["symbol"]:
                            actions.append({"via": "code", "symbol": canon_symbol(s, secs)})
    return {"task_id": traj["task_id"], "run": traj["run"], "prices": prices, "fred": fred, "actions": actions,
            "unresolved": sorted(bad), "longest_args": longest, "finishes": finishes}


def load_runs(tasks, cache):
    """{run name: {(task_id, run): (result row, summary)}} over the five runs, the last row of each key (as
    judge_sft.py and filter_sft.py keep it); summaries are cached as JSONL under cache."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    secs = {r[0] for r in con.execute("SELECT symbol FROM securities")}
    con.close()
    out = {}
    for name in RUNS:
        d = f"{MK}/runs/{name}"
        res = {(r["task_id"], r["run"]): r for r in jsonl(f"{d}/results.jsonl")}
        cpath = cache and os.path.join(cache, f"{name}.summaries.jsonl")
        if cpath and os.path.exists(cpath):
            summ = {(s["task_id"], s["run"]): s for s in jsonl(cpath)}
        else:
            summ = {}
            for t in jsonl(f"{d}/trajectories.jsonl"):
                if t["task_id"] in tasks:
                    summ[(t["task_id"], t["run"])] = summarize(t, secs)
            if cpath:
                os.makedirs(cache, exist_ok=True)
                with open(cpath + ".tmp", "w") as f:
                    for s in summ.values():
                        f.write(json.dumps(s) + "\n")
                os.replace(cpath + ".tmp", cpath)
        out[name] = {k: (res[k], s) for k, s in summ.items() if k in res}
    return out


# -- the snapshot: trading days and price levels per adjustment ----------------------------------------------------

class Snap:
    """Read-only lookups on market_v1.sqlite for the rules' requirements and materiality tests."""

    def __init__(self):
        self.con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        self._days = {}

    def days(self, sym):
        if sym not in self._days:
            self._days[sym] = [r[0] for r in self.con.execute(
                "SELECT date FROM prices WHERE symbol=? ORDER BY date", (sym,))]
        return self._days[sym]

    def on_or_before(self, sym, d):
        ds = self.days(sym)
        i = bisect_right(ds, d) - 1
        return ds[i] if i >= 0 else None

    def shift(self, sym, d, n):
        """The trading day n trading days after (n > 0) or before (n < 0) trading day d."""
        ds = self.days(sym)
        i = ds.index(d) + n
        return ds[i] if 0 <= i < len(ds) else None

    def between(self, sym, a, b):
        ds = self.days(sym)
        return ds[bisect_left(ds, a):bisect_right(ds, b)]

    def levels(self, sym, a, b):
        """{date: {"raw", "split", "total": {field: value}}} from a to b, each adjustment's open, high, low, close
        and volume as market_query.prices() builds them: the raw_* columns; open..volume (split-adjusted); open,
        high and low scaled by adj_close/close, adj_close, and the split-adjusted volume (total)."""
        rows = self.con.execute("SELECT date, open, high, low, close, adj_close, volume, raw_open, raw_high, raw_low, "
                                "raw_close, raw_volume FROM prices WHERE symbol=? AND date BETWEEN ? AND ?", (sym, a, b))
        out = {}
        for d, o, h, lo, c, ac, v, ro, rh, rl, rc, rv in rows:
            f = ac / c if (ac is not None and c) else None
            sc = [None if (x is None or f is None) else x * f for x in (o, h, lo)]
            out[d] = {"raw": dict(zip(FIELDS, (ro, rh, rl, rc, rv))), "split": dict(zip(FIELDS, (o, h, lo, c, v))),
                      "total": dict(zip(FIELDS, (*sc, ac, v)))}
        return out

    def cpi(self, month):
        """(as_of_cutoff, latest) CPIAUCSL for a YYYY-MM month (market_query.fred: value_vintage, value)."""
        r = self.con.execute("SELECT value_vintage, value FROM fred WHERE series_id='CPIAUCSL' AND date=?",
                             (month + "-01",)).fetchone()
        return r if r else (None, None)


def month_end(m):
    y, mo = int(m[:4]), int(m[5:7])
    nxt = datetime.date(y + mo // 12, mo % 12 + 1, 1)
    return (nxt - datetime.timedelta(days=1)).isoformat()


def period_ends(snap, sym, first, last, freq):
    """The last trading day of each calendar month or ISO week from first to last (the last one cut at last): the
    closes a month or week return reads. The previous period's close is left out, so that a first period computed
    from its own first close, which changes the value only when that period is the extreme, never fires a rule."""
    ends = {}
    for d in snap.between(sym, first, last):
        k = d[:7] if freq == "month" else tuple(datetime.date.fromisoformat(d).isocalendar()[:2])
        ends[k] = d
    return sorted(ends.values())


def prev_month(m):
    y, mo = int(m[:4]), int(m[5:7])
    return f"{y - 1}-12" if mo == 1 else f"{y}-{mo - 1:02d}"


# -- what each template's reference computation reads (harness/market_reference.py) ---------------------------------

LEVEL_TEMPLATES = {"price_lookup", "market_cap", "split_event", "reinvest_return"}  # price levels, not returns


def needs(t, snap):
    """The price series and FRED series the task's reference code reads (market_reference._c_<template>):
    prices: [{"symbol", "adj", "span": (first, last) or None, "points": [trading days] or None, "lookback": n,
    "fields": the columns it reads, close unless the template reads others}],
    fred: [{"series", "vintage" (None when the question leaves it open), "months"}]. Dates of the span are calendar
    dates; points are trading days of the symbol as the reference picks them."""
    tpl, p = t["template"], t["meta"]["params"]
    P, F = [], []

    def pt(sym, *ds):
        return [snap.on_or_before(sym, d) for d in ds]

    if tpl == "price_lookup":
        P.append({"symbol": p["symbol"], "adj": "raw" if p["adj"] == "raw" else "split",
                  "points": pt(p["symbol"], p["date"]), "fields": list(p["fields"])})
    elif tpl == "period_return":
        for m in p["measures"]:
            P.append({"symbol": p["symbol"], "adj": "total" if m == "total" else "split",
                      "points": pt(p["symbol"], p["start"], p["end"])})
    elif tpl == "nth_day_move":
        ds = [d for d in snap.days(p["symbol"]) if d > p["date"]]
        day = ds[p["n"] - 1]
        P.append({"symbol": p["symbol"], "adj": "split",
                  "points": [day] + ([snap.shift(p["symbol"], day, -1)] if p["mode"] == "cc" else []),
                  "fields": ["close"] if p["mode"] == "cc" else ["open", "close"]})
    elif tpl == "extreme_period":
        if p["freq"] == "month":
            first = p["first"][:7] + "-01"
        else:
            d0 = datetime.date.fromisoformat(p["first"])
            first = (d0 - datetime.timedelta(days=d0.weekday())).isoformat()
        P.append({"symbol": p["symbol"], "adj": "split", "points": period_ends(snap, p["symbol"], first, p["last"],
                                                                              p["freq"])})
    elif tpl == "threshold_scan":
        P.append({"symbol": p["symbol"], "adj": "raw" if p["mode"] == "usd_drop" else "split",
                  "span": (p["start"], p["end"])})
    elif tpl == "max_drawdown":
        P.append({"symbol": p["symbol"], "adj": "total" if p["measure"] == "total" else "split",
                  "span": (p["start"], p["end"])})
    elif tpl == "rolling_vol":
        for s in p["symbols"]:
            P.append({"symbol": s, "adj": "split", "span": (p["start"], p["end"]), "lookback": 20})
    elif tpl == "split_event":
        E = p["effective"]
        P.append({"symbol": p["symbol"], "adj": "raw", "points": [E, snap.shift(p["symbol"], E, -1)],
                  "fields": ["open", "close"]})  # the open on E, the close of the day before
    elif tpl == "cross_rank":
        for s in p["symbols"]:
            if p["metric"] == "vol":
                P.append({"symbol": s, "adj": "split", "span": (p["start"], p["end"])})
            else:
                P.append({"symbol": s, "adj": "total" if p["metric"] == "total" else "split",
                          "points": pt(s, p["start"], p["end"])})
    elif tpl == "fx_index_return":
        P.append({"symbol": p["symbol"], "adj": "split", "points": pt(p["symbol"], p["start"], p["end"])})
        F.append({"series": p["series"], "vintage": None})
    elif tpl == "real_return":
        P.append({"symbol": p["symbol"], "adj": "total" if p["total"] else "split",
                  "points": pt(p["symbol"], month_end(p["m0"]), month_end(p["m1"]))})
        F.append({"series": "CPIAUCSL", "vintage": None})
    elif tpl == "market_cap":
        P.append({"symbol": p["symbol"], "adj": "raw", "points": pt(p["symbol"], p["date"])})
    elif tpl == "two_hop":
        ch = p["chain"]
        if ch == "bestmonth_unrate":
            P.append({"symbol": p["symbol"], "adj": "split", "points": period_ends(
                snap, p["symbol"], p["first_month"] + "-01", month_end(p["last_month"]), "month")})
        else:
            P.append({"symbol": p["symbol"], "adj": "split", "span": (p["start"], p["end"])})
        F += [{"series": s, "vintage": None} for s in {"mdd_dgs10": ["DGS10"], "peak_dff": ["DFF"],
                                                        "bestmonth_unrate": ["UNRATE"]}.get(ch, [])]
        if ch == "worstday_vix":
            P.append({"symbol": "^VIX", "adj": "split", "points": [t["keys"][0]["value"]], "fixed": False})
    elif tpl == "risk_rank":
        for s in p["symbols"]:
            P.append({"symbol": s, "adj": "total", "span": (p["start"], p["end"])})
        if p["rf"]:
            F.append({"series": "DTB3", "vintage": None})
    elif tpl == "rolling_beta":
        for s in (p["symbol"], p["bench"]):
            P.append({"symbol": s, "adj": "total" if p["total"] else "split", "span": (p["start"], p["end"]),
                      "lookback": p["window"]})
    elif tpl == "ma_cross":
        if p["mode"] == "backtest":
            P.append({"symbol": p["symbol"], "adj": "split", "span": (p["start"], p["end"]), "lookback": p["long"]})
        else:  # crossings after the date, up to the last one the keys date (the third)
            last = [k["value"] for k in t["keys"] if k["kind"] == "date"][-1]
            P.append({"symbol": p["symbol"], "adj": "split", "span": (p["date"], last), "lookback": p["long"]})
    elif tpl == "reinvest_return":
        P.append({"symbol": p["symbol"], "adj": "raw", "points": pt(p["symbol"], p["start"], p["end"])})
        if p["fx"]:
            F.append({"series": p["fx"], "vintage": None})
    elif tpl == "filing_reaction":
        sym = t["keys"][0]["value"]
        filed = [k["value"] for k in t["keys"] if k["kind"] == "date"][0]
        ds = snap.days(sym)
        day0 = ds[bisect_left(ds, filed)]
        pts = [snap.shift(sym, day0, -1), snap.shift(sym, day0, p["n"])]
        # the days follow from the growth leader and its filing date, which the judge would have to re-solve to
        # check, so they are not points a date rule may test ("fixed": False)
        P.append({"symbol": sym, "adj": "total", "points": pts, "fixed": False, "symbol_fixed": False})
        P.append({"symbol": "^GSPC", "adj": "split", "points": pts, "fixed": False})
    elif tpl == "month_chain":
        P.append({"symbol": p["symbol"], "adj": "total" if p["total"] else "split", "points": period_ends(
            snap, p["symbol"], p["first_month"] + "-01", month_end(p["last_month"]), "month")})
        ym = t["keys"][0]["value"][:7]
        F.append({"series": "CPIAUCSL", "vintage": p["vintage"], "months": [prev_month(ym), ym]})
        if p["ask_yield"]:
            F.append({"series": "DGS10", "vintage": None})
    for n in P:
        n.setdefault("span", None)
        n.setdefault("points", None)
        n.setdefault("lookback", 0)
        n.setdefault("fixed", True)
        n.setdefault("symbol_fixed", True)
        n.setdefault("fields", ["close"])
    return {"prices": P, "fred": F}


# -- C2 rules -------------------------------------------------------------------------------------------------------

MIN_MISSING = 5  # trading days of a span left uncovered: chosen above the 1-2 days a holiday or the start date's
#                  previous close can account for, so a window rule fires only on a shifted or cut window
MATERIAL = 1e-3  # change in the ratio of two adjustments' closes over the dates used: a split, or about one
#                  dividend (95% of the snapshot's 35,125 dividends are at least 0.11% of the ex-date close); the
#                  ratio's rounding noise is below 1e-6 (GOOGL total/split, 2015-2023)
CPI_MATERIAL = 1e-4  # a CPI m/m ratio that differs by 0.01 percentage points, the real return's tolerance
PRICE_KINDS = ("code_get_prices", "get_prices", "run_python")
ACTION_KINDS = ("code_get_corporate_actions", "get_corporate_actions", "run_python")
FRED_KINDS = ("code_get_fred", "get_fred", "run_python")


def covered(fetches, days):
    spans = [f["cover"] for f in fetches]
    return {d for d in days if any(a <= d <= b for a, b in spans)}


def rule_price_type(t, nd, s, snap):
    """The reference reads closes of adjustment X ('raw', 'split' or 'total'); the episode fetched the symbol only
    with other adjustments, never X, and fetched no corporate actions for it (with which it could adjust by hand;
    reinvest_return and split_event need the actions anyway, so for them actions do not excuse it). Material: for
    every adjustment used, the ratio of its values to X's changes by at least MATERIAL across the dates and fields
    the reference reads (for the level templates, differs from 1 on one of them); otherwise the finding is marked
    immaterial. The fields matter because the adjustments scale them differently: total-return volume is the
    split-adjusted volume, so a volume lookup read in total instead of split gives the same value."""
    out = []
    for n in nd["prices"]:
        fs = [f for f in s["prices"] if f["symbol"] == n["symbol"]]
        used = sorted({f["adj"] for f in fs})
        if not fs or n["adj"] in used:
            continue
        if t["template"] not in ("reinvest_return", "split_event") and \
                any(a["symbol"] == n["symbol"] for a in s["actions"]):
            continue
        dates = n["points"] if n["points"] else snap.between(n["symbol"], *n["span"])
        dates = [d for d in dates if d]
        if not dates:
            continue
        lv = snap.levels(n["symbol"], min(dates), max(dates))
        sizes = []
        for u in used:
            f = [lv[d][u][k] / lv[d][n["adj"]][k] for d in dates if d in lv for k in n["fields"]
                 if lv[d][u][k] and lv[d][n["adj"]][k]]
            if not f:
                break
            sizes.append(max(abs(x - 1) for x in f) if t["template"] in LEVEL_TEMPLATES else max(f) / min(f) - 1)
        if len(sizes) == len(used):
            out.append({"rule": "price_type", "symbol": n["symbol"], "required": n["adj"], "used": used,
                        "factor_change": round(min(sizes), 6), "immaterial": min(sizes) < MATERIAL})
    return out


def rule_window(t, nd, s, snap):
    """A span the reference reads (a whole window of closes) has at least MIN_MISSING of its trading days outside
    every fetch of the symbol: the episode computed over a shifted or shorter window."""
    out = []
    for n in nd["prices"]:
        fs = [f for f in s["prices"] if f["symbol"] == n["symbol"]]
        if not fs or not n["span"]:
            continue
        days = snap.between(n["symbol"], *n["span"])
        cov = covered(fs, days)
        miss = [d for d in days if d not in cov]
        if len(miss) >= MIN_MISSING:
            side = "start" if miss[0] == days[0] else "end" if miss[-1] == days[-1] else "middle"
            out.append({"rule": "date_window", "symbol": n["symbol"], "window": list(n["span"]),
                        "missing_days": len(miss), "of": len(days), "first_missing": miss[0],
                        "last_missing": miss[-1], "side": side})
    return out


def rule_dates(t, nd, s, snap):
    """A trading day whose close the reference reads and that the task's parameters fix (the start or end of a
    return, the looked-up day, with the task's rule for non-trading days applied) is outside every window the
    episode asked for that symbol: it used another date."""
    out = []
    for n in nd["prices"]:
        fs = [f for f in s["prices"] if f["symbol"] == n["symbol"]]
        if not fs or not n["points"] or not all(n["points"]) or not n["fixed"]:
            continue
        cov = covered(fs, n["points"])
        miss = [d for d in n["points"] if d not in cov]
        if miss:
            out.append({"rule": "date_points", "symbol": n["symbol"], "points": n["points"], "missing": miss})
    return out


def rule_lookback(t, nd, s, snap):
    """A rolling statistic (rolling_vol, rolling_beta, a moving average of ma_cross) needs the closes before the
    window's first day; the episode fetched no close of the symbol before that day, so its first windows are short
    or empty."""
    out = []
    for n in nd["prices"]:
        if not n["lookback"]:
            continue
        fs = [f for f in s["prices"] if f["symbol"] == n["symbol"]]
        first = n["span"][0]
        days = snap.days(n["symbol"])
        before = days[max(0, bisect_left(days, first) - n["lookback"]):bisect_left(days, first)]
        if fs and before and not covered(fs, before):
            out.append({"rule": "rolling_lookback", "symbol": n["symbol"], "window_start": first,
                        "lookback_days": n["lookback"], "earliest_fetch": min(f["cover"][0] for f in fs)})
    return out


def rule_vintage(t, nd, s, snap):
    """month_chain names the FRED vintage of CPIAUCSL (as_of_cutoff or latest); the episode fetched CPIAUCSL only in
    the other vintage. Material: the two vintages' m/m CPI ratio for the answer month differs by CPI_MATERIAL;
    otherwise the finding is marked immaterial."""
    out = []
    for n in nd["fred"]:
        if not n["vintage"]:
            continue
        fs = [f for f in s["fred"] if f["series"] == n["series"]]
        used = sorted({f["vintage"] for f in fs})
        if not fs or n["vintage"] in used:
            continue
        (c0, l0), (c1, l1) = snap.cpi(n["months"][0]), snap.cpi(n["months"][1])
        if None in (c0, l0, c1, l1):
            continue
        diff = abs(c1 / c0 - l1 / l0)
        out.append({"rule": "fred_vintage", "series": n["series"], "required": n["vintage"], "used": used,
                    "months": n["months"], "mm_ratio_diff": round(diff, 6), "immaterial": diff < CPI_MATERIAL})
    return out


def rule_instrument(t, nd, s, snap):
    """A symbol the task's parameters fix (not filing_reaction's growth leader, which only re-solving finds) or a
    FRED series the reference reads is never fetched, while the episode fetched another symbol or series that is
    neither among the task's entities nor read by the reference: it substituted an instrument."""
    out = []
    ent = {str(e).upper() for e in t["meta"].get("entities") or []}
    need_sym = {n["symbol"] for n in nd["prices"] if n["symbol_fixed"]}
    need_ser = {n["series"] for n in nd["fred"]}
    got_sym = {f["symbol"] for f in s["prices"] if f["symbol"]}
    got_ser = {f["series"] for f in s["fred"]}
    for sym in sorted(need_sym - got_sym):
        other = sorted(got_sym - {n["symbol"] for n in nd["prices"]} - ent)
        if other:
            out.append({"rule": "instrument", "required": sym, "fetched_instead": other})
    for ser in sorted(need_ser - got_ser):
        other = sorted(got_ser - need_ser - ent)
        if other:
            out.append({"rule": "instrument", "required": ser, "fetched_instead": other})
    return out


# rule name -> (function, call kinds that must be resolved for it to apply)
RULES = {"price_type": (rule_price_type, PRICE_KINDS + ACTION_KINDS),
         "date_window": (rule_window, PRICE_KINDS),
         "date_points": (rule_dates, PRICE_KINDS),
         "rolling_lookback": (rule_lookback, PRICE_KINDS),
         "fred_vintage": (rule_vintage, FRED_KINDS),
         "instrument": (rule_instrument, PRICE_KINDS + FRED_KINDS)}


PRIMARY = ["instrument", "fred_vintage", "date_window", "price_type", "date_points", "rolling_lookback"]  # rarest first


def findings(t, s, snap, nd=None):
    """(findings, immaterial findings, rules the episode could not be checked against for unresolved calls). An
    immaterial finding breaks a convention where both conventions give the same values, which the judge's prompt
    calls fine, so it labels nothing and only enters the manifest's counts."""
    nd = nd or needs(t, snap)
    found, minor, skipped = [], [], []
    for name, (fn, kinds) in RULES.items():
        if set(kinds) & set(s["unresolved"]):
            skipped.append(name)
            continue
        for x in fn(t, nd, s, snap):
            (minor if x.pop("immaterial", False) else found).append(x)
    return found, minor, skipped


# -- C1 transplant pairs --------------------------------------------------------------------------------------------

# parameters that move together count as one group: a ticker and its CIK, a window's two ends
GROUP = {"symbol": "instrument", "cik": "instrument", "symbols": "instruments", "ciks": "instruments",
         "start": "window", "end": "window", "first": "window", "last": "window", "first_month": "window",
         "last_month": "window", "m0": "window", "m1": "window", "d0": "dates", "d1": "dates",
         "window": "rolling_length"}
# parameters that only choose which values the answer lists (equal key kinds already make the shapes match) or that
# follow from others (straddles_split from the symbol and window)
NOT_COMPARED = {"with_peak", "list", "want_pr", "straddles_split", "vol_scale"}


def differing(a, b):
    """{group: [param keys]} of the parameters in which tasks a and b differ."""
    pa, pb = a["meta"]["params"], b["meta"]["params"]
    out = collections.defaultdict(list)
    for k in sorted(pa):
        if k.startswith("ask") or k in NOT_COMPARED:
            continue
        if json.dumps(pa[k], sort_keys=True, default=str) != json.dumps(pb.get(k), sort_keys=True, default=str):
            out[GROUP.get(k, k)].append(k)
    return dict(out)


def differing_not_compared(a, b):
    """{param: {"question", "episode"}} of the NOT_COMPARED parameters in which question task a and episode task b
    differ (e.g. price_lookup's vol_scale): a note on the item only, never part of the pair selection."""
    pa, pb = a["meta"]["params"], b["meta"]["params"]
    return {k: {"question": pa.get(k), "episode": pb.get(k)} for k in sorted(NOT_COMPARED & (set(pa) | set(pb)))
            if json.dumps(pa.get(k), sort_keys=True, default=str) != json.dumps(pb.get(k), sort_keys=True, default=str)}


def distance(a, b, d):
    """How far apart two tasks' differing parameters are: the summed days between their dates (a YYYY-MM month as
    its first day) plus the summed absolute differences of numbers; instruments (CIKs are ids, not magnitudes) and
    other values add nothing."""
    pa, pb, out = a["meta"]["params"], b["meta"]["params"], 0.0
    for k in (k for g, ks in d.items() if g not in ("instrument", "instruments") for k in ks):
        x, y = pa[k], pb[k]
        if isinstance(x, str) and isinstance(y, str) and x[:4].isdigit() and y[:4].isdigit():
            try:
                fx = datetime.date.fromisoformat(x if len(x) == 10 else x + "-01")
                fy = datetime.date.fromisoformat(y if len(y) == 10 else y + "-01")
                out += abs((fx - fy).days)
            except ValueError:
                pass
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            out += abs(x - y)
    return out


def same_shape(a, b):
    pa, pb = a["meta"]["params"], b["meta"]["params"]
    return [k["kind"] for k in a["keys"]] == [k["kind"] for k in b["keys"]] and set(pa) == set(pb) and \
        pa.get("op") == pb.get("op")


def key_differs(ka, kb):
    """Whether two reference keys of the same kind differ by more than ka's tolerance."""
    if ka["kind"] == "number":
        tol = ka.get("tol", abs(ka["value"]) * ka.get("rel_tol", 0))
        return abs(float(ka["value"]) - float(kb["value"])) > tol
    if ka["kind"] in ("ticker", "text"):
        names = {str(x).lower() for x in [ka["value"], *ka.get("aliases", [])]}
        return str(kb["value"]).lower() not in names
    return json.dumps(ka["value"], sort_keys=True) != json.dumps(kb["value"], sort_keys=True)


def pick_c1(tasks, tier, eps, per_tpl, rng, used, grade_keys):
    """Transplant pairs per template: (target task T, passed teacher episode E of another task of the same template
    and tier). A pair needs the same answer shape, a reference value that differs beyond tolerance, and E's own
    answer failing T's keys. Pairs that differ in the fewest parameter groups come first, and within a group the
    closest pairs (distance()), so that the mismatch is as small as the tasks allow; the differing group rotates so
    that one template's items do not all differ in the same way. A task enters C1 once, as T or as E's task."""
    by_tpl = collections.defaultdict(list)
    for t in tasks.values():
        if t["task_id"] in tier:
            by_tpl[(tier[t["task_id"]], t["template"])].append(t)
    out = []
    for (tr, tpl), ts in sorted(by_tpl.items()):
        mine = [(run, k) for run, k in eps if tier.get(k[0]) == tr and tasks[k[0]]["template"] == tpl
                and (run, k) not in used]
        cands = collections.defaultdict(list)
        for run, k in mine:
            e_task = tasks[k[0]]
            for t in ts:
                if t["task_id"] == k[0] or not same_shape(t, e_task):
                    continue
                d = differing(t, e_task)
                if d and any(key_differs(a, b) for a, b in zip(t["keys"], e_task["keys"])):
                    cands[len(d)].append((t, run, k, d))
        n = per_tpl[tr]
        got, t_used = [], set()
        for size in sorted(cands):
            pool = cands[size]
            rng.shuffle(pool)
            pool.sort(key=lambda c: -distance(c[0], tasks[c[2][0]], c[3]))  # stable: random among equal distances
            by_group = collections.defaultdict(list)
            for c in pool:
                by_group["+".join(sorted(c[3]))].append(c)
            groups = sorted(by_group)
            rng.shuffle(groups)
            while len(got) < n and any(by_group.values()):
                for g in groups:
                    while by_group[g] and len(got) < n:
                        t, run, k, d = by_group[g].pop()
                        if t["task_id"] in t_used or k[0] in t_used or (run, k) in used:
                            continue
                        if grade_keys(t["keys"], eps[(run, k)][0].get("answer"))["passed"]:
                            continue
                        got.append((t, run, k, d))
                        t_used |= {t["task_id"], k[0]}
                        used.add((run, k))
                        break
            if len(got) >= n:
                break
        out += [(tr, tpl) + g for g in got]
    return out


# -- assembling the set ---------------------------------------------------------------------------------------------

def traj_of(keys_by_run):
    """{(run name, (task_id, run)): messages} of the wanted episodes, the last trajectory of each key."""
    out = {}
    for name, keys in keys_by_run.items():
        for t in jsonl(f"{MK}/runs/{name}/trajectories.jsonl"):
            k = (t["task_id"], t["run"])
            if k in keys:
                out[(name, k)] = t["messages"]
    return out


def render(task, msgs, cut):
    return [{"role": "system", "content": judge_sft.SYSTEM},
            {"role": "user", "content": judge_sft.transcript(task, msgs, cut)}]


def main(a):
    rng = random.Random(a.seed)
    cut = judge_sft.parse(["--source", "x:y:z", "--out", "x", "--base-url", "x", "--model", "x"]).result_chars
    sys.path.insert(0, V3)
    from harness.envs.market import grade_keys  # the harness's own grading of a market answer
    tasks, tier = load_tasks()
    runs = load_runs(tasks, a.cache)
    snap = Snap()
    nd = {}
    flagged, cands, stats = {}, collections.defaultdict(list), collections.Counter()
    for name, eps in runs.items():
        for k, (r, s) in eps.items():
            if k[0] not in tier:
                continue
            t = tasks[k[0]]
            nd.setdefault(k[0], needs(t, snap))
            f, minor, skipped = findings(t, s, snap, nd[k[0]])
            for x in skipped:
                stats[f"unchecked {x}"] += 1
            for rule in dict.fromkeys(x["rule"] for x in minor):
                stats[f"{rule} immaterial, {'passed' if r['passed'] else 'failed'}"] += 1
            if not f:
                continue
            flagged[(name, k)] = f
            eligible = r["end"] == "finished" and s["finishes"] >= 1 and s["longest_args"] <= CUT
            outcome = "passed" if r["passed"] else "failed"
            for rule in dict.fromkeys(x["rule"] for x in f):
                stats[f"{rule} {'eligible' if eligible else 'not eligible'}, {outcome}"] += 1
            if eligible:  # one item per episode, under its rarest rule
                cands[min((x["rule"] for x in f), key=PRIMARY.index)].append((name, k))
    per = {k: tuple(int(x) for x in getattr(a, k).split(":")) for k in ("c0_per_template", "c1_per_template")}
    teacher = {(name, k): v for name, eps in runs.items() if RUNS[name] == "teacher" for k, v in eps.items()
               if k[0] in tier and v[0]["passed"] and (name, k) not in flagged}

    # C0: a seeded draw per template among the unflagged passed teacher episodes, never reading a verdict
    c0, used = [], set()
    by_tpl = collections.defaultdict(list)
    for key in sorted(teacher):
        by_tpl[(tier[key[1][0]], tasks[key[1][0]]["template"])].append(key)
    for (tr, tpl), keys in sorted(by_tpl.items()):
        n = per["c0_per_template"][0 if tr == "tier1" else 1]
        for key in rng.sample(keys, min(n, len(keys))):
            c0.append((tr, tpl, key))
            used.add(key)

    # C1
    c1 = pick_c1(tasks, tier, teacher, {"tier1": per["c1_per_template"][0], "hard": per["c1_per_template"][1]},
                 rng, used, grade_keys)

    # C2: an equal share per rule (what a rule cannot fill goes to the others); within a rule the passed episodes
    # first, since stage 8 only ever judges passed ones, each rotating over templates, and one item per task
    rules = [r for r in PRIMARY if cands.get(r)]
    room = {r: len({x[1][0] for x in cands[r]}) for r in rules}
    quota = {r: 0 for r in rules}
    left = a.c2_total
    while left > 0 and any(quota[r] < room[r] for r in rules):
        for r in rules:
            if left > 0 and quota[r] < room[r]:
                quota[r] += 1
                left -= 1
    c2, c2_tasks = [], set()
    for r in rules:
        picked = []
        for want_passed in (True, False):
            rest = collections.defaultdict(list)
            for x in sorted(cands[r]):
                if runs[x[0]][x[1]][0]["passed"] == want_passed:
                    rest[tasks[x[1][0]]["template"]].append(x)
            tpls = sorted(rest)
            rng.shuffle(tpls)
            for tp in tpls:
                rng.shuffle(rest[tp])
            while len(picked) < quota[r] and any(rest.values()):
                for tp in tpls:
                    while rest[tp] and len(picked) < quota[r]:
                        x = rest[tp].pop()
                        if x[1][0] not in c2_tasks:  # one item per task, whichever run it came from
                            picked.append(x)
                            c2_tasks.add(x[1][0])
                            break
        c2 += [(r, x) for x in picked]
    for r in rules:  # a rule left short by tasks another rule took first: fill from the rules with room
        for x in sorted(cands[r], key=lambda x: (not runs[x[0]][x[1]][0]["passed"], rng.random())):
            if len(c2) < a.c2_total and x[1][0] not in c2_tasks:
                c2.append((r, x))
                c2_tasks.add(x[1][0])

    want = collections.defaultdict(set)
    for _, _, (name, k) in c0:
        want[name].add(k)
    for _, _, _, name, k, _ in c1:
        want[name].add(k)
    for _, (name, k) in c2:
        want[name].add(k)
    msgs = traj_of(want)

    items = []

    def src(name, k):
        r = runs[name][k][0]
        return {"run_dir": name, "task_id": k[0], "run": k[1], "model": r["model"], "passed": r["passed"],
                "end": r["end"]}

    for i, (tr, tpl, (name, k)) in enumerate(c0):
        items.append({"id": f"C0-{i:03d}", "category": "C0", "template": tpl, "tier": tr, "label": "ok",
                      "true_label": "presumably ok", "source": src(name, k), "question_task": k[0],
                      "messages": render(tasks[k[0]], msgs[(name, k)], cut)})
    for i, (tr, tpl, t, name, k, d) in enumerate(c1):
        pe, pt = tasks[k[0]]["meta"]["params"], t["meta"]["params"]
        nc = differing_not_compared(t, tasks[k[0]])
        items.append({"id": f"C1-{i:03d}", "category": "C1", "template": tpl, "tier": tr, "label": "flawed",
                      "differing": sorted(d), "n_differing": len(d), "distance": distance(t, tasks[k[0]], d),
                      **({"differing_not_compared": nc} if nc else {}),
                      "evidence": {g: {p: {"question": pt[p], "episode": pe[p]} for p in ks} for g, ks in d.items()},
                      "source": src(name, k), "question_task": t["task_id"],
                      "messages": render(t, msgs[(name, k)], cut)})
    for i, (rule, (name, k)) in enumerate(c2):
        t = tasks[k[0]]
        items.append({"id": f"C2-{i:03d}", "category": "C2", "template": t["template"], "tier": tier[k[0]],
                      "label": "flawed", "rule": rule, "evidence": [x for x in flagged[(name, k)] if x["rule"] == rule],
                      "other_findings": [x for x in flagged[(name, k)] if x["rule"] != rule],
                      "source": src(name, k), "question_task": k[0],
                      "messages": render(t, msgs[(name, k)], cut)})
    for it in items:
        it["user_sha256"] = hashlib.sha256(it["messages"][1]["content"].encode()).hexdigest()

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "judge_cal_set.jsonl")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    os.replace(path + ".tmp", path)
    count = lambda xs, key: dict(sorted(collections.Counter(key(x) for x in xs).items()))  # noqa: E731
    manifest = {
        "log": "research/log/2026-11-01-0010.md, 11-02 06:0x UTC",
        "args": {k: v for k, v in vars(a).items() if k != "cache"}, "result_chars": cut, "call_args_cut": CUT,
        "counts": {"total": len(items), "by_category": count(items, lambda x: x["category"]),
                   "C0_by_template": count([x for x in items if x["category"] == "C0"], lambda x: x["template"]),
                   "C1_by_template": count([x for x in items if x["category"] == "C1"], lambda x: x["template"]),
                   "C1_by_n_differing": count([x for x in items if x["category"] == "C1"], lambda x: x["n_differing"]),
                   "C1_by_differing": count([x for x in items if x["category"] == "C1"],
                                            lambda x: "+".join(x["differing"])),
                   "C1_by_differing_not_compared": count(
                       [x for x in items if x.get("differing_not_compared")],
                       lambda x: "+".join(x["differing_not_compared"])),
                   "C2_by_rule": count([x for x in items if x["category"] == "C2"], lambda x: x["rule"]),
                   "C2_by_template": count([x for x in items if x["category"] == "C2"], lambda x: x["template"]),
                   "C2_by_run": count([x for x in items if x["category"] == "C2"], lambda x: x["source"]["run_dir"]),
                   "C2_passed": count([x for x in items if x["category"] == "C2"], lambda x: x["source"]["passed"])},
        "rule_candidates": dict(sorted(stats.items())),
        "rule_constants": {"MIN_MISSING": MIN_MISSING, "MATERIAL": MATERIAL, "CPI_MATERIAL": CPI_MATERIAL},
        "inputs_sha256": {**{k: sha256(p) for k, p in TASK_FILES.items()},
                          **{f"{n}/results.jsonl": sha256(f"{MK}/runs/{n}/results.jsonl") for n in RUNS},
                          **{f"{n}/trajectories.jsonl": sha256(f"{MK}/runs/{n}/trajectories.jsonl") for n in RUNS},
                          "judge_sft.py": sha256(os.path.join(HERE, "judge_sft.py"))},
        "set_sha256": sha256(path)}
    with open(os.path.join(a.out, "judge_cal_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    print(json.dumps(manifest["counts"], indent=1))


if __name__ == "__main__":
    main(parse())
