"""Reference trajectories for the wealth tasks gpt-oss-120b cannot solve (research/log/2026-10-29-1725.md).

A scripted policy per template stands in for the model: it calls the agent's own tools one step at a time, reads their
results and decides the next call from them, as the reference solution in env/tasks.py does from the database. It runs
inside agent.Episode through Scripted, an OpenAI-compatible client, so a reference trajectory has the same system prompt,
tool results, context management, checklist and finish verification as a sampled one, and the same grader scores it.
Every value in the answer comes from a tool result: arithmetic goes through the calculator, and the minimum initial
subscription through the knowledge base. Each step carries a one-line reasoning_content saying why the call is made.

A policy that meets a result it does not expect (a rejected write, a rule check that disagrees with the task) raises,
and the task gets no reference trajectory. The test-only templates have no policy.

Version 3 (research/log/2026-10-31-0150.md): the policies are version 2's, step for step; they run inside the v3
loop (append-only context, the checklist turns outside the step count) with the Qwen provider's message format by
default, so each step's reasoning is kept in its assistant message as a sampled Qwen trajectory has it.

The market kind (research/log/2026-10-31-0150.md, W7) has one policy for every task, in-house and external
(harness/market_reference.py): one run_python call whose code computes the answer from the snapshot with the
sandbox's data object, then finish with the printed values in the prompt's units and rounding; under H1 a checklist
first and a done mark before finish, so that its answer must pass H1's verification. --concurrency runs that many
episodes at once (run_python runs in worker threads, so they overlap).

Usage: python -B golden.py --tasks <tasks.jsonl> --db <instance.sqlite> --out <dir> [--config H1] [--limit N]
       [--max-steps 100] [--provider qwen|vllm] [--tool-result-chars 8000] [--concurrency 1]
       (writes <out>/results.jsonl and <out>/trajectories.jsonl as run.py does, model "golden")
"""
import argparse
import asyncio
import datetime
import json
import os
import re
from types import SimpleNamespace

from harness import agent
from harness.envs.wealth import core

DEC = json.JSONDecoder()
CUST, PROD, DATE = re.compile(r"\b[ABC]-C\d{4}\b"), re.compile(r"\b[ABC]-P\d{4}\b"), re.compile(r"\d{4}-\d{2}-\d{2}")
MINIMUM = re.compile(r"\$([\d,]+)")  # the dealing terms state one amount, the minimum, in one of three wordings


class Scripted:
    """An OpenAI-compatible client whose replies come from a policy generator: the generator yields
    (reasoning, [(tool, arguments), ...]) and is sent the parsed results of those calls."""

    def __init__(self, policy):
        self.policy, self.started, self.n, self.error = policy, False, 0, None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, messages, **_):
        tail = []
        for m in reversed(messages):
            if m["role"] != "tool":
                break
            tail.append(DEC.raw_decode(m["content"])[0])
        try:
            thought, calls = self.policy.send(tail[::-1]) if self.started else next(self.policy)
        except Exception as e:  # agent.Episode reads any client error as a server error; keep the first one
            self.error = self.error or f"{type(e).__name__}: {e}"[:300]
            raise
        self.started, self.n = True, self.n + 1
        tcs = [SimpleNamespace(id=f"golden-{self.n}-{i}", type="function",
                               function=SimpleNamespace(name=name, arguments=json.dumps(args)))
               for i, (name, args) in enumerate(calls)]
        msg = SimpleNamespace(content="", tool_calls=tcs or None, reasoning_content=thought)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def act(thought, *calls):
    return (yield thought, list(calls))


def act_many(thought, calls, size=8):
    """Many calls in turns of at most size (version 2 chose 8 so that its context management kept every result of a
    turn whole; kept, so that the trajectories keep version 2's turns)."""
    out = []
    for i in range(0, len(calls), size):
        out += yield from act(thought, *calls[i:i + size])
    return out


def one(thought, name, **args):
    (r,) = yield from act(thought, (name, args))
    return r


def money(x):
    return f"{'-' if x < 0 else ''}${abs(x):,.2f}"


def calc(thought, exprs):
    r = yield from one(thought, "calculator", expressions=exprs)
    assert all("result" in x for x in r["results"]), r
    return [x["result"] for x in r["results"]]


def holdings(c):
    items, page = [], 1
    while True:
        r = yield from one(f"List {c}'s holdings" + (f" (page {page})." if page > 1 else "."), "list_holdings",
                           customer_id=c, page=page)
        items += r["items"]
        if page >= r["pages"]:
            return items
        page += 1


def funds(ids, why):
    out = {}
    for i in range(0, len(ids), core.MAX_FUNDS):
        r = yield from one(why, "get_fund", product_ids=ids[i:i + core.MAX_FUNDS])
        out |= {f["product_id"]: f for f in r["products"]}
    return out


def minimum(p):
    """The minimum initial subscription from the prospectus: a search, then the passage that states it."""
    yield from one(f"Rule 7: a first subscription to {p} needs its prospectus minimum; search the knowledge base.",
                   "search_knowledge", query=f"{p} minimum initial subscription")
    d = yield from one(f"Read the dealing terms of {p}.", "get_document", doc_id=f"{p}-2")
    return float(MINIMUM.search(d["text"]).group(1).replace(",", ""))


def checklist(items):
    yield from one("Write the steps of the request as a checklist.", "set_checklist", items=items)


def done(items):
    yield from one("Every step is done; mark the checklist.", "set_checklist", items=items,
                   done=list(range(1, len(items) + 1)))


def rate_of(h):
    days = (core.TODAY - datetime.date.fromisoformat(h["purchase_date"])).days
    return core.redemption_fee_rate(h["purchase_date"]), days


def total_expr(cash, hs, f):
    return " + ".join([str(cash)] + [f"{h['units']}*{f[h['product_id']]['nav']}" for h in hs])


# policies: each takes the task and the tools offered, yields steps and returns the final answer -----------------------

def p_value_pnl(t):
    c = CUST.findall(t["prompt"])[0]
    hs = yield from holdings(c)
    f = yield from funds([h["product_id"] for h in hs], "Today's NAV of each holding.")
    mv = " + ".join(f"{h['units']}*{f[h['product_id']]['nav']}" for h in hs)
    cost = " + ".join(str(h["cost_basis"]) for h in hs)
    v, g = yield from calc("Market value is units times NAV; the gain is market value minus cost basis.",
                           [f"round({mv}, 2)", f"round({mv} - ({cost}), 2)"])
    return f"{c}'s holdings have a total market value of {money(v)} at today's NAV, and a total unrealized gain or loss of {money(g)}."


def p_period_return(t):
    p, (d1, d2) = PROD.findall(t["prompt"])[0], DATE.findall(t["prompt"])[:2]
    a, b = yield from act(f"The NAVs of {p} on {d1} and {d2}.",
                          ("get_nav_history", {"product_id": p, "start_date": d1, "end_date": d1}),
                          ("get_nav_history", {"product_id": p, "start_date": d2, "end_date": d2}))
    n1, n2 = a["items"][0]["nav"], b["items"][0]["nav"]
    (r,) = yield from calc("Return in percent: end NAV over start NAV, minus 1.", [f"round(({n2}/{n1} - 1)*100, 4)"])
    return f"The return of {p} from {d1} to {d2} was {r:.2f}% (NAV {n1} to {n2})."


def p_prospectus(t):
    p = PROD.findall(t["prompt"])[0]
    yield from one(f"The prospectus of {p} is in the knowledge base.", "search_knowledge",
                   query=f"{p} minimum initial subscription settlement")
    d = yield from one(f"Read the dealing terms of {p}.", "get_document", doc_id=f"{p}-2")
    m = MINIMUM.search(d["text"]).group(1)
    s = re.search(r"T\+(\d+) business days", d["text"]).group(1)
    return f"According to its prospectus, the minimum initial subscription for {p} is ${m}, and redemption proceeds are paid T+{s}, that is {s} business days after the order."


def p_fee_quote(t):
    c, p = CUST.findall(t["prompt"])[0], PROD.findall(t["prompt"])[0]
    hs = yield from holdings(c)
    h = next(x for x in hs if x["product_id"] == p)
    f = yield from funds([p], f"Today's NAV of {p}.")
    rate, days = rate_of(h)
    g = f"{h['units']}*{f[p]['nav']}"
    fee, net = yield from calc(f"Held since {h['purchase_date']} ({days} days), so rule 5 gives a {rate:.1%} fee.",
                               [f"round({g}*{rate}, 2)", f"round({g} - round({g}*{rate}, 2), 2)"])
    return f"This is a quote only; nothing was redeemed. Redeeming all {h['units']} units of {p} today would charge a redemption fee of {money(fee)}, and {c} would receive {money(net)}."


def p_redeem_all(t):
    c, p = CUST.findall(t["prompt"])[0], PROD.findall(t["prompt"])[0]
    hs = yield from holdings(c)
    h = next(x for x in hs if x["product_id"] == p)
    f = yield from funds([p], f"Check whether {p} is open for redemption.")
    if f[p]["redemption_suspended"]:
        assert t["violated_rules"] == [4]
        return f"I made no trade: redemption of {p} is suspended, and rule 4 does not allow redeeming a product whose redemption is suspended."
    r = yield from one(f"{p} is open; redeem all {h['units']} units.", "redeem", customer_id=c, product_id=p, units=h["units"])
    assert r["status"] == "executed", r
    return f"Redeemed all {h['units']} units of {p} for {c}. After a redemption fee of {money(r['fee'])}, the customer receives {money(r['proceeds'])}."


def p_drawdown_fee(t):
    c, (d1, d2) = CUST.findall(t["prompt"])[0], DATE.findall(t["prompt"])[:2]
    hs = yield from holdings(c)
    ms = yield from act_many(f"Maximum drawdown of each holding from {d1} to {d2}.",
                             [("fund_metrics", {"product_id": h["product_id"], "start_date": d1, "end_date": d2}) for h in hs])
    i = max(range(len(hs)), key=lambda k: ms[k]["max_drawdown_pct"])
    h, d = hs[i], ms[i]["max_drawdown_pct"]
    p = h["product_id"]
    f = yield from funds([p], f"{p} has the largest drawdown; its NAV today.")
    rate, days = rate_of(h)
    (fee,) = yield from calc(f"Held {days} days, so the fee rate is {rate:.1%}; the fee on all units.",
                             [f"round({h['units']}*{f[p]['nav']}*{rate}, 2)"])
    return f"{p} had the largest maximum drawdown among {c}'s funds from {d1} to {d2}: {d:.2f}%. Redeeming all of it today would cost a redemption fee of {money(fee)}. No trade was made."


def cheapest_bond(c, level):
    r = yield from one(f"{c} may buy up to risk R{level}; list open bond funds by management fee.", "search_funds",
                       type="bond", max_risk_level=f"R{level}", open_for_subscription=True, sort_by="management_fee")
    return r["items"][0]["product_id"]


def p_cheapest_bond(t):
    c, a = CUST.findall(t["prompt"])[0], float(re.search(r"\$([\d,]+)", t["prompt"]).group(1).replace(",", ""))
    cu = yield from one(f"{c}'s risk tolerance.", "get_customer", customer_id=c)
    b = yield from cheapest_bond(c, int(cu["risk_tolerance"][1:]))
    f = yield from funds([b], f"Subscription fee and NAV of {b}.")
    sf, nav = f[b]["subscription_fee"], f[b]["nav"]
    fee, u = yield from calc("The fee is charged on the amount; the rest buys units at today's NAV.",
                             [f"round({a}*{sf}, 2)", f"round(({a} - round({a}*{sf}, 2))/{nav}, 4)"])
    return f"The open bond fund with the lowest management fee that {c} may buy is {b}. On ${a:,.0f} the subscription fee would be {money(fee)}, and the rest would buy {u:,.4f} units at today's NAV. Nothing was bought."


def p_redeem_worst(t):
    c, (d1, d2) = CUST.findall(t["prompt"])[0], DATE.findall(t["prompt"])[:2]
    hs = yield from holdings(c)
    ms = yield from act_many(f"Return of each holding from {d1} to {d2}.",
                             [("fund_metrics", {"product_id": h["product_id"], "start_date": d1, "end_date": d2}) for h in hs])
    i = min(range(len(hs)), key=lambda k: ms[k]["return_pct"])
    h, p = hs[i], hs[i]["product_id"]
    f = yield from funds([p], f"{p} has the lowest return ({ms[i]['return_pct']}%); check it is open for redemption.")
    if f[p]["redemption_suspended"]:
        assert t["violated_rules"] == [4]
        return f"The fund with the lowest return is {p} ({ms[i]['return_pct']:.2f}%), but its redemption is suspended, so rule 4 does not allow redeeming it. I made no trade."
    r = yield from one(f"Redeem all {h['units']} units of {p}.", "redeem", customer_id=c, product_id=p, units=h["units"])
    assert r["status"] == "executed", r
    return f"The fund with the lowest return is {p} ({ms[i]['return_pct']:.2f}%). I redeemed all {h['units']} units; after a fee of {money(r['fee'])}, {c} receives {money(r['proceeds'])}."


def subscription_rules(cu, hs, f, p, a, held, minimum_, base_total=None, cash=None):
    """Rules 1-4 and 7 for subscribing a dollars of p, as env.core.check reads them, from tool results and the
    calculator; returns (broken rules, share of p after, cash share after)."""
    cash = cu["cash"] if cash is None else cash
    fp = f[p]
    total = base_total if base_total is not None else total_expr(cash, hs, f)
    fee = f"{a}*{fp['subscription_fee']}"
    after = f"({total}) - {fee}"
    value = f"{held}*{fp['nav']} + ({a} - {fee})"
    share, cash_share = yield from calc(
        f"After buying ${a:,.0f} of {p}: its share of total assets (rule 2) and the cash share (rule 3), in percent.",
        [f"round(({value})/({after})*100, 4)", f"round(({cash} - {a})/({after})*100, 4)"])
    broken = []
    if int(fp["risk_level"][1:]) > int(cu["risk_tolerance"][1:]):
        broken.append(1)
    if fp["subscription_suspended"]:
        broken.append(4)
    if share > 30 + 1e-4:
        broken.append(2)
    if cash_share < 5 - 1e-4:
        broken.append(3)
    if not held and a < minimum_ - 1e-9:
        broken.append(7)
    return sorted(broken), share, cash_share


WHY = {1: "the product's risk level is above the customer's risk tolerance (rule 1)",
       2: "the product would be more than 30% of total assets (rule 2)",
       3: "available cash would fall below 5% of total assets (rule 3)",
       4: "subscription to the product is suspended (rule 4)",
       7: "the amount is below the prospectus minimum initial subscription (rule 7)"}


def p_subscribe(t):
    c, p = CUST.findall(t["prompt"])[0], PROD.findall(t["prompt"])[0]
    a = float(re.search(r"\$([\d,]+)", t["prompt"]).group(1).replace(",", ""))
    cu = yield from one(f"{c}'s risk tolerance and cash.", "get_customer", customer_id=c)
    hs = yield from holdings(c)
    f = yield from funds(sorted({h["product_id"] for h in hs} | {p}), "NAVs of the holdings and the terms of the product.")
    held = next((h["units"] for h in hs if h["product_id"] == p), 0)
    m = 0 if held else (yield from minimum(p))
    broken, _, _ = yield from subscription_rules(cu, hs, f, p, a, held, m)
    assert broken == sorted(t["violated_rules"]), (broken, t["violated_rules"])
    if broken:
        return f"I placed no order for {c}: " + "; ".join(WHY[r] for r in broken) + "."
    r = yield from one("Every rule allows it; place the order.", "subscribe", customer_id=c, product_id=p, amount=a)
    assert r["status"] == "executed", r
    return f"Subscribed ${a:,.0f} of {p} for {c}: the subscription fee was {money(r['fee'])} and {r['units']:,.4f} units were bought at NAV {r['nav']}."


def p_switch(t):
    c, p = CUST.findall(t["prompt"])[0], PROD.findall(t["prompt"])[0]
    a = float(re.search(r"\$([\d,]+)", t["prompt"]).group(1).replace(",", ""))
    cu = yield from one(f"{c}'s risk tolerance and cash.", "get_customer", customer_id=c)
    hs = yield from holdings(c)
    b = yield from cheapest_bond(c, int(cu["risk_tolerance"][1:]))
    f = yield from funds(sorted({h["product_id"] for h in hs} | {b}), f"NAVs of the holdings and the terms of {b}.")
    h = next(x for x in hs if x["product_id"] == p)
    rate, days = rate_of(h)
    g = f"{h['units']}*{f[p]['nav']}"
    proceeds, total = yield from calc(
        f"Redeeming {p} (held {days} days, fee {rate:.1%}) first: the proceeds, and total assets after it, since the fee leaves.",
        [f"round({g} - round({g}*{rate}, 2), 2)", f"({total_expr(cu['cash'], hs, f)}) - round({g}*{rate}, 2)"])
    held = next((x["units"] for x in hs if x["product_id"] == b), 0)
    m = 0 if held else (yield from minimum(b))
    broken, _, _ = yield from subscription_rules(cu, hs, f, b, a, held, m, base_total=total, cash=cu["cash"] + proceeds)
    assert broken == sorted(t["violated_rules"]), (broken, t["violated_rules"])
    if broken:
        return (f"I made no trade for {c}: after redeeming {p}, investing ${a:,.0f} in {b}, the cheapest eligible bond fund, "
                f"would break the rules: " + "; ".join(WHY[r] for r in broken) + ". So neither the redemption nor the subscription was placed.")
    r1 = yield from one(f"The switch keeps every rule; redeem all of {p} first.", "redeem", customer_id=c, product_id=p,
                        units=h["units"])
    r2 = yield from one(f"Now buy ${a:,.0f} of {b}.", "subscribe", customer_id=c, product_id=b, amount=a)
    assert r1["status"] == r2["status"] == "executed", (r1, r2)
    return f"Redeemed all {h['units']} units of {p} for {c}, with proceeds of {money(r1['proceeds'])}; then bought {r2['units']:,.4f} units of {b}, the open bond fund with the lowest management fee the customer may buy, for ${a:,.0f} (fee {money(r2['fee'])})."


def p_redeem_units(t):
    c, p = CUST.findall(t["prompt"])[0], PROD.findall(t["prompt"])[0]
    u = float(re.search(r"Redeem ([\d,]+\.\d+) units", t["prompt"]).group(1).replace(",", ""))
    f = yield from funds([p], f"Check whether {p} is open for redemption.")
    if f[p]["redemption_suspended"]:
        assert t["violated_rules"] == [4]
        return f"I made no trade: redemption of {p} is suspended, so rule 4 does not allow it."
    r = yield from one(f"{p} is open; redeem {u} units.", "redeem", customer_id=c, product_id=p, units=u)
    assert r["status"] == "executed", r
    return f"Redeemed {u:,.2f} units of {p} for {c}. The redemption fee was {money(r['fee'])}, and the customer receives {money(r['proceeds'])}."


def rebalance_one(c):
    """One customer of a concentration review; returns the line of the answer."""
    cu = yield from one(f"{c}'s cash.", "get_customer", customer_id=c)
    hs = yield from holdings(c)
    f = yield from funds([h["product_id"] for h in hs], f"Today's NAVs of {c}'s holdings.")
    total = total_expr(cu["cash"], hs, f)
    vals = yield from calc(f"{c}'s total assets, and each holding's share of them in percent.",
                           [f"round({total}, 2)"] + [f"round({h['units']}*{f[h['product_id']]['nav']}/({total})*100, 4)" for h in hs])
    over = [h for h, s in zip(hs, vals[1:]) if s > 30]
    if not over:
        return f"{c}: no holding is above 30% of total assets, so no units were redeemed; cash after the review is {money(cu['cash'])}."
    h = over[0]
    p = h["product_id"]
    if f[p]["redemption_suspended"]:
        return f"{c}: {p} is above 30% of total assets, but its redemption is suspended (rule 4), so it was skipped; no units were redeemed and cash stays {money(cu['cash'])}."
    rate, days = rate_of(h)
    (u,) = yield from calc(f"{p} is above 30%; held {days} days, so f = {rate}. Units to redeem by the request's formula.",
                           [f"round(({h['units']}*{f[p]['nav']} - 0.25*({total}))/({f[p]['nav']}*(1 - 0.25*{rate})), 2)"])
    r = yield from one(f"Redeem {u} units of {p}.", "redeem", customer_id=c, product_id=p, units=u)
    assert r["status"] == "executed", r
    (cash,) = yield from calc("Cash after the review: cash plus the redemption proceeds.", [f"round({cu['cash']} + {r['proceeds']}, 2)"])
    return f"{c}: redeemed {u:,.2f} units of {p}; cash after the review is {money(cash)}."


def p_rebalance(t):
    lines = []
    for c in t["customers"]:
        lines.append((yield from rebalance_one(c)))
    return "Concentration review:\n" + "\n".join(lines)


def p_review_switch(t):
    """t5_review_switch (research/log/2026-10-30-0855.md): redeem the worst holding, then subscribe the requested amount
    of the cheapest eligible bond fund, or the largest whole $100 that rules 2 and 3 allow, or nothing under rule 7."""
    c, (d1, d2) = CUST.findall(t["prompt"])[0], DATE.findall(t["prompt"])[:2]
    a = float(re.search(r"invest \$([\d,]+)", t["prompt"]).group(1).replace(",", ""))
    cu = yield from one(f"{c}'s risk tolerance and cash.", "get_customer", customer_id=c)
    hs = yield from holdings(c)
    ms = yield from act_many(f"Return of each holding from {d1} to {d2}.",
                             [("fund_metrics", {"product_id": h["product_id"], "start_date": d1, "end_date": d2}) for h in hs])
    i = min(range(len(hs)), key=lambda k: ms[k]["return_pct"])
    h, p = hs[i], hs[i]["product_id"]
    b = yield from cheapest_bond(c, int(cu["risk_tolerance"][1:]))
    f = yield from funds(sorted({x["product_id"] for x in hs} | {b}), f"NAVs of the holdings and the terms of {b}.")
    rate, days = rate_of(h)
    g = f"{h['units']}*{f[p]['nav']}"
    (total,) = yield from calc(f"{p} has the lowest return ({ms[i]['return_pct']}%). Total assets after redeeming it, since its "
                               f"fee ({rate:.1%}, held {days} days) leaves.",
                               [f"({total_expr(cu['cash'], hs, f)}) - round({g}*{rate}, 2)"])
    r1 = yield from one(f"Redeem all {h['units']} units of {p}.", "redeem", customer_id=c, product_id=p, units=h["units"])
    assert r1["status"] == "executed", r1
    cash = round(cu["cash"] + r1["proceeds"], 2)
    held = next((x["units"] for x in hs if x["product_id"] == b), 0)
    m = 0 if held else (yield from minimum(b))
    sf, nav = f[b]["subscription_fee"], f[b]["nav"]
    broken, _, _ = yield from subscription_rules(cu, hs, f, b, a, held, m, base_total=total, cash=cash)
    amount = a
    if broken:
        # the largest x with value <= 30% of assets after the fee (rule 2) and cash >= 5% of them (rule 3)
        b2, b3 = yield from calc("The rules do not allow the amount asked; the largest amounts rules 2 and 3 allow.",
                                 [f"(0.3*({total}) - {held}*{nav})/(1 - 0.7*{sf})", f"({cash} - 0.05*({total}))/(1 - 0.05*{sf})"])
        amount = max(0, int(min(b2, b3) // 100) * 100)
        while amount > 0 and (yield from subscription_rules(cu, hs, f, b, amount, held, 0, base_total=total, cash=cash))[0]:
            amount -= 100  # the bound is exact up to rounding; step down if the check disagrees
        if amount < m or amount <= 0:
            why = (yield from subscription_rules(cu, hs, f, b, max(amount, 100), held, m, base_total=total, cash=cash))[0]
            assert why == sorted(t["violated_rules"]), (why, t["violated_rules"])
            return (f"The fund {c} holds with the lowest return from {d1} to {d2} is {p} ({ms[i]['return_pct']:.2f}%); I redeemed "
                    f"all {h['units']} units, which returned {money(r1['proceeds'])}. No subscription of {b}, the open bond fund with "
                    f"the lowest management fee the customer may buy, is allowed: " + "; ".join(WHY[r] for r in why) +
                    f". So I made no subscription; {c}'s cash at the end is {money(cash)}.")
    r2 = yield from one(f"Subscribe ${amount:,.0f} of {b}.", "subscribe", customer_id=c, product_id=b, amount=amount)
    assert r2["status"] == "executed", r2
    (end,) = yield from calc("Cash at the end.", [f"round({cash} - {amount}, 2)"])
    note = "" if amount == a else f" (the rules do not allow ${a:,.0f}; ${amount:,.0f} is the largest whole $100 they allow)"
    return (f"The fund {c} holds with the lowest return from {d1} to {d2} is {p} ({ms[i]['return_pct']:.2f}%); I redeemed all "
            f"{h['units']} units, which returned {money(r1['proceeds'])}. Then I invested ${amount:,.0f}{note} in {b}, the open bond "
            f"fund with the lowest management fee the customer may buy, buying {r2['units']:,.4f} units. {c}'s cash at the end is "
            f"{money(end)}.")


POLICIES = {"t1_value_pnl": p_value_pnl, "t1_period_return": p_period_return, "t1_prospectus": p_prospectus,
            "t1_fee_quote": p_fee_quote, "t1_redeem_all": p_redeem_all, "t2_drawdown_fee": p_drawdown_fee,
            "t2_cheapest_bond": p_cheapest_bond, "t2_redeem_worst": p_redeem_worst, "t3_subscribe": p_subscribe,
            "t3_switch": p_switch, "t3_redeem_units": p_redeem_units, "t4_rebalance": p_rebalance,
            "t4_rebalance_one": p_rebalance, "t5_review_switch": p_review_switch}


def market_episode(task, checklist_on):
    """A market task: one run_python call computing every asked value, then finish (harness/market_reference.py)."""
    from harness import market_reference
    items = ["compute the asked values from the database with run_python", "report them"]
    if checklist_on:
        yield from checklist(items)
    r = yield from one("Compute every value the question asks for from the database in one run_python call, and "
                       "print them.", "run_python", code=market_reference.code_of(task))
    if "stdout" not in r:
        raise AssertionError(f"run_python failed: {str(r.get('error', r))[-400:]}")
    answer = market_reference.answer_of(task, json.loads(r["stdout"].strip().splitlines()[-1]))
    if checklist_on:
        yield from done(items)
    yield from act("Report the printed values, rounded as the question asks.", ("finish", {"answer": answer}))
    raise AssertionError(f"finish was not accepted: {answer}")


def episode(task, checklist_on):
    """The whole scripted episode: a checklist under context management, the policy's steps, then finish."""
    items = ["gather the data the request needs", "check the business rules and compute the values",
             "act or decline, then report"]
    if checklist_on:
        yield from checklist(items)
    answer = yield from POLICIES[task["template"]](task)
    if checklist_on:
        yield from done(items)
    yield from act("Report the result.", ("finish", {"answer": answer}))
    raise AssertionError("finish was not accepted")


async def main(a):
    settings = agent.Settings(provider=a.provider, max_steps=a.max_steps, tool_result_chars=a.tool_result_chars,
                              token_records=False)
    tasks = [json.loads(line) for line in open(a.tasks, encoding="utf-8")][:a.limit]
    os.makedirs(a.out, exist_ok=True)
    res_f = open(os.path.join(a.out, "results.jsonl"), "w", encoding="utf-8")
    traj_f = open(os.path.join(a.out, "trajectories.jsonl"), "w", encoding="utf-8")
    on = agent.CONFIGS[a.config]["context"]
    sem = asyncio.Semaphore(a.concurrency)

    async def run(t):
        market = t.get("env") == "market"
        if not market and t["template"] not in POLICIES:
            return
        async with sem:
            client = Scripted(market_episode(t, on) if market else episode(t, on))
            row, traj = await agent.run_one(t, a.config, a.db, client, "golden", 1, settings)
        if client.error:
            row |= {"end": "policy_error", "failure": client.error}
        row["source"] = traj["source"] = "golden"
        res_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        traj_f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        res_f.flush()
        traj_f.flush()

    await asyncio.gather(*(run(t) for t in tasks))


def cli(argv=None):
    ap = argparse.ArgumentParser(prog="golden.py")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="H1", choices=list(agent.CONFIGS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--provider", choices=["vllm", "qwen"], default="qwen",
                    help="the message format: qwen keeps each step's reasoning in its message, vllm (gpt-oss) does not")
    ap.add_argument("--tool-result-chars", type=int, default=agent.Settings.tool_result_chars)
    ap.add_argument("--concurrency", type=int, default=1, help="episodes at once (1: in task order, as before)")
    asyncio.run(main(ap.parse_args(argv)))
