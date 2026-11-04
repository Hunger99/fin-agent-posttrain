"""Generate the wealth-management tasks of one split, with their gold outcome, per docs/03_harness.md section 3.

Each template draws entities from a database instance, states the request, and runs a reference solution with the
environment's own tools on a copy, so the gold state and key numbers carry exactly the rounding the tools apply.
A task records:
- prompt: the request, in English, amounts in US dollars;
- customers: the customers whose cash, holdings and trades the grader compares;
- gold_state: their state after the reference solution (the initial state when nothing may be executed);
- keys: values the final answer must contain, each a product id or a number with its kind (money within $0.01,
  percent within 0.1 percentage points, units within 0.01), per the harness task success rate in docs/results.md;
- feasible: false when the business rules forbid the request, in whole or for one customer; then violated_rules
  lists the rules, and the answer must name at least one of them while the forbidden trade is not made.

Four types, each about 15% infeasible: T1 query and compute (1-3 tool calls; t1_prospectus needs the knowledge base), T2 multi-hop analysis (3-8), T3 trades
under the rules (5-15), T4 long rebalancing over five customers (20-40). Two templates, t1_cash_share and
t2_concentration, appear only in the test split, so the test set holds request forms the harness was never tuned on.

Usage: python tasks.py <instance.sqlite> <split train|dev|test> <n tasks> <seed> <out.jsonl>
"""
import datetime
import json
import random
import sys

import core

YEAR_AGO = "2025-06-30"
TEST_ONLY = {"t1_cash_share", "t2_concentration"}
INFEASIBLE_SHARE = 0.15


def holdings(env, c):
    rows = env.db.execute("SELECT h.*, p.nav, p.redemption_suspended, p.subscription_suspended, p.type, p.risk_level, "
                          "p.mgmt_fee, p.subscription_fee FROM holdings h JOIN products p USING (product_id) "
                          "WHERE h.customer_id=? ORDER BY h.product_id", (c,)).fetchall()
    return [dict(r) | {"value": r["units"] * r["nav"]} for r in rows]


def customer(env, c):
    return dict(env.db.execute("SELECT * FROM customers WHERE customer_id=?", (c,)).fetchone())


def all_customers(env):
    return [r["customer_id"] for r in env.db.execute("SELECT customer_id FROM customers ORDER BY customer_id")]


def cheapest_bond(env, level, exclude=()):
    """The open bond fund the customer may buy with the lowest management fee, or None when the lowest is tied."""
    rows = [r for r in env.db.execute("SELECT * FROM products WHERE type='bond' AND risk_level<=? AND "
                                      "subscription_suspended=0 ORDER BY mgmt_fee, product_id", (level,))
            if r["product_id"] not in exclude]
    if len(rows) < 2 or rows[0]["mgmt_fee"] == rows[1]["mgmt_fee"]:
        return None
    return dict(rows[0])


def nav_on(env, p, date):
    return env.db.execute("SELECT nav FROM nav_history WHERE product_id=? AND date=?", (p, date)).fetchone()["nav"]


_METRICS = {}


def mdd(env, p, start=YEAR_AGO, end=core.TODAY.isoformat()):
    key = (id(env), p, start, end)
    if key not in _METRICS:
        navs = [r["nav"] for r in env.db.execute("SELECT nav FROM nav_history WHERE product_id=? AND date BETWEEN ? AND ? "
                                                  "ORDER BY date", (p, start, end))]
        _METRICS[key] = core.metrics(navs)
    return _METRICS[key]


_WORST = {}


def worst_holding(env, c, start=YEAR_AGO):
    """The holding with the lowest return since start, or None with fewer than 3 holdings or a near tie."""
    hs = holdings(env, c)
    if len(hs) < 3:
        return None
    scored = sorted(((mdd(env, h["product_id"], start)["return_pct"], h) for h in hs), key=lambda x: x[0])
    return None if scored[1][0] - scored[0][0] < 0.2 else scored[0][1]


PERIODS = [YEAR_AGO, "2025-12-31", "2026-03-31"]


def worst_candidates(env, suspended):
    """(customer, start) pairs whose lowest-return holding is (or is not) redemption-suspended; the suspended case is rare."""
    key = (id(env), suspended)
    if key not in _WORST:
        _WORST[key] = [(c, s) for s in PERIODS for c in all_customers(env)
                       if (h := worst_holding(env, c, s)) and bool(h["redemption_suspended"]) == suspended]
    return _WORST[key]


def money(x):
    return {"kind": "money", "value": round(x, 2)}


def pct(x):
    return {"kind": "percent", "value": round(x, 4)}


def units(x):
    return {"kind": "units", "value": round(x, 4)}


def pid(x):
    return {"kind": "id", "value": x}


# templates: each returns (prompt, customers, keys, violated rules, solve(env)) or None when the draw does not fit ----

def t1_value_pnl(env, rng, feasible):
    c = rng.choice(all_customers(env))
    hs = holdings(env, c)
    if len(hs) < 2:
        return None
    mv = sum(h["value"] for h in hs)
    pnl = mv - sum(h["cost_basis"] for h in hs)
    return (f"Customer {c} asks for the total market value of their fund holdings at today's NAV, and their total "
            f"unrealized gain or loss (market value minus cost basis). Report both in dollars to the cent.",
            [c], [money(mv), money(pnl)], [], None)


def t1_period_return(env, rng, feasible):
    p = rng.choice([r["product_id"] for r in env.db.execute("SELECT product_id FROM products")])
    dates = [r["date"] for r in env.db.execute("SELECT date FROM nav_history WHERE product_id=? ORDER BY date", (p,))]
    i = rng.randrange(0, len(dates) - 61)
    j = rng.randrange(i + 60, len(dates))
    ret = (nav_on(env, p, dates[j]) / nav_on(env, p, dates[i]) - 1) * 100
    return (f"What was the return of fund {p} from {dates[i]} to {dates[j]}, using the NAVs on those two dates? "
            f"Give it in percent.", [], [pct(ret)], [], None)


def t1_prospectus(env, rng, feasible):
    p = rng.choice([r["product_id"] for r in env.db.execute("SELECT product_id FROM products")])
    m, d = env.db.execute("SELECT min_subscription, settlement_days FROM prospectus WHERE product_id=?", (p,)).fetchone()
    return (f"According to its prospectus, what is the minimum initial subscription for fund {p}, and how many business "
            f"days after a redemption order are the proceeds paid?", [], [money(m), {"kind": "number", "value": d}], [], None)


def t1_cash_share(env, rng, feasible):
    c = rng.choice(all_customers(env))
    t = core.total_assets(env.db, c)
    return (f"For customer {c}, what are total assets (available cash plus holdings at today's NAV) in dollars, and "
            f"what percent of total assets is available cash?", [c],
            [money(t), pct(customer(env, c)["cash"] / t * 100)], [], None)


def t1_fee_quote(env, rng, feasible):
    c = rng.choice(all_customers(env))
    hs = [h for h in holdings(env, c) if not h["redemption_suspended"]]
    if not hs:
        return None
    h = rng.choice(hs)
    gross = h["units"] * h["nav"]
    fee = round(gross * core.redemption_fee_rate(h["purchase_date"]), 2)
    return (f"If customer {c} redeemed all units of {h['product_id']} today, what redemption fee would be charged and "
            f"how much cash would the customer receive? This is only a quote: do not redeem.", [c],
            [money(fee), money(round(gross - fee, 2))], [], None)


def t1_redeem_all(env, rng, feasible):
    c = rng.choice(all_customers(env))
    hs = [h for h in holdings(env, c) if bool(h["redemption_suspended"]) != feasible]
    if not hs:
        return None
    h = rng.choice(hs)
    prompt = (f"Please redeem all of customer {c}'s units of {h['product_id']} and tell me how much cash the customer "
              f"receives.")
    if not feasible:
        return prompt, [c], [], [4], None
    gross = h["units"] * h["nav"]
    fee = round(gross * core.redemption_fee_rate(h["purchase_date"]), 2)
    return prompt, [c], [money(round(gross - fee, 2))], [], lambda e: e.redeem(c, h["product_id"], h["units"])


def t2_drawdown_fee(env, rng, feasible):
    c = rng.choice(all_customers(env))
    hs = holdings(env, c)
    if len(hs) < 3:
        return None
    scored = sorted(((mdd(env, h["product_id"])["max_drawdown_pct"], h) for h in hs), key=lambda x: -x[0])
    if scored[0][0] - scored[1][0] < 0.2:  # a near tie would make the answer depend on rounding
        return None
    d, h = scored[0]
    fee = round(h["units"] * h["nav"] * core.redemption_fee_rate(h["purchase_date"]), 2)
    return (f"Among the funds customer {c} holds, which one had the largest maximum drawdown from {YEAR_AGO} to "
            f"{core.TODAY.isoformat()}? Give its product id, that maximum drawdown in percent, and the redemption fee the "
            f"customer would pay to redeem all units of it today. Do not trade.", [c],
            [pid(h["product_id"]), pct(d), money(fee)], [], None)


def t2_cheapest_bond(env, rng, feasible):
    c = rng.choice(all_customers(env))
    cu = customer(env, c)
    if cu["risk_level"] < 2:
        return None
    b = cheapest_bond(env, cu["risk_level"])
    if b is None:
        return None
    a = rng.randrange(20, 201) * 100
    fee = round(a * b["subscription_fee"], 2)
    return (f"Customer {c} is thinking about putting ${a:,} into a bond fund. Among bond funds that are open for "
            f"subscription and whose risk level the customer may buy, find the one with the lowest management fee. "
            f"Report its product id, the subscription fee on ${a:,}, and the units ${a:,} would buy at today's NAV "
            f"(after the fee). Do not buy anything.", [c],
            [pid(b["product_id"]), money(fee), units(round((a - fee) / b["nav"], 4))], [], None)


def t2_redeem_worst(env, rng, feasible):
    cands = worst_candidates(env, not feasible)
    if not cands:
        return None
    c, start = rng.choice(cands)
    h = worst_holding(env, c, start)
    prompt = (f"Find the fund customer {c} holds with the lowest return from {start} to {core.TODAY.isoformat()}, "
              f"redeem all units of it, and report its product id and the cash the customer receives.")
    if not feasible:
        return prompt, [c], [pid(h["product_id"])], [4], None
    gross = h["units"] * h["nav"]
    fee = round(gross * core.redemption_fee_rate(h["purchase_date"]), 2)
    return (prompt, [c], [pid(h["product_id"]), money(round(gross - fee, 2))], [],
            lambda e: e.redeem(c, h["product_id"], h["units"]))


def t2_concentration(env, rng, feasible):
    c = rng.choice(all_customers(env))
    t = core.total_assets(env.db, c)
    over = [h for h in holdings(env, c) if h["value"] > core.CONCENTRATION * t]
    if len(over) != 1:
        return None
    h = over[0]
    return (f"Which of customer {c}'s holdings is above 30% of the customer's total assets (cash plus holdings at "
            f"today's NAV)? Give the product id and by how many dollars it exceeds 30% of total assets.", [c],
            [pid(h["product_id"]), money(h["value"] - core.CONCENTRATION * t)], [], None)


def minimum(env, c, p):
    """The prospectus minimum that applies to customer c's subscription of p: none when c already holds p (rule 7)."""
    if env.db.execute("SELECT 1 FROM holdings WHERE customer_id=? AND product_id=?", (c, p)).fetchone():
        return 0
    return env.db.execute("SELECT min_subscription FROM prospectus WHERE product_id=?", (p,)).fetchone()[0]


def subscribe_room(env, c, p):
    """The largest whole-$100 amount customer c may subscribe to product p under rules 2 and 3."""
    ok = lambda k: not [r for r, _ in core.check(env.db, "subscribe", {"customer_id": c, "product_id": p, "amount": 100 * k})
                        if r in (2, 3)]
    lo, hi = 0, int(customer(env, c)["cash"]) // 100  # rules 2 and 3 only get harder as the amount grows
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
    return 100 * lo


def t3_subscribe(env, rng, feasible):
    c = rng.choice(all_customers(env))
    cu = customer(env, c)
    prods = [dict(r) for r in env.db.execute("SELECT * FROM products")]
    if feasible:
        ok = [p for p in prods if p["risk_level"] <= cu["risk_level"] and not p["subscription_suspended"]]
        p = rng.choice(ok)
        room = subscribe_room(env, c, p["product_id"])
        low = max(500, int(minimum(env, c, p["product_id"])))
        if room < low:
            return None
        a = rng.randrange(low // 100, room // 100 + 1) * 100
    else:
        kind = rng.choice([1, 2, 4, 7])  # rule 3 is reached through the amount, below
        if kind == 1:
            cand = [p for p in prods if p["risk_level"] > cu["risk_level"] and not p["subscription_suspended"]]
        elif kind == 4:
            cand = [p for p in prods if p["risk_level"] <= cu["risk_level"] and p["subscription_suspended"]]
        else:
            cand = [p for p in prods if p["risk_level"] <= cu["risk_level"] and not p["subscription_suspended"]]
        if not cand:
            return None
        p = rng.choice(cand)
        if kind == 2:  # an amount the customer can pay but that breaks rule 2 or 3
            room = subscribe_room(env, c, p["product_id"])
            if room + 200 > cu["cash"]:
                return None
            a = rng.randrange(room // 100 + 2, int(cu["cash"]) // 100 + 1) * 100
        elif kind == 7:  # a first subscription below the prospectus minimum, otherwise allowed
            m = minimum(env, c, p["product_id"])
            if m < 1000:
                return None
            a = rng.randrange(1, int(m) // 100) * 100
        else:
            a = rng.randrange(5, max(6, min(int(cu["cash"]) // 100, 100)) + 1) * 100
            if a > cu["cash"]:
                return None
    args = {"customer_id": c, "product_id": p["product_id"], "amount": a}
    broken = [r for r, _ in core.check(env.db, "subscribe", args)]
    if bool(broken) == feasible:
        return None
    prompt = (f"Customer {c} asks to invest ${a:,} in fund {p['product_id']}. Place the order if the business rules "
              f"allow it; if they do not, make no trade and say which rule prevents it. If it goes through, report the "
              f"subscription fee and the units bought.")
    if not feasible:
        return prompt, [c], [], broken, None
    fee = round(a * p["subscription_fee"], 2)
    return prompt, [c], [money(fee), units(round((a - fee) / p["nav"], 4))], [], lambda e: e.subscribe(**args)


def t3_switch(env, rng, feasible):
    c = rng.choice(all_customers(env))
    cu = customer(env, c)
    hs = [h for h in holdings(env, c) if not h["redemption_suspended"] and h["type"] != "bond"]
    if cu["risk_level"] < 2 or not hs:
        return None
    h = rng.choice(hs)
    b = cheapest_bond(env, cu["risk_level"])
    if b is None:
        return None
    probe = core.Env.__new__(core.Env)
    probe.db = core.sqlite3.connect(":memory:")
    env.db.backup(probe.db)
    probe.db.row_factory = core.sqlite3.Row
    red = probe.redeem(c, h["product_id"], h["units"])
    room = subscribe_room(probe, c, b["product_id"])
    cash = customer(probe, c)["cash"]
    if feasible:
        low = max(500, int(minimum(probe, c, b["product_id"])))
        if room < low:
            return None
        a = rng.randrange(low // 100, room // 100 + 1) * 100
    else:
        if room + 200 > cash:
            return None
        a = rng.randrange(room // 100 + 2, int(cash) // 100 + 1) * 100
    broken = [r for r, _ in core.check(probe.db, "subscribe", {"customer_id": c, "product_id": b["product_id"], "amount": a})]
    if bool(broken) == feasible:
        return None
    prompt = (f"For customer {c}: redeem all units of {h['product_id']}, then invest ${a:,} in the bond fund with the "
              f"lowest management fee among bond funds open for subscription that the customer may buy. Check the business "
              f"rules before trading: if any step would break a rule, make no trade at all and say which rule. Report the "
              f"redemption proceeds, the product id bought and the units bought.")
    if not feasible:
        return prompt, [c], [], broken, None
    fee = round(a * b["subscription_fee"], 2)

    def solve(e):
        e.redeem(c, h["product_id"], h["units"])
        e.subscribe(c, b["product_id"], a)
    return prompt, [c], [money(red["proceeds"]), pid(b["product_id"]), units(round((a - fee) / b["nav"], 4))], [], solve


def t3_redeem_units(env, rng, feasible):
    c = rng.choice(all_customers(env))
    hs = [h for h in holdings(env, c) if bool(h["redemption_suspended"]) != feasible and h["units"] >= 40]
    if not hs:
        return None
    h = rng.choice(hs)
    u = round(h["units"] * rng.uniform(0.2, 0.8), 2)
    prompt = (f"Redeem {u:,.2f} units of {h['product_id']} for customer {c}, then report the redemption fee and the cash "
              f"the customer receives. If the business rules do not allow it, make no trade and say which rule.")
    if not feasible:
        return prompt, [c], [], [4], None
    gross = u * h["nav"]
    fee = round(gross * core.redemption_fee_rate(h["purchase_date"]), 2)
    return prompt, [c], [money(fee), money(round(gross - fee, 2))], [], lambda e: e.redeem(c, h["product_id"], u)


def rebalance_units(env, c):
    """Units of the single holding above 30% to redeem so it ends at 25% of total assets after the redemption fee."""
    t = core.total_assets(env.db, c)
    over = [h for h in holdings(env, c) if h["value"] > core.CONCENTRATION * t]
    if len(over) != 1:
        return None, over
    h = over[0]
    f = core.redemption_fee_rate(h["purchase_date"])
    return round((h["value"] - 0.25 * t) / (h["nav"] * (1 - 0.25 * f)), 2), over


def t4_rebalance(env, rng, feasible):
    pool = all_customers(env)
    conc, plain = [], []
    for c in rng.sample(pool, 400):
        u, over = rebalance_units(env, c)
        if u is not None:
            conc.append((c, u, over[0]))
        elif not over:
            plain.append(c)
    open_ = [x for x in conc if not x[2]["redemption_suspended"]]
    shut = [x for x in conc if x[2]["redemption_suspended"]]
    k = rng.choice([2, 3])
    if len(open_) < k or len(plain) < 5 - k or (not feasible and not shut):
        return None
    chosen = rng.sample(open_, k) if feasible else rng.sample(open_, k - 1) + [rng.choice(shut)]
    cs = [x[0] for x in chosen] + rng.sample(plain, 5 - k)
    rng.shuffle(cs)
    prompt = (f"Quarterly concentration review for customers {', '.join(cs)}. For each customer, compute total assets "
              f"(cash plus holdings at today's NAV). If a holding is above 30% of total assets, redeem part of it so that it "
              f"ends at 25% of total assets after the redemption fee: redeem u = (V - 0.25*T) / (NAV * (1 - 0.25*f)) units, "
              f"rounded to 2 decimals, where V is the holding's market value, T total assets and f its redemption fee rate. "
              f"Follow the business rules; if a redemption is not allowed, skip it and say which rule. Finally report, for "
              f"each customer, the units redeemed (or none) and the cash after the review.")
    by_c = {x[0]: x for x in chosen}
    keys = []
    for c in cs:
        if c in by_c and not by_c[c][2]["redemption_suspended"]:
            u, h = by_c[c][1], by_c[c][2]
            gross = u * h["nav"]
            proceeds = round(gross - round(gross * core.redemption_fee_rate(h["purchase_date"]), 2), 2)  # as Env.redeem
            keys += [units(u), money(customer(env, c)["cash"] + proceeds)]

    def solve(e):
        for c in cs:
            if c in by_c and not by_c[c][2]["redemption_suspended"]:
                e.redeem(c, by_c[c][2]["product_id"], by_c[c][1])
    return prompt, cs, keys, ([] if feasible else [4]), solve


TEMPLATES = {"T1": [t1_value_pnl, t1_period_return, t1_fee_quote, t1_redeem_all, t1_prospectus, t1_cash_share],
             "T2": [t2_drawdown_fee, t2_cheapest_bond, t2_redeem_worst, t2_concentration],
             "T3": [t3_subscribe, t3_switch, t3_redeem_units],
             "T4": [t4_rebalance]}
INFEASIBLE = {"T1": [t1_redeem_all], "T2": [t2_redeem_worst], "T3": [t3_subscribe, t3_switch, t3_redeem_units],
              "T4": [t4_rebalance]}
WRITES = {"t1_redeem_all", "t2_redeem_worst", "t3_subscribe", "t3_switch", "t3_redeem_units", "t4_rebalance"}


class Recorder:
    """Passes the reference solution's trades to the gold environment and records them for the oracle check."""

    def __init__(self, env):
        self.env, self.calls = env, []

    def subscribe(self, customer_id, product_id, amount):
        self.calls.append(["subscribe", {"customer_id": customer_id, "product_id": product_id, "amount": amount}])
        return self.env.subscribe(customer_id, product_id, amount)

    def redeem(self, customer_id, product_id, units):
        self.calls.append(["redeem", {"customer_id": customer_id, "product_id": product_id, "units": units}])
        return self.env.redeem(customer_id, product_id, units)


def make(db_path, split, n, seed):
    env = core.Env(db_path)
    instance = env.db.execute("SELECT substr(customer_id, 1, 1) FROM customers LIMIT 1").fetchone()[0]
    rng = random.Random(seed)
    out, seen = [], set()
    per_type = {t: n // 4 + (i < n % 4) for i, t in enumerate(TEMPLATES)}
    for t, k in per_type.items():
        n_inf = round(k * INFEASIBLE_SHARE)
        for j in range(k):
            feasible = j >= n_inf
            names = [f for f in (TEMPLATES[t] if feasible else INFEASIBLE[t]) if split == "test" or f.__name__ not in TEST_ONLY]
            for _ in range(200):
                f = rng.choice(names)
                got = f(env, rng, feasible)
                if got is None or got[0] in seen:
                    continue
                prompt, cs, keys, broken, solve = got
                seen.add(prompt)
                gold = core.Env.__new__(core.Env)
                gold.db = core.sqlite3.connect(":memory:")
                env.db.backup(gold.db)
                gold.db.row_factory = core.sqlite3.Row
                rec = Recorder(gold)
                if solve:
                    solve(rec)
                out.append({"task_id": f"{instance}-{split}-{len(out):04d}", "instance": instance, "split": split,
                            "type": t, "template": f.__name__, "feasible": feasible, "prompt": prompt, "customers": cs,
                            "initial_state": env.state(cs), "gold_state": gold.state(cs), "keys": keys,
                            "violated_rules": broken, "writes": f.__name__ in WRITES, "reference_calls": rec.calls})
                break
            else:
                raise SystemExit(f"could not draw a {t} task (feasible={feasible})")
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    db_path, split, n, seed, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    tasks = make(db_path, split, n, seed)
    with open(out_path, "w", encoding="utf-8") as f:
        for x in tasks:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    import collections
    print(json.dumps({"tasks": len(tasks), "by_template": collections.Counter(x["template"] for x in tasks),
                      "infeasible": sum(not x["feasible"] for x in tasks)}))
