"""The wealth-management environment kind ("wealth", the default): the system prompt, tools, rule checks, verification
and grading that version 2 kept in agent.py and env/, behind the EnvKind interface of harness/envs/base.py.

The system prompt and tool descriptions are version 2.1's, except that get_fund says how many products one call takes
(core.MAX_FUNDS). The five-customer review t4_rebalance and the composite t5_review_switch are templates of this kind.
"""
from harness.envs import base
from harness.envs.base import B, DATE, FINISH, I, S, schema
from harness.envs.wealth import core, grade

SYSTEM = f"""You are an operations assistant on a fund platform, serving requests about customers' accounts with the tools provided. Today is {core.TODAY.isoformat()}; every NAV the tools return is today's unless dated otherwise. Amounts are US dollars.

Business rules:
1. A customer may subscribe only to products whose risk level (R1-R5) is at most the customer's risk tolerance (C1-C5).
2. After a subscription, the subscribed product's market value may not exceed 30% of the customer's total assets.
3. After a subscription, the customer's available cash must be at least 5% of total assets.
4. A product whose subscription (or redemption) is suspended cannot be subscribed (or redeemed).
5. Redemption fee by holding period since the purchase date: under 7 days 1.5%, 7 days to under 1 year 0.5%, 1 year or more none.
6. An operation that breaks a rule must not be executed; say which rule prevents it.
7. A first subscription to a product (the customer holds none of it) must be at least the minimum initial subscription in the product's prospectus; prospectuses are in the knowledge base.
Total assets are available cash plus all holdings at today's NAV. A subscription fee is charged on the amount; the rest buys units at today's NAV.

Act on the request as stated without asking questions. Do all arithmetic with the calculator, not in your reasoning. Use as few tool calls as you can: one get_fund call can look up many products, one calculator call can evaluate many expressions, and one expression can combine many values, such as a customer's total assets. When done, call finish with the final answer, giving every value the request asks for (amounts to the cent)."""

TOOLS = [
    schema("get_customer", "A customer's name, risk tolerance and available cash.", {"customer_id": S}, ["customer_id"]),
    schema("list_holdings", "A customer's holdings (units, cost basis, purchase date), 20 per page.",
           {"customer_id": S, "page": I}, ["customer_id"]),
    schema("get_fund", f"Products' type, risk level, today's NAV, fees and suspensions; up to {core.MAX_FUNDS} products in "
           "one call, e.g. all of a customer's holdings. It does not give the minimum initial subscription, which is in "
           "the product's prospectus in the knowledge base.",
           {"product_ids": {"type": "array", "items": S, "minItems": 1, "maxItems": core.MAX_FUNDS}}, ["product_ids"]),
    schema("search_funds", "List products, 20 per page, optionally filtered and sorted; use get_fund for details.",
           {"type": {"type": "string", "enum": ["money_market", "bond", "mixed", "index", "equity"]},
            "max_risk_level": {"type": "string", "enum": ["R1", "R2", "R3", "R4", "R5"]}, "open_for_subscription": B,
            "sort_by": {"type": "string", "enum": ["product_id", "management_fee", "subscription_fee"]}, "page": I}, []),
    schema("get_nav_history", "A product's daily NAVs between two dates, 60 per page.",
           {"product_id": S, "start_date": DATE, "end_date": DATE, "page": I}, ["product_id", "start_date", "end_date"]),
    schema("fund_metrics", "Return, maximum drawdown and annualized volatility (percent) between two dates.",
           {"product_id": S, "start_date": DATE, "end_date": DATE}, ["product_id", "start_date", "end_date"]),
    schema("calculator", "Evaluate one or more arithmetic expressions in one call, e.g. [\"(2.7124*1196.65)*(1-0.005)\", "
           "\"5055.3 + 48744.1891*1.3714 + 737.5851*3.1643\"]; put every value you need at this point into one call.",
           {"expressions": {"type": "array", "items": S, "minItems": 1}}, ["expressions"]),
    schema("search_knowledge", "Search the knowledge base of fund prospectuses; returns passage ids, titles and snippets.",
           {"query": S, "top_k": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
    schema("get_document", "The full text of a knowledge-base passage.", {"doc_id": S}, ["doc_id"]),
    schema("subscribe", "Buy a product for a customer with a dollar amount (fee included). Executes immediately. "
           "Before a first subscription to a product, look up its minimum initial subscription in the knowledge base "
           "(rule 7).",
           {"customer_id": S, "product_id": S, "amount": {"type": "number", "exclusiveMinimum": 0}},
           ["customer_id", "product_id", "amount"]),
    schema("redeem", "Sell units of a customer's holding at today's NAV, minus the redemption fee. Executes immediately.",
           {"customer_id": S, "product_id": S, "units": {"type": "number", "exclusiveMinimum": 0}},
           ["customer_id", "product_id", "units"]),
    FINISH,
]
EXAMPLES = {"get_customer": {"customer_id": "B-C0001"}, "list_holdings": {"customer_id": "B-C0001", "page": 1},
            "get_fund": {"product_ids": ["B-P0006", "B-P0170"]},
            "search_funds": {"type": "bond", "max_risk_level": "R2", "open_for_subscription": True, "sort_by": "management_fee"},
            "get_nav_history": {"product_id": "B-P0006", "start_date": "2026-06-01", "end_date": "2026-06-30"},
            "fund_metrics": {"product_id": "B-P0006", "start_date": "2025-06-30", "end_date": "2026-06-30"},
            "calculator": {"expressions": ["1196.65 * 2.7124 * (1 - 0.005)", "5055.3 + 48744.1891 * 1.3714"]},
            "search_knowledge": {"query": "B-P0006 minimum initial subscription", "top_k": 5},
            "get_document": {"doc_id": "B-P0006-2"},
            "subscribe": {"customer_id": "B-C0001", "product_id": "B-P0170", "amount": 5000},
            "redeem": {"customer_id": "B-C0001", "product_id": "B-P0006", "units": 1196.65},
            "finish": {"answer": "..."}}


def largest_allowed(db, c, p):
    lo, hi = 0, int(db.execute("SELECT cash FROM customers WHERE customer_id=?", (c,)).fetchone()[0] * 100)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        broken = {r for r, _ in core.check(db, "subscribe", {"customer_id": c, "product_id": p, "amount": mid / 100})}
        lo, hi = (mid, hi) if not broken & {2, 3} else (lo, mid - 1)  # rules 2 and 3 only get harder as the amount grows
    return lo / 100


class WealthKind(base.EnvKind):
    name = "wealth"
    SYSTEM, TOOLS, EXAMPLES, WRITES = SYSTEM, TOOLS, EXAMPLES, ("subscribe", "redeem")

    def make_env(self, task, db_path):
        return core.Env(task.get("db") or db_path)

    def check_write(self, env, name, args):
        return core.check(env.db, name, args)

    def rejection_details(self, env, name, args, broken):
        if name == "subscribe" and {r for r, _ in broken} <= {2, 3}:  # rule 7 has a floor, not a ceiling
            return {"largest_allowed_amount": largest_allowed(env.db, args["customer_id"], args["product_id"])}
        return {}

    def trade_info(self, env, name, args):
        """Whether a subscription is the customer's first of the product, for rule 7 at verification."""
        return name == "subscribe" and env.db.execute(
            "SELECT 1 FROM holdings WHERE customer_id=? AND product_id=?",
            (args.get("customer_id"), args.get("product_id"))).fetchone() is None

    def verify(self, ep, answer):
        """Each trade of the episode must satisfy the rules on the final state (version 2's check)."""
        problems, db = [], ep.env.db
        for name, args, first in ep.trades:
            c, p = args["customer_id"], args["product_id"]
            cust = db.execute("SELECT * FROM customers WHERE customer_id=?", (c,)).fetchone()
            prod = db.execute("SELECT * FROM products WHERE product_id=?", (p,)).fetchone()
            if name == "subscribe":
                t = core.total_assets(db, c)
                v = (db.execute("SELECT units FROM holdings WHERE customer_id=? AND product_id=?", (c, p)).fetchone() or [0])[0] * prod["nav"]
                if prod["risk_level"] > cust["risk_level"]:
                    problems.append(f"the subscription of {p} for {c} broke rule 1")
                if prod["subscription_suspended"]:
                    problems.append(f"the subscription of {p} for {c} broke rule 4")
                if v > core.CONCENTRATION * t + 1e-6:
                    problems.append(f"after the subscription, {p} is {v / t:.1%} of {c}'s total assets (rule 2)")
                if cust["cash"] < core.CASH_FLOOR * t - 1e-6:
                    problems.append(f"after the subscription, {c}'s cash is {cust['cash'] / t:.1%} of total assets (rule 3)")
                m = db.execute("SELECT min_subscription FROM prospectus WHERE product_id=?", (p,)).fetchone()[0]
                if first and float(args["amount"]) < m - 1e-9:
                    problems.append(f"the first subscription of {p} for {c} is below its prospectus minimum (rule 7)")
            elif prod["redemption_suspended"]:
                problems.append(f"the redemption of {p} for {c} broke rule 4")
        return problems

    def grade(self, task, env):
        return grade.grade(task, env.state(task["customers"]), env.answer)

    def categorize(self, ep, g):
        """The main failure category from the log (docs/03_harness.md section 6), version 2's rules."""
        if g["passed"]:
            return None
        t = ep.task
        if ep.end in ("context_overflow", "step_limit", "hard_cap"):
            return ep.end
        if max(ep.calls.values(), default=0) >= 3:
            return "loop"
        forbidden = {c for c in t["customers"] if t["gold_state"][c] == t["initial_state"][c]}
        if any(a["customer_id"] in forbidden and not t["feasible"] for _, a, _ in ep.trades):
            return "rule_violation"
        gold_trades = sum(len(s["transactions"]) for s in t["gold_state"].values())
        if g["state"] and len(ep.trades) < gold_trades:
            return "missed_subtask" if ep.trades else "premature_finish"
        if g["state"]:
            return "argument_error"
        if g["missing_keys"] or g["names_rule"] is False:
            return "calculation_error" if any(k["kind"] != "id" for k in g["missing_keys"]) else "other"
        return "other"
