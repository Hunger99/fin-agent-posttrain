"""The wealth-management environment: an in-memory copy of one database instance, its tools and its business rules.

docs/03_harness.md sections 1-2. The environment is what every harness configuration shares; the harness decides
what the model sees and checks. So the write tools do not check business rules 1-4: they execute what they are given
(only impossible requests fail, such as spending more cash than the customer has or redeeming more units than are
held), charge rule 5's redemption fee, and record the trade. rules.check() is used by the harness's constraint part
before a write and by the grader after the episode.

Business rules (the project's own, after the idea of investor suitability; not real regulation):
1. A customer may subscribe only to products whose risk level is at most the customer's risk tolerance.
2. After a subscription, no product's market value exceeds 30% of the customer's total assets.
3. After a subscription, available cash is at least 5% of total assets.
4. A product with subscription (redemption) suspended cannot be subscribed (redeemed).
5. Redemption fee by holding period: under 7 days 1.5%, 7 days to under 1 year 0.5%, 1 year or more none.
6. An operation that breaks a rule is not executed, and the answer names the rule. (Checked by the grader.)
7. A first subscription to a product (the customer holds none of it) is at least the minimum initial subscription
   stated in the product's prospectus. The minimum is found only in the knowledge base (search_knowledge), so trades
   need retrieval; this rule is the English environment's addition (research/log/2026-10-27-0307.md).

Total assets are available cash plus the market value of all holdings at the latest NAV.
"""
import ast
import datetime
import json
import math
import operator
import re
import sqlite3

TODAY = datetime.date(2026, 6, 30)
CONCENTRATION, CASH_FLOOR = 0.30, 0.05
PAGE = 20


def load(path):
    """An in-memory copy of a database instance, so an episode's trades never touch the file."""
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db = sqlite3.connect(":memory:")
    src.backup(db)
    src.close()
    db.row_factory = sqlite3.Row
    return db


def redemption_fee_rate(purchase_date, on=TODAY):
    days = (on - datetime.date.fromisoformat(purchase_date)).days
    return 0.015 if days < 7 else 0.005 if days < 365 else 0.0


def total_assets(db, customer_id):
    cash = db.execute("SELECT cash FROM customers WHERE customer_id=?", (customer_id,)).fetchone()["cash"]
    mv = db.execute("SELECT COALESCE(SUM(h.units * p.nav), 0) v FROM holdings h JOIN products p USING (product_id) "
                    "WHERE h.customer_id=?", (customer_id,)).fetchone()["v"]
    return cash + mv


def check(db, tool, args):
    """The business rules a write would break, as [(rule number, reason)]; empty when it is allowed."""
    out = []
    cust = db.execute("SELECT * FROM customers WHERE customer_id=?", (args.get("customer_id"),)).fetchone()
    prod = db.execute("SELECT * FROM products WHERE product_id=?", (args.get("product_id"),)).fetchone()
    if cust is None or prod is None:
        return out
    if tool == "subscribe":
        amount = float(args.get("amount", 0))
        if prod["risk_level"] > cust["risk_level"]:
            out.append((1, f"product risk R{prod['risk_level']} is above the customer's tolerance C{cust['risk_level']}"))
        if prod["subscription_suspended"]:
            out.append((4, "subscription to this product is suspended"))
        fee = amount * prod["subscription_fee"]
        total_after = total_assets(db, cust["customer_id"]) - fee
        held = db.execute("SELECT units FROM holdings WHERE customer_id=? AND product_id=?",
                          (cust["customer_id"], prod["product_id"])).fetchone()
        value_after = (held["units"] if held else 0) * prod["nav"] + (amount - fee)
        if value_after > CONCENTRATION * total_after + 1e-6:
            out.append((2, f"the product would be {value_after / total_after:.1%} of total assets, above 30%"))
        if cust["cash"] - amount < CASH_FLOOR * total_after - 1e-6:
            out.append((3, f"cash would be {(cust['cash'] - amount) / total_after:.1%} of total assets, below 5%"))
        minimum = db.execute("SELECT min_subscription FROM prospectus WHERE product_id=?", (prod["product_id"],)).fetchone()[0]
        if held is None and amount < minimum - 1e-9:
            out.append((7, f"a first subscription must be at least the prospectus minimum of ${minimum:,.0f}"))
    elif tool == "redeem" and prod["redemption_suspended"]:
        out.append((4, "redemption of this product is suspended"))
    return out


class Env:
    """One episode's environment. Each tool takes keyword arguments and returns a JSON-serializable dict."""

    def __init__(self, db_path):
        self.db = load(db_path)
        self.finished, self.answer = False, None

    # read tools -------------------------------------------------------------------------------------------------
    def get_customer(self, customer_id):
        r = self.db.execute("SELECT * FROM customers WHERE customer_id=?", (customer_id,)).fetchone()
        if r is None:
            raise ToolError(f"no customer {customer_id!r}")
        return {"customer_id": r["customer_id"], "name": r["name"], "risk_tolerance": f"C{r['risk_level']}",
                "cash": round(r["cash"], 2)}

    def list_holdings(self, customer_id, page=1):
        self.get_customer(customer_id)
        rows = self.db.execute("SELECT product_id, units, cost_basis, purchase_date FROM holdings WHERE customer_id=? "
                               "ORDER BY product_id", (customer_id,)).fetchall()
        return paged([dict(r) for r in rows], page)

    def fund(self, product_id):
        r = self.db.execute("SELECT * FROM products WHERE product_id=?", (product_id,)).fetchone()
        if r is None:
            raise ToolError(f"no product {product_id!r}")
        return {"product_id": r["product_id"], "name": r["name"], "type": r["type"], "risk_level": f"R{r['risk_level']}",
                "nav": r["nav"], "nav_date": TODAY.isoformat(), "management_fee": r["mgmt_fee"],
                "custody_fee": r["custody_fee"], "subscription_fee": r["subscription_fee"],
                "subscription_suspended": bool(r["subscription_suspended"]),
                "redemption_suspended": bool(r["redemption_suspended"])}

    def get_fund(self, product_ids):
        """The get_fund tool: up to 20 products in one call (research/log/2026-10-27-1945.md); an unknown id gives its
        error in place of the product, so the others still count."""
        if isinstance(product_ids, str):
            product_ids = [product_ids]
        if not isinstance(product_ids, list) or not 1 <= len(product_ids) <= 20:
            raise ToolError("product_ids must be a list of 1 to 20 product ids")
        out = []
        for p in product_ids:
            try:
                out.append(self.fund(p))
            except ToolError as e:
                out.append({"product_id": p, "error": str(e)})
        return {"products": out}

    def search_funds(self, type=None, max_risk_level=None, open_for_subscription=None, sort_by="product_id", page=1):
        q, params = "SELECT product_id, name, type, risk_level, mgmt_fee, subscription_fee, subscription_suspended FROM products WHERE 1=1", []
        if type:
            if type not in ("money_market", "bond", "mixed", "index", "equity"):
                raise ToolError("type must be one of money_market, bond, mixed, index, equity")
            q += " AND type=?"
            params.append(type)
        if max_risk_level:
            q += " AND risk_level<=?"
            params.append(int(str(max_risk_level).lstrip("Rr")))
        if open_for_subscription:
            q += " AND subscription_suspended=0"
        order = {"product_id": "product_id", "management_fee": "mgmt_fee, product_id",
                 "subscription_fee": "subscription_fee, product_id"}.get(sort_by)
        if order is None:
            raise ToolError("sort_by must be product_id, management_fee or subscription_fee")
        rows = self.db.execute(q + " ORDER BY " + order, params).fetchall()
        return paged([{"product_id": r["product_id"], "name": r["name"], "type": r["type"],
                       "risk_level": f"R{r['risk_level']}", "management_fee": r["mgmt_fee"],
                       "subscription_fee": r["subscription_fee"],
                       "subscription_suspended": bool(r["subscription_suspended"])} for r in rows], page)

    def get_nav_history(self, product_id, start_date, end_date, page=1):
        self.fund(product_id)
        rows = self.db.execute("SELECT date, nav FROM nav_history WHERE product_id=? AND date BETWEEN ? AND ? ORDER BY date",
                               (product_id, start_date, end_date)).fetchall()
        return paged([dict(r) for r in rows], page, size=60)

    def fund_metrics(self, product_id, start_date, end_date):
        """Return over the period, maximum drawdown and annualized volatility of daily returns, in percent."""
        self.fund(product_id)
        navs = [r["nav"] for r in self.db.execute("SELECT nav FROM nav_history WHERE product_id=? AND date BETWEEN ? AND ? "
                                                    "ORDER BY date", (product_id, start_date, end_date))]
        if len(navs) < 2:
            raise ToolError("fewer than two NAVs in the period")
        return metrics(navs) | {"start_nav": navs[0], "end_nav": navs[-1], "days": len(navs)}

    def calculator(self, expressions):
        return calculate(expressions)

    def search_knowledge(self, query, top_k=5):
        """BM25 over the prospectus passages; returns passage ids, titles and the first 160 characters."""
        if not hasattr(self, "_bm25"):
            self._bm25 = BM25([(r["doc_id"], r["title"], r["text"]) for r in self.db.execute("SELECT * FROM documents")])
        hits = self._bm25.search(str(query), max(1, min(int(top_k), 10)))
        return {"results": [{"doc_id": d, "title": t, "snippet": x[:160]} for d, t, x in hits]}

    def get_document(self, doc_id):
        r = self.db.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if r is None:
            raise ToolError(f"no document {doc_id!r}")
        return {"doc_id": r["doc_id"], "title": r["title"], "text": r["text"]}

    # write tools ------------------------------------------------------------------------------------------------
    def subscribe(self, customer_id, product_id, amount):
        amount = float(amount)
        cust, prod = self.get_customer(customer_id), self.fund(product_id)
        if amount <= 0:
            raise ToolError("amount must be positive")
        if amount > cust["cash"] + 1e-9:
            raise ToolError(f"insufficient cash: {cust['cash']:.2f} available")
        fee = round(amount * prod["subscription_fee"], 2)
        units = round((amount - fee) / prod["nav"], 4)
        self.db.execute("UPDATE customers SET cash=cash-? WHERE customer_id=?", (amount, customer_id))
        held = self.db.execute("SELECT 1 FROM holdings WHERE customer_id=? AND product_id=?", (customer_id, product_id)).fetchone()
        if held:
            self.db.execute("UPDATE holdings SET units=units+?, cost_basis=cost_basis+? WHERE customer_id=? AND product_id=?",
                            (units, amount, customer_id, product_id))
        else:
            self.db.execute("INSERT INTO holdings VALUES (?,?,?,?,?)", (customer_id, product_id, units, amount, TODAY.isoformat()))
        self.db.execute("INSERT INTO transactions (customer_id, product_id, type, amount, units, nav, fee, date) "
                        "VALUES (?,?,?,?,?,?,?,?)", (customer_id, product_id, "subscribe", amount, units, prod["nav"], fee,
                                                     TODAY.isoformat()))
        return {"status": "executed", "type": "subscribe", "amount": amount, "fee": fee, "units": units, "nav": prod["nav"],
                "cash_after": round(cust["cash"] - amount, 2)}

    def redeem(self, customer_id, product_id, units):
        units = float(units)
        self.get_customer(customer_id)
        prod = self.fund(product_id)
        h = self.db.execute("SELECT * FROM holdings WHERE customer_id=? AND product_id=?", (customer_id, product_id)).fetchone()
        if h is None:
            raise ToolError(f"{customer_id} holds no {product_id}")
        if units <= 0 or units > h["units"] + 1e-9:
            raise ToolError(f"units must be positive and at most the {h['units']} held")
        gross = units * prod["nav"]
        rate = redemption_fee_rate(h["purchase_date"])
        fee = round(gross * rate, 2)
        proceeds = round(gross - fee, 2)
        left = round(h["units"] - units, 4)
        if left <= 1e-4:
            self.db.execute("DELETE FROM holdings WHERE customer_id=? AND product_id=?", (customer_id, product_id))
        else:
            self.db.execute("UPDATE holdings SET units=?, cost_basis=cost_basis*? WHERE customer_id=? AND product_id=?",
                            (left, left / h["units"], customer_id, product_id))
        self.db.execute("UPDATE customers SET cash=cash+? WHERE customer_id=?", (proceeds, customer_id))
        self.db.execute("INSERT INTO transactions (customer_id, product_id, type, amount, units, nav, fee, date) "
                        "VALUES (?,?,?,?,?,?,?,?)", (customer_id, product_id, "redeem", proceeds, units, prod["nav"], fee,
                                                     TODAY.isoformat()))
        return {"status": "executed", "type": "redeem", "units": units, "nav": prod["nav"], "fee_rate": rate, "fee": fee,
                "proceeds": proceeds}

    def finish(self, answer):
        self.finished, self.answer = True, str(answer)
        return {"status": "finished"}

    # state ------------------------------------------------------------------------------------------------------
    def state(self, customer_ids):
        """The parts of the database a task can change, for the customers it concerns."""
        out = {}
        for c in customer_ids:
            cash = self.db.execute("SELECT cash FROM customers WHERE customer_id=?", (c,)).fetchone()["cash"]
            hold = {r["product_id"]: r["units"] for r in
                    self.db.execute("SELECT product_id, units FROM holdings WHERE customer_id=?", (c,))}
            txns = [(r["product_id"], r["type"]) for r in
                    self.db.execute("SELECT product_id, type FROM transactions WHERE customer_id=? ORDER BY txn_id", (c,))]
            out[c] = {"cash": cash, "holdings": hold, "transactions": sorted(txns)}
        return out


class ToolError(Exception):
    pass


TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class BM25:
    """Okapi BM25 (k1 1.2, b 0.75) over (id, title, text) passages, title and text indexed together."""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.toks = [TOKEN.findall((t + " " + x).lower()) for _, t, x in docs]
        self.avg = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = {}
        for ts in self.toks:
            for w in set(ts):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.tf = [{w: ts.count(w) for w in set(ts)} for ts in self.toks]

    def search(self, query, k):
        q = TOKEN.findall(query.lower())
        scores = []
        for i, tf in enumerate(self.tf):
            n_i = len(self.toks[i])
            s = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * n_i / self.avg))
                    for w in q if w in tf)
            if s > 0:
                scores.append((s, i))
        return [self.docs[i] for _, i in sorted(scores, reverse=True)[:k]]


def paged(rows, page, size=PAGE):
    page = max(1, int(page))
    pages = max(1, math.ceil(len(rows) / size))
    return {"page": page, "pages": pages, "total": len(rows), "items": rows[(page - 1) * size:page * size]}


def metrics(navs):
    peak, mdd = navs[0], 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = max(mdd, 1 - v / peak)
    rets = [b / a - 1 for a, b in zip(navs, navs[1:])]
    mean = sum(rets) / len(rets)
    vol = math.sqrt(sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)) * math.sqrt(252)
    return {"return_pct": round((navs[-1] / navs[0] - 1) * 100, 4), "max_drawdown_pct": round(mdd * 100, 4),
            "annualized_volatility_pct": round(vol * 100, 4)}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp, "abs": abs, "round": round, "min": min, "max": max,
          "sum": lambda *a: sum(a)}


def calc(expression):
    """Evaluate an arithmetic expression: numbers, + - * / ** %, parentheses and a few functions."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ToolError("only numbers, + - * / ** %, parentheses and sqrt, log, exp, abs, round, min, max, sum")
    try:
        v = ev(ast.parse(re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(expression)).replace("^", "**"), mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError, ValueError, TypeError) as e:
        raise ToolError(f"cannot evaluate: {e}")
    return round(v, 10) if isinstance(v, float) else v


def calculate(expressions):
    """The calculator tool: one or more expressions in one call (research/log/2026-10-27-1825.md); one that fails
    gives its error in place of a result, so the others still count."""
    if isinstance(expressions, str):
        expressions = [expressions]
    if not isinstance(expressions, list) or not expressions:
        raise ToolError("expressions must be a non-empty list of strings")
    out = []
    for e in expressions:
        try:
            out.append({"expression": e, "result": calc(e)})
        except ToolError as x:
            out.append({"expression": e, "error": str(x)})
    return {"results": out}


def to_json(x):
    return json.dumps(x, ensure_ascii=False)
