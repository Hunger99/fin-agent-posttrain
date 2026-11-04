"""Generate one instance of the wealth-management database, per docs/03_harness.md section 2.

The English environment keeps the Chinese design's tables, rules and sizes, in US dollars (docs/results.md,
Overview Version 4). An instance is a SQLite file with five tables:

- customers: id, name, risk tolerance C1-C5 (stored 1-5), available cash;
- products: id, name, type, risk level R1-R5 (stored 1-5), annual management and custody fees, subscription fee,
  latest NAV, whether subscription or redemption is suspended;
- nav_history: two years of business-day NAVs per product, ending on the environment's date TODAY;
- holdings: one lot per customer and product, with units, cost basis and purchase date;
- transactions: the trades the agent makes, empty when generated;
- prospectus: each product's minimum initial subscription, settlement period, benchmark, manager and dividend
  policy, which the tools do not return: they are found only in the knowledge base;
- documents: the knowledge base, three prospectus passages per product.

Three instances with different seeds and id prefixes (A for training tasks, B for development, C for testing) share
no customer or product. NAVs follow a geometric random walk whose drift and volatility depend on the product type.
About 5% of products have subscription suspended and 8% redemption
suspended, so that tasks the rules forbid can be drawn. Holdings respect each customer's risk tolerance; about a fifth of customers hold one product above 30% of their
total assets, so that rebalancing and concentration tasks have cases to find.

Usage: python -B envtool.py generate-db <out.sqlite> <prefix A|B|C> <seed> [--customers 2000] [--products 300]
"""
import argparse
import datetime
import math
import os
import sqlite3

import numpy as np

TODAY = datetime.date(2026, 6, 30)
DAYS = 504  # two years of business days
# type: (risk levels, annual drift, annual volatility, subscription fee, management fee, custody fee)
TYPES = {"money_market": ((1,), 0.025, 0.003, 0.0, 0.0030, 0.0008),
         "bond": ((2,), 0.035, 0.035, 0.006, 0.0060, 0.0015),
         "mixed": ((3,), 0.060, 0.120, 0.012, 0.0120, 0.0020),
         "index": ((3, 4), 0.070, 0.180, 0.010, 0.0050, 0.0010),
         "equity": ((4, 5), 0.080, 0.260, 0.015, 0.0150, 0.0025)}
TYPE_SHARE = {"money_market": 0.08, "bond": 0.27, "mixed": 0.25, "index": 0.15, "equity": 0.25}
WORDS = ["Harbor", "Summit", "Evergreen", "Pioneer", "Atlas", "Beacon", "Cedar", "Granite", "Horizon", "Liberty",
         "Meridian", "Northstar", "Oakridge", "Prairie", "Redwood", "Sterling", "Trident", "Vanguardia", "Willow", "Zenith"]
SUFFIX = {"money_market": "Money Market Fund", "bond": "Bond Fund", "mixed": "Balanced Fund", "index": "Index Fund",
          "equity": "Equity Fund"}
FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth", "William",
         "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Daniel", "Karen", "Wei", "Priya", "Carlos",
         "Fatima", "Hiroshi", "Olga", "Kwame", "Sofia", "Ahmed", "Mei"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Lee",
        "Walker", "Hall", "Allen", "Young", "King", "Wright", "Scott", "Green", "Baker", "Chen", "Patel", "Nguyen", "Kim"]


def business_days(end, n):
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= datetime.timedelta(days=1)
    return out[::-1]


BENCH = {"money_market": ["the 3-month US Treasury bill rate"], "bond": ["the Bloomberg US Aggregate Bond Index",
          "the ICE BofA US Corporate Index", "the Bloomberg US Treasury Index"],
          "mixed": ["a 60/40 blend of the S&P 500 and the Bloomberg US Aggregate Bond Index"],
          "index": ["the S&P 500 Index", "the Nasdaq-100 Index", "the Russell 2000 Index", "the MSCI EAFE Index"],
          "equity": ["the S&P 500 Index", "the Russell 1000 Growth Index", "the MSCI World Index"]}
OBJECTIVE = {"money_market": "to preserve capital and provide liquidity", "bond": "to provide current income",
             "mixed": "to balance growth and income", "index": "to track its benchmark index before fees",
             "equity": "to seek long-term capital growth"}
MIN_SENT = ["The minimum initial subscription is ${m:,}; later subscriptions to an existing position have no minimum.",
            "Investors opening a position must subscribe at least ${m:,}. Additional purchases may be of any amount.",
            "Initial purchases below ${m:,} are not accepted, while top-ups of an existing holding have no minimum."]
DIVIDEND = ["Income is distributed monthly.", "Income is distributed quarterly.", "Income is reinvested; the fund pays no dividends.",
            "Any income is distributed once a year in December."]


def prospectuses(products, rng):
    """Dealing terms found only in each product's prospectus (the knowledge base), and the prospectus passages."""
    terms, docs = [], []
    for pid, name, t, risk, mgmt, custody, sub, nav, *_ in products:
        m = int(rng.choice([100, 500, 1000, 2500, 5000, 10000], p=[0.1, 0.2, 0.3, 0.2, 0.12, 0.08]))
        d = int(rng.choice([1, 2, 3, 5]))
        bench = str(rng.choice(BENCH[t]))
        manager = f"{WORDS[int(rng.integers(len(WORDS)))]} Asset Management"
        div = str(rng.choice(DIVIDEND))
        terms.append((pid, m, d, bench, manager, div))
        docs += [(f"{pid}-1", pid, f"{name}: overview",
                  f"{name} ({pid}) is a {t.replace('_', ' ')} fund managed by {manager}. Its objective is {OBJECTIVE[t]}. "
                  f"Benchmark: {bench}. Risk level R{risk}."),
                 (f"{pid}-2", pid, f"{name}: dealing terms",
                  f"Dealing terms for {pid}. " + str(rng.choice(MIN_SENT)).format(m=m) +
                  f" Redemption proceeds are paid T+{d} business days after the order. {div}"),
                 (f"{pid}-3", pid, f"{name}: fees",
                  f"Fees for {pid}: management fee {mgmt * 100:.3f}% a year, custody fee {custody * 100:.2f}% a year, "
                  f"subscription fee {sub * 100:.2f}% of the amount. Redemption fees follow the platform schedule.")]
    return terms, docs


def build(path, prefix, seed, n_customers, n_products):
    rng = np.random.default_rng(seed)
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.executescript("""
    CREATE TABLE customers (customer_id TEXT PRIMARY KEY, name TEXT, risk_level INTEGER, cash REAL);
    CREATE TABLE products (product_id TEXT PRIMARY KEY, name TEXT, type TEXT, risk_level INTEGER, mgmt_fee REAL,
        custody_fee REAL, subscription_fee REAL, nav REAL, subscription_suspended INTEGER, redemption_suspended INTEGER);
    CREATE TABLE nav_history (product_id TEXT, date TEXT, nav REAL, PRIMARY KEY (product_id, date));
    CREATE TABLE holdings (customer_id TEXT, product_id TEXT, units REAL, cost_basis REAL, purchase_date TEXT,
        PRIMARY KEY (customer_id, product_id));
    CREATE TABLE transactions (txn_id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT, product_id TEXT, type TEXT,
        amount REAL, units REAL, nav REAL, fee REAL, date TEXT);
    CREATE TABLE prospectus (product_id TEXT PRIMARY KEY, min_subscription REAL, settlement_days INTEGER, benchmark TEXT,
        manager TEXT, dividend TEXT);
    CREATE TABLE documents (doc_id TEXT PRIMARY KEY, product_id TEXT, title TEXT, text TEXT);
    """)
    days = business_days(TODAY, DAYS)
    types = rng.choice(list(TYPE_SHARE), size=n_products, p=list(TYPE_SHARE.values()))
    products, navs = [], {}
    for i, t in enumerate(types):
        risks, mu, sigma, sub_fee, mgmt, custody = TYPES[t]
        pid = f"{prefix}-P{i + 1:04d}"
        start = round(float(rng.uniform(0.8, 3.0)), 4) if t != "money_market" else 1.0
        steps = rng.normal((mu - sigma ** 2 / 2) / 252, sigma / math.sqrt(252), size=DAYS - 1)
        path_ = start * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))
        path_ = np.round(path_, 4)
        navs[pid] = path_
        name = f"{WORDS[int(rng.integers(len(WORDS)))]} {WORDS[int(rng.integers(len(WORDS)))]} {SUFFIX[t]} {prefix}{i + 1}"
        jitter = float(rng.uniform(0.8, 1.2))
        products.append((pid, name, str(t), int(rng.choice(risks)), round(mgmt * jitter, 5), custody,
                         round(sub_fee * float(rng.uniform(0.6, 1.0)), 4) if sub_fee else 0.0, float(path_[-1]),
                         int(rng.random() < 0.05), int(rng.random() < 0.08)))
    db.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?)", products)
    terms, docs = prospectuses(products, np.random.default_rng(seed + 1))
    db.executemany("INSERT INTO prospectus VALUES (?,?,?,?,?,?)", terms)
    db.executemany("INSERT INTO documents VALUES (?,?,?,?)", docs)
    db.executemany("INSERT INTO nav_history VALUES (?,?,?)",
                   [(pid, d.isoformat(), float(v)) for pid, p in navs.items() for d, v in zip(days, p)])
    by_risk = {r: [p for p in products if p[3] <= r] for r in range(1, 6)}
    customers, holdings = [], []
    for i in range(n_customers):
        cid = f"{prefix}-C{i + 1:04d}"
        level = int(rng.choice([1, 2, 3, 4, 5], p=[0.10, 0.25, 0.30, 0.25, 0.10]))
        invest = float(np.exp(rng.normal(math.log(80_000), 0.8)))
        k = int(rng.integers(2, 7))
        pool = by_risk[level]
        picks = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        weights = rng.dirichlet(np.ones(len(picks)))
        if rng.random() < 0.2:  # one concentrated position
            weights = weights * 0.4
            weights[0] += 0.6
        for w, j in zip(weights, picks):
            p = pool[int(j)]
            day = int(rng.integers(0, DAYS - 1))
            if rng.random() < 0.08:  # a recent purchase, inside the 7-day redemption-fee tier
                day = DAYS - 1 - int(rng.integers(0, 4))
            nav_then = float(navs[p[0]][day])
            amount = round(invest * float(w), 2)
            units = round(amount / nav_then, 4)
            holdings.append((cid, p[0], units, amount, days[day].isoformat()))
        cash = round(invest * float(rng.uniform(0.04, 0.35)), 2)
        customers.append((cid, f"{FIRST[int(rng.integers(len(FIRST)))]} {LAST[int(rng.integers(len(LAST)))]}", level, cash))
    db.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
    db.executemany("INSERT INTO holdings VALUES (?,?,?,?,?)", holdings)
    db.commit()
    db.close()
    return {"customers": len(customers), "products": len(products), "holdings": len(holdings)}


def main(argv):
    ap = argparse.ArgumentParser(prog="envtool.py generate-db")
    ap.add_argument("out")
    ap.add_argument("prefix", choices=["A", "B", "C"])
    ap.add_argument("seed", type=int)
    ap.add_argument("--customers", type=int, default=2000)
    ap.add_argument("--products", type=int, default=300)
    a = ap.parse_args(argv)
    print(build(a.out, a.prefix, a.seed, a.customers, a.products))
