"""Check the market_v1 snapshot against known facts and store the results.

Each check compares a stored value with a published figure within a
relative tolerance and prints expected, got and pass/fail; the results are
also written to meta key ``validation`` as JSON.

Usage:
    python -B validate.py [--db PATH] [--no-write]
"""

import argparse
import json
import os
import sqlite3
import sys

DB = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1/market_v1.sqlite"

# (name, sql, expected, relative tolerance). The price checks compare the snapshot with itself (the raw close drops
# by about the split ratio on a split day; the 2020 low of ^GSPC falls on 2020-03-23) rather than with quoted prices.
CHECKS = [
    ("AAPL raw close 2020-08-28 / 2020-08-31 (the 4:1 split)",
     "SELECT a.raw_close / b.raw_close FROM prices a, prices b WHERE a.symbol='AAPL' AND a.date='2020-08-28' "
     "AND b.symbol='AAPL' AND b.date='2020-08-31'", 4.0, 0.15),
    ("AAPL split 2020-08-31 ratio",
     "SELECT value FROM actions WHERE symbol='AAPL' AND date='2020-08-31' AND kind='split'", 4.0, 1e-9),
    ("AAPL raw close 2014-06-06 / 2014-06-09 (the 7:1 split)",
     "SELECT a.raw_close / b.raw_close FROM prices a, prices b WHERE a.symbol='AAPL' AND a.date='2014-06-06' "
     "AND b.symbol='AAPL' AND b.date='2014-06-09'", 7.0, 0.15),
    ("AAPL split 2014-06-09 ratio",
     "SELECT value FROM actions WHERE symbol='AAPL' AND date='2014-06-09' AND kind='split'", 7.0, 1e-9),
    ("NVDA raw close 2024-06-07 / 2024-06-10 (the 10:1 split)",
     "SELECT a.raw_close / b.raw_close FROM prices a, prices b WHERE a.symbol='NVDA' AND a.date='2024-06-07' "
     "AND b.symbol='NVDA' AND b.date='2024-06-10'", 10.0, 0.15),
    ("NVDA split 2024-06-10 ratio",
     "SELECT value FROM actions WHERE symbol='NVDA' AND date='2024-06-10' AND kind='split'", 10.0, 1e-9),
    ("TSLA raw close 2022-08-24 / 2022-08-25 (the 3:1 split)",
     "SELECT a.raw_close / b.raw_close FROM prices a, prices b WHERE a.symbol='TSLA' AND a.date='2022-08-24' "
     "AND b.symbol='TSLA' AND b.date='2022-08-25'", 3.0, 0.15),
    ("TSLA split 2022-08-25 ratio",
     "SELECT value FROM actions WHERE symbol='TSLA' AND date='2022-08-25' AND kind='split'", 3.0, 1e-9),
    ("GOOGL raw close 2022-07-15 / 2022-07-18 (the 20:1 split)",
     "SELECT a.raw_close / b.raw_close FROM prices a, prices b WHERE a.symbol='GOOGL' AND a.date='2022-07-15' "
     "AND b.symbol='GOOGL' AND b.date='2022-07-18'", 20.0, 0.15),
    ("GOOGL split 2022-07-18 ratio",
     "SELECT value FROM actions WHERE symbol='GOOGL' AND date='2022-07-18' AND kind='split'", 20.0, 1e-9),
    ("^GSPC 2020 low on 2020-03-23 (1 = yes)",
     "SELECT (SELECT date FROM prices WHERE symbol='^GSPC' AND date BETWEEN '2020-01-01' AND '2020-12-31' "
     "ORDER BY close LIMIT 1) = '2020-03-23'", 1.0, 1e-9),
    ("DGS10 on 2023-10-19",
     "SELECT value FROM fred WHERE series_id='DGS10' AND date='2023-10-19'", 4.99, 0.005),
    ("NVIDIA us-gaap Goodwill, period end 2022-01-30, FY 10-K (USD)",
     "SELECT val FROM facts WHERE cik=1045810 AND taxonomy='us-gaap' AND tag='Goodwill' "
     "AND \"end\"='2022-01-30' AND form='10-K' AND fy=2022 AND fp='FY'", 4349e6, 1e-9),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    con = sqlite3.connect(args.db, timeout=600)
    results, n_fail = [], 0
    for name, sql, exp, tol in CHECKS:
        row = con.execute(sql).fetchone()
        got = row[0] if row else None
        ok = got is not None and abs(got - exp) <= tol * abs(exp)
        n_fail += not ok
        rel = None if got is None else (got - exp) / exp
        results.append({"check": name, "expected": exp, "got": got,
                        "rel_err": rel, "pass": ok})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: expected {exp:,.4f} got "
              f"{'None' if got is None else f'{got:,.4f}'}"
              + ("" if rel is None else f" (rel err {rel:+.4%})"))
    if not args.no_write:
        con.execute("INSERT OR REPLACE INTO meta VALUES('validation', ?)",
                    (json.dumps(results),))
        con.commit()
    print(f"{len(CHECKS) - n_fail}/{len(CHECKS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
