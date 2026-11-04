"""The three hand-written market tasks of the tests, with the keys of the two price tasks read from the snapshot.

The snapshot's daily bars come from Yahoo, whose terms do not allow redistributing them, so the values those two keys
check are not written in the repository: path() reads them from market_v1.sqlite (read-only), writes the three tasks
to a temporary JSONL file once per process and returns its path. The third task's key comes from NVIDIA's 10-K (SEC
XBRL data) and is written out.
"""
import atexit
import calendar
import json
import os
import sqlite3
import tempfile

DB = os.environ.get("FP_DATA_D2", "<FP_DATA_D2 unset>") + "/market_v1/market_v1.sqlite"
_PATH = []


def _one(c, sql, *args):
    row = c.execute(sql, args).fetchone()
    if row is None:
        raise LookupError(f"no row for {sql} {args}")
    return row[0]


def hand_tasks(db=DB):
    """The three tasks: AAPL's raw and split-adjusted close on 2020-08-28, the S&P 500's largest monthly decline of
    2022, and NVDA's goodwill share of total assets at the end of fiscal 2022."""
    if not os.path.exists(db):
        raise FileNotFoundError(db)
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        q = "select %s from prices where symbol=? and date=?"
        raw = round(_one(c, q % "raw_close", "AAPL", "2020-08-28"), 2)
        split = round(_one(c, q % "close", "AAPL", "2020-08-28"), 2)  # the bars' close is split-adjusted
        ends = c.execute("select substr(date, 1, 7), max(date) from prices where symbol='^GSPC' and date between "
                         "'2021-12-01' and '2022-12-31' group by substr(date, 1, 7) order by 1").fetchall()
        closes = [(m, _one(c, q % "raw_close", "^GSPC", d)) for m, d in ends]
    finally:
        c.close()
    change = {closes[i][0]: (closes[i][1] / closes[i - 1][1] - 1) * 100 for i in range(1, len(closes))}
    month = min(change, key=change.get)
    drop = round(change[month], 2)
    name, ab = calendar.month_name[int(month[5:])], calendar.month_abbr[int(month[5:])]
    common = {"env": "market", "instance": "market_v1", "split": "hand", "feasible": True}
    return [
        dict(common, task_id="market-hand-lookup", type="T2", template="hand_lookup",
             prompt="What was Apple's (AAPL) unadjusted closing price on August 28, 2020, and what is that day's close "
                    "adjusted for later stock splits (but not dividends)? Give both in US dollars, rounded to 2 "
                    "decimals.",
             keys=[{"kind": "number", "value": raw, "tol": 0.005}, {"kind": "number", "value": split, "tol": 0.005}],
             answer_text=f"Unadjusted close: ${raw:.2f}; split-adjusted close: ${split:.2f}."),
        dict(common, task_id="market-hand-monthly", type="T3", template="hand_monthly_drop",
             prompt="In calendar year 2022, in which month did the S&P 500 index (^GSPC) have its largest monthly "
                    "decline, measured from the last close of the previous month to the last close of the month? Give "
                    "the month as YYYY-MM and the monthly change as a signed percentage rounded to 2 decimals.",
             keys=[{"kind": "text", "value": month, "aliases": [f"{name} {month[:4]}", f"{ab} {month[:4]}"]},
                   {"kind": "number", "value": drop, "tol": 0.005}],
             answer_text=f"{month}, with a change of {drop:.2f}%."),
        dict(common, task_id="market-hand-sec-ratio", type="T2", template="hand_sec_ratio",
             prompt="In NVIDIA's (NVDA) annual report for fiscal year 2022 (fiscal year ended January 30, 2022), what "
                    "percentage of total assets was goodwill at the end of the fiscal year? Round to 2 decimals.",
             keys=[{"kind": "number", "value": 9.84, "tol": 0.005}],
             answer_text="Goodwill was 9.84% of total assets ($4,349M of $44,187M)."),
    ]


def path(db=DB):
    """A JSONL file of hand_tasks(db), written once per process."""
    if not _PATH:
        fd, p = tempfile.mkstemp(prefix="market_tasks_", suffix=".jsonl")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for t in hand_tasks(db):
                f.write(json.dumps(t) + "\n")
        _PATH.append(p)
        atexit.register(os.remove, p)
    return _PATH[0]
