"""The market environment kind ("market"): questions answered from the offline market_v1 snapshot (docs/03_harness.md
section 9; research/log/2026-10-31-0150.md, W4).

A task is {"task_id", "env": "market", "prompt", "keys": [...]} plus "db" (the snapshot's path; run.py's --db when
left out) and, for the result rows, "instance", "split", "type", "template" and "feasible" when it has them.

The environment opens the snapshot read-only (harness/envs/market_query.py holds the queries and their conventions),
so episodes never change it and need no copy. The tools are read-only lookups that page long results, and run_python,
which runs the model's code in a separate process (harness/envs/market_sandbox.py) with a preloaded `data` object:
inside bubblewrap when it works on the host (no network, the whole file system read-only except a fresh work
directory, its own process ids), in any case with sockets blocked and writes outside the work directory refused in
the prelude, proxy variables cleared, an address-space limit (MEMORY_BYTES) and a timeout (TIMEOUT_S). run_python is
the kind's one ASYNC_TOOLS entry, so the loop runs it in a worker thread and the other episodes go on meanwhile; it
touches only its own subprocess and work directory, never the shared snapshot connection.

H1's verification of a finish answer (unsupported_in below, research/log/2026-10-31-0150.md, W7): a number written
in the answer is supported when some number in an earlier tool result (run_python's printed output included) equals
it, in absolute value, after scaling by 1, 1e3, 1e6 or 1e9 (either way) and/or by 100, and rounding to the decimals the
answer writes; numbers inside dates (2022-03-31, March 31, 2022, March 2022, 2022-09, FY2022, Q3 2023, a year) and
numbers that the question itself contains are not checked. So "$4.349 billion" is supported by 4349000000, "9.84%" by
a printed 0.09843, and a computed value by run_python printing it; the message asks for exactly that.

Grading is grade_keys over the task's keys ("number" with "tol" or "rel_tol", "date", "text" or "ticker", "set"),
plus an optional "aliases" list on a text or ticker key (any of value and aliases counts), also inside a set, an
optional "sign_free": true on a number key (absolute values compared), and dates also read in the YYYY/M/D form.
There are no writes, so the constraints part has nothing to check.
"""
import bisect
import datetime
import functools
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile

from harness.envs import base
from harness.envs.base import DATE, FINISH, I, S, ToolError, schema
from harness.envs.market_query import QueryError, Snapshot
from harness.envs.wealth import grade

SANDBOX_PYTHON = os.environ.get("MARKET_SANDBOX_PYTHON", os.environ.get("FP_VENVS", "<FP_VENVS unset>") + "/mkt/bin/python")
HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX, QUERY = os.path.join(HERE, "market_sandbox.py"), os.path.join(HERE, "market_query.py")
TIMEOUT_S = 30
MEMORY_BYTES = 3 << 30
STDOUT_CHARS, STDERR_CHARS = 6000, 2000
PAGE_FRED, PAGE_ACTIONS, PAGE_FACTS, PAGE_SEARCH = 200, 150, 80, 20

SYSTEM = """You are a financial data analyst answering one question from an offline database that covers 2000-01-01 to 2025-10-30: daily prices of US equities, major indices, ETFs, continuous futures, FX pairs and crypto; FRED economic series; and SEC XBRL facts from company filings. Look every value up with the tools rather than from memory, and compute with run_python, printing the final values rounded as you will give them. State an assumption only when the question leaves a convention open (such as the price adjustment or fiscal versus calendar year). Write decreases, losses and outflows as negative numbers unless the question asks for a magnitude, and give dates as YYYY-MM-DD. Act without asking questions. End with finish, giving the answer in the units and rounding the question asks for. Keep it short."""

ADJ_DOC = ("adjustment: 'raw' = unadjusted, the prices actually traded that day (default); 'split' = adjusted for "
           "every split up to 2025-10-30 but not for dividends; 'total' = adjusted for splits and dividends (Yahoo "
           "adjusted close; use it for total return); on 2025-10-30 split and total equal the price of that day.")
SEC_DOC = ("Values are in base units (USD, shares, USD per share), never thousands or millions. fy and fp describe "
           "the filing that reported the value (fy 2023 fp Q1 = the first 10-Q of fiscal 2023), not the value's own "
           "period, which is start..end (start is null for balance-sheet values; months = length of the period).")
TOOLS = [
    schema("search_securities",
           "Find securities by ticker or name. Returns symbol, name, kind (equity, index, etf, future, fx, crypto), "
           "exchange, currency, first and last date with data, and CIK (SEC id) for companies. Symbols are Yahoo's: "
           "^GSPC (S&P 500), ^IXIC, ^DJI, EURUSD=X, JPY=X, GC=F (continuous gold future), BTC-USD, BRK-B.",
           {"query": S, "kind": {"type": "string", "enum": ["equity", "index", "etf", "future", "fx", "crypto"]}},
           ["query"]),
    schema("get_prices",
           "Price history of one symbol from start to end (inclusive; only trading days have rows; an end after "
           "2025-10-30 is cut there). " + ADJ_DOC + " interval: 'day' (default), 'week' or 'month': a bar has the "
           "open of its first trading day, the highest high, the lowest low, the close of its last trading day and "
           "the summed volume, and is labelled by its last trading day; bars are built from the days inside "
           "start..end, so a bar cut by the range is listed in partial_bars. Returns columns, rows, total_rows and "
           "pages; a page has up to 200 rows with one or two fields, fewer with more fields, so ask only for the "
           "fields you need. For many rows or any calculation, use data.get_prices in run_python instead.",
           {"symbol": S, "start": DATE, "end": DATE,
            "adjustment": {"type": "string", "enum": ["raw", "split", "total"]},
            "fields": {"type": "array", "items": {"type": "string", "enum": ["open", "high", "low", "close", "volume"]},
                       "description": "default: all five"},
            "interval": {"type": "string", "enum": ["day", "week", "month"]}, "page": I},
           ["symbol", "start", "end"]),
    schema("get_corporate_actions",
           "Splits and dividends of one symbol, by ex-date (default: all dates). A split has ratio = new shares per "
           "old share (4.0 for 4-for-1, 0.2 for 1-for-5), Yahoo's ratio_text, and split_type: split, reverse_split, "
           "or adjustment (a spin-off or stock dividend Yahoo records as a split; not a real split). A dividend has "
           "amount_raw (cash per share as paid then) and amount_split_adjusted (per share after later splits, "
           "comparable with split-adjusted prices).",
           {"symbol": S, "start": DATE, "end": DATE, "kind": {"type": "string", "enum": ["all", "split", "dividend"]},
            "page": I},
           ["symbol"]),
    schema("trading_day",
           "The trading day of a symbol that rule gives for a date ('on_or_before', default: the date itself if it "
           "is a trading day, else the last trading day before it; 'on_or_after': the next one), then moved by "
           "offset trading days (1 = the next trading day, -1 = the previous one). Use it for weekends, holidays "
           "and 'N trading days after'. Returns trading_day, base_trading_day and date_is_trading_day.",
           {"symbol": S, "date": DATE, "offset": {"type": "integer"},
            "rule": {"type": "string", "enum": ["on_or_before", "on_or_after"]}},
           ["symbol", "date"]),
    schema("count_trading_days",
           "The number of trading days of a symbol from start to end, both included, with the first and last.",
           {"symbol": S, "start": DATE, "end": DATE}, ["symbol", "start", "end"]),
    schema("search_fred",
           "Find FRED economic series by id or title words. Returns series_id, title, units, frequency, seasonal "
           "adjustment and the date range.",
           {"query": S}, ["query"]),
    schema("get_fred",
           "Observations of one FRED series from start to end (inclusive; an observation is dated by the start of "
           "its period, e.g. 2023-07-01 for July or Q3 2023), with title, units, frequency and seasonal "
           "adjustment. vintage: 'as_of_cutoff' (default: the values as published on 2025-10-30, what was known "
           "then) or 'latest' (today's values, including revisions and observations published after 2025-10-30). "
           "Up to 200 rows per page.",
           {"series_id": S, "start": DATE, "end": DATE,
            "vintage": {"type": "string", "enum": ["as_of_cutoff", "latest"]}, "page": I},
           ["series_id", "start", "end"]),
    schema("search_financial_tags",
           "Find the XBRL tags a company reports (us-gaap, ifrs-full for IFRS filers such as TSM, dei for cover data "
           "such as EntityCommonStockSharesOutstanding). company: ticker or name. query: words such as 'goodwill' "
           "or 'operating income'. Returns tag, label, units, number of facts and the first and last period end, "
           "best matches first, 20 per page.",
           {"company": S, "query": S, "page": I}, ["company", "query"]),
    schema("get_financials",
           "Reported values of one XBRL tag of a company (ticker or name; predecessor registrants included). "
           + SEC_DOC + " filing: 'latest' (default) gives one row per period (start, end) from the latest filing "
           "that reports it, so a restated value replaces the original; 'original' gives the first filing that "
           "reported it (the value as originally published); 'all' lists every filing. Each row names its filing "
           "(form, filed date, accession number accn). fiscal_year, period (FY, Q1, Q2, Q3) and form (10-K, 10-Q, "
           "20-F, ...) filter by the filing; e.g. fiscal_year=2022 form='10-K' gives the FY2022 annual report's "
           "values, including its prior-year comparatives (check end). Fiscal years often differ from calendar "
           "years (NVIDIA's FY2022 ended 2022-01-30). 80 rows per page.",
           {"company": S, "tag": {"type": "string", "description": "e.g. Goodwill or us-gaap:Goodwill"},
            "fiscal_year": {"type": "integer"}, "period": {"type": "string", "enum": ["FY", "Q1", "Q2", "Q3"]},
            "form": S, "filing": {"type": "string", "enum": ["latest", "original", "all"]}, "page": I},
           ["company", "tag"]),
    schema("run_python",
           "Run Python code in a fresh process (nothing is kept between calls: load the data again each time) and "
           "return its printed output; the value of a last line that is an expression is printed too. Preloaded: "
           "pd (pandas), np (numpy) and data, whose methods mirror the tools but return all rows as pandas objects, "
           "with no paging: data.get_prices(symbol, start=None, end=None, adjustment='raw', fields=None, "
           "interval='day') -> DataFrame indexed by date; data.get_corporate_actions(symbol, start=None, end=None, "
           "kind='all'); data.trading_day(symbol, date, offset=0, rule='on_or_before') -> 'YYYY-MM-DD'; "
           "data.count_trading_days(symbol, start, end) -> int; data.trading_days(symbol, start=None, end=None) -> "
           "DatetimeIndex; data.search_securities(query, kind=None); data.search_fred(query); "
           "data.get_fred(series_id, start=None, end=None, vintage='as_of_cutoff') -> DataFrame with column value; "
           "data.search_financial_tags(company, query); data.get_financials(company, tag, fiscal_year=None, "
           "period=None, form=None, filing='latest'). The same conventions hold (adjustment, base units, fy/fp "
           "of the filing). Limits: 30 seconds, no network, files only in the working directory; print only what "
           "you need, since long output is cut.",
           {"code": S}, ["code"]),
    FINISH,
]
EXAMPLES = {
    "search_securities": {"query": "apple"},
    "get_prices": {"symbol": "AAPL", "start": "2020-08-24", "end": "2020-08-28", "adjustment": "raw",
                   "fields": ["close"]},
    "get_corporate_actions": {"symbol": "AAPL", "start": "2020-01-01", "end": "2020-12-31", "kind": "split"},
    "trading_day": {"symbol": "^GSPC", "date": "2020-08-29", "offset": 0, "rule": "on_or_before"},
    "count_trading_days": {"symbol": "^GSPC", "start": "2020-01-01", "end": "2020-12-31"},
    "search_fred": {"query": "10-year treasury"},
    "get_fred": {"series_id": "DGS10", "start": "2023-10-16", "end": "2023-10-20"},
    "search_financial_tags": {"company": "NVDA", "query": "goodwill"},
    "get_financials": {"company": "NVDA", "tag": "Goodwill", "fiscal_year": 2022, "form": "10-K"},
    "run_python": {"code": "df = data.get_prices('AAPL', '2020-01-01', '2020-12-31', adjustment='total')\n"
                           "print(round((df.close.iloc[-1] / df.close.iloc[0] - 1) * 100, 2))"},
    "finish": {"answer": "..."},
}


@functools.lru_cache(maxsize=4)
def snapshot(path):
    """One open snapshot per path in this process, shared by the episodes (read-only)."""
    return Snapshot(path)


def page_of(rows, page, size):
    try:
        p = int(page or 1)
    except (TypeError, ValueError):
        raise ToolError(f"page must be a whole number from 1; got {page!r}") from None
    pages = max(1, math.ceil(len(rows) / size))
    if not 1 <= p <= pages:
        raise ToolError(f"page {p} does not exist; there {'is' if pages == 1 else 'are'} {pages} page(s)")
    info = {"total_rows": len(rows), "page": p, "pages": pages}
    if p < pages:
        info["next"] = f"page={p + 1} for more rows"
    return rows[(p - 1) * size:p * size], info


def compact(d):
    return {k: v for k, v in d.items() if v not in (None, [], "")}


@functools.lru_cache(maxsize=1)
def bwrap_works():
    exe = shutil.which("bwrap")
    if exe is None:
        return False
    try:
        r = subprocess.run([exe, "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-net", "true"], capture_output=True,
                           timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def clip(text, n, keep_tail):
    if len(text) <= n:
        return text
    if keep_tail:
        return f"[first {len(text) - n:,} characters cut]\n" + text[-n:]
    head, tail = n * 2 // 3, n // 3
    return text[:head] + f"\n[... {len(text) - head - tail:,} characters cut; print less ...]\n" + text[-tail:]


def run_sandboxed(code, db_path, timeout=TIMEOUT_S, memory=MEMORY_BYTES):
    """(return code, stdout, stderr) of the code in the sandbox; return code None on timeout."""
    work = tempfile.mkdtemp(prefix="market_py_")
    env = {"PATH": "/usr/bin:/bin", "HOME": work, "TMPDIR": work, "LANG": "C.UTF-8", "MPLCONFIGDIR": work,
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    cmd = [SANDBOX_PYTHON, "-I", "-B", SANDBOX, os.path.abspath(db_path), QUERY, work]
    if bwrap_works():
        cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--bind", work, work,
               "--unshare-net", "--unshare-pid", "--unshare-ipc", "--die-with-parent", "--new-session",
               "--chdir", work] + cmd

    def limits():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout + 5, timeout + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=work,
                             env=env, preexec_fn=limits, start_new_session=True)
        try:
            out, err = p.communicate(code.encode("utf-8"), timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
            rc = None
        return rc, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    finally:
        shutil.rmtree(work, ignore_errors=True)


class MarketEnv(base.Env):
    def __init__(self, db_path):
        if not db_path or not os.path.exists(db_path):
            raise FileNotFoundError(f"market snapshot not found: {db_path!r} (give the task a \"db\" or run.py --db)")
        self.db_path = db_path
        self.snap = snapshot(os.path.abspath(db_path))

    def call(self, fn, *a):
        try:
            return fn(*a)
        except QueryError as e:
            raise ToolError(str(e)) from None

    def search_securities(self, query, kind=None, page=1):
        hits = self.call(self.snap.search_securities, query, kind)
        if not hits:
            raise ToolError(f"no security matches {query!r}" + (f" with kind {kind!r}" if kind else "") +
                            "; try a ticker, fewer words, or another kind")
        rows, info = page_of(hits, page, PAGE_SEARCH)
        return {"results": [compact(dict(r)) for r in rows], **info}

    def get_prices(self, symbol, start, end, adjustment="raw", fields=None, interval="day", page=1):
        r = self.call(self.snap.prices, symbol, start, end, adjustment, fields, interval)
        size = min(200, 6500 // (16 + 10 * (len(r["columns"]) - 1)))
        rows, info = page_of(r["rows"], page, size)
        return compact({"symbol": r["symbol"], "name": r["name"], "currency": r["currency"],
                        "adjustment": r["adjustment"], "interval": r["interval"], "columns": r["columns"], "rows": rows,
                        **info, "partial_bars": r["partial_bars"], "note": r["note"]}) | {"rows": rows}

    def get_corporate_actions(self, symbol, start=None, end=None, kind="all", page=1):
        r = self.call(self.snap.actions, symbol, start, end, kind)
        both = [("split", x) for x in r["splits"]] + [("dividend", x) for x in r["dividends"]]
        both.sort(key=lambda x: x[1][0])
        rows, info = page_of(both, page, PAGE_ACTIONS)
        out = {"symbol": r["symbol"], "start": r["start"], "end": r["end"]}
        if r["splits"] or kind in ("split", "all", None):
            out["splits"] = {"columns": ["date", "ratio", "ratio_text", "split_type"],
                             "rows": [x for k, x in rows if k == "split"]}
        if r["dividends"] or kind in ("dividend", "all", None):
            out["dividends"] = {"columns": ["date", "amount_raw", "amount_split_adjusted"],
                                "rows": [x for k, x in rows if k == "dividend"]}
        out |= {"total_splits": len(r["splits"]), "total_dividends": len(r["dividends"]), **info}
        out.pop("total_rows")
        return compact(out | ({"note": r["note"]} if r["note"] else {}))

    def trading_day(self, symbol, date, offset=0, rule="on_or_before"):
        return self.call(self.snap.trading_day, symbol, date, offset, rule)

    def count_trading_days(self, symbol, start, end):
        return compact(self.call(self.snap.count_trading_days, symbol, start, end))

    def search_fred(self, query, page=1):
        hits = self.call(self.snap.search_fred, query)
        if not hits:
            raise ToolError(f"no FRED series matches {query!r}; try fewer or other words, or a series id")
        rows, info = page_of(hits, page, PAGE_SEARCH)
        return {"results": [compact(dict(r)) for r in rows], **info}

    def get_fred(self, series_id, start, end, vintage="as_of_cutoff", page=1):
        r = self.call(self.snap.fred, series_id, start, end, vintage)
        rows, info = page_of(r["rows"], page, PAGE_FRED)
        head = {k: r[k] for k in ("series_id", "title", "units", "frequency", "seasonal_adjustment", "vintage")}
        return compact(head | {"columns": ["date", "value"], "rows": rows, **info, "note": r["note"]}) | {"rows": rows}

    def search_financial_tags(self, company, query, page=1):
        tick, regs, hits = self.call(self.snap.search_financial_tags, company, query)
        if not hits:
            raise ToolError(f"no tag of {tick} matches {query!r}; try other words (e.g. 'revenue', 'assets', 'shares')")
        rows, info = page_of(hits, page, PAGE_SEARCH)
        return {"ticker": tick, "registrants": [r["name"] for r in regs],
                "results": [compact(dict(r)) for r in rows], **info}

    def get_financials(self, company, tag, fiscal_year=None, period=None, form=None, filing="latest", page=1):
        r = self.call(self.snap.financials, company, tag, fiscal_year, period, form, filing)
        rows, info = page_of(r["rows"], page, PAGE_FACTS)
        fl = str(filing or "latest").strip().lower()
        note = ("values in base units; fy/fp/form/filed/accn describe the filing; " +
                {"all": "every filing listed", "original": "one row per period, from the first filing that reported it"
                 }.get(fl, "one row per period, from the latest filing that reports it"))
        return compact({k: r[k] for k in ("ticker", "registrants", "tag", "taxonomy", "label")} |
                       {"note": note, "columns": r["columns"], "rows": rows, **info})

    def run_python(self, code):
        if not isinstance(code, str) or not code.strip():
            raise ToolError("code is empty; pass Python source, e.g. print(data.get_prices('AAPL', '2020-08-28', "
                            "'2020-08-28'))")
        rc, out, err = run_sandboxed(code, self.db_path)
        out_c, err_c = clip(out, STDOUT_CHARS, False), clip(err, STDERR_CHARS, True)
        if rc is None:
            raise ToolError(f"the code ran longer than {TIMEOUT_S} seconds and was stopped; load less data or avoid "
                            f"slow loops (use pandas operations)" + (f"\nstdout before the stop:\n{out_c}" if out else ""))
        if rc != 0:
            what = {1: "the code raised an error"}.get(rc, f"the process ended with status {rc}" +
                                                       (" (killed: out of memory or CPU time?)" if rc < 0 or rc > 128 else ""))
            raise ToolError(f"{what}:\n{err_c.strip()}" + (f"\nstdout before the error:\n{out_c}" if out.strip() else ""))
        res = {"stdout": out_c if out else "(nothing printed; use print() or end with an expression)"}
        if err.strip():
            res["stderr"] = err_c
        return res


SLASH_DATE = re.compile(r"(?<![\d/])(\d{4})/(\d{1,2})/(\d{1,2})(?![\d/])")


def dates(text):
    """grade.dates plus the YYYY/M/D form ("2023/1/10")."""
    out = grade.dates(text)
    for m in SLASH_DATE.finditer(text or ""):
        try:
            out.append(datetime.date(*map(int, m.groups())))
        except ValueError:
            pass
    return out


def key_found(key, text, nums):
    """grade.key_found with "aliases" on text and ticker keys (also inside a set; a bare string is a text key),
    "sign_free" on number keys (absolute values compared) and dates also written YYYY/M/D."""
    if isinstance(key, str):
        key = {"kind": "text", "value": key}
    k = key["kind"]
    if k in ("text", "ticker"):
        return any(grade.word_found(v, text) for v in [key["value"], *key.get("aliases", [])])
    if k == "set":
        return all(key_found(m, text, nums) for m in key["value"])
    if k == "date":
        return datetime.date.fromisoformat(key["value"]) in dates(text)
    if key.get("sign_free") and k in grade.TOL:
        return grade.key_found(dict(key, value=abs(key["value"])), text, [abs(n) for n in nums])
    return grade.key_found(key, text, nums)


def grade_keys(keys, answer):
    """{"passed": bool, "missing_keys": [...]}: every key value is in the answer (grade.grade_keys plus aliases)."""
    text = None if answer is None else grade.normalize(answer)
    nums = grade.numbers(text)
    missing = [k for k in keys if not key_found(k, text, nums)]
    return {"passed": answer is not None and not missing, "missing_keys": missing}


MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
         r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
DATE_PARTS = re.compile("|".join([
    r"\b\d{4}-\d{1,2}-\d{1,2}\b", r"\b\d{4}/\d{1,2}/\d{1,2}\b", r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # ISO, Y/M/D, M/D/Y
    r"\b\d{4}-\d{2}\b(?!\.\d)",  # a month, 2022-09
    rf"\b{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",  # March 31, 2022
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH},?\s+\d{{4}}\b",  # 31 March 2022
    rf"\b{MONTH}\s+\d{{4}}\b", rf"\b{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b",  # March 2022, March 31
    r"\b(?:FY|CY|Q[1-4]|H[12])\s*'?\d{2,4}\b", r"\bfiscal(?:\s+year)?\s+\d{4}\b",  # FY2022, Q3 2023, fiscal 2022
    r"(?<![\d.,$])(?:19|20)\d{2}(?![\d%]|[.,]\d)",  # a year
]), re.I)
ANSWER_NUM = re.compile(r"(?<![A-Za-z_^\d.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
RESULT_NUM = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?")
SCALES = sorted({m * 10.0 ** e for m in (1, 100) for e in (-9, -6, -3, 0, 3, 6, 9)})


def written_numbers(text):
    """(value, decimals, as written) of each number in an answer, dates left out."""
    text = DATE_PARTS.sub(" ", grade.normalize(text or "").translate(grade.DASHES))
    return [(float(m[1].replace(",", "") + ("." + m[2] if m[2] else "")), len(m[2] or ""), m[0])
            for m in ANSWER_NUM.finditer(text)]


def result_numbers(texts):
    """The absolute value of every number in the tool results, sorted (a grouped 1,234 also read as 1234)."""
    out = set()
    for t in texts:
        for m in RESULT_NUM.finditer(t):
            try:
                out.add(float(m[0].replace(",", "")))
            except ValueError:
                pass
    return sorted(out)


def unsupported_in(answer, results, question):
    """The numbers of the answer (as written) that no tool result supports, by the rule in the module docstring."""
    seen = result_numbers(results)
    asked = {float(m[0].replace(",", "")) for m in RESULT_NUM.finditer(question or "")}
    bad = []
    for v, dec, written in written_numbers(answer):
        if v in asked:
            continue
        half = 0.5 * 10.0 ** -dec * (1 + 1e-9) + 1e-12
        # t * s rounds to v at dec decimals: t in [(v - half) / s, (v + half) / s]
        if not any(bisect.bisect_right(seen, (v + half) / s) > bisect.bisect_left(seen, (v - half) / s) for s in SCALES):
            bad.append(written)
    return bad


class MarketKind(base.EnvKind):
    name = "market"
    SYSTEM, TOOLS, EXAMPLES = SYSTEM, TOOLS, EXAMPLES
    ASYNC_TOOLS = ("run_python",)

    def make_env(self, task, db_path):
        return MarketEnv(task.get("db") or db_path)

    def unsupported_numbers(self, ep, answer):
        return unsupported_in(answer, ep.results, ep.task.get("prompt", ""))

    def unsupported_message(self, bad):
        return (f"these numbers in the answer match no value in any tool result, even read in thousands, millions or "
                f"billions, or as a percentage, at the answer's rounding: {', '.join(bad[:8])}; compute each with "
                f"run_python, print it, and give the printed value (dates and numbers from the question are not "
                f"checked)")

    def grade(self, task, env):
        return grade_keys(task["keys"], env.answer)


def check_answers(path):
    """Grade every task's own answer_text: (number passed, [(task_id, missing keys)] of the others)."""
    ok, bad = 0, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            g = grade_keys(t["keys"], t.get("answer_text"))
            if g["passed"]:
                ok += 1
            else:
                bad.append((t.get("task_id", t.get("prompt_id")), g["missing_keys"]))
    return ok, bad
