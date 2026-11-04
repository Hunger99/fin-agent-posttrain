"""The child process of the market kind's run_python tool; harness/envs/market.py runs it, never imports it.

Usage (by run_python only): <python with pandas> -I -B market_sandbox.py <snapshot.sqlite> <market_query.py> <work dir>
with the model's code on stdin. The prelude blocks sockets, clears proxy variables, refuses writes outside the work
directory, opens the snapshot read-only, and runs the code with `data` (the snapshot's queries as DataFrames), `pd`
and `np`; a last line that is an expression is printed, as in a notebook. An exception prints its traceback (without
the prelude's frames) on stderr and exits with status 1. The parent adds the hard limits: a bubblewrap sandbox with
no network and a read-only file system where available, a memory limit, and the timeout.
"""
import ast
import builtins
import importlib.util
import io
import os
import socket
import sys
import traceback

ME = os.path.abspath(__file__)


def block_network():
    def refuse(*a, **k):
        raise OSError("network access is disabled in run_python")

    class NoSocket(socket.socket):
        def __init__(self, *a, **k):
            refuse()

    socket.socket = NoSocket
    for name in ("create_connection", "create_server", "getaddrinfo", "gethostbyname", "socketpair", "fromfd"):
        setattr(socket, name, refuse)
    import _socket
    _socket.socket = NoSocket
    for k in list(os.environ):
        if k.lower() in ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy"):
            del os.environ[k]


def guard_writes(work):
    work = os.path.realpath(work)

    def inside(p):
        if isinstance(p, int):
            return True
        p = os.fsdecode(os.fspath(p))
        r = os.path.realpath(p)
        return r == work or r.startswith(work + os.sep)

    def refuse(p):
        raise PermissionError(f"run_python can write only inside its working directory; not {os.fsdecode(os.fspath(p))!r}")

    real_open = builtins.open

    def g_open(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+") and not inside(file):
            refuse(file)
        return real_open(file, mode, *a, **k)

    builtins.open = io.open = g_open
    real_os_open = os.open
    W = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def g_os_open(path, flags, *a, **k):
        if flags & W and not inside(path):
            refuse(path)
        return real_os_open(path, flags, *a, **k)

    os.open = g_os_open

    def wrap(fn, n_paths):
        def g(*a, **k):
            for p in a[:n_paths]:
                if not inside(p):
                    refuse(p)
            return fn(*a, **k)
        return g

    for name, n in (("remove", 1), ("unlink", 1), ("rmdir", 1), ("mkdir", 1), ("chmod", 1), ("truncate", 1),
                    ("rename", 2), ("replace", 2), ("symlink", 2), ("link", 2), ("utime", 1)):
        if hasattr(os, name):
            setattr(os, name, wrap(getattr(os, name), n))
    import sqlite3
    real_connect = sqlite3.connect

    def g_connect(database, *a, **k):
        if str(database) != ":memory:" and not inside(str(database)):
            refuse(database)
        return real_connect(database, *a, **k)

    sqlite3.connect = g_connect


def make_data(mq, snap, pd):
    def frame(cols, rows, index="date"):
        df = pd.DataFrame(rows, columns=cols)
        if index in df.columns:
            df[index] = pd.to_datetime(df[index])
            df = df.set_index(index)
        return df

    class Data:
        """The snapshot's queries as DataFrames (no paging): see the run_python tool's description."""

        def search_securities(self, query, kind=None):
            return pd.DataFrame(snap.search_securities(query, kind))

        def get_prices(self, symbol, start=None, end=None, adjustment="raw", fields=None, interval="day"):
            r = snap.prices(symbol, start, end, adjustment, fields, interval, required=False)
            return frame(r["columns"], r["rows"])

        def get_corporate_actions(self, symbol, start=None, end=None, kind="all"):
            r = snap.actions(symbol, start, end, kind)
            rows = [{"date": d, "kind": "split", "ratio": v, "ratio_text": t, "split_type": st} for d, v, t, st in r["splits"]]
            rows += [{"date": d, "kind": "dividend", "amount_raw": a, "amount_split_adjusted": b} for d, a, b in r["dividends"]]
            cols = ["date", "kind", "ratio", "ratio_text", "split_type", "amount_raw", "amount_split_adjusted"]
            return frame(cols, sorted(([x.get(c) for c in cols] for x in rows), key=lambda x: x[0]))

        def trading_day(self, symbol, date, offset=0, rule="on_or_before"):
            return snap.trading_day(symbol, date, offset, rule)["trading_day"]

        def count_trading_days(self, symbol, start, end):
            return snap.count_trading_days(symbol, start, end)["count"]

        def trading_days(self, symbol, start=None, end=None):
            s, e, _ = snap.window(start, end, required=False)
            return pd.DatetimeIndex([d for d in snap.trading_days(snap.symbol(symbol)) if s <= d <= e], name="date")

        def search_fred(self, query):
            return pd.DataFrame(snap.search_fred(query))

        def get_fred(self, series_id, start=None, end=None, vintage="as_of_cutoff"):
            r = snap.fred(series_id, start, end, vintage, required=False)
            return frame(["date", "value"], r["rows"])

        def search_financial_tags(self, company, query=""):
            return pd.DataFrame(snap.search_financial_tags(company, query)[2])

        def get_financials(self, company, tag, fiscal_year=None, period=None, form=None, filing="latest"):
            r = snap.financials(company, tag, fiscal_year, period, form, filing)
            df = pd.DataFrame(r["rows"], columns=r["columns"])
            for c in ("start", "end", "filed"):
                df[c] = pd.to_datetime(df[c])
            return df

        def __repr__(self):
            return ("data: get_prices, get_corporate_actions, trading_day, count_trading_days, trading_days, "
                    "search_securities, search_fred, get_fred, search_financial_tags, get_financials")

    return Data()


def run(code, ns):
    tree = ast.parse(code, "<run_python>")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, "<run_python>", "exec"), ns)
    if last is not None:
        val = eval(compile(ast.Expression(last.value), "<run_python>", "eval"), ns)
        if val is not None:
            print(repr(val))


def main():
    db, query_path, work = sys.argv[1:4]
    code = sys.stdin.read()
    block_network()
    spec = importlib.util.spec_from_file_location("market_query", query_path)
    mq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mq)
    snap = mq.Snapshot(db)
    import numpy as np
    import pandas as pd
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 60)
    ns = {"__name__": "__main__", "data": make_data(mq, snap, pd), "pd": pd, "np": np}
    guard_writes(work)
    try:
        run(code, ns)
    except SystemExit as e:
        sys.stdout.flush()
        os._exit(e.code if isinstance(e.code, int) else (0 if e.code is None else 1))
    except BaseException as e:
        # the model's own frames and those of the libraries it called; the prelude's and the queries' are noise
        te = traceback.TracebackException(type(e), e, e.__traceback__)
        own = (ME, os.path.abspath(query_path))
        te.stack = traceback.StackSummary.from_list(
            [f for f in te.stack if os.path.abspath(f.filename) not in own and not f.filename.endswith(os.sep + "ast.py")])
        sys.stdout.flush()
        print("".join(te.format()).replace("market_query.QueryError", "QueryError"), end="", file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
