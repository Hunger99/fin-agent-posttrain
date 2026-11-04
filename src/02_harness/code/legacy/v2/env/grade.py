"""Grade one episode of a wealth-management task, by the harness task success rate in docs/results.md.

An executable task passes when, at the end, the cash, holdings and trades of every customer it concerns agree with the
gold state, and the final answer contains every key value: a product id verbatim, an amount within $0.01, a percentage
within 0.1 percentage points, a number of units within 0.01. A task the rules forbid (in whole, or for one customer of a
T4 review) passes when the state agrees with the gold state, in which the forbidden trade is not made, and the answer
names one of the broken rules ("rule 4"). The state comparison allows $0.06 of cash and 0.011 units, the most that a
0.01-unit rounding of a redemption moves them; trades are compared as a multiset of (product, buy or sell).

Version 2 (research/log/2026-10-29-0836.md): gpt-oss writes ids as "B‑P0023" with U+2011 (research/log/2026-10-29-0743.md),
so the answer is normalized first. The spaces U+00A0, U+2009 and U+202F become a plain space. For ids, every dash
U+2010-U+2015 and U+2212 counts as "-". For numbers, only U+2010-U+2012 and U+2212 directly before a digit or "$" become
a minus sign, since in the dev answers the en and em dashes and the spaced minus sign are punctuation before a positive
number ("B‑P0128 – 13,158.98 units", "$10,900 − $63.22"). Everything else is version 1's.
"""
import re

SPACES = dict.fromkeys(map(ord, "\u00a0\u2009\u202f"), " ")
DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
MINUS = re.compile(r"[\u2010-\u2012\u2212](?=\$?\d)")
NUM = re.compile(r"(?<![\w.])-?\$?\s?\d[\d,]*(?:\.\d+)?")
TOL = {"money": 0.01, "percent": 0.1, "units": 0.01, "number": 0.0}  # number: a count, exact


def normalize(text):
    """The answer as numbers() and names_rule() read it."""
    return MINUS.sub("-", (text or "").translate(SPACES))


def numbers(text):
    out = []
    for m in NUM.finditer(text or ""):
        s = m.group().replace("$", "").replace(",", "").replace(" ", "")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def state_diff(final, gold):
    """Differences between the final and gold states, as readable strings; empty when they agree."""
    out = []
    for c, g in gold.items():
        f = final[c]
        if abs(f["cash"] - g["cash"]) > 0.06:
            out.append(f"{c} cash {f['cash']:.2f} vs {g['cash']:.2f}")
        for p in set(f["holdings"]) | set(g["holdings"]):
            a, b = f["holdings"].get(p, 0.0), g["holdings"].get(p, 0.0)
            if abs(a - b) > 0.011:
                out.append(f"{c} {p} units {a} vs {b}")
        if sorted(map(tuple, f["transactions"])) != sorted(map(tuple, g["transactions"])):
            out.append(f"{c} trades {f['transactions']} vs {g['transactions']}")
    return out


def key_found(key, answer, nums):
    if key["kind"] == "id":
        return key["value"].translate(DASHES).lower() in (answer or "").translate(DASHES).lower()
    v, tol = key["value"], TOL[key["kind"]]
    cands = nums + ([n * 100 for n in nums] if key["kind"] == "percent" else [])  # 0.1634 for 16.34%
    return any(abs(n - v) <= tol + 1e-9 for n in cands)


def names_rule(answer, rules):
    return any(re.search(rf"\brule\s*(?:no\.?\s*|number\s*|#\s*)?{n}\b", answer or "", re.I) for n in rules)


def grade(task, final_state, answer):
    """{"passed": bool, "state": [...differences], "missing_keys": [...], "names_rule": bool or None}"""
    diff = state_diff(final_state, task["gold_state"])
    answer = None if answer is None else normalize(answer)
    nums = numbers(answer)
    missing = [k for k in task["keys"] if not key_found(k, answer, nums)]
    rule = names_rule(answer, task["violated_rules"]) if task["violated_rules"] else None
    passed = answer is not None and not diff and not missing and rule is not False
    return {"passed": passed, "state": diff, "missing_keys": missing, "names_rule": rule}
