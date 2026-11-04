"""A strict reading of the market grader, which answers that list many values without the reference fail
(research/log/2026-11-01-0010.md, the grader audit of 11-02 20:0x UTC and the request of 22:3x UTC).

market.grade_keys (src/02_harness/code/v3/harness/envs/market.py, over envs/wealth/grade.py) passes an answer when every key
value occurs anywhere in it. In the audit an answer that knows no reference value (the prompt restated, the dates of
its window, and grids of the values the training tasks' keys take) passed 209 of the 939 GRPO training tasks (22.3%;
risk_rank 111/111). Here an answer passes when it passes market.grade_keys, which this module calls unchanged, and
four rules hold:
  F1   at most MAX_NUMBERS numbers outside dates (market.written_numbers);
  F2   at most MAX_DATES distinct dates (market.dates);
  F3b  every number key is found with the numeric dates (2022-03-31, 2022/3/31, 3/31/2022, 2022-09) and grade.DATES'
       month-name dates (March 31, 2022; 31 March 2022) blanked, so that a count of 3 is not read off 2021-03-19;
  F4d  on a task with 2 or more candidates (meta.params.symbols), the first candidate the answer names is a key
       ticker, and every ticker key after the first is followed within PAIR_SPAN characters, before any other
       candidate's ticker, by the number key that follows it in the key list; a candidate is named by its ticker in
       capitals as a whole word, a key ticker also by an alias in any case, and a one-letter ticker counts as named
       only when it is a key ("C" is also a word). On a task without candidates every ticker key must be written so.
The rules are checked only on an answer that market.grade_keys passes, and "failed_rules" lists the ones it breaks.

The rules and thresholds are the audit's (its followup.py, F1+F2+F3b+F4d), measured on train_market.jsonl's 2,350
tasks: all of its reference-free answers fail; of M1's 4,188 passing screening answers (screen_m1_train) 10 fail, 7
of them wrong (6 name a wrong company first and list the right one with its value later, 1 has a wrong count whose
right value is part of a date) and 3 right ones that name another candidate before the conclusion; none of the 2,350
answer_text references fails. The caps sit well above honest answers: the 2,948 answer_text references of the
train, dev, test and FinSearchComp files write at most 28 numbers and 20 distinct dates, M1's passing screening
answers 99% at most 32 numbers (one 61) and at most 21 dates, while F1, F2 and F3b together already fail all 892
sprays that passed on the GRPO training tasks. PAIR_SPAN 200 is F4d's; the audit's F4c used 80. Only ticker keys after the first are
paired: pairing the first one too fails 11 more M1 screening answers, all filing_reaction, whose first ticker key is
followed in the key list by the revenue growth, which answers often give away from the ticker.

Limits: an answer that knows a value only roughly and lists its neighbours ("10.19 (alternatives 10.11, 10.13, ...)")
still passes, on every training task in the audit; the audit's F5, which caps the numbers near a key value, catches it
but fails 14 reference answers. A non-key candidate is recognized by its ticker only, and a one-letter one not at
all, so an answer that names a wrong company first by its name alone, or by a one-letter ticker ("Pick: O" for
Realty Income when the key is PLD), passes F4d.

Usage: grade_task_strict(task, answer), or grade_keys_strict(keys, answer, candidates) as market.grade_keys is called.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "02_harness", "code", "v3"))
from harness.envs import market  # noqa: E402
from harness.envs.wealth import grade  # noqa: E402

MAX_NUMBERS = 50
MAX_DATES = 40
PAIR_SPAN = 200
RULES = ("F1", "F2", "F3b", "F4d")
NUMERIC_DATE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{4}/\d{1,2}/\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b|"
                          r"\b\d{4}-\d{2}\b(?!\.\d)")


def candidates(task):
    """The task's candidate tickers (meta.params.symbols); empty when it has none."""
    return list(((task.get("meta") or {}).get("params") or {}).get("symbols") or [])


def word(s, flags=0):
    return re.compile(rf"(?<![\w]){re.escape(s.translate(grade.DASHES))}(?![\w])", flags)


def first_mention(ticker, text, aliases=()):
    """Start of the first mention of the ticker (capitals) or an alias (any case) as a whole word, or None."""
    t = text.translate(grade.DASHES)
    found = [m.start() for m in [word(ticker).search(t)] + [word(a, re.I).search(t) for a in aliases] if m]
    return min(found, default=None)


def paired(ticker, aliases, key, text, cands):
    """Some mention of the ticker or an alias is followed within PAIR_SPAN characters, before any other candidate's
    ticker, by a number that matches the number key."""
    t = text.translate(grade.DASHES)
    others = [word(c) for c in cands if c != ticker]
    for pat in [word(ticker)] + [word(a, re.I) for a in aliases]:
        for m in pat.finditer(t):
            seg = t[m.end():m.end() + PAIR_SPAN]
            cut = min([o.search(seg).start() for o in others if o.search(seg)], default=len(seg))
            if market.key_found(key, None, grade.numbers(text[m.end():m.end() + cut])):
                return True
    return False


def numbers_outside_dates_ok(answer):
    """F1."""
    return len(market.written_numbers(answer)) <= MAX_NUMBERS


def distinct_dates_ok(text):
    """F2."""
    return len(set(market.dates(text))) <= MAX_DATES


def number_keys_outside_dates_ok(keys, text):
    """F3b. The spans are found on the dash-normalized copy, which has the text's length."""
    chars, t = list(text), text.translate(grade.DASHES)
    spans = [m.span() for m in NUMERIC_DATE.finditer(t)] + [m.span() for p, _ in grade.DATES for m in p.finditer(t)]
    for s, e in spans:
        chars[s:e] = " " * (e - s)
    nums = grade.numbers("".join(chars))
    return all(market.key_found(k, text, nums) for k in keys if isinstance(k, dict) and k["kind"] == "number")


def pick_ok(keys, text, cands):
    """F4d."""
    tk = [i for i, k in enumerate(keys) if isinstance(k, dict) and k["kind"] == "ticker"]
    if len(cands) < 2:
        return all(first_mention(keys[i]["value"], text, keys[i].get("aliases", ())) is not None for i in tk)
    if not tk:
        return True
    al = {keys[i]["value"]: keys[i].get("aliases", ()) for i in tk}
    pos = {}
    for s in cands:
        if len(s) > 1 or s in al:
            p = first_mention(s, text, al.get(s, ()))
            if p is not None:
                pos[s] = p
    if not pos or min(pos, key=pos.get) not in al:
        return False
    for i in tk[1:]:
        k = keys[i]
        nxt = keys[i + 1] if i + 1 < len(keys) and isinstance(keys[i + 1], dict) and keys[i + 1]["kind"] == "number" \
            else None
        if nxt is None:
            if first_mention(k["value"], text, k.get("aliases", ())) is None:
                return False
        elif not paired(k["value"], k.get("aliases", ()), nxt, text, cands):
            return False
    return True


def grade_keys_strict(keys, answer, cands=()):
    """{"passed", "missing_keys", "failed_rules"}: market.grade_keys' verdict and missing keys, with "passed" also
    requiring every rule of RULES; "failed_rules" is empty when market.grade_keys fails."""
    g = market.grade_keys(keys, answer)
    failed = []
    if g["passed"]:
        text = grade.normalize(answer)
        ok = {"F1": numbers_outside_dates_ok(answer), "F2": distinct_dates_ok(text),
              "F3b": number_keys_outside_dates_ok(keys, text), "F4d": pick_ok(keys, text, list(cands))}
        failed = [r for r in RULES if not ok[r]]
    return {"passed": g["passed"] and not failed, "missing_keys": g["missing_keys"], "failed_rules": failed}


def grade_task_strict(task, answer):
    return grade_keys_strict(task["keys"], answer, candidates(task))
