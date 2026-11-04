"""Read and score the answer of a numeric question over a report excerpt (the English domain score).

The prediction is the first number after the last "Answer:"; an exponent written right after it, as in
"3.87688e+06", is part of the number (research/log/2026-10-29-0442.md). It is correct when it, or it times 100,
or it divided by 100, is within 1% of the gold value (0.005 when the gold value is 0): FinQA writes a percentage
as a ratio (0.0676 for 6.76%) and TAT-QA as the percent figure, and an answer rounded to the excerpt's precision
should count. A yes/no gold value needs the same word. An answer with no number counts as wrong.
"""
import re

NUMBER = re.compile(r"\(?-?\$?\s*\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?\)?")
EXPONENT = re.compile(r"[eE]([+-]?\d+)")


def parse(text, gold):
    tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else ""
    if isinstance(gold, str):
        m = re.search(r"\b(yes|no)\b", tail.lower())
        return m.group(1) if m else None
    m = NUMBER.search(tail)
    if not m:
        return None
    s = m.group(0)
    e = EXPONENT.search(s)
    s = EXPONENT.sub("", s)  # before the sign test, so that the "-" of "e-05" does not read as a negative number
    neg = s.startswith("(") and s.endswith(")") or "-" in s
    s = re.sub(r"[^\d.]", "", s).rstrip(".")
    try:
        v = float(s) * 10 ** int(e.group(1)) if e else float(s)
    except (ValueError, OverflowError):
        return None
    return -v if neg else v


def correct(pred, gold):
    if pred is None:
        return False
    if isinstance(gold, str):
        return pred == gold
    tol = 0.005 if gold == 0 else 0.01 * abs(gold)
    return any(abs(p - gold) <= tol for p in (pred, pred * 100, pred / 100))
