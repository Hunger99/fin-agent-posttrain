"""Sampling with the reference solution's steps as a hint, for the wealth tasks gpt-oss-120b has not solved in 8 tries
(research/log/2026-10-29-1725.md).

The request is followed by the steps golden.py's policy takes for the task's template, written without any value: which
tools to call, in what order, and which rule each check serves. The model still looks up and computes every value
itself, and its finish goes through the same verification. A trajectory kept for training has the hint removed from
the request, so it is dropped when the model's own text refers to the hint (HINT_WORDS).
"""
import re

import agent

HEAD = "\n\nSteps to follow (look up and compute every value with the tools yourself):\n"
CHECK = "With the calculator, the product's share of total assets after the purchase (rule 2) and the cash share (rule 3)"
HINTS = {
    "t1_value_pnl": ["list_holdings for the customer", "get_fund for every holding, for today's NAV",
                     "calculator: market value is the sum of units times NAV; the gain is market value minus the sum of cost bases",
                     "finish with both amounts"],
    "t1_period_return": ["get_nav_history for each of the two dates", "calculator: (end NAV / start NAV - 1) * 100",
                         "finish with the return in percent"],
    "t1_prospectus": ["search_knowledge for the fund's dealing terms", "get_document for the passage <fund>-2",
                      "finish with the minimum and the settlement days it states"],
    "t1_fee_quote": ["list_holdings for the units and the purchase date", "get_fund for today's NAV",
                     "rule 5 gives the fee rate from the days held", "calculator: the fee on units times NAV, and the cash received",
                     "finish without redeeming"],
    "t1_redeem_all": ["list_holdings for the units", "get_fund to check that redemption is not suspended (rule 4)",
                      "redeem all units if allowed; otherwise make no trade and name rule 4", "finish with the cash received"],
    "t2_drawdown_fee": ["list_holdings", "fund_metrics for every holding over the stated period",
                        "take the largest max_drawdown_pct", "get_fund for its NAV; rule 5 gives the fee rate",
                        "calculator: the fee on all its units", "finish with the id, the drawdown and the fee"],
    "t2_cheapest_bond": ["get_customer for the risk tolerance",
                         "search_funds: bond, max_risk_level at the tolerance, open for subscription, sorted by management fee; the first is the fund",
                         "get_fund for its subscription fee and NAV",
                         "calculator: fee = amount * fee rate; units = (amount - fee) / NAV", "finish without buying"],
    "t2_redeem_worst": ["list_holdings", "fund_metrics for every holding over the stated period; take the lowest return_pct",
                        "get_fund to check that its redemption is not suspended (rule 4)",
                        "redeem all its units if allowed; otherwise make no trade and name rule 4", "finish"],
    "t3_subscribe": ["get_customer for the risk tolerance and cash", "list_holdings",
                     "get_fund for every holding and the product (risk level, suspension, fee, NAV)",
                     "if the customer holds none of the product, search_knowledge and get_document <product>-2 for its minimum (rule 7)",
                     CHECK, "subscribe only if rules 1, 2, 3, 4 and 7 all hold; otherwise make no trade and name the rules", "finish"],
    "t3_switch": ["get_customer for the risk tolerance and cash", "list_holdings",
                  "search_funds: bond, max_risk_level at the tolerance, open for subscription, sorted by management fee; the first is the target",
                  "get_fund for every holding and the target",
                  "if the customer holds none of the target, search_knowledge and get_document <target>-2 for its minimum (rule 7)",
                  "calculator: the redemption proceeds (rule 5 fee) and total assets after the redemption",
                  CHECK + ", both after the redemption",
                  "if every rule holds, redeem first and then subscribe; otherwise make no trade at all and name the rules",
                  "finish"],
    "t3_redeem_units": ["get_fund to check that redemption is not suspended (rule 4)",
                        "redeem the units if allowed; otherwise make no trade and name rule 4", "finish with the fee and the cash received"],
}
REVIEW = ["for each customer: get_customer, list_holdings, get_fund for every holding",
          "calculator: total assets and each holding's share of them",
          "for a holding above 30%: if its redemption is suspended, skip it and name rule 4; otherwise compute u by the "
          "request's formula with the calculator (f by rule 5) and redeem u units",
          "calculator: cash after the review", "finish with every customer's units redeemed and cash after"]
HINTS["t4_rebalance"] = HINTS["t4_rebalance_one"] = REVIEW
HINT_WORDS = re.compile(r"\b(hint|steps to follow|the steps given|as instructed above)\b", re.I)


def hint_text(template):
    return HEAD + "\n".join(f"{i}. {s}" for i, s in enumerate(HINTS[template], 1))


class HintEpisode(agent.Episode):
    """An Episode whose request carries the template's steps; the trajectory keeps them in the request, and the
    row's "hint" field marks it."""

    def __init__(self, task, *args, **kw):
        super().__init__(task, *args, **kw)
        self.msgs[1]["content"] = task["prompt"] + hint_text(task["template"])


def strip(msgs, task):
    """The trajectory for training: the request without the hint; None when the model's own text refers to it."""
    for m in msgs:
        if m["role"] == "assistant" and HINT_WORDS.search((m.get("content") or "") + " " + (m.get("reasoning_content") or "")):
            return None
    return [dict(m, content=task["prompt"]) if i == 1 else m for i, m in enumerate(msgs)]
