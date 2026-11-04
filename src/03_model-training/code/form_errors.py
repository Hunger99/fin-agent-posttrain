"""Which assistant turns of a trajectory got a form error back from a tool call, for build_agent_sft.py,
build_agent_sft_v3.py and rl_data.py (research/log/2026-11-01-0010.md, 20:2x UTC).

A form error is a call the harness cannot execute as written: a name the episode does not offer, arguments that are
not a JSON object, or arguments Python cannot bind to the tool function's parameters. Both harness versions
(src/02_harness/code/v3/harness/agent.py and harness_v2's agent.py) answer it before any tool code runs on the arguments,
as a JSON object whose first key is "error", followed by any loop hint and the checklist, in a tool message that
carries its call's tool_call_id:
  unknown_tool   {"error": "unknown tool <name>"}: a tool name the episode does not offer, after the harness has
                 read a name with a harmony suffix as the tool it starts with;
  args_not_json  {"error": "arguments are not a JSON object"};
  bad_args       {"error": "<Class>.<tool>() got an unexpected keyword argument 'x'", ...} and the other TypeErrors
                 Python raises when it binds the arguments to the tool function's parameters (a missing required
                 argument, a second value for one), where the function the message names is the tool called.
A write that the harness's schema check rejects before running it, {"status": "rejected", "error": "missing [...] /
unexpected [...]"} with a key missing or unexpected, is the same mismatch and counts as bad_args, while a write
rejected for a rule it breaks or a value it cannot take ("amount must be a positive number") and a finish that
verification rejected are not form errors and are left to each builder's rejection rule. The builders take a turn with
a form error out of the loss and keep it in the context, so the call is not learned but the recovery after it is; a
call whose arguments are not a JSON object cannot be rendered by the chat template (call_args), so they drop its
trajectory instead.
Errors from running a call keep their loss as before (whether run_python's should is left to an ablation):
run_python's report of the model's code failing, whose message starts with "the code raised an error" even when that
code raised the same TypeError; a ToolError such as "no security matches 'X'"; a value the tool or the checklist
cannot use, such as "done": 3. Only the result's leading JSON object is read, so a result whose data merely mentions
"unknown tool" does not count.
"""
import json
import re

KINDS = ("unknown_tool", "args_not_json", "bad_args")
DECODER = json.JSONDecoder()
# the first key and its string value of a json.dumps object, for a result v3 cut at its length cap mid-object
ERROR = re.compile(r'\{"error": ("(?:[^"\\]|\\.)*")\s*[,}]')
# CPython's messages when binding keyword arguments to a function fails (the same in 3.10, 3.11 and 3.12), after the
# function's qualified name, which has a <locals> part for a class defined in a function; "takes N positional
# arguments" cannot occur, since both harnesses call fn(**args)
BINDING = re.compile(r"(?:[\w<>]+\.)*(\w+)\(\) (?:got an unexpected keyword argument |missing \d+ required "
                     r"(?:positional|keyword-only) arguments?: |got multiple values for argument |got some "
                     r"positional-only arguments passed as keyword arguments: )")
SCHEMA = re.compile(r"missing \[(.*)\] / unexpected \[(.*)\]", re.S)  # valid_args() in both harnesses


def leading_object(text):
    """The JSON object a tool result starts with, or None; of a result cut off inside it, only an "error" first key."""
    s = (text or "").lstrip()
    try:
        obj = DECODER.raw_decode(s)[0]
    except ValueError:
        m = ERROR.match(s)
        return {"error": json.loads(m.group(1))} if m else None
    return obj if isinstance(obj, dict) else None


def form_error(text, call_name):
    """The kind of form error (one of KINDS) that a tool result reports for a call to call_name, or None."""
    obj = leading_object(text)
    keys = list(obj or ())
    err = obj.get("error") if obj else None
    if not isinstance(err, str):
        return None
    if keys[:2] == ["status", "error"] and obj["status"] == "rejected":
        s = SCHEMA.fullmatch(err)
        return "bad_args" if s and (s.group(1) or s.group(2)) else None
    if keys[0] != "error":
        return None
    if err.startswith("unknown tool "):
        return "unknown_tool"
    if err == "arguments are not a JSON object":
        return "args_not_json"
    b = BINDING.match(err)
    tool = re.match(r"\w*", call_name or "").group(0)  # the name as the harness executed it (harmony suffix dropped)
    if b and tool and b.group(1) == tool:
        return "bad_args"
    return None


def call_args(call):
    """A tool call's arguments as the harness reads them (none for an empty string), or None when they are not a JSON
    object, the call the harness answers with args_not_json."""
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except (TypeError, ValueError):
        return None
    return args if isinstance(args, dict) else None


def form_error_turns(msgs):
    """For each assistant message in order: the sorted kinds of form error its calls got back (empty for none)."""
    out = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        calls = m.get("tool_calls") or []
        by_id = {c["id"]: c for c in calls if c.get("id") is not None}
        kinds, j = set(), i + 1
        while j < len(msgs) and msgs[j]["role"] == "tool":
            c = by_id.get(msgs[j].get("tool_call_id"))
            if c is None and j - i - 1 < len(calls):  # no id to go by: the k-th result answers the k-th call
                c = calls[j - i - 1]
            k = form_error(msgs[j].get("content"), c["function"]["name"] if c else None)
            if k:
                kinds.add(k)
            j += 1
        out.append(sorted(kinds))
    return out


def mask_form_errors(msgs, train=None):
    """train (one flag per assistant message, None for all True) with every turn that got a form error back turned
    off, and the number of turns this turned off."""
    form = [bool(k) for k in form_error_turns(msgs)]
    base = [True] * len(form) if train is None else list(train)
    return [t and not f for t, f in zip(base, form)], sum(t and f for t, f in zip(base, form))
