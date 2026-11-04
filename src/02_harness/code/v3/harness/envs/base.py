"""The interface between the harness and an environment kind (docs/03_harness.md, Version 3).

An environment kind is an EnvKind subclass registered in harness/envs/registry.py under its name. A task names its kind
in the field "env" (no field: "wealth"); the harness builds one environment object per episode with
kind.make_env(task, db_path), where db_path is the task's own "db" field when it has one and run.py's --db otherwise.

What a kind declares:
- SYSTEM: the system prompt, the same in every configuration (H1 adds its checklist note after it);
- TOOLS: the tools as OpenAI function schemas (schema() below); one must be "finish" with a string "answer";
- EXAMPLES: one valid call per tool, {name: arguments}, shown with a tool error when the correction part is on;
- WRITES: the names of the tools that change the environment; the constraints part checks them with check_write()
  before they run, and the episode records each executed one as a trade;
- grade(task, env): the grade of the finished episode, a dict with at least "passed"; tasks graded by key values use
  harness.envs.wealth.grade.grade_keys(task["keys"], env.answer), whose key kinds are: "id" (verbatim), "text" or
  "ticker" (a whole word), "money", "percent", "units", "number" (each a number within a tolerance: the kind's default,
  or the key's own "tol" (absolute) and "rel_tol" (relative)), "date" (YYYY-MM-DD, found in any common writing) and
  "set" (every member found; members are keys or strings);
- optionally check_write(), rejection_details(), trade_info(), verify() and categorize(), whose defaults do nothing or
  the generic thing;
- optionally unsupported_numbers() and unsupported_message(), H1's rule for which numbers of a finish answer the tool
  results support and the message naming the others (default: the shared rule of agent.Episode.untraced, written for
  the wealth tasks and their calculator);
- optionally ASYNC_TOOLS, the names of slow tools that the loop runs in a worker thread (asyncio.to_thread) so that
  they do not stall the other episodes; such a tool must not touch an object the event loop's thread also uses
  without a lock (a tool not listed always runs on the loop's thread, so an episode's database connection is safe).

What the environment object provides: one method per tool, taking the tool's arguments as keyword arguments and
returning a JSON-serializable value; finish(answer), which sets .finished and .answer (Env below does this); and
ToolError (below) for a request it cannot serve, whose message the model gets back. The harness also turns TypeError and
ValueError (wrong or malformed arguments) and, flagged as internal errors, any other exception into a tool error, so
no environment error ends an episode or a run. The harness caps every result at its tool-result limit when it is
produced, so a tool that can return much text should page or take narrower arguments, and say how in its description.
"""
import json


class ToolError(Exception):
    """A request the environment cannot serve; its message goes back to the model as {"error": message}."""


def schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required, "additionalProperties": False}}}


S, I, N, B = {"type": "string"}, {"type": "integer", "minimum": 1}, {"type": "number"}, {"type": "boolean"}
DATE = {"type": "string", "description": "YYYY-MM-DD"}
FINISH = schema("finish", "End the task with the final answer to the request.", {"answer": S}, ["answer"])


class Env:
    """The part every environment object shares: finish() ends the episode with the answer."""

    finished, answer = False, None

    def finish(self, answer):
        self.finished, self.answer = True, str(answer)
        return {"status": "finished"}


class EnvKind:
    """One environment kind; the harness keeps one instance per episode and holds no state in it."""

    name = None
    SYSTEM = ""
    TOOLS = []
    EXAMPLES = {}
    WRITES = ()
    ASYNC_TOOLS = ()

    def make_env(self, task, db_path):
        raise NotImplementedError

    def check_write(self, env, name, args):
        """The constraints part: the rules a write would break, as [(rule number, reason)]; empty when allowed."""
        return []

    def rejection_details(self, env, name, args, broken):
        """Extra fields of a rule rejection when the correction part is on, such as the largest allowed amount."""
        return {}

    def trade_info(self, env, name, args):
        """What the kind wants recorded with a write before it runs (kept in Episode.trades with the call)."""
        return None

    def verify(self, ep, answer):
        """The kind's own checks at finish, after the shared ones: a list of problems, empty when it passes."""
        return []

    def unsupported_numbers(self, ep, answer):
        """H1's verification: the numbers written in the answer that no tool result supports (strings, as written)."""
        return ep.untraced(answer)

    def unsupported_message(self, bad):
        """The verification problem naming the unsupported numbers, for the model."""
        return (f"these numbers in the answer appear in no tool result: {', '.join(bad[:8])}; compute them with the "
                f"calculator or correct them")

    def grade(self, task, env):
        raise NotImplementedError

    def categorize(self, ep, g):
        """The main failure category of a failed episode; None when it passed."""
        if g["passed"]:
            return None
        if ep.end in ("context_overflow", "step_limit", "hard_cap"):
            return ep.end
        if max(ep.calls.values(), default=0) >= 3:
            return "loop"
        return "wrong_answer"


def to_json(x):
    return json.dumps(x, ensure_ascii=False)
