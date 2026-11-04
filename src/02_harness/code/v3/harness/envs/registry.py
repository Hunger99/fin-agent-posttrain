"""Every environment kind by the name a task gives in its "env" field; a new kind is one module and one line here."""
from harness.envs.docfinqa import DocKind
from harness.envs.market import MarketKind
from harness.envs.ml import MLKind
from harness.envs.report_gate import GateKind
from harness.envs.wealth.kind import WealthKind

KINDS = {k.name: k for k in (WealthKind, MLKind, DocKind, GateKind, MarketKind)}


def kind_of(task):
    """The EnvKind instance for a task; a task without "env" is a wealth-management one."""
    name = task.get("env", "wealth")
    if name not in KINDS:
        raise KeyError(f"task {task.get('task_id')}: unknown environment kind {name!r}; known: {sorted(KINDS)}")
    return KINDS[name]()
