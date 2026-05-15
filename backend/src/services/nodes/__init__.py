"""Node subsystem with Protocol-based interfaces. See base.py.

Import each node module here so their @register(...) side effects populate
the registry. WorkflowExecutor._execute_inline_node then dispatches via
registry.get_node_class.
"""

# Eagerly import node modules so @register decorators run.
from . import audio, llm, logic, text_io  # noqa: F401
