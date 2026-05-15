"""Registry mapping node type string → class.

Populated by each node module's import (side effect). Lookup is done by
WorkflowExecutor._execute_inline_node (Lane S split: dispatch nodes go
to RunnerClient, inline nodes resolve through this registry).
"""

from __future__ import annotations

_NODE_CLASSES: dict[str, type] = {}


def register(node_type: str):
    """Decorator to register a node class."""
    def _inner(cls: type) -> type:
        _NODE_CLASSES[node_type] = cls
        return cls
    return _inner


def get_node_class(node_type: str) -> type | None:
    return _NODE_CLASSES.get(node_type)


def all_registered() -> dict[str, type]:
    return dict(_NODE_CLASSES)
