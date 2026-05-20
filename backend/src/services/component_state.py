"""PR-5a: in-memory mirror of runner component load state (spec §6.1).

The runner subprocess owns the real L1 cache; it emits ComponentEvent on every
state transition. The backend keeps this best-effort mirror so GET
/api/v1/models/components/state can answer without a blocking RPC. Unknown keys
default to 'cold'. Lost on backend restart (matches L1 being in-memory)."""
from __future__ import annotations

import time
from typing import Any


class ComponentStateRegistry:
    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    def update(self, key: str, state: str, error: str | None = None) -> None:
        self._states[key] = {"key": key, "state": state, "error": error, "updated_at": time.time()}

    def get(self, key: str) -> dict[str, Any]:
        entry = self._states.get(key)
        if entry is None:
            return {"key": key, "state": "cold", "error": None}
        return {"key": key, "state": entry["state"], "error": entry["error"]}

    def query(self, keys: list[str]) -> list[dict[str, Any]]:
        return [self.get(k) for k in keys]

    def all(self) -> list[dict[str, Any]]:
        return [{"key": k, "state": v["state"], "error": v["error"]} for k, v in self._states.items()]
