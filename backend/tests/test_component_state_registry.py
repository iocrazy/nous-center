"""PR-5a: ComponentStateRegistry mirror."""
from __future__ import annotations

from src.services.component_state import ComponentStateRegistry


def test_registry_defaults_cold_and_updates():
    reg = ComponentStateRegistry()
    assert reg.get("/m/u|cuda:1|bfloat16|") == {"key": "/m/u|cuda:1|bfloat16|", "state": "cold", "error": None}
    reg.update("/m/u|cuda:1|bfloat16|", "loading", None)
    reg.update("/m/u|cuda:1|bfloat16|", "loaded", None)
    assert reg.get("/m/u|cuda:1|bfloat16|")["state"] == "loaded"


def test_registry_query_many_and_all():
    reg = ComponentStateRegistry()
    reg.update("a", "loaded", None)
    reg.update("b", "failed", "boom")
    rows = reg.query(["a", "b", "c"])
    by = {r["key"]: r for r in rows}
    assert by["a"]["state"] == "loaded"
    assert by["b"]["state"] == "failed" and by["b"]["error"] == "boom"
    assert by["c"]["state"] == "cold"
    assert len(reg.all()) == 2
