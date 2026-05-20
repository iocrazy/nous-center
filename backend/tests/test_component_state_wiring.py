"""PR-5a: image client component-event callback updates registry + schedules WS."""
from __future__ import annotations

import asyncio
import pytest

from src.runner import protocol as P
from src.services.component_state import ComponentStateRegistry
from src.api.main import _make_component_event_handler


@pytest.mark.asyncio
async def test_handler_updates_registry_and_broadcasts():
    reg = ComponentStateRegistry()
    broadcasts = []

    class _WS:
        async def broadcast_component_state(self, key, state, error=None):
            broadcasts.append((key, state, error))

    handler = _make_component_event_handler(reg, _WS())
    handler(P.ComponentEvent(component_key="k1", state="loaded", error=None))
    await asyncio.sleep(0.05)  # let the scheduled broadcast run
    assert reg.get("k1")["state"] == "loaded"
    assert broadcasts == [("k1", "loaded", None)]
