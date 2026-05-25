"""PR-5a §9: preload → ComponentEvent → registry → /state, end-to-end (no GPU)."""
from __future__ import annotations

import asyncio
import pytest

from src.runner import protocol as P
from src.runner.client import RunnerClient
from src.runner.runner_process import _RunnerState, _handle_preload_components
from src.services.component_state import ComponentStateRegistry
from src.api.main import _make_component_event_handler


class _PairChannel:
    """One-way in-memory channel: runner .send_message → client .recv_message."""
    def __init__(self):
        self.to_client = asyncio.Queue()
    async def send_message(self, m):   # runner side
        self.to_client.put_nowait(m)
    async def recv_message(self):      # client side
        return await self.to_client.get()


class _FakeMM:
    async def get_or_load_image_adapter(self, components, pipeline_class, on_event=None):
        from src.services.inference.component_spec import component_state_key
        for k in ("diffusion_models", "clip", "vae"):
            key = component_state_key(components[k])
            await on_event(key, "loading", None)
            await on_event(key, "loaded", None)
        return object()


class _WS:
    def __init__(self):
        self.calls = []
    async def broadcast_component_state(self, key, state, error=None):
        self.calls.append((key, state, error))


@pytest.mark.asyncio
async def test_preload_to_registry_e2e():
    ch = _PairChannel()
    registry = ComponentStateRegistry()
    ws = _WS()

    client = RunnerClient(None, runner_id="test")
    client._ch = ch
    client._connected = True
    client.on_component_event = _make_component_event_handler(registry, ws)
    demux = asyncio.create_task(client._demux_loop())

    state = _RunnerState("r", "image", [0, 1, 2], _FakeMM())
    comps = {
        "diffusion_models": {"kind": "diffusion_models", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
    }
    await _handle_preload_components(state, ch, P.PreloadComponents(task_id=1, components=comps))
    await asyncio.sleep(0.1)
    demux.cancel()

    from src.services.inference.component_spec import ComponentSpec, component_state_key
    unet_key = component_state_key(ComponentSpec(**comps["diffusion_models"]))
    assert registry.get(unet_key)["state"] == "loaded"
    assert any(s == "loaded" for (_k, s, _e) in ws.calls)
