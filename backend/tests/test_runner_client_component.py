"""PR-5a: RunnerClient routes ComponentEvent → on_component_event; preload_components sends."""
from __future__ import annotations

import asyncio
import pytest

from src.runner import protocol as P
from src.runner.client import RunnerClient


class _FakeChannel:
    def __init__(self):
        self.sent = []
        self._incoming = asyncio.Queue()

    async def send_message(self, m):
        self.sent.append(m)

    async def recv_message(self):
        return await self._incoming.get()

    def feed(self, m):
        self._incoming.put_nowait(m)


def _make_client(ch):
    # RunnerClient.__init__ 需要 conn + runner_id（keyword-only）。
    # PipeChannel 会在 __init__ 里起 writer thread 但不立即读 conn，
    # 所以传 None 作为 conn 占位即可。
    # 构造完后把 _ch 替换为 fake channel，绕过真实 pipe 逻辑。
    client = RunnerClient(None, runner_id="test-runner")
    client._ch = ch
    client._connected = True
    return client


@pytest.mark.asyncio
async def test_component_event_routed_to_callback():
    ch = _FakeChannel()
    client = _make_client(ch)
    got = []
    client.on_component_event = lambda evt: got.append(evt)
    task = asyncio.create_task(client._demux_loop())
    ch.feed(P.ComponentEvent(component_key="/m/u|cuda:1|bfloat16|", state="loaded", error=None))
    await asyncio.sleep(0.05)
    task.cancel()
    assert got and got[0].state == "loaded" and got[0].component_key == "/m/u|cuda:1|bfloat16|"


@pytest.mark.asyncio
async def test_preload_components_sends_message():
    ch = _FakeChannel()
    client = _make_client(ch)
    await client.preload_components(task_id=9, components={"unet": {}, "clip": {}, "vae": {}}, pipeline_class="Flux2KleinPipeline")
    sent = [m for m in ch.sent if isinstance(m, P.PreloadComponents)]
    assert sent and sent[0].task_id == 9
