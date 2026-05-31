"""_broadcast 死连接剔除 — bug hunt round2 #6。"""
import pytest


class _DeadWS:
    async def send_json(self, ev):
        raise RuntimeError("conn reset")  # 非干净断开


class _LiveWS:
    def __init__(self):
        self.got = []

    async def send_json(self, ev):
        self.got.append(ev)


@pytest.mark.asyncio
async def test_broadcast_removes_dead_keeps_live():
    from src.api import main as main_mod
    from src.services.workflow_runner import _broadcast
    live, dead = _LiveWS(), _DeadWS()
    main_mod._ws_connections["chan1"] = [live, dead]
    await _broadcast("chan1", {"x": 1})
    bucket = main_mod._ws_connections.get("chan1", [])
    assert dead not in bucket and live in bucket  # 死的剔、活的留
    assert live.got == [{"x": 1}]
    main_mod._ws_connections.pop("chan1", None)


@pytest.mark.asyncio
async def test_broadcast_drops_empty_bucket():
    from src.api import main as main_mod
    from src.services.workflow_runner import _broadcast
    main_mod._ws_connections["chan2"] = [_DeadWS()]
    await _broadcast("chan2", {"y": 2})
    assert "chan2" not in main_mod._ws_connections  # 全死 → 桶删
