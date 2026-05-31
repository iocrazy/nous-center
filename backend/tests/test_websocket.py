import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.api.websocket import ConnectionManager


def test_connection_manager_init():
    manager = ConnectionManager()
    assert len(manager.active_connections) == 0


@pytest.mark.asyncio
async def test_websocket_endpoint():
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ):
        # Just verify the websocket route is registered
        routes = [r.path for r in app.routes]
        assert "/ws/tasks/{task_id}" in routes


@pytest.mark.asyncio
async def test_broadcast_snapshot_does_not_skip_on_concurrent_unsubscribe():
    """round4 #4:广播迭代订阅 list 时,若某订阅者 send 触发对**另一个**订阅者的
    unsubscribe(模拟 await 间隙的并发断开),不应漏发后续订阅者(老的索引迭代会跳过)。"""
    mgr = ConnectionManager()

    received: list[str] = []

    class _FakeWS:
        def __init__(self, name):
            self.name = name

        async def send_text(self, msg):
            received.append(self.name)
            # 第一个订阅者发送时移除**自己**(模拟 await 间隙的并发断开)→ 老的索引
            # 迭代会因 list 前移跳过紧跟的下一个(b)。
            if self.name == "a":
                mgr.unsubscribe_global(ws_a)

    ws_a, ws_b, ws_c = _FakeWS("a"), _FakeWS("b"), _FakeWS("c")
    mgr._global_subscribers.extend([ws_a, ws_b, ws_c])

    await mgr.broadcast_task_update("updated", {"id": 1})

    # 快照迭代:a 移除自己后 b、c 仍在快照里被发到。老代码索引前移会跳过 b。
    assert received == ["a", "b", "c"]
