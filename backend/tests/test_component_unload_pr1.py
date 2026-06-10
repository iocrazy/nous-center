"""统一模型管理收尾 PR-1:已预加载组件卸载 —— 协议 + runner handler + client + 端点 + mm 行为。

引擎库组件卡「出缓存」:把预加载进 L1 的组件卸载(出池 + 释放显存);combo 在用则只清常驻待自然释放。
CI 安全:protocol/client/engines/runner 源码检查;mm.unload_image_component 真行为 mock build seam。
跨进程载卸真机另验(smoke;CI runner mock torch 测不了显存)。
"""
from __future__ import annotations

import pathlib

import pytest

import src.services.inference.image_modular as IM
import src.services.model_manager as MM
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager

_SRC = pathlib.Path(__file__).parent.parent / "src"


def test_protocol_message():
    from src.runner import protocol as P  # noqa: PLC0415

    m = P.UnloadComponent(state_key="/m/c.safe|cuda:1|bfloat16|")
    assert m.kind == "unload_component"
    assert P._KIND_TO_CLASS.get("unload_component") is P.UnloadComponent
    assert P.UnloadComponent in P.Message.__args__
    back = P.decode(P.encode(m))
    assert isinstance(back, P.UnloadComponent) and back.state_key == "/m/c.safe|cuda:1|bfloat16|"


def test_runner_handler_wired():
    src = (_SRC / "runner/runner_process.py").read_text()
    assert "_handle_unload_component(" in src
    assert "unload_image_component(" in src
    assert "isinstance(msg, P.UnloadComponent)" in src


def test_runner_client_has_method():
    src = (_SRC / "runner/client.py").read_text()
    assert "async def unload_component(" in src
    assert "P.UnloadComponent(" in src


def test_engines_route_registered_before_param_route():
    """`/component/unload` 必须定义在 `/{name}/unload` **之前**,否则参数路由抢先匹配致 404。"""
    src = (_SRC / "api/routes/engines.py").read_text()
    assert '"/component/unload"' in src
    assert "client.unload_component(" in src
    assert src.index('"/component/unload"') < src.index('"/{name}/unload"')


# ---- mm.unload_image_component 真行为 ---------------------------------------

class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


@pytest.fixture
def mm(monkeypatch):
    m = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    monkeypatch.setattr(MM, "_reference_repo_for_arch", lambda arch: "/fake/repo")
    monkeypatch.setattr(IM, "build_bridged_text_encoder", lambda spec, repo, dev: object())
    return m


def _clip(file="/m/clipU.safe", dev="cuda:1"):
    return ComponentSpec(kind="clip", file=file, device=dev, dtype="bfloat16")


@pytest.mark.asyncio
async def test_unload_preloaded_component_frees_pool(mm):
    """预加载组件(refs 空)→ unload 命中 → 出 L1 池(_components 清空)。"""
    res = await mm.preload_image_component(_clip(), resident=True)
    sk = res["key"]
    assert len(mm._components) == 1
    assert mm.unload_image_component(sk) is True
    assert len(mm._components) == 0  # refs 空 → 真出池


def test_unload_unknown_component_is_noop(mm):
    """没加载该组件 → 返回 False(no-op,不抛)。"""
    assert mm.unload_image_component("/nope|cuda:1|bfloat16|") is False


@pytest.mark.asyncio
async def test_unload_in_use_component_keeps_module_clears_resident(mm):
    """组件被某 combo 引用(refs 非空)→ unload 只清常驻 + 留模块(不硬拔在用组件),待 combo 释放自然出池。"""
    res = await mm.preload_image_component(_clip(), resident=True)
    sk = res["key"]
    comp = next(iter(mm._components.values()))
    comp["refs"].add("combo-X")  # 模拟某 combo 在用
    assert mm.unload_image_component(sk) is True
    assert len(mm._components) == 1          # 没出池(refs 非空)
    assert comp["resident"] is False         # 但清了常驻
    assert comp["module"] is not None        # 模块还在(combo 在用)
