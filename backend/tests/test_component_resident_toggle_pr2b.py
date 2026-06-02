"""组件 L1 PR-2b:已加载组件常驻 toggle —— 协议 + runner handler + client + 端点 + mm 行为。

引擎库组件卡「常驻」开关:把随工作流/预加载进 L1 的组件钉常驻(不被 LRU 让出)或取消。
CI 安全:protocol/client/engines/runner 源码检查;mm.set_component_resident 真行为 mock build seam。
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

    m = P.SetComponentResident(state_key="/m/c.safe|cuda:1|bfloat16|", resident=True)
    assert m.kind == "set_component_resident"
    assert P._KIND_TO_CLASS.get("set_component_resident") is P.SetComponentResident
    assert P.SetComponentResident in P.Message.__args__
    back = P.decode(P.encode(m))
    assert isinstance(back, P.SetComponentResident) and back.resident is True


def test_runner_handler_wired():
    src = (_SRC / "runner/runner_process.py").read_text()
    assert "_handle_set_component_resident(" in src
    assert "set_component_resident(" in src
    assert "isinstance(msg, P.SetComponentResident)" in src


def test_runner_client_has_method():
    src = (_SRC / "runner/client.py").read_text()
    assert "async def set_component_resident(" in src
    assert "P.SetComponentResident(" in src


def test_engines_route_registered():
    src = (_SRC / "api/routes/engines.py").read_text()
    assert '"/component/resident"' in src
    assert "client.set_component_resident(" in src


# ---- mm.set_component_resident 真行为 ---------------------------------------

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


def _clip(file="/m/clipY.safe", dev="cuda:1"):
    return ComponentSpec(kind="clip", file=file, device=dev, dtype="bfloat16")


@pytest.mark.asyncio
async def test_toggle_resident_on_loaded_component(mm):
    """预加载组件(非常驻)→ set_resident(True) 命中切位 → 再 False 取消。"""
    res = await mm.preload_image_component(_clip(), resident=False)
    sk = res["key"]
    comp = next(iter(mm._components.values()))
    assert comp["resident"] is False and comp["state_key"] == sk

    assert mm.set_component_resident(sk, True) is True
    assert comp["resident"] is True
    assert mm.set_component_resident(sk, False) is True
    assert comp["resident"] is False


def test_toggle_unknown_component_is_noop(mm):
    """没加载该组件 → 返回 False(no-op,不抛)。"""
    assert mm.set_component_resident("/nope|cuda:1|bfloat16|", True) is False


@pytest.mark.asyncio
async def test_state_key_stable_across_build_and_status(mm):
    """存进 dict 的 state_key 与 preload 返回的 key、get_status 露的 state_key 一致(单一来源)。"""
    res = await mm.preload_image_component(_clip(), resident=True)
    comp = next(iter(mm._components.values()))
    assert comp["state_key"] == res["key"]
    st = mm.get_status()
    assert st["components"][0]["state_key"] == res["key"]
    assert st["components"][0]["resident"] is True
