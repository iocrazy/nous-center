"""组件 L1 PR-2a:单组件预加载 + 常驻 pin —— 协议 + runner handler + client + 端点 + mm 行为。

CI 安全:protocol/client/engines/runner 用源码检查避 torch mock 边界;mm.preload_image_component
真行为用 mock build seam(同 test_component_l1_cache 套路)。引擎库点单个 clip/vae「预加载/常驻」。
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


# ---- 协议 / wiring 源码检查 -------------------------------------------------

def test_preload_component_protocol_message():
    from src.runner import protocol as P  # noqa: PLC0415

    msg = P.PreloadComponent(spec={"kind": "clip", "file": "/m/c.safe"}, resident=True)
    assert msg.kind == "preload_component"
    assert msg.resident is True
    assert P._KIND_TO_CLASS.get("preload_component") is P.PreloadComponent
    assert P.PreloadComponent in P.Message.__args__
    # msgpack 收发 round-trip(过进程边界)
    back = P.decode(P.encode(msg))
    assert isinstance(back, P.PreloadComponent)
    assert back.spec["file"] == "/m/c.safe" and back.resident is True


def test_runner_handler_wired():
    src = (_SRC / "runner/runner_process.py").read_text()
    assert "async def _handle_preload_component(" in src
    assert "preload_image_component(" in src
    assert "isinstance(msg, P.PreloadComponent)" in src  # _pipe_reader 派发


def test_runner_client_has_preload_component():
    src = (_SRC / "runner/client.py").read_text()
    assert "async def preload_component(" in src
    assert "P.PreloadComponent(" in src


def test_engines_route_registered():
    src = (_SRC / "api/routes/engines.py").read_text()
    assert '"/component/preload"' in src
    assert "client.preload_component(" in src


# ---- mm.preload_image_component 真行为 --------------------------------------

class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


@pytest.fixture
def mm(monkeypatch):
    m = ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())
    # repo 反推 + build seam mock(无 torch)
    monkeypatch.setattr(MM, "_reference_repo_for_arch", lambda arch: "/fake/repo")
    calls = {"transformer": [], "text_encoder": [], "vae": []}

    def _mk(role):
        def _fn(spec, repo, device):
            mod = object()
            calls[role].append({"file": spec.file, "device": device})
            return mod
        return _fn

    monkeypatch.setattr(IM, "build_bridged_transformer", _mk("transformer"))
    monkeypatch.setattr(IM, "build_bridged_text_encoder", _mk("text_encoder"))
    monkeypatch.setattr(IM, "build_bridged_vae", _mk("vae"))
    return m, calls


def _clip(file="/m/clipY.safe", dev="cuda:1"):
    # clip/vae 的 ComponentSpec 不带 adapter_arch(diffusion_models-only);arch 经参数传。
    return ComponentSpec(kind="clip", file=file, device=dev, dtype="bfloat16")


@pytest.mark.asyncio
async def test_preload_builds_and_stores_with_empty_refs(mm):
    """预加载单组件 → 进 L1 池,refs 空(无 combo 引用),resident 按参数。"""
    m, calls = mm
    res = await m.preload_image_component(_clip(), resident=False)
    assert res["state"] == "loaded" and res["role"] == "text_encoder"
    assert len(calls["text_encoder"]) == 1
    comp = next(iter(m._components.values()))
    assert comp["refs"] == set()  # 预加载无 combo 引用
    assert comp["resident"] is False
    assert comp["device"] == "cuda:1"


@pytest.mark.asyncio
async def test_preload_resident_pins(mm):
    """resident=True 预加载 → 组件常驻(卸 combo / refs 空也不被释放)。"""
    m, _calls = mm
    await m.preload_image_component(_clip(), resident=True)
    comp = next(iter(m._components.values()))
    assert comp["resident"] is True


@pytest.mark.asyncio
async def test_preload_hit_upgrades_resident_no_rebuild(mm):
    """组件已在池(非常驻)→ 再 preload(resident=True)只升常驻,不重 build。"""
    m, calls = mm
    await m.preload_image_component(_clip(), resident=False)
    await m.preload_image_component(_clip(), resident=True)
    assert len(calls["text_encoder"]) == 1, "命中不重 build"
    comp = next(iter(m._components.values()))
    assert comp["resident"] is True


@pytest.mark.asyncio
async def test_preloaded_component_reused_by_matching_combo(mm):
    """预加载的组件被后来跑的匹配 combo 复用(refs 加上该 combo)—— 预加载→工作流命中。"""
    m, calls = mm
    await m.preload_image_component(_clip(dev="cuda:1"), resident=True)
    # 模拟 combo 路径调 _get_or_build_image_component(同 key)→ 应命中预加载的
    spec = _clip(dev="cuda:1")
    mod = await m._get_or_build_image_component(
        "text_encoder", IM.build_bridged_text_encoder, spec, "/fake/repo", "cuda:1", "none", "combo-A")
    assert len(calls["text_encoder"]) == 1, "复用预加载的,不重 build"
    comp = next(iter(m._components.values()))
    assert "combo-A" in comp["refs"]  # combo 引用上了
    assert comp["resident"] is True  # 仍常驻
    assert mod is comp["module"]


@pytest.mark.asyncio
async def test_preload_diffusion_uses_spec_arch(mm):
    """diffusion_models 的 spec 带 adapter_arch → 优先用它反推 repo(arch 参数兜底)。"""
    m, calls = mm
    dm = ComponentSpec(kind="diffusion_models", file="/m/X.safe", device="cuda:1",
                       dtype="bfloat16", adapter_arch="flux2")
    res = await m.preload_image_component(dm, resident=True)
    assert res["role"] == "transformer"
    assert len(calls["transformer"]) == 1
