"""组件 L1 PR-3a:单组件 L1 池跨进程快照露出 —— snapshot + Pong 字段 + aggregate + catalog 标 loaded/resident。

引擎库要能显示**预加载的孤组件**(不属任何 combo)loaded@卡 + resident。CI 安全(无 torch)。
"""
from __future__ import annotations

import types

import pytest

import src.services.inference.image_modular as IM
import src.services.model_manager as MM
from src.services.gpu_allocator import GPUAllocator
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


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


def _clip():
    return ComponentSpec(kind="clip", file="/m/qwen_3_8b.safetensors", device="cuda:1", dtype="bfloat16")


# ---- snapshot 形状 ----------------------------------------------------------

@pytest.mark.asyncio
async def test_loaded_components_snapshot_shape(mm):
    await mm.preload_image_component(_clip(), resident=True)
    snap = mm.loaded_components_snapshot()
    assert len(snap) == 1
    e = snap[0]
    assert e["role"] == "text_encoder"
    assert e["file"] == "/m/qwen_3_8b.safetensors"
    assert e["device"] == "cuda:1"
    assert e["resident"] is True
    assert e["refs_count"] == 0          # 预加载孤组件,无 combo 引用
    assert e["state_key"].endswith("|")  # 无 LoRA
    assert e["last_used_ago_sec"] >= 0


# ---- Pong 字段 + 向后兼容 ---------------------------------------------------

def test_pong_carries_loaded_components_backward_compat():
    from src.runner import protocol as P  # noqa: PLC0415

    # 新 Pong 带字段
    m = P.Pong(runner_id="r", loaded_components=[{"file": "/m/c.safe", "resident": True}])
    assert P.decode(P.encode(m)).loaded_components[0]["resident"] is True
    # 老 Pong 不带 → 默认 []
    old = P.Pong(runner_id="r", loaded_models=[])
    assert P.decode(P.encode(old)).loaded_components == []


# ---- aggregate ----------------------------------------------------------

def test_aggregate_runner_components_merges_sup_and_main(mm, monkeypatch):
    from src.services import runner_models as RM  # noqa: PLC0415

    sup = types.SimpleNamespace(
        group_id="image",
        loaded_components=[{"file": "/m/clipA.safe", "role": "text_encoder", "resident": False}],
    )
    app_state = types.SimpleNamespace(runner_supervisors=[sup], model_manager=mm)
    # 主进程 mm 也放一个组件(模拟主进程加载,虽实际在 runner)
    mm._components[("X", "cuda:0", "bfloat16", frozenset())] = {
        "module": object(), "role": "vae", "key": ("/m/vaeM.safe", "cuda:0", "bfloat16", frozenset()),
        "state_key": "/m/vaeM.safe|cuda:0|bfloat16|", "refs": set(), "resident": True,
        "last_used": 0.0, "device": "cuda:0",
    }
    out = RM.aggregate_runner_components(app_state)
    groups = {e["group_id"] for e in out}
    assert "image" in groups and "main" in groups
    files = {e["file"] for e in out}
    assert "/m/clipA.safe" in files and "/m/vaeM.safe" in files


# ---- catalog 标 loaded/resident --------------------------------------------

def test_component_catalog_marks_preloaded_loaded_resident(monkeypatch):
    from src.services import engine_catalog as EC  # noqa: PLC0415

    # scan 出一个 clip 文件
    monkeypatch.setattr(EC, "_COMPONENT_ROLES", [("clip", "component")])
    monkeypatch.setattr(
        "src.services.component_scanner.scan_components",
        lambda role: [{"filename": "qwen_3_8b.safetensors",
                       "abs_path": "/m/text_encoders/qwen_3_8b.safetensors", "size_mb": 8000}],
    )
    # 无 combo loaded;单组件 L1 池里有它(预加载 + 常驻 @cuda:1)
    monkeypatch.setattr(EC, "_loaded_index", lambda st: ({}, []))
    monkeypatch.setattr(EC, "_component_loaded_index", lambda st: {
        "qwen_3_8b.safetensors": {
            "file": "/m/text_encoders/qwen_3_8b.safetensors", "device": "cuda:1",
            "resident": True, "role": "text_encoder",
            "state_key": "/m/text_encoders/qwen_3_8b.safetensors|cuda:1|bfloat16|",
        }
    })
    out = EC.component_catalog_entries(app_state=None)
    e = next(x for x in out if x.display_name == "qwen_3_8b.safetensors")
    assert e.status == "loaded"
    assert e.resident is True
    assert e.loaded_gpu == 1
    assert e.kind == "component"
    # state_key 露给前端常驻 toggle 精确匹配(含真实 device)
    assert e.state_key == "/m/text_encoders/qwen_3_8b.safetensors|cuda:1|bfloat16|"


def test_component_catalog_unloaded_when_absent(monkeypatch):
    from src.services import engine_catalog as EC  # noqa: PLC0415

    monkeypatch.setattr(EC, "_COMPONENT_ROLES", [("vae", "component")])
    monkeypatch.setattr(
        "src.services.component_scanner.scan_components",
        lambda role: [{"filename": "flux2-vae.safetensors", "abs_path": "/m/vae/flux2-vae.safetensors", "size_mb": 300}],
    )
    monkeypatch.setattr(EC, "_loaded_index", lambda st: ({}, []))
    monkeypatch.setattr(EC, "_component_loaded_index", lambda st: {})
    out = EC.component_catalog_entries(app_state=None)
    e = out[0]
    assert e.status == "unloaded" and e.resident is False and e.loaded_gpu is None
