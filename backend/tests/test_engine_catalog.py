"""统一引擎库目录扩展(SeedVR2 + 单文件组件)+ VRAM 残留状态多键匹配 —— CI 安全
(engine_catalog 顶层只 schemas;seedvr2/component/runner_models 顶层无 torch;mock 数据源)。"""
from __future__ import annotations

import pytest


class _FakeSup:
    def __init__(self, group_id, loaded_models):
        self.group_id = group_id
        self.loaded_models = loaded_models


class _FakeState:
    def __init__(self, sups):
        self.runner_supervisors = sups


@pytest.fixture
def patched_sources(monkeypatch):
    """mock SeedVR2 磁盘状态 + 组件扫描,不依赖真模型盘。"""
    import src.services.inference.image_seedvr2 as sv2
    import src.services.component_scanner as cs

    monkeypatch.setattr(sv2, "seedvr2_dit_models_with_disk_status", lambda model_dir=None: [
        {"filename": "seedvr2_7b.safetensors", "label": "7B fp8-mixed", "present": True, "size_mb": 8466, "is_default": True},
        {"filename": "seedvr2_3b.safetensors", "label": "3B fp16", "present": False, "size_mb": None, "is_default": False},
    ])
    monkeypatch.setattr(cs, "scan_components", lambda role, **kw: {
        "diffusion_models": [{"filename": "flux-unet.safetensors", "abs_path": "/m/flux-unet.safetensors", "size_mb": 18000}],
        "clip": [{"filename": "clipL.safetensors", "abs_path": "/m/clipL.safetensors", "size_mb": 200}],
        "vae": [], "loras": [{"filename": "style.safetensors", "abs_path": "/m/loras/style.safetensors", "size_mb": 150}],
    }.get(role, []))


def test_engine_info_has_kind_field():
    from src.models.schemas import EngineInfo  # noqa: PLC0415

    e = EngineInfo(name="x", display_name="X", type="image", status="unloaded", gpu=0, vram_gb=1.0, resident=False)
    assert e.kind == "model"  # 默认


def test_catalog_type_filter_non_image_empty(patched_sources):
    from src.services.engine_catalog import catalog_extra_engines  # noqa: PLC0415

    assert catalog_extra_engines(_FakeState([]), "llm") == []  # 非 image 过滤 → 空


def test_seedvr2_catalog_only_present_loadable(patched_sources):
    """SeedVR2:只列磁盘已有(present)的;kind=upscale,可独立加载(has_adapter)。"""
    from src.services.engine_catalog import seedvr2_catalog_entries  # noqa: PLC0415

    rows = seedvr2_catalog_entries(_FakeState([]))
    assert [r.display_name for r in rows] == ["SeedVR2 7B fp8-mixed"]  # 3B present=False 不列
    r = rows[0]
    assert r.kind == "upscale" and r.has_adapter is True
    assert r.status == "unloaded"  # 没加载


def test_components_not_independently_loadable(patched_sources):
    """单文件组件:kind=component/lora,has_adapter=False(不独立可加载,UI 禁用加载按钮)。"""
    from src.services.engine_catalog import component_catalog_entries  # noqa: PLC0415

    rows = component_catalog_entries(_FakeState([]))
    by_name = {r.display_name: r for r in rows}
    assert by_name["flux-unet.safetensors"].kind == "component"
    assert by_name["flux-unet.safetensors"].has_adapter is False
    assert by_name["style.safetensors"].kind == "lora"
    assert all(r.has_adapter is False for r in rows)


def test_vram_residency_multikey_match(patched_sources):
    """VRAM 残留状态:loaded adapter 的 source_files 含 SeedVR2 DiT / 组件文件 → status=loaded@gpu。
    这是用户要的「哪些常驻显存」—— 多键匹配 aggregate_runner_loaded。"""
    from src.services.engine_catalog import catalog_extra_engines  # noqa: PLC0415

    # 一个 image runner 加载了 SeedVR2(source_files 含 7b)+ 一个 flux combo(含 flux-unet)
    state = _FakeState([_FakeSup("image", [
        {"model_id": "image:SeedVR2:abc", "model_type": "image", "gpu_index": 1,
         "source_files": ["seedvr2_7b.safetensors", "ema_vae.safetensors"], "vram_mb": 8000},
        {"model_id": "image:Flux2Klein:def", "model_type": "image", "gpu_index": 0,
         "source_files": ["/m/flux-unet.safetensors", "/m/clipL.safetensors", "/m/v.safe"], "vram_mb": 18000},
    ])])
    rows = catalog_extra_engines(state, None)
    by_name = {r.display_name: r for r in rows}
    # SeedVR2 7B → loaded@cuda:1
    assert by_name["SeedVR2 7B fp8-mixed"].status == "loaded"
    assert by_name["SeedVR2 7B fp8-mixed"].loaded_gpu == 1
    # flux-unet 组件 → loaded@cuda:0(随 flux combo 加载)
    assert by_name["flux-unet.safetensors"].status == "loaded"
    assert by_name["flux-unet.safetensors"].loaded_gpu == 0
    # 没加载的组件 → unloaded
    assert by_name["clipL.safetensors"].status == "loaded"  # clipL 也在 flux combo source_files
    assert by_name["style.safetensors"].status == "unloaded"  # LoRA 没加载
