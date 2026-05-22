"""PR-1 T5: 整模型单卡统一 + LLM 卡显存前置保护。"""
from __future__ import annotations

import pytest

from src.services.gpu_allocator import GPUAllocator
from src.services.inference.component_spec import ComponentSpec
from src.services.inference.registry import ModelRegistry
from src.services.model_manager import ModelManager


class _EmptyRegistry(ModelRegistry):
    def __init__(self):
        self._config_path = ""
        self._specs = {}


@pytest.fixture
def mm():
    return ModelManager(registry=_EmptyRegistry(), allocator=GPUAllocator())


def _comps(dev="cuda:1"):
    return {
        "unet": ComponentSpec(kind="unet", file="/m/u.safe", device=dev, dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }


@pytest.mark.asyncio
async def test_insufficient_vram_raises_clear_error(mm, monkeypatch):
    # 目标卡(unet 的 cuda:1)空闲显存严重不足(LLM 占着)
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: 500 if dev == "cuda:1" else 90000)
    monkeypatch.setattr(mm, "_estimate_image_vram_mb", lambda resolved: 20000)
    with pytest.raises(RuntimeError, match="显存不足|cuda:1|LLM"):
        await mm.get_or_load_image_adapter(_comps("cuda:1"), "Flux2KleinPipeline")


@pytest.mark.asyncio
async def test_guard_skipped_when_free_unknown(mm, monkeypatch):
    # 无 GPU / 查询失败 → free=None → 跳过保护(不阻塞)。装配 stub 让流程走通。
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)
    monkeypatch.setattr(mm, "_estimate_image_vram_mb", lambda resolved: 99999)
    monkeypatch.setattr(mm, "_load_component_module",
                        lambda spec: {"module": object(), "tokenizer": None})
    monkeypatch.setattr(
        "src.services.inference.image_diffusers.DiffusersImageBackend.from_loaded_components",
        staticmethod(lambda modules, components, pc: object()))
    adapter = await mm.get_or_load_image_adapter(_comps("cuda:1"), "Flux2KleinPipeline")
    assert adapter is not None


@pytest.mark.asyncio
async def test_single_card_unifies_clip_vae_to_unet_device(mm, monkeypatch):
    # 三组件传入不同卡(unet cuda:1, clip cuda:0, vae cuda:2)→ 统一到 unet 的 cuda:1
    seen = []
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)
    monkeypatch.setattr(mm, "_load_component_module",
                        lambda spec: (seen.append((spec.kind, spec.device)),
                                      {"module": object(), "tokenizer": None})[1])
    monkeypatch.setattr(
        "src.services.inference.image_diffusers.DiffusersImageBackend.from_loaded_components",
        staticmethod(lambda modules, components, pc: object()))
    await mm.get_or_load_image_adapter(_comps("cuda:1"), "Flux2KleinPipeline")
    assert {dev for _kind, dev in seen} == {"cuda:1"}  # 三件同卡
