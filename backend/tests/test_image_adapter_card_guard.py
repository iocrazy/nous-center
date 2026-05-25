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
        "diffusion_models": ComponentSpec(kind="diffusion_models", file="/m/u.safe", device=dev, dtype="bfloat16", adapter_arch="flux2"),
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
    # 无 GPU / 查询失败 → free=None → 跳过保护(不阻塞)。modular 装配 stub 让流程走通。
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)
    monkeypatch.setattr(mm, "_estimate_image_vram_mb", lambda resolved: 99999)

    async def _fake_modular(resolved, combo_key, pc, target, emit):
        return object()

    monkeypatch.setattr(mm, "_get_or_load_modular_adapter", _fake_modular)
    adapter = await mm.get_or_load_image_adapter(_comps("cuda:1"), "Flux2KleinPipeline")
    assert adapter is not None


def test_estimate_vram_fp8_halves_transformer_and_clip(mm, tmp_path):
    """fp8 weight-only:transformer + clip 按 file bytes 的一半估(vae 不量化全计),
    所以 fp8 估算 ≈ bf16 估算 - (transformer+clip)/2 的余量倍数。"""
    t = tmp_path / "t.safe"
    t.write_bytes(b"\0" * 8_000_000)   # 8MB transformer
    c = tmp_path / "c.safe"
    c.write_bytes(b"\0" * 4_000_000)   # 4MB clip
    v = tmp_path / "v.safe"
    v.write_bytes(b"\0" * 1_000_000)   # 1MB vae

    def comps(dt):
        return {
            "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(t), device="cuda:1", dtype=dt, adapter_arch="flux2"),
            "clip": ComponentSpec(kind="clip", file=str(c), device="cuda:1", dtype=dt),
            "vae":  ComponentSpec(kind="vae",  file=str(v), device="cuda:1", dtype="bfloat16"),  # vae 永不 fp8
        }

    bf16 = mm._estimate_image_vram_mb(comps("bfloat16"))
    fp8 = mm._estimate_image_vram_mb(comps("fp8_e4m3"))
    # bf16: (8+4+1)MB*1.3 ; fp8: (4+2+1)MB*1.3 —— transformer/clip 减半
    assert bf16 is not None and fp8 is not None
    assert fp8 < bf16
    assert fp8 == int((4_000_000 + 2_000_000 + 1_000_000) / (1024 * 1024) * 1.3)


@pytest.mark.asyncio
async def test_single_card_unifies_clip_vae_to_unet_device(mm, monkeypatch):
    # 三组件传入不同卡(unet cuda:1, clip cuda:0, vae cuda:2)→ 统一到 unet 的 cuda:1
    seen = {}
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)

    async def _fake_modular(resolved, combo_key, pc, target, emit):
        seen["resolved"] = resolved
        return object()

    monkeypatch.setattr(mm, "_get_or_load_modular_adapter", _fake_modular)
    await mm.get_or_load_image_adapter(_comps("cuda:1"), "Flux2KleinPipeline")
    assert {s.device for s in seen["resolved"].values()} == {"cuda:1"}  # 三件同卡
