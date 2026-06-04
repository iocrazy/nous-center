"""PR-1 T5: 整模型单卡统一 + LLM 卡显存前置保护。"""
from __future__ import annotations

from unittest.mock import MagicMock

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
async def test_insufficient_vram_raises_clear_error(mm, monkeypatch, tmp_path):
    # 逐卡守卫(2026-06-04):unet 落 cuda:1,该卡空闲严重不足(LLM 占着)→ 装载前清晰报错。
    # 真文件让逐卡守卫能估出该卡需求(守卫读 file bytes 分卡求和)。
    u = tmp_path / "u.safe"
    u.write_bytes(b"\0" * 40_000_000)  # 40MB transformer → 约需 ~50MB
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file=str(u), device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file=str(u), device="cuda:0", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file=str(u), device="cuda:2", dtype="bfloat16"),
    }
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: 1 if dev == "cuda:1" else 90000)
    with pytest.raises(RuntimeError, match="显存不足|cuda:1"):
        await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline")


@pytest.mark.asyncio
async def test_guard_skipped_when_free_unknown(mm, monkeypatch):
    # 无 GPU / 查询失败 → free=None → 跳过保护(不阻塞)。modular 装配 stub 让流程走通。
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)

    async def _fake_modular(resolved, combo_key, pc, target, emit, offload="none", comp_devices=None, comp_offloads=None):
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
async def test_explicit_per_component_cards_honored(mm, monkeypatch):
    # 逐组件选卡(2026-06-04):三组件显式不同卡(unet cuda:1, clip cuda:0, vae cuda:2)
    # → **各落各的卡**(不再统一到 unet 卡)。
    seen = {}
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)

    async def _fake_modular(resolved, combo_key, pc, target, emit, offload="none", comp_devices=None, comp_offloads=None):
        seen["resolved"] = resolved
        seen["comp_devices"] = comp_devices
        return object()

    monkeypatch.setattr(mm, "_get_or_load_modular_adapter", _fake_modular)
    await mm.get_or_load_image_adapter(_comps("cuda:1"), "Flux2KleinPipeline")
    assert {s.device for s in seen["resolved"].values()} == {"cuda:1", "cuda:0", "cuda:2"}
    assert seen["comp_devices"] == {
        "transformer": "cuda:1", "text_encoder": "cuda:0", "vae": "cuda:2"}


@pytest.mark.asyncio
async def test_auto_clip_vae_follow_unet_card(mm, monkeypatch):
    # 逐组件选卡零回归:clip/vae device=auto → 跟随 transformer 解析出的卡(整模型单卡)。
    seen = {}
    monkeypatch.setattr(mm, "_free_vram_mb", lambda dev: None)

    async def _fake_modular(resolved, combo_key, pc, target, emit, offload="none", comp_devices=None, comp_offloads=None):
        seen["comp_devices"] = comp_devices
        return object()

    monkeypatch.setattr(mm, "_get_or_load_modular_adapter", _fake_modular)
    comps = {
        "diffusion_models": ComponentSpec(kind="diffusion_models", file="/m/u.safe", device="cuda:1", dtype="bfloat16", adapter_arch="flux2"),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="auto", dtype="bfloat16"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="auto", dtype="bfloat16"),
    }
    await mm.get_or_load_image_adapter(comps, "Flux2KleinPipeline")
    assert seen["comp_devices"] == {
        "transformer": "cuda:1", "text_encoder": "cuda:1", "vae": "cuda:1"}


# --- PR-2: 单文件装配辅助(架构参考整模型 + 单文件检测)---

def test_reference_repo_for_arch_matches_class(tmp_path, monkeypatch):
    """PR-B 后:flux2 优先返回仓内 bundle(几 MB);未知架构 fallback 扫 LOCAL_MODELS_PATH。"""
    from src.services import model_manager as mm_mod
    base = tmp_path / "image" / "diffusers"
    (base / "ERNIE-Image").mkdir(parents=True)
    (base / "ERNIE-Image" / "model_index.json").write_text('{"_class_name": "ErnieImagePipeline"}')
    settings = MagicMock()
    settings.LOCAL_MODELS_PATH = str(tmp_path)
    monkeypatch.setattr("src.config.get_settings", lambda: settings)
    # flux2 → 优先 bundle(无需 LOCAL_MODELS_PATH/diffusers/Flux2-klein-9B);
    assert mm_mod._reference_repo_for_arch("flux2").endswith("configs/image_arch/flux2")
    # ernie 未 bundle → fallback 扫 LOCAL_MODELS_PATH。
    assert mm_mod._reference_repo_for_arch("ernie").endswith("ERNIE-Image")
    assert mm_mod._reference_repo_for_arch("nope") is None


def test_is_standalone_single_file(tmp_path):
    from src.services.model_manager import _is_standalone_single_file
    sf = tmp_path / "diffusion_models" / "flux" / "x.safetensors"
    sf.parent.mkdir(parents=True)
    sf.write_text("x")
    assert _is_standalone_single_file(
        ComponentSpec(kind="diffusion_models", file=str(sf), device="cuda:0", dtype="bfloat16"))
    hf = tmp_path / "diffusers" / "M" / "transformer" / "y.safetensors"
    hf.parent.mkdir(parents=True)
    hf.write_text("y")
    (hf.parent / "config.json").write_text("{}")
    assert not _is_standalone_single_file(
        ComponentSpec(kind="diffusion_models", file=str(hf), device="cuda:0", dtype="bfloat16"))
