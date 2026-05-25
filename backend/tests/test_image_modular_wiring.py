"""PR-1 wiring 测(CI 可跑,无 GPU/真 diffusers)。

monkeypatch `_import_modular` → fake ModularPipeline/ComponentsManager,断言
ImageRequest → pipe() 参数映射正确。验「接线」回归,不验真出图(真出图走 standalone
A/B smoke,见 plan Task 4)。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.services.inference.base import ImageRequest
from src.services.inference import image_modular


def _fake_modular(monkeypatch):
    """装一对 fake (ModularPipeline, ComponentsManager) 并返回内部 pipe mock。"""
    pipe = MagicMock(name="pipe")
    out = MagicMock(name="out")
    img = MagicMock(name="img")
    img.save.side_effect = lambda buf, format: buf.write(b"\x89PNG\r\n\x1a\n")  # noqa: A002
    out.images = [img]
    pipe.return_value = out  # pipe(...) → out

    modular_cls = MagicMock(name="ModularPipeline")
    modular_cls.from_pretrained.return_value = pipe
    cm_cls = MagicMock(name="ComponentsManager")

    monkeypatch.setattr(image_modular, "_import_modular", lambda: (modular_cls, cm_cls))
    return modular_cls, cm_cls, pipe


@pytest.mark.asyncio
async def test_modular_backend_maps_request_to_pipe(monkeypatch):
    modular_cls, cm_cls, pipe = _fake_modular(monkeypatch)
    # device="cpu":wiring 测不碰真 CUDA(全套里 torch 可能是真的,CI 无 GPU →
    # torch.Generator(device="cuda") 会 AcceleratorError)。CPU generator 真假 torch 都行。
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu", dtype="bfloat16")

    res = await be.infer(
        ImageRequest(request_id="t1", prompt="a fox", steps=7, width=512, height=512, seed=42)
    )

    # 结果信封
    assert res.media_type == "image/png"
    assert res.data.startswith(b"\x89PNG")
    assert res.usage.image_count == 1
    assert res.metadata["engine"] == "modular"

    # 构建链:from_pretrained(repo, components_manager=...) + load_components + to(device)
    modular_cls.from_pretrained.assert_called_once()
    assert modular_cls.from_pretrained.call_args.args[0] == "/m/flux2"
    assert "components_manager" in modular_cls.from_pretrained.call_args.kwargs
    pipe.load_components.assert_called_once()
    pipe.to.assert_called_once_with("cpu")

    # 参数映射:ImageRequest → pipe(...)
    kw = pipe.call_args.kwargs
    assert kw["prompt"] == "a fox"
    assert kw["num_inference_steps"] == 7
    assert kw["width"] == 512 and kw["height"] == 512
    assert "generator" in kw


@pytest.mark.asyncio
async def test_modular_backend_reuses_pipe_across_infers(monkeypatch):
    """同一 backend 多次 infer 复用已建 pipe(不重复 from_pretrained)。"""
    modular_cls, _cm, pipe = _fake_modular(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")

    await be.infer(ImageRequest(request_id="a", prompt="x", steps=2, width=64, height=64))
    await be.infer(ImageRequest(request_id="b", prompt="y", steps=2, width=64, height=64))

    modular_cls.from_pretrained.assert_called_once()  # 只建一次
    assert be.is_loaded


@pytest.mark.asyncio
async def test_transformer_override_calls_update_components(monkeypatch):
    """PR-2:comfy 量化桥接 → transformer_override 经 update_components 替换 HF transformer。"""
    modular_cls, _cm, pipe = _fake_modular(monkeypatch)
    override = MagicMock(name="bridged_transformer")
    be = image_modular.ModularImageBackend(
        repo="/m/flux2", device="cpu", transformer_override=override)

    await be.infer(ImageRequest(request_id="q", prompt="x", steps=3, width=64, height=64))

    pipe.update_components.assert_called_once_with(transformer=override)


@pytest.mark.asyncio
async def test_no_override_skips_update_components(monkeypatch):
    """HF-layout(无 override)不调 update_components(transformer=)。"""
    _m, _cm, pipe = _fake_modular(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    await be.infer(ImageRequest(request_id="h", prompt="x", steps=3, width=64, height=64))
    pipe.update_components.assert_not_called()


def test_wants_fp8_detects_fp8_dtypes():
    assert image_modular._wants_fp8("fp8_e4m3")
    assert image_modular._wants_fp8("fp8_e5m2")
    assert image_modular._wants_fp8("fp8")
    assert not image_modular._wants_fp8("bfloat16")
    assert not image_modular._wants_fp8("float16")
    assert not image_modular._wants_fp8("")
    assert not image_modular._wants_fp8(None)


@pytest.mark.asyncio
async def test_fp8_dtype_triggers_weight_only_quant(monkeypatch):
    """weight_dtype=fp8 + 无 override → _quantize_fp8_weight_only(pipe)(torchao 真量化由 smoke 验)。"""
    _m, _cm, pipe = _fake_modular(monkeypatch)
    spy = MagicMock(name="quantize_fp8")
    monkeypatch.setattr(image_modular, "_quantize_fp8_weight_only", spy)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu", dtype="fp8_e4m3")
    await be.infer(ImageRequest(request_id="q8", prompt="x", steps=2, width=64, height=64))
    spy.assert_called_once_with(pipe)
    pipe.update_components.assert_not_called()


@pytest.mark.asyncio
async def test_bf16_dtype_no_fp8_quant(monkeypatch):
    _m, _cm, pipe = _fake_modular(monkeypatch)
    spy = MagicMock(name="quantize_fp8")
    monkeypatch.setattr(image_modular, "_quantize_fp8_weight_only", spy)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu", dtype="bfloat16")
    await be.infer(ImageRequest(request_id="b16", prompt="x", steps=2, width=64, height=64))
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_override_takes_precedence_over_fp8_quant(monkeypatch):
    """comfy 桥接 override + fp8 → 走 update_components(桥接),不再 torchao 量化(elif)。"""
    _m, _cm, pipe = _fake_modular(monkeypatch)
    spy = MagicMock(name="quantize_fp8")
    monkeypatch.setattr(image_modular, "_quantize_fp8_weight_only", spy)
    override = MagicMock(name="bridged")
    be = image_modular.ModularImageBackend(
        repo="/m/flux2", device="cpu", dtype="fp8_e4m3", transformer_override=override)
    await be.infer(ImageRequest(request_id="ov", prompt="x", steps=2, width=64, height=64))
    pipe.update_components.assert_called_once_with(transformer=override)
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_apply_loras_loads_and_sets_adapters(monkeypatch):
    """PR-3:req.loras → pipe.load_lora_weights + set_adapters(Flux2Klein 经同 LoRA mixin)。
    fake pipe(type 非 Flux2)→ 跳过 comfy 转换分支(CI 不碰 safetensors/diffusers);
    comfy 转换由 #125 + PR-3 真模型 smoke 验。"""
    from src.services.inference.base import LoRASpec

    _m, _cm, pipe = _fake_modular(monkeypatch)
    pipe.get_active_adapters.return_value = ["turbo"]  # 零匹配检查通过
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")

    await be.infer(ImageRequest(
        request_id="L", prompt="x", steps=4, width=64, height=64,
        loras=[LoRASpec(name="turbo", path="/m/turbo.safetensors", strength=0.8)]))

    pipe.load_lora_weights.assert_called_once()
    assert pipe.load_lora_weights.call_args.kwargs.get("adapter_name") == "turbo"
    pipe.set_adapters.assert_called_once_with(["turbo"], adapter_weights=[0.8])


@pytest.mark.asyncio
async def test_no_loras_does_not_load(monkeypatch):
    _m, _cm, pipe = _fake_modular(monkeypatch)
    pipe.get_active_adapters.return_value = []
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    await be.infer(ImageRequest(request_id="n", prompt="x", steps=2, width=64, height=64))
    pipe.load_lora_weights.assert_not_called()


@pytest.mark.asyncio
async def test_modular_backend_rejects_non_image_request(monkeypatch):
    _fake_modular(monkeypatch)
    from src.services.inference.base import AudioRequest

    be = image_modular.ModularImageBackend(repo="/m/flux2")
    with pytest.raises(TypeError, match="ImageRequest"):
        await be.infer(AudioRequest(request_id="t", text="hi"))
