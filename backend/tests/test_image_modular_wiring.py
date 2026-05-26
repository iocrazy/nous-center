"""图像后端 wiring 测(CI 可跑,无 GPU/真 diffusers)。

PR「true-cfg 修复」后:Flux2 comfy 单文件走**标准** `Flux2KleinPipeline(is_distilled=False)`
(monkeypatch `_import_klein_pipeline`);非 Flux2(ERNIE)留 modular fallback(monkeypatch
`_import_modular`)。断言 ImageRequest → pipe() 参数映射 + 构建链;真出图走 standalone smoke
(spike_true_cfg.py / smoke_single_file_prod.py)。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.services.inference.base import ImageRequest, LoRASpec
from src.services.inference import image_modular


def _fake_klein(monkeypatch):
    """装 fake (Flux2KleinPipeline, AutoTokenizer, FlowMatchEulerDiscreteScheduler) 并返回 pipe mock。

    pipe 既是 `klein_cls(...)`(直接装配)又是 `klein_cls.from_pretrained(...)`(HF-layout)的返回值。
    """
    pipe = MagicMock(name="pipe")
    out = MagicMock(name="out")
    img = MagicMock(name="img")
    img.save.side_effect = lambda buf, format: buf.write(b"\x89PNG\r\n\x1a\n")  # noqa: A002
    out.images = [img]
    pipe.return_value = out  # pipe(...) → out

    klein_cls = MagicMock(name="Flux2KleinPipeline")
    klein_cls.return_value = pipe                 # 直接装配(全单文件 override)
    klein_cls.from_pretrained.return_value = pipe  # HF-layout
    tokenizer_cls = MagicMock(name="AutoTokenizer")
    scheduler_cls = MagicMock(name="FlowMatchEulerDiscreteScheduler")

    monkeypatch.setattr(
        image_modular, "_import_klein_pipeline",
        lambda: (klein_cls, tokenizer_cls, scheduler_cls))
    return klein_cls, tokenizer_cls, scheduler_cls, pipe


def _fake_modular(monkeypatch):
    """非 Flux2 fallback:fake (ModularPipeline, ComponentsManager) → pipe mock。"""
    pipe = MagicMock(name="modular_pipe")
    out = MagicMock(name="out")
    img = MagicMock(name="img")
    img.save.side_effect = lambda buf, format: buf.write(b"\x89PNG\r\n\x1a\n")  # noqa: A002
    out.images = [img]
    pipe.return_value = out

    modular_cls = MagicMock(name="ModularPipeline")
    modular_cls.from_pretrained.return_value = pipe
    cm_cls = MagicMock(name="ComponentsManager")
    monkeypatch.setattr(image_modular, "_import_modular", lambda: (modular_cls, cm_cls))
    return modular_cls, cm_cls, pipe


# device="cpu":wiring 测不碰真 CUDA(CI 无 GPU);torch 在 conftest 被 mock。


@pytest.mark.asyncio
async def test_klein_backend_maps_request_to_pipe(monkeypatch):
    """HF-layout(无 override)→ from_pretrained;ImageRequest → pipe() 参数映射。"""
    klein_cls, _tok, _sch, pipe = _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu", dtype="bfloat16")

    res = await be.infer(
        ImageRequest(request_id="t1", prompt="a fox", steps=7, width=512, height=512, seed=42)
    )

    assert res.media_type == "image/png"
    assert res.data.startswith(b"\x89PNG")
    assert res.usage.image_count == 1
    assert res.metadata["engine"] == "flux2klein"

    klein_cls.from_pretrained.assert_called_once()
    assert klein_cls.from_pretrained.call_args.args[0] == "/m/flux2"
    pipe.to.assert_called_once_with("cpu")

    kw = pipe.call_args.kwargs
    assert kw["prompt"] == "a fox"
    assert kw["num_inference_steps"] == 7
    assert kw["width"] == 512 and kw["height"] == 512
    assert kw["guidance_scale"] == 7.0  # cfg_scale 默认 7.0 → guidance_scale
    assert "generator" in kw
    assert "negative_prompt_embeds" not in kw  # 无 negative


@pytest.mark.asyncio
async def test_klein_negative_encoded_to_embeds_at_cfg_gt_1(monkeypatch):
    """cfg>1 + negative → pipe.encode_prompt(neg) → negative_prompt_embeds(true-CFG)。"""
    _klein, _tok, _sch, pipe = _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")

    await be.infer(ImageRequest(
        request_id="neg", prompt="a fox", negative_prompt="blurry, ugly",
        steps=4, width=64, height=64, cfg_scale=4.0))

    pipe.encode_prompt.assert_called_once()
    assert pipe.encode_prompt.call_args.kwargs.get("prompt") == "blurry, ugly"
    assert "negative_prompt_embeds" in pipe.call_args.kwargs


@pytest.mark.asyncio
async def test_klein_negative_skipped_at_cfg_1(monkeypatch):
    """cfg=1 → 无 CFG → negative 被忽略(不编码、不传 embeds),对齐 ComfyUI。"""
    _klein, _tok, _sch, pipe = _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")

    await be.infer(ImageRequest(
        request_id="n1", prompt="a fox", negative_prompt="blurry",
        steps=4, width=64, height=64, cfg_scale=1.0))

    pipe.encode_prompt.assert_not_called()
    assert "negative_prompt_embeds" not in pipe.call_args.kwargs
    assert pipe.call_args.kwargs["guidance_scale"] == 1.0


@pytest.mark.asyncio
async def test_klein_all_three_overrides_assemble_true_cfg(monkeypatch):
    """全单文件三 override → 直接装配 klein_cls(is_distilled=False),不 from_pretrained。"""
    klein_cls, tok_cls, sch_cls, _pipe = _fake_klein(monkeypatch)
    t, c, v = MagicMock(name="t"), MagicMock(name="c"), MagicMock(name="v")
    be = image_modular.ModularImageBackend(
        repo="/m/flux2", device="cpu",
        transformer_override=t, text_encoder_override=c, vae_override=v)

    await be.infer(ImageRequest(request_id="sf", prompt="x", steps=2, width=64, height=64))

    klein_cls.from_pretrained.assert_not_called()
    klein_cls.assert_called_once()
    kw = klein_cls.call_args.kwargs
    assert kw["is_distilled"] is False
    assert kw["transformer"] is t and kw["text_encoder"] is c and kw["vae"] is v
    tok_cls.from_pretrained.assert_called_once()
    sch_cls.from_pretrained.assert_called_once()


@pytest.mark.asyncio
async def test_klein_partial_override_registers_modules(monkeypatch):
    """部分 override(仅 transformer)→ from_pretrained 后 register_modules 换入。"""
    klein_cls, _tok, _sch, pipe = _fake_klein(monkeypatch)
    t = MagicMock(name="t")
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu", transformer_override=t)

    await be.infer(ImageRequest(request_id="p", prompt="x", steps=2, width=64, height=64))

    klein_cls.from_pretrained.assert_called_once()
    pipe.register_modules.assert_called_once_with(transformer=t)


@pytest.mark.asyncio
async def test_klein_reuses_pipe_across_infers(monkeypatch):
    klein_cls, _tok, _sch, _pipe = _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")

    await be.infer(ImageRequest(request_id="a", prompt="x", steps=2, width=64, height=64))
    await be.infer(ImageRequest(request_id="b", prompt="y", steps=2, width=64, height=64))

    klein_cls.from_pretrained.assert_called_once()  # 只建一次
    assert be.is_loaded


@pytest.mark.asyncio
async def test_fp8_dtype_triggers_weight_only_quant(monkeypatch):
    _fake_klein(monkeypatch)
    spy = MagicMock(name="quantize_fp8")
    monkeypatch.setattr(image_modular, "_quantize_fp8_weight_only", spy)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu", dtype="fp8_e4m3")
    await be.infer(ImageRequest(request_id="q8", prompt="x", steps=2, width=64, height=64))
    spy.assert_called_once()


@pytest.mark.asyncio
async def test_bf16_dtype_no_fp8_quant(monkeypatch):
    _fake_klein(monkeypatch)
    spy = MagicMock(name="quantize_fp8")
    monkeypatch.setattr(image_modular, "_quantize_fp8_weight_only", spy)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu", dtype="bfloat16")
    await be.infer(ImageRequest(request_id="b16", prompt="x", steps=2, width=64, height=64))
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_scheduler_swap_sigma_schedule(monkeypatch):
    """PR-2:scheduler→use_*_sigmas 互斥开关;换 FlowMatchEuler.from_config;保留原 config(shift)。"""
    _klein, _tok, _sch, pipe = _fake_klein(monkeypatch)
    pipe.scheduler.config = {"shift": 3.0, "use_dynamic_shifting": True}
    euler, heun, lcm = MagicMock(name="EulerCls"), MagicMock(name="HeunCls"), MagicMock(name="LCMCls")
    monkeypatch.setattr(image_modular, "_import_flow_schedulers",
                        lambda: {"euler": euler, "heun": heun, "lcm": lcm})
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")

    await be.infer(ImageRequest(request_id="s", prompt="x", steps=2, width=64, height=64,
                                sampler_name="euler", scheduler="karras"))

    euler.from_config.assert_called_once()
    cfg = euler.from_config.call_args.args[0]
    assert cfg["use_karras_sigmas"] is True
    assert cfg["use_exponential_sigmas"] is False and cfg["use_beta_sigmas"] is False
    assert cfg["shift"] == 3.0  # 原 config 保留
    assert be._sched_key == ("euler", "karras")


@pytest.mark.asyncio
async def test_unsupported_sampler_fails_loud(monkeypatch):
    """选了本架构不支持的采样器(heun 之于 Flux2)→ 清晰报错,不出图(不崩在 diffusers 深处)。"""
    _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    with pytest.raises(ValueError, match="采样器 'heun' 不被"):
        await be.infer(ImageRequest(request_id="h", prompt="x", steps=2, width=64, height=64,
                                    sampler_name="heun", scheduler="normal"))


@pytest.mark.asyncio
async def test_unsupported_scheduler_fails_loud(monkeypatch):
    _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    with pytest.raises(ValueError, match="调度器 'sgm_uniform' 不被"):
        await be.infer(ImageRequest(request_id="g", prompt="x", steps=2, width=64, height=64,
                                    sampler_name="euler", scheduler="sgm_uniform"))


@pytest.mark.asyncio
async def test_scheduler_default_euler_normal_no_swap(monkeypatch):
    """默认 euler+normal = 参考库现状 → 不换 scheduler(不 import,出图基线不动)。"""
    _fake_klein(monkeypatch)
    spy = MagicMock(name="import_flow_schedulers")
    monkeypatch.setattr(image_modular, "_import_flow_schedulers", spy)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    await be.infer(ImageRequest(request_id="d", prompt="x", steps=2, width=64, height=64))
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_apply_loras_loads_and_sets_adapters(monkeypatch):
    """req.loras → pipe.load_lora_weights + set_adapters(标准 Flux2KleinPipeline 经 Flux2LoraLoaderMixin)。"""
    _klein, _tok, _sch, pipe = _fake_klein(monkeypatch)
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
    _klein, _tok, _sch, pipe = _fake_klein(monkeypatch)
    pipe.get_active_adapters.return_value = []
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    await be.infer(ImageRequest(request_id="n", prompt="x", steps=2, width=64, height=64))
    pipe.load_lora_weights.assert_not_called()


@pytest.mark.asyncio
async def test_non_flux2_falls_back_to_modular(monkeypatch):
    """pipeline_class != Flux2KleinPipeline(ERNIE 等)→ modular fallback 装配链。"""
    modular_cls, _cm, pipe = _fake_modular(monkeypatch)
    be = image_modular.ModularImageBackend(
        repo="/m/ernie", device="cpu", pipeline_class="ErnieImagePipeline")

    res = await be.infer(ImageRequest(request_id="e", prompt="x", steps=3, width=64, height=64))

    modular_cls.from_pretrained.assert_called_once()
    pipe.load_components.assert_called_once()
    assert res.metadata["engine"] == "modular"


@pytest.mark.asyncio
async def test_backend_rejects_non_image_request(monkeypatch):
    _fake_klein(monkeypatch)
    from src.services.inference.base import AudioRequest

    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    with pytest.raises(TypeError, match="ImageRequest"):
        await be.infer(AudioRequest(request_id="t", text="hi"))


def test_wants_fp8_detects_fp8_dtypes():
    assert image_modular._wants_fp8("fp8_e4m3")
    assert image_modular._wants_fp8("fp8_e5m2")
    assert image_modular._wants_fp8("fp8")
    assert not image_modular._wants_fp8("bfloat16")
    assert not image_modular._wants_fp8("float16")
    assert not image_modular._wants_fp8("")
    assert not image_modular._wants_fp8(None)
