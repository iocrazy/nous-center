"""图像后端 wiring 测(CI 可跑,无 GPU/真 diffusers)。

PR「true-cfg 修复」后:Flux2 comfy 单文件走**标准** `Flux2KleinPipeline(is_distilled=False)`
(monkeypatch `_import_klein_pipeline`);PR-A 起非 Flux2 raise NotImplementedError
(modular 死代码已退役,新架构经 PR-C 的 ImageArchSpec 注册表接)。断言 ImageRequest → pipe() 参数映射 + 构建链;真出图走 standalone smoke
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


# device="cpu":wiring 测不碰真 CUDA(CI 无 GPU);torch 在 conftest 被 mock。
# PR-A:删了 _fake_modular(modular 死代码已退役;非 Flux2 改 raise NotImplementedError)。


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
async def test_callback_on_step_end_forwards_progress(monkeypatch):
    """PR-3:infer 给 pipe 装 callback_on_step_end → 每步调 progress_callback(done, total)。"""
    _k, _t, _s, pipe = _fake_klein(monkeypatch)
    progress: list[tuple[int, int]] = []

    def on_p(step: int, total: int) -> None:
        progress.append((step, total))

    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    await be.infer(
        ImageRequest(request_id="p", prompt="x", steps=3, width=64, height=64),
        progress_callback=on_p,
    )
    cb = pipe.call_args.kwargs.get("callback_on_step_end")
    assert cb is not None
    # 模拟 diffusers pipe 每步调 callback
    cb(pipe, 0, None, {})
    cb(pipe, 1, None, {})
    cb(pipe, 2, None, {})
    assert progress == [(1, 3), (2, 3), (3, 3)]


@pytest.mark.asyncio
async def test_callback_raises_cancelled_on_flag_set(monkeypatch):
    """PR-3:cancel_flag 置位 → callback raise CancelledError(穿出 pipe(),runner 落 cancelled)。"""
    import asyncio
    import threading

    _k, _t, _s, pipe = _fake_klein(monkeypatch)
    cancel = threading.Event()
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    await be.infer(
        ImageRequest(request_id="c", prompt="x", steps=3, width=64, height=64),
        cancel_flag=cancel,
    )
    cb = pipe.call_args.kwargs.get("callback_on_step_end")
    assert cb is not None
    # 未置位 → 正常通过
    cb(pipe, 0, None, {})
    # 置位 → raise CancelledError
    cancel.set()
    with pytest.raises(asyncio.CancelledError):
        cb(pipe, 1, None, {})


@pytest.mark.asyncio
async def test_no_callback_attached_when_progress_and_cancel_absent(monkeypatch):
    """没传 progress_callback / cancel_flag → pipe() 不带 callback_on_step_end(默认行为)。"""
    _k, _t, _s, pipe = _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    await be.infer(ImageRequest(request_id="n", prompt="x", steps=2, width=64, height=64))
    assert "callback_on_step_end" not in pipe.call_args.kwargs


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
async def test_offload_none_calls_pipe_to(monkeypatch):
    """PR-D:offload=none → pipe.to(device)(普通路径,不挂 accelerate hook)。"""
    _klein, _tok, _sch, pipe = _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu", offload="none")
    await be.infer(ImageRequest(request_id="o1", prompt="x", steps=2, width=64, height=64))
    pipe.to.assert_called_once_with("cpu")
    pipe.enable_model_cpu_offload.assert_not_called()


@pytest.mark.asyncio
async def test_offload_cpu_calls_enable_model_cpu_offload(monkeypatch):
    """PR-D:offload=cpu → pipe.enable_model_cpu_offload(gpu_id=N)(替代 .to)。"""
    _klein, _tok, _sch, pipe = _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cuda:2", offload="cpu")
    await be.infer(ImageRequest(request_id="o2", prompt="x", steps=2, width=64, height=64))
    pipe.enable_model_cpu_offload.assert_called_once_with(gpu_id=2)
    pipe.to.assert_not_called()


@pytest.mark.asyncio
async def test_offload_cuda_not_implemented(monkeypatch):
    """PR-D:offload=cuda:N 跨卡 offload 留 PR-D2 → fail-loud。"""
    _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cuda:0", offload="cuda:1")
    with pytest.raises(NotImplementedError, match="offload='cuda:1'"):
        await be.infer(ImageRequest(request_id="o3", prompt="x", steps=2, width=64, height=64))


@pytest.mark.asyncio
async def test_non_flux2_raises_not_implemented(monkeypatch):
    """PR-A:pipeline_class != Flux2KleinPipeline → fail-loud NotImplementedError
    (modular 死代码已退役;ERNIE/Qwen-Image/AuraFlow 等待 PR-C 经 ImageArchSpec 注册表接入)。"""
    _fake_klein(monkeypatch)  # 即便走 klein seam 也无所谓 —— 不应被调
    be = image_modular.ModularImageBackend(
        repo="/m/ernie", device="cpu", pipeline_class="ErnieImagePipeline")

    with pytest.raises(NotImplementedError, match="ErnieImagePipeline"):
        await be.infer(ImageRequest(request_id="e", prompt="x", steps=3, width=64, height=64))


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
