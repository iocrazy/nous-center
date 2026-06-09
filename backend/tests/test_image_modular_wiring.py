"""图像后端 wiring 测(CI 可跑,无 GPU/真 diffusers)。

PR「true-cfg 修复」后:Flux2 comfy 单文件走**标准** `Flux2KleinPipeline(is_distilled=False)`
(monkeypatch `_import_klein_pipeline`);PR-A 起非 Flux2 raise NotImplementedError
(modular 死代码已退役,新架构经 PR-C 的 ImageArchSpec 注册表接)。断言 ImageRequest → pipe() 参数映射 + 构建链;真出图走 standalone smoke
(spike_true_cfg.py / smoke_single_file_prod.py)。
"""
from __future__ import annotations

import asyncio
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
    """PR-3 + PR-1a:infer 给 pipe 装 callback_on_step_end → 每步调 progress_callback;
    PR-1a 加 stage 三态(text_encode 在 pipe() 前发 / dit_denoise 每步 / vae_decode 在 pipe() 后发)
    + step_latency_ms + eta_ms。本测试覆盖**新契约**(老 (done, total) 兼容由 _make_emit 兜底)。"""
    _k, _t, _s, pipe = _fake_klein(monkeypatch)
    events: list[dict] = []

    def on_p(step: int, total: int, **extras) -> None:
        events.append({"step": step, "total": total, **extras})

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
    # #PR flux2 to_thread:进度回调改 loop.call_soon_threadsafe 异步调度 → flush 一拍再断言。
    await asyncio.sleep(0)

    # pipe() 前 text_encode + pipe() 后 vae_decode + 3 个 dit_denoise = 5 个 event
    stages = [e["stage"] for e in events]
    assert stages == ["text_encode", "vae_decode", "dit_denoise", "dit_denoise", "dit_denoise"]
    # dit_denoise 三步 step(callback 位置参数 `done`)单调 + total 一致(`total` = total_steps)。
    dit = [e for e in events if e["stage"] == "dit_denoise"]
    assert [e["step"] for e in dit] == [1, 2, 3]
    assert all(e["total"] == 3 for e in dit)
    # 每步带 step_latency_ms(int)+ ETA(int 非负);最后一步 ETA = 0(total-step=0)。
    assert all(isinstance(e["step_latency_ms"], int) and e["step_latency_ms"] >= 0 for e in dit)
    assert dit[-1]["eta_ms"] == 0


@pytest.mark.asyncio
async def test_callback_legacy_done_total_signature_backcompat(monkeypatch):
    """PR-1a:老 fake 只接 `(done, total)` 不收 **extras —— _make_emit 应 TypeError 降级两次后,
    用最简 (done, total) 调用。验向后兼容,旧 fake 不必改就能存活。"""
    _k, _t, _s, pipe = _fake_klein(monkeypatch)
    calls: list[tuple[int, int]] = []

    def on_p(step: int, total: int) -> None:  # 老契约
        calls.append((step, total))

    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    await be.infer(
        ImageRequest(request_id="legacy", prompt="x", steps=3, width=64, height=64),
        progress_callback=on_p,
    )
    cb = pipe.call_args.kwargs.get("callback_on_step_end")
    cb(pipe, 0, None, {})
    cb(pipe, 1, None, {})
    cb(pipe, 2, None, {})
    await asyncio.sleep(0)  # call_soon_threadsafe 异步调度,flush 再断言

    # text_encode (0,3) + vae_decode (3,3) + 三步 dit_denoise (1,3)/(2,3)/(3,3)
    assert calls == [(0, 3), (3, 3), (1, 3), (2, 3), (3, 3)]


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
    # sgm_uniform 现已支持(#285 补齐 ComfyUI 9 个)→ 用真不存在的名字验 fail loud。
    _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    with pytest.raises(ValueError, match="调度器 'nonexistent_sched' 不被"):
        await be.infer(ImageRequest(request_id="g", prompt="x", steps=2, width=64, height=64,
                                    sampler_name="euler", scheduler="nonexistent_sched"))


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
async def test_offload_cross_gpu_routes_to_cross_gpu_helper(monkeypatch):
    """PR-D2:offload=cuda:N(≠device)→ 走 `_enable_cross_gpu_offload` 而非
    pipe.to / pipe.enable_model_cpu_offload(后两条是 PR-D 的 none/cpu 路径)。

    cross-GPU 的真实正确性(accelerate CpuOffload 子类、pipe._execution_device 解析、
    forward 跨卡 align)依赖真 accelerate + 真模型;conftest mock torch 跑不了,
    见 `tests/manual/smoke_cross_gpu_offload.py`(feedback_verify_real_model)。
    """
    from unittest.mock import patch  # noqa: PLC0415

    _klein, _tok, _sch, pipe = _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cuda:0", offload="cuda:1")

    # stub helper:验「被调一次 + 参数正确」就够。
    with patch.object(image_modular, "_enable_cross_gpu_offload") as spy:
        await be.infer(ImageRequest(request_id="o3", prompt="x", steps=2, width=64, height=64))
        spy.assert_called_once_with(pipe, compute_device="cuda:0", stash_device="cuda:1")

    # 不走 cpu / none 路径。
    pipe.to.assert_not_called()
    pipe.enable_model_cpu_offload.assert_not_called()


@pytest.mark.asyncio
async def test_offload_cross_gpu_rejects_same_card(monkeypatch):
    """PR-D2:offload=cuda:0 = device=cuda:0 → fail-loud(同卡无意义)。"""
    _fake_klein(monkeypatch)
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cuda:0", offload="cuda:0")
    with pytest.raises(ValueError, match="同卡"):
        await be.infer(ImageRequest(request_id="o4", prompt="x", steps=2, width=64, height=64))


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


def test_unload_clears_pipe_and_loras():
    """round3 #3:unload 必须拆 _pipe/_model + 清 lora,否则换模型时 GPU 显存不降。"""
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    pipe = MagicMock(name="pipe")
    be._pipe = pipe
    be._model = pipe
    be._loaded_loras.add("some-lora")
    assert be.is_loaded

    be.unload()

    assert be._pipe is None
    assert be._model is None
    assert not be.is_loaded  # _model is None
    assert be._loaded_loras == set()
    pipe.unload_lora_weights.assert_called_once()


def test_unload_is_idempotent_and_safe_when_never_loaded():
    """从没 _ensure_pipe 过(_pipe=None)时 unload 不该崩。"""
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    be.unload()  # no raise
    be.unload()  # 再次也安全
    assert be._pipe is None


def test_apply_loras_deletes_stale_on_switch():
    """round3 #6:切换 LoRA 时删掉不再请求的旧 LoRA(set_adapters 只停用、不释放)。"""
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    pipe = MagicMock(name="pipe")
    pipe.get_active_adapters.return_value = ['A', 'B']
    pipe.peft_config = {}
    be._pipe = pipe

    def _lora(name):
        s = MagicMock()
        s.name = name
        s.path = f"/l/{name}.bin"  # 非 .safetensors → 跳过 comfy convert 分支
        s.strength = 1.0
        return s

    be._apply_loras([_lora('A'), _lora('B')])
    assert be._loaded_loras == {'A', 'B'}
    pipe.delete_adapters.assert_not_called()  # 首次无 stale

    # 只请求 A → B 应被删除
    be._apply_loras([_lora('A')])
    pipe.delete_adapters.assert_called_once_with(['B'])
    assert be._loaded_loras == {'A'}


def test_unload_resets_sched_key():
    """round5:unload 复位 _sched_key,否则同实例 rebuild 后 _apply_scheduler 缓存早退
    会漏装实际 scheduler。"""
    be = image_modular.ModularImageBackend(repo="/m/flux2", device="cpu")
    be._sched_key = ("heun", "karras")  # 模拟上一轮装了非默认 scheduler
    be._pipe = MagicMock(name="pipe")
    be._model = be._pipe
    be.unload()
    assert be._sched_key == ("euler", "normal")


# ---- PR-A2:img2img(z-image strength)门控 + img2img pipe 复用组件 ----------------

import pytest as _pytest  # noqa: E402


@_pytest.mark.parametrize("pipeline_class,input_image,strength,expected", [
    ("ZImagePipeline", "/tmp/a.png", 0.6, True),    # z-image 有 img2img 变体 + 图 + 0<s<1 → img2img
    ("ZImagePipeline", "/tmp/a.png", 1.0, False),   # strength=1 = 全量去噪 ≈ 文生图(零回归)
    ("ZImagePipeline", "/tmp/a.png", 0.0, False),   # strength=0 边界 → 不 img2img
    ("ZImagePipeline", None, 0.6, False),           # 没连输入图 → 不 img2img
    ("Flux2KleinPipeline", "/tmp/a.png", 0.6, False),  # flux2 无 img2img 变体(接图走多参考编辑)
    ("QwenImageEditPlusPipeline", "/tmp/a.png", 0.6, False),  # qwen-edit 无 img2img 变体
])
def test_wants_img2img_gating(pipeline_class, input_image, strength, expected):
    """img2img 门控:仅 arch 注册了 img2img_pipeline_class + 连了 input_image + 0<strength<1。"""
    be = image_modular.ModularImageBackend(repo="/m/x", device="cpu", pipeline_class=pipeline_class)
    req = ImageRequest(request_id="t", prompt="x", input_image=input_image, strength=strength)
    assert be._wants_img2img(req) is expected


def test_ensure_img2img_pipe_reuses_components(monkeypatch):
    """_ensure_img2img_pipe 用 text2img pipe 的 components 构建 img2img 类(不重载权重)。"""
    base = MagicMock(name="base_pipe")
    base.components = {"transformer": "T", "vae": "V", "text_encoder": "E"}
    img_pipe = MagicMock(name="img2img_pipe")
    img_cls = MagicMock(name="ZImageImg2ImgPipeline", return_value=img_pipe)
    monkeypatch.setattr(image_modular, "_import_img2img_pipeline", lambda cls_name: img_cls)

    be = image_modular.ModularImageBackend(repo="/m/z", device="cpu", pipeline_class="ZImagePipeline")
    be._pipe = base  # _ensure_pipe 早退返回它(不构建真 pipe)
    be._model = base

    got = be._ensure_img2img_pipe()
    assert got is img_pipe
    img_cls.assert_called_once_with(transformer="T", vae="V", text_encoder="E")  # 复用组件
    # 二次调用缓存,不重建
    assert be._ensure_img2img_pipe() is img_pipe
    img_cls.assert_called_once()


def test_unload_clears_img2img_pipe():
    """unload 清 _img2img_pipe(与 _pipe 共享组件,随之释放)。"""
    be = image_modular.ModularImageBackend(repo="/m/z", device="cpu", pipeline_class="ZImagePipeline")
    be._pipe = MagicMock(name="pipe")
    be._img2img_pipe = MagicMock(name="img2img")
    be._model = be._pipe
    be.unload()
    assert be._img2img_pipe is None


# ---- PR-B2:留噪 latent 接力 / 分段采样门控 + 派发前 arch 校验 -----------------------


@_pytest.mark.parametrize("pipeline_class,kwargs,expected", [
    # z-image + 各分段字段非默认 → 走手写分段循环
    ("ZImagePipeline", {"start_at_step": 5}, True),                       # 续采段
    ("ZImagePipeline", {"end_at_step": 5, "steps": 12}, True),            # base 留噪截断段
    ("ZImagePipeline", {"add_noise": False}, True),                       # 注入原样续采
    ("ZImagePipeline", {"init_latent_ref": {"path": "/t/x.safetensors"}}, True),
    # 全默认 → 整段采样(零回归)
    ("ZImagePipeline", {}, False),
    ("ZImagePipeline", {"start_at_step": 0, "add_noise": True}, False),
    ("ZImagePipeline", {"end_at_step": 12, "steps": 12}, False),          # end>=steps 不截断 → 不触发
    # 跨模型 latent 不兼容:Flux2 / qwen 即便带分段字段也不走此路(只 z-image 同 16ch 空间)
    ("Flux2KleinPipeline", {"start_at_step": 5, "init_latent_ref": {"path": "/t/x"}}, False),
    ("QwenImageEditPlusPipeline", {"add_noise": False}, False),
])
def test_wants_segmented_gating(pipeline_class, kwargs, expected):
    """分段采样门控:仅 ZImagePipeline + 任一分段字段真正非默认。"""
    be = image_modular.ModularImageBackend(repo="/m/x", device="cpu", pipeline_class=pipeline_class)
    req = ImageRequest(request_id="t", prompt="x", **kwargs)
    assert be._wants_segmented(req) is expected


def test_load_init_latent_rejects_cross_arch():
    """跨架构 latent 注入(flux2 latent → z-image 段)派发前人话报错,不崩 transformer 深处
    (对齐 [[project_anima_arch_mismatch]])。校验在读文件前 → 无需真 latent 文件/torch。"""
    be = image_modular.ModularImageBackend(repo="/m/z", device="cpu", pipeline_class="ZImagePipeline")
    req = ImageRequest(request_id="t", prompt="x",
                       init_latent_ref={"path": "/t/x.safetensors", "arch": "flux2", "latent_channels": 128})
    with pytest.raises(ValueError, match="架构不匹配|不兼容"):
        be._load_init_latent(req, "cpu")


def test_load_init_latent_rejects_missing_path():
    be = image_modular.ModularImageBackend(repo="/m/z", device="cpu", pipeline_class="ZImagePipeline")
    req = ImageRequest(request_id="t", prompt="x", init_latent_ref={"arch": "z-image"})
    with pytest.raises(ValueError, match="缺 path"):
        be._load_init_latent(req, "cpu")


def test_segmented_fields_default_zero_regression():
    """ImageRequest 分段字段默认值 = 不触发分段(契约零回归)。"""
    req = ImageRequest(request_id="t", prompt="x")
    assert req.start_at_step == 0
    assert req.end_at_step is None
    assert req.add_noise is True
    assert req.return_with_leftover_noise is False
    assert req.init_latent_ref is None
