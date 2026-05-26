"""Modular Diffusers 图像引擎(ModularPipeline + ComponentsManager)。

spec: 2026-05-22-image-engine-modular-diffusers-design.md。PR-1:与现有
`DiffusersImageBackend`(自写 ImageSampler)**并存灰度**,经 `NOUS_IMAGE_ENGINE`
选择(默认 legacy)。PR-4 才删旧。

约束(plan-eng-review D2):`diffusers.modular*` / `diffusers` 的 import **只允许经
`_import_modular()` 这一个函数**(blast-radius 隔离 + conftest mock torch 时模块顶层
不 import diffusers/torch,避免 collection 崩)。
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, ClassVar

from src.services.inference.base import (
    ImageRequest,
    InferenceAdapter,
    InferenceRequest,
    InferenceResult,
    MediaModality,
    UsageMeter,
)


def _import_modular() -> tuple[Any, Any]:
    """Lazy import —— diffusers 的 Modular API 只在这里取(D2 隔离)。

    测试可 monkeypatch 本函数返回 fake,从而无需真 diffusers/GPU 验证参数映射。
    """
    from diffusers import ComponentsManager, ModularPipeline  # noqa: PLC0415

    return ModularPipeline, ComponentsManager


def _import_flux2_transformer() -> Any:
    """Lazy import Flux2Transformer2DModel(D2:diffusers import 只在本文件)。"""
    from diffusers import Flux2Transformer2DModel  # noqa: PLC0415

    return Flux2Transformer2DModel


def _import_klein_pipeline() -> tuple[Any, Any, Any]:
    """Lazy import 标准(非 modular)Flux2 Klein pipeline + tokenizer/scheduler 类(D2 隔离)。

    comfy 单文件类别走**标准** `Flux2KleinPipeline`(true-cfg / negative / 逐步回调内置,行为对齐
    ComfyUI),取代 modular 蒸馏管线 —— 后者把 cfg/negative 掐了,是图像质量/控制不如 ComfyUI 的根因
    (真模型 A/B 已证,见 tests/manual/spike_true_cfg.py + plan 2026-05-25)。测试 monkeypatch 本函数注入 fake。
    """
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    return Flux2KleinPipeline, AutoTokenizer, FlowMatchEulerDiscreteScheduler


def _import_flow_schedulers() -> dict:
    """Lazy import diffusers flow-matching scheduler 类(PR-2 采样器选择;D2 隔离)。

    复刻 ComfyUI KSampler 的 sampler_name 下拉,但选项是 diffusers 的:Flux2 是 flow-matching,
    只有这 3 个兼容(不追 ComfyUI 40 个 k-diffusion 采样器)。测试可 monkeypatch 本函数。
    """
    from diffusers import (  # noqa: PLC0415
        FlowMatchEulerDiscreteScheduler,
        FlowMatchHeunDiscreteScheduler,
        FlowMatchLCMScheduler,
    )

    return {
        "euler": FlowMatchEulerDiscreteScheduler,
        "heun": FlowMatchHeunDiscreteScheduler,
        "lcm": FlowMatchLCMScheduler,
    }


def _maybe_convert_comfy_flux2_lora(state_dict: dict):
    """ComfyUI/BFL 格式 Flux2 LoRA(`diffusion_model.` 前缀 + `lora_down/up`)→ diffusers
    `transformer.*` 格式;否则 None(走原 load_lora_weights)。绕 Flux2LoraLoaderMixin 的
    is_kohya 误判(把 ComfyUI LoRA 误路由到 lora_unet_ 转换器→零匹配)。#125 实测 242→306 键。
    PR-4 从 image_diffusers 搬来(legacy 删除后存活;diffusers import 仍只在本文件)。
    """
    if not any(k.startswith("diffusion_model.") for k in state_dict):
        return None
    if not any(".lora_down.weight" in k or ".lora_up.weight" in k for k in state_dict):
        return None
    from diffusers.loaders.lora_conversion_utils import (  # noqa: PLC0415
        _convert_non_diffusers_flux2_lora_to_diffusers,
    )
    return _convert_non_diffusers_flux2_lora_to_diffusers(dict(state_dict))


def build_bridged_transformer(unet_spec: Any, repo: str, device: str) -> Any:
    """comfy 量化单文件 → diffusers transformer module(PR-2 桥接)。

    dequant_and_convert(quant_loaders 反量化 + diffusers 转键)→ from_config(HF repo 的
    transformer config)→ load_state_dict → 落 device。喂 `ModularImageBackend(transformer_override=)`。
    """
    from src.services.inference.quant_loaders import dequant_and_convert  # noqa: PLC0415

    transformer_cls = _import_flux2_transformer()
    sd = dequant_and_convert(unet_spec)
    cfg = transformer_cls.load_config(str(Path(repo) / "transformer"))
    module = transformer_cls.from_config(cfg).to(_torch_dtype(unet_spec.dtype))
    missing, unexpected = module.load_state_dict(sd, strict=False)
    if missing or unexpected:
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).warning(
            "build_bridged_transformer: missing=%d unexpected=%d(键不全,查转换器/config)",
            len(missing), len(unexpected),
        )
    return module.to(device)


def _materialize_meta_params(module: Any) -> None:
    """加载后仍在 meta 设备的 param(未被 state_dict 覆盖,如 comfy 省掉的 tied lm_head)→
    零初始化兜底(这些权重在图像文本编码里不用)。只动 meta,不碰已加载权重。"""
    import torch  # noqa: PLC0415

    for name, p in list(module.named_parameters()):
        if not p.is_meta:
            continue
        parent = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
        attr = name.rsplit(".", 1)[1]
        setattr(parent, attr, torch.nn.Parameter(
            torch.zeros(p.shape, dtype=torch.bfloat16), requires_grad=False))


def build_bridged_text_encoder(clip_spec: Any, repo: str, device: str) -> Any:
    """comfy 单文件 text encoder → 参考整模型 config 建模型 + comfy 逐张量反量化(标准键,无需转)。

    repo `text_encoder/config.json` 建空模型(init_empty_weights)→ `dequant_comfy_mixed` 反量化单文件
    (fp8/nvfp4/plain)→ load_state_dict(assign)→ tie_weights(lm_head)→ meta 兜底 → 落 device。
    真模型验过(spike_single_file_assembly)。当前 Flux2=Qwen3;架构扩展见 spec。
    """
    import torch  # noqa: PLC0415
    from accelerate import init_empty_weights  # noqa: PLC0415
    from transformers import AutoConfig, AutoModelForCausalLM  # noqa: PLC0415

    from src.services.inference.quant_loaders import dequant_comfy_mixed  # noqa: PLC0415

    sd = dequant_comfy_mixed(clip_spec)
    cfg = AutoConfig.from_pretrained(str(Path(repo) / "text_encoder"))
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if unexpected:
        import logging  # noqa: PLC0415
        logging.getLogger(__name__).warning(
            "build_bridged_text_encoder: unexpected=%d(键不符 config?)", len(unexpected))
    model.tie_weights()
    _materialize_meta_params(model)
    return model.to(device, dtype=torch.bfloat16)


def build_bridged_vae(vae_spec: Any, repo: str, device: str) -> Any:
    """comfy 单文件 vae → 参考整模型 vae config 建 + 反量化单文件权重(plain/comfy 标准键)。"""
    import torch  # noqa: PLC0415
    from accelerate import init_empty_weights  # noqa: PLC0415
    from diffusers import AutoencoderKLFlux2  # noqa: PLC0415

    from src.services.inference.quant_loaders import dequant_comfy_mixed  # noqa: PLC0415

    sd = dequant_comfy_mixed(vae_spec)
    cfg = AutoencoderKLFlux2.load_config(str(Path(repo) / "vae"))
    with init_empty_weights():
        vae = AutoencoderKLFlux2.from_config(cfg)
    vae.load_state_dict(sd, strict=False, assign=True)
    _materialize_meta_params(vae)
    return vae.to(device, dtype=torch.bfloat16)


def _torch_dtype(dtype_str: str) -> Any:
    """'bfloat16'/'float16'/'float32' → torch.dtype;'default' → None(原生精度)。
    fp8_* → bfloat16(**compute dtype**):fp8 经 torchao weight-only 量化实现(权重 fp8 存储 /
    bf16 计算),组件先按 bf16 加载再量化,所以这里 fp8 映射到 bf16。"""
    import torch  # noqa: PLC0415

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "default": None,
    }.get(dtype_str, torch.bfloat16)


def _wants_fp8(dtype_str: str) -> bool:
    """weight_dtype 是否要求 fp8(fp8_e4m3 / fp8_e5m2 / fp8)。"""
    return (dtype_str or "").lower().startswith("fp8")


def _quantize_fp8_weight_only(pipe: Any) -> None:
    """torchao weight-only fp8:transformer + text_encoder 权重 fp8 存储(省 ~½ 显存),bf16 计算。

    量化权重包在 tensor subclass 里,对外 `model.dtype` 仍报 bf16 → **不破坏 modular pipe 的
    noise/latents dtype 推导**(spec 2026-05-25;raw layerwise-casting 会让 dtype 变 fp8 → randn 崩)。
    注:Ampere(sm<8.9,如 3090)无 fp8 matmul 核 → fp8 **只省显存不加速**(正常,见 spec);
    fp8 真加速要 sm≥8.9(Ada/Hopper/Blackwell)。
    """
    from torchao.quantization import Float8WeightOnlyConfig, quantize_  # noqa: PLC0415

    cfg = Float8WeightOnlyConfig()
    for name in ("transformer", "text_encoder"):
        mod = getattr(pipe, name, None)
        if mod is not None:
            quantize_(mod, cfg)


class ModularImageBackend(InferenceAdapter):
    """ModularPipeline 后端,实现现有 `InferenceAdapter.infer` 接口(与 TTS/LLM 一致)。

    PR-1 只做 HF-layout 基线(bf16);量化桥接 PR-2、LoRA PR-3。
    """

    modality: ClassVar[MediaModality] = MediaModality.IMAGE
    estimated_vram_mb: ClassVar[int] = 0

    def __init__(
        self,
        repo: str,
        device: str = "cuda",
        *,
        dtype: str = "bfloat16",
        pipeline_class: str = "Flux2KleinPipeline",
        components_manager: Any = None,
        transformer_override: Any = None,
        text_encoder_override: Any = None,
        vae_override: Any = None,
        **params: Any,
    ):
        super().__init__(paths={"main": repo}, device=device, **params)
        self.repo = repo
        self.dtype = dtype
        # comfy 单文件 Flux2 → 标准 Flux2KleinPipeline(true-cfg);其它架构(ERNIE 等)→ modular fallback。
        self.pipeline_class = pipeline_class
        self._cm = components_manager
        # comfy 单文件桥接 override:transformer(量化/comfy)+ text_encoder/vae(单文件装配,PR-2)。
        self._transformer_override = transformer_override
        self._text_encoder_override = text_encoder_override
        self._vae_override = vae_override
        self._pipe: Any = None
        self._loaded_loras: set[str] = set()  # PR-3
        # PR-2 当前已装的 (sampler_name, scheduler);初值 = 参考库默认(FlowMatchEuler + normal sigmas)
        # → 默认请求不触发换 scheduler。
        self._sched_key: tuple[str, str] = ("euler", "normal")

    async def load(self, device: str) -> None:
        """对齐 ABC;实际 pipeline 构建在首次 infer 时 lazy(_ensure_pipe)。"""
        self.device = device

    def _ensure_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        # Flux2 comfy 单文件 → 标准 Flux2KleinPipeline(true-cfg,行为对齐 ComfyUI);
        # 其它架构(ERNIE 等)暂留 modular fallback(架构收口 spec 后统一)。
        if self.pipeline_class == "Flux2KleinPipeline":
            pipe = self._build_klein_pipe()
        else:
            pipe = self._build_modular_pipe()
        if _wants_fp8(self.dtype):
            # fp8 weight-only(torchao):transformer + text_encoder 权重 fp8 存储,省 ~½ 显存,
            # 让大模型塞进 24GB 3090(出图正确,见 spec 2026-05-25)。作用于最终组件(override 或 repo)。
            _quantize_fp8_weight_only(pipe)
        pipe.to(self.device)
        self._pipe = pipe
        self._model = pipe  # is_loaded → True
        return pipe

    def _build_klein_pipe(self) -> Any:
        """comfy Flux2 单文件 → 标准 `Flux2KleinPipeline(is_distilled=False)` = true-CFG。

        全单文件(三桥接 override):直接用桥接组件 + 参考库的 tokenizer/scheduler 装配 ——
        免加载参考库的整 transformer(省 ~18GB),且 `is_distilled=False` 让 cfg/negative 真生效
        (用户用 cfg 控:cfg=1 退化无 CFG,cfg>1+negative=true-CFG,对齐 ComfyUI)。
        HF-layout 整模型(无 override):`from_pretrained` 加载并尊重其 model_index 的 is_distilled
        (官方蒸馏 klein 仍蒸馏);有部分 override 则换入。
        """
        klein_cls, tokenizer_cls, scheduler_cls = _import_klein_pipeline()
        overrides = {k: v for k, v in (
            ("transformer", self._transformer_override),
            ("text_encoder", self._text_encoder_override),
            ("vae", self._vae_override),
        ) if v is not None}
        if len(overrides) == 3:
            tokenizer = tokenizer_cls.from_pretrained(str(Path(self.repo) / "tokenizer"))
            scheduler = scheduler_cls.from_pretrained(str(Path(self.repo) / "scheduler"))
            return klein_cls(
                scheduler=scheduler,
                vae=overrides["vae"],
                text_encoder=overrides["text_encoder"],
                tokenizer=tokenizer,
                transformer=overrides["transformer"],
                is_distilled=False,
            )
        pipe = klein_cls.from_pretrained(self.repo, torch_dtype=_torch_dtype(self.dtype))
        if overrides:
            pipe.register_modules(**overrides)
        return pipe

    def _build_modular_pipe(self) -> Any:
        """非 Flux2(ERNIE 等)的 modular fallback(原 _ensure_pipe 装配链)。架构收口后退役。"""
        modular_pipeline_cls, components_manager_cls = _import_modular()
        cm = self._cm or components_manager_cls()
        pipe = modular_pipeline_cls.from_pretrained(self.repo, components_manager=cm)
        pipe.load_components(torch_dtype=_torch_dtype(self.dtype))
        overrides = {k: v for k, v in (
            ("transformer", self._transformer_override),
            ("text_encoder", self._text_encoder_override),
            ("vae", self._vae_override),
        ) if v is not None}
        if overrides:
            pipe.update_components(**overrides)
        return pipe

    def _apply_loras(self, loras: list) -> None:
        """LoRA(含 ComfyUI 格式)接 Flux2Klein modular pipe(经 Flux2LoraLoaderMixin,
        与 DiffusionPipeline 同 API)。复用 #125 `_maybe_convert_comfy_flux2_lora`(绕 is_kohya
        误判)。PR-3:无 cpu_offload(.to(device)),省 legacy 的 offload dance。"""
        pipe = self._pipe
        if not loras:
            active = pipe.get_active_adapters() if hasattr(pipe, "get_active_adapters") else []
            if active:
                pipe.set_adapters([])
            return
        for spec in loras:
            if spec.name in self._loaded_loras:
                continue
            lora_path = getattr(spec, "path", None)
            if not lora_path:
                raise ValueError(f"LoRA {spec.name!r} 无 path")
            converted = None
            if (isinstance(lora_path, str) and lora_path.endswith(".safetensors")
                    and type(pipe).__name__.startswith("Flux2")):
                from safetensors.torch import load_file  # noqa: PLC0415
                converted = _maybe_convert_comfy_flux2_lora(load_file(lora_path))
            if converted is not None:
                pipe.load_lora_weights(converted, adapter_name=spec.name)
            else:
                pipe.load_lora_weights(lora_path, adapter_name=spec.name)
            active = pipe.get_active_adapters() or []
            if spec.name not in active and spec.name not in (getattr(pipe, "peft_config", {}) or {}):
                raise ValueError(
                    f"LoRA {spec.name!r} 零匹配(键不符 {type(pipe).__name__})—— 用对架构的 LoRA")
            self._loaded_loras.add(spec.name)
        pipe.set_adapters([s.name for s in loras], adapter_weights=[s.strength for s in loras])

    def _apply_scheduler(self, pipe: Any, sampler_name: str, scheduler: str) -> None:
        """PR-2:换 pipe.scheduler 实现采样器/调度器选择(复刻 ComfyUI KSampler 两下拉)。

        通用节点:下拉列候选(可扩展),但本架构**真能用**的子集由 model_arch_adapter 注册表声明。
        选了不支持的 → **fail loud 清晰报错,不出图**(用户决策:别静默 fallback,也别崩在 diffusers 深处)。
        sampler_name → scheduler **类**(euler/heun/lcm);scheduler → **sigma 调度** config
        (normal/karras/exponential/beta → use_*_sigmas 互斥)。`from_config(原 config)` 保留 shift /
        use_dynamic_shifting(不改已验好的基线),只覆盖 use_*_sigmas + 换类。带缓存键避免重复换。
        """
        sampler_name = sampler_name or "euler"
        scheduler = scheduler or "normal"
        self._validate_sampler_scheduler(sampler_name, scheduler)
        key = (sampler_name, scheduler)
        if key == self._sched_key:
            return
        cls_map = _import_flow_schedulers()
        cls = cls_map.get(key[0]) or cls_map["euler"]
        cfg = dict(pipe.scheduler.config)
        cfg["use_karras_sigmas"] = key[1] == "karras"
        cfg["use_exponential_sigmas"] = key[1] == "exponential"
        cfg["use_beta_sigmas"] = key[1] == "beta"
        pipe.scheduler = cls.from_config(cfg)
        self._sched_key = key

    def _validate_sampler_scheduler(self, sampler_name: str, scheduler: str) -> None:
        """据 model_arch_adapter 注册表校验本架构是否支持该采样器/调度器;不支持 → 清晰报错。
        未注册的 pipeline_class → 放行(无约束信息,不拦)。"""
        from src.services.inference.model_arch_adapter import MODEL_ARCH_REGISTRY  # noqa: PLC0415

        arch = MODEL_ARCH_REGISTRY.get(self.pipeline_class)
        if arch is None:
            return
        samplers = arch.supported_samplers()
        if sampler_name not in samplers:
            raise ValueError(
                f"采样器 {sampler_name!r} 不被 {self.pipeline_class} 支持(当前支持:{sorted(samplers)})。"
                f"diffusers flow-matching 这条路上 Flux2 只 euler 可用;其它采样器待对应模型/架构接入。")
        scheds = arch.supported_schedulers()
        if scheduler not in scheds:
            raise ValueError(
                f"调度器 {scheduler!r} 不被 {self.pipeline_class} 支持(当前支持:{sorted(scheds)})。")

    async def infer(self, req: InferenceRequest) -> InferenceResult:
        if not isinstance(req, ImageRequest):
            raise TypeError(f"ModularImageBackend 只接受 ImageRequest,收到 {type(req).__name__}")
        import torch  # noqa: PLC0415

        pipe = self._ensure_pipe()
        self._apply_loras(list(getattr(req, "loras", None) or []))
        if self.pipeline_class == "Flux2KleinPipeline":
            self._apply_scheduler(
                pipe, getattr(req, "sampler_name", "euler"), getattr(req, "scheduler", "normal"))
        gen = torch.Generator(device=self.device)
        if req.seed is not None:
            gen = gen.manual_seed(req.seed)

        t = time.monotonic()
        cfg = float(req.cfg_scale)
        # cfg → guidance_scale。标准 Flux2KleinPipeline(is_distilled=False):cfg>1 → 跑 cond+uncond 真 CFG
        # (cfg=1 退化单次前向);negative 走 **预编码 negative_prompt_embeds**(klein __call__ 无 negative 字符串
        # 入参,内部 do_cfg 时用它做 true-CFG)。真模型 A/B 已证 cfg/negative 生效(spike_true_cfg.py)。
        call_kwargs: dict[str, Any] = dict(
            prompt=req.prompt,
            num_inference_steps=req.steps,
            width=req.width,
            height=req.height,
            guidance_scale=cfg,
            generator=gen,
        )
        neg = (getattr(req, "negative_prompt", "") or "").strip()
        if (neg and cfg > 1.0 and self.pipeline_class == "Flux2KleinPipeline"
                and hasattr(pipe, "encode_prompt")):
            call_kwargs["negative_prompt_embeds"] = pipe.encode_prompt(
                prompt=neg, device=self.device)[0]
        out = pipe(**call_kwargs)
        latency_ms = int((time.monotonic() - t) * 1000)

        img = out.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return InferenceResult(
            media_type="image/png",
            data=buf.getvalue(),
            metadata={
                "width": req.width,
                "height": req.height,
                "seed": req.seed,
                "engine": ("flux2klein" if self.pipeline_class == "Flux2KleinPipeline" else "modular"),
            },
            usage=UsageMeter(image_count=1, latency_ms=latency_ms),
        )
