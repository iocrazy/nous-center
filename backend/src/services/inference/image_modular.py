"""图像引擎(标准 diffusers pipeline)—— 文件名 `image_modular` 是历史包袱(早期走 modular_pipelines)。

变迁:
- #128-132:迁到 diffusers Modular Diffusers(experimental,ModularPipeline + ComponentsManager)。
- #144:翻案 —— modular 蒸馏 block 把 cfg/negative 掐了(质量根因);改走**标准** Flux2KleinPipeline。
- **PR-A(本文件 modular 退役第二刀)**:删 `_import_modular` / `_build_modular_pipe` 死代码,
  纯标准 pipeline 路径;非 Flux2(ERNIE/Qwen-Image/AuraFlow)等待 PR-C 经 ImageArchSpec 注册表接入。

约束(plan-eng-review D2):`diffusers` import **只允许经 `_import_*` 这些 lazy seam 函数**
(blast-radius 隔离 + conftest mock torch 时模块顶层不 import diffusers/torch,避免 collection 崩)。
class 名 `ModularImageBackend` 也是历史名,实际是标准 diffusers pipeline 引擎;改名是更大的 churn,
留给架构收口后(per #145 spec)统一处理。
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


def _enable_cross_gpu_offload(pipe: Any, *, compute_device: str, stash_device: str) -> None:
    """跨卡 offload:三组件常驻 stash_device,forward 时挪 compute_device(PR-D2)。

    实现复用 diffusers `enable_model_cpu_offload` 的 chained hook 设计(accelerate
    `CpuOffload` ModelHook),但**子类化让 offload destination 可配 cuda:N 替代 cpu**。
    关键:diffusers `pipe._execution_device` 通过 `_hf_hook.execution_device` 解析正确
    compute_device,latents / random noise 也分配到 compute_device(否则跟 forward output
    设备不一致,scheduler.step 算式 `sample + dt * model_output` 冲突)。

    主用例:stash=Pro 6000(96GB)+ compute=3090(24GB)—— stash 一卡装下所有组件常驻,
    compute 卡装下单个最大组件。Pro 6000 当「显存银行」,小卡当算力。
    PCIe 4.0(~16GB/s)挪 18GB transformer 约 1.1s,20 步多 ~22s —— 比 CPU offload 快(CPU 经
    PCIe 同速但 host RAM 比 GPU HBM 慢得多),比单卡 OOM 强。

    **不**适用于「2×3090 协作跑 34GB 模型」 —— 那需要 stash 卡装下所有组件常驻,
    24GB stash 装不下 34GB。layer-level offload(单组件再切块)留之后。

    chain 顺序按 `pipe.model_cpu_offload_seq`(diffusers 标准约定,比如
    "text_encoder->transformer->vae"):每个组件 pre_forward 时先 offload 上一个回 stash,
    再加载自己到 compute。任一时刻 compute 卡只有一个组件,不会一起爆。
    """
    import torch  # noqa: PLC0415
    from accelerate.hooks import (  # noqa: PLC0415
        CpuOffload,
        UserCpuOffloadHook,
        add_hook_to_module,
    )

    compute_t = torch.device(compute_device)
    stash_t = torch.device(stash_device)

    class _GpuStashOffload(CpuOffload):
        """CpuOffload 子类:init_hook destination 改 stash_device 而非 cpu。

        关键:用 `UserCpuOffloadHook(model, hook)` 包传给下一个 hook 的 `prev_module_hook`
        参数 —— accelerate `CpuOffload.pre_forward` 严格 `isinstance(prev, UserCpuOffloadHook)` 检查,
        传 CpuOffload 子类不通过 → prev.offload() 不触发 → 上一个组件不挪回 stash → OOM。
        `UserCpuOffloadHook.offload()` 内部就是调 `self.hook.init_hook(self.model)` —— 即我们
        重写的 init_hook,挪 stash。

        diffusers `pipe._execution_device` 通过 `_hf_hook.execution_device` 解析到 compute_device,
        latents/noise 在 compute,跟 forward output 一致 —— 不再触发 scheduler.step 跨设备相加。
        """

        def init_hook(self, module: Any) -> Any:
            # offload destination = stash_device(覆盖 base 写死的 "cpu")。
            return module.to(stash_t)

    # chain hooks 按 model_cpu_offload_seq;diffusers Flux2KleinPipeline 有此属性。
    seq = getattr(pipe, "model_cpu_offload_seq", None)
    if not seq:
        raise NotImplementedError(
            f"{type(pipe).__name__} 无 model_cpu_offload_seq,跨卡 offload 暂不支持;"
            f"用 offload=cpu 暂代,或为该 pipeline 显式定义组件 chain 顺序。")
    prev_user_hook: Any = None
    for model_str in seq.split("->"):
        name = model_str.strip()
        model = getattr(pipe, name, None)
        if not isinstance(model, torch.nn.Module):
            continue
        # 链路:每个 CpuOffload 子类的 prev_module_hook 必须是 UserCpuOffloadHook(accelerate 用
        # isinstance 严格检查);UserCpuOffloadHook.offload() 调 hook.init_hook(model) → 我们的
        # init_hook 挪 stash → 上一个组件让位给下一个。
        hook = _GpuStashOffload(execution_device=compute_t, prev_module_hook=prev_user_hook)
        add_hook_to_module(model, hook, append=True)
        prev_user_hook = UserCpuOffloadHook(model=model, hook=hook)


def _move_to_device(obj: Any, device: Any) -> Any:
    """Recursively move tensors in nested args/kwargs/outputs to `device`,
    leaving non-tensors (and tensors already there) untouched.

    dict-likes are mutated **in place** so transformers `ModelOutput` (a dict
    subclass whose fields double as attributes — `output.hidden_states`) keeps
    its type. Rebuilding as a plain dict drops the attribute access and breaks
    diffusers' `output.hidden_states` use in encode_prompt."""
    import torch  # noqa: PLC0415
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = _move_to_device(obj[k], device)
        return obj
    if isinstance(obj, list):
        return [_move_to_device(o, device) for o in obj]
    if isinstance(obj, tuple):
        moved = [_move_to_device(o, device) for o in obj]
        try:
            return type(obj)(moved)  # plain tuple; namedtuple needs *moved
        except TypeError:
            return type(obj)(*moved)
    return obj


def _place_components_per_device(
    pipe: Any, *, compute_device: str, comp_devices: dict[str, str],
) -> None:
    """逐组件跨卡放置(2026-06-04 spec):每个组件常驻**自己选的卡**并在那张卡上 forward,
    跨卡张量流由 forward 边界的 accelerate ModelHook 透明搬运 —— 不依赖 diffusers 内部
    对 mixed-device 的支持(那正是 2026-05-21 放弃逐组件跨卡的脆弱点)。

    comp_devices: {"transformer": dev, "text_encoder": dev, "vae": dev}(都是解析后的具体卡)。
    compute_device = transformer 的卡 = denoise / latents 的锚点(pipe._execution_device)。

    机制(在 nn.Module.forward 边界挂 hook,与上游 encode_prompt/__call__ 怎么调无关):
      - 非锚点组件(text_encoder/vae):pre_forward 把输入挪到该组件自己的卡跑,post_forward
        把输出挪回 compute 卡 —— 这样 pipe 主流(latents/embeds 流)始终在 compute 卡,
        encode_prompt 即便把 token ids 挪到 compute,text_encoder 的 hook 也会再挪到它的卡。
      - 锚点组件(transformer):只设 execution_device=compute(让 diffusers `_execution_device`
        无歧义解析到 compute,latents/noise 分配在 compute),不改输出。
    """
    import torch  # noqa: PLC0415
    from accelerate.hooks import ModelHook, add_hook_to_module  # noqa: PLC0415

    compute_t = torch.device(compute_device)

    class _CrossDeviceHook(ModelHook):
        def __init__(self, run_device: Any, return_device: Any, execution_device: Any = None):
            self.run_device = torch.device(run_device)
            self.return_device = torch.device(return_device) if return_device is not None else None
            # diffusers `_execution_device` 扫描 `_hf_hook.execution_device`;只在锚点设值,
            # 其余设 None 让扫描跳过 → _execution_device 无歧义解析到 compute 卡。
            self.execution_device = torch.device(execution_device) if execution_device is not None else None

        def pre_forward(self, module: Any, *args: Any, **kwargs: Any) -> Any:
            return (_move_to_device(args, self.run_device), _move_to_device(kwargs, self.run_device))

        def post_forward(self, module: Any, output: Any) -> Any:
            if self.return_device is not None:
                return _move_to_device(output, self.return_device)
            return output

    def _wrap_method(module: Any, method_name: str, run_device: Any, return_device: Any) -> None:
        """包住非 forward 入口(vae.decode / vae.encode):accelerate hook 只拦 forward,
        而 diffusers 经 `vae.decode(latents)` 调用 → 绕过 forward hook → 权重在卡 X、输入在卡 Y
        的 conv device mismatch。包方法:输入挪到组件卡,输出挪回 compute。"""
        orig = getattr(module, method_name, None)
        if orig is None or getattr(orig, "_per_component_wrapped", False):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            args = _move_to_device(args, run_device)
            kwargs = _move_to_device(kwargs, run_device)
            out = orig(*args, **kwargs)
            return _move_to_device(out, return_device) if return_device is not None else out

        wrapped._per_component_wrapped = True  # type: ignore[attr-defined]
        setattr(module, method_name, wrapped)

    for name, dev in comp_devices.items():
        module = getattr(pipe, name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        dev_t = torch.device(dev)
        module.to(dev_t)  # 常驻自己的卡
        if name == "transformer":
            # 锚点:固定 _execution_device=compute,不改输出(latents 留在 compute)。
            add_hook_to_module(module, _CrossDeviceHook(compute_t, None, execution_device=compute_t), append=True)
        elif dev_t != compute_t:
            # 跨卡组件:输入挪到自己的卡,输出挪回 compute(主流回到锚点卡)。
            add_hook_to_module(module, _CrossDeviceHook(dev_t, compute_t), append=True)
            # vae 经 .decode()/.encode() 方法调用(非 forward)→ 额外包方法对齐设备。
            if name == "vae":
                _wrap_method(module, "decode", dev_t, compute_t)
                _wrap_method(module, "encode", dev_t, compute_t)


class ModularImageBackend(InferenceAdapter):
    """图像引擎(标准 diffusers Flux2KleinPipeline);class 名是历史包袱(原 ModularPipeline,见模块 docstring)。

    实现 `InferenceAdapter.infer` 接口(与 TTS/LLM 一致)。PR-A 起,modular 死代码已清,
    非 Flux2 pipeline_class 由 PR-C 的 ImageArchSpec 注册表接入(见 plan 2026-05-26)。
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
        offload: str = "none",
        transformer_override: Any = None,
        text_encoder_override: Any = None,
        vae_override: Any = None,
        comp_devices: dict[str, str] | None = None,
        **params: Any,
    ):
        # PR-A:删了 components_manager 参数(modular 才需要的;吞剩余 kwargs 兼容旧调用)。
        params.pop("components_manager", None)
        super().__init__(paths={"main": repo}, device=device, **params)
        self.repo = repo
        self.dtype = dtype
        self.pipeline_class = pipeline_class
        # 逐组件跨卡放置(2026-06-04):{"transformer":dev,"text_encoder":dev,"vae":dev}(解析后具体卡)。
        # None 或三者同卡 = 整模型单卡(旧路径,零回归)。device(=transformer 卡)是 compute 锚点。
        self.comp_devices = comp_devices or {}
        # PR-D:权重 offload 策略 — "none"(全程 GPU)/ "cpu"(enable_model_cpu_offload,大模型塞小卡)。
        # "cuda:N" 跨卡 offload 留 PR-D2(需 accelerate 自定义 hook)。
        self.offload = offload
        # comfy 单文件桥接 override:transformer(量化/comfy)+ text_encoder/vae(单文件装配,PR-2)。
        self._transformer_override = transformer_override
        self._text_encoder_override = text_encoder_override
        self._vae_override = vae_override
        self._pipe: Any = None
        self._loaded_loras: set[str] = set()  # PR-3
        # PR-2 当前已装的 (sampler_name, scheduler);初值 = 参考库默认(FlowMatchEuler + normal sigmas)
        # → 默认请求不触发换 scheduler。
        self._sched_key: tuple[str, str] = ("euler", "normal")
        # injected scheduler 名(simple/sgm_uniform/...);非 None 时 infer 传 pipe(sigmas=...)。
        self._injected_scheduler: str | None = None

    async def load(self, device: str) -> None:
        """对齐 ABC;实际 pipeline 构建在首次 infer 时 lazy(_ensure_pipe)。"""
        self.device = device

    def unload(self) -> None:
        """真正释放 GPU pipeline —— base.unload 只置 `_model=None`,远远不够。

        round3 #3(70G 累积真根因):`_pipe` 持着 transformer(~18GB)+ vae +
        text_encoder + 已装 LoRA,base.unload 不 teardown 它,且全工程零
        `torch.cuda.empty_cache()` → adapter 被换出/驱逐后 `is_loaded` 翻 False
        (manager 以为卡空了、往同卡再装),但 CUDA caching allocator 仍持旧 pipe
        的块 → 显存只增不减。这里显式拆 pipe + 清缓存,让换模型时显存真降。
        """
        pipe = self._pipe
        self._pipe = None
        self._model = None
        self._loaded_loras.clear()
        # round5:复位 scheduler 缓存键。_apply_scheduler 用 _sched_key 做缓存早退不重装;
        # unload 后同实例 rebuild 出的新 pipe 是参考库默认(normal sigmas),若 _sched_key
        # 还停在上轮值,请求同 key 时会漏装实际 scheduler → 静默用错 sigma 调度出错图。
        # 复位回 __init__ 的默认,使缓存键与新 pipe 的真实默认重新同步。
        self._sched_key = ("euler", "normal")
        self._injected_scheduler = None
        if pipe is not None:
            # 先卸 LoRA(peft adapter 也占显存);失败不该挡卸载主流程。
            try:
                if hasattr(pipe, "unload_lora_weights"):
                    pipe.unload_lora_weights()
            except Exception:  # noqa: BLE001
                pass
        del pipe
        try:
            import gc  # noqa: PLC0415

            import torch  # noqa: PLC0415
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — 清缓存 best-effort,不可因它崩卸载
            pass

    def _ensure_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        # PR-A:diffusers modular 退役。Flux2 → 标准 Flux2KleinPipeline。
        # 非 Flux2 架构(ERNIE / Qwen-Image / AuraFlow 等)留待 PR-C 经 ImageArchSpec 注册表统一接入。
        if self.pipeline_class != "Flux2KleinPipeline":
            raise NotImplementedError(
                f"pipeline_class {self.pipeline_class!r} 暂未接入;Flux2KleinPipeline 之外的架构"
                f"(ERNIE / Qwen-Image / AuraFlow 等)需经 PR-C 的 ImageArchSpec 注册表加入"
                f"(见 plans/2026-05-26-image-engine-ux-consolidation.md)。")
        pipe = self._build_klein_pipe()
        if _wants_fp8(self.dtype):
            # fp8 weight-only(torchao):transformer + text_encoder 权重 fp8 存储,省 ~½ 显存,
            # 让大模型塞进 24GB 3090(出图正确,见 spec 2026-05-25)。作用于最终组件(override 或 repo)。
            # offload 时:在 enable_model_cpu_offload 之前量化(accelerate hooks 才看到量化后权重)。
            _quantize_fp8_weight_only(pipe)
        # 逐组件跨卡放置(2026-06-04):任一组件落与 transformer 不同的卡 → 走逐组件路径。
        # 全同卡(含全 auto→同 target)→ 落下面的整模型单卡 offload 路径(零回归)。
        import torch  # noqa: PLC0415
        hetero = bool(self.comp_devices) and any(
            torch.device(d) != torch.device(self.device) for d in self.comp_devices.values()
        )
        if hetero:
            if self.offload != "none":
                raise NotImplementedError(
                    "逐组件跨卡放置暂只支持 offload=none(组件常驻各自卡);"
                    "想 offload 大模型请把组件放同一张卡再用 offload=cpu/cuda:N。"
                    "(逐组件放置 + offload 组合见 spec 2026-06-04 §5 后续项)")
            _place_components_per_device(
                pipe, compute_device=self.device, comp_devices=self.comp_devices)
            self._pipe = pipe
            self._model = pipe
            return pipe
        # PR-D / PR-D2:offload 策略(整模型单卡)。
        #   none      → 普通 .to(device),全程在 device。最快。
        #   cpu       → enable_model_cpu_offload(gpu_id=N):accelerate hooks 不用时挪 CPU。慢 3-5×。
        #   cuda:N    → 跨卡 stash:三大组件常驻 cuda:N,forward 前 to(device) 后 to(stash)。
        #               用 2×3090 协作跑 34GB 模型(权重 stash 一张,forward 另一张)。
        if self.offload == "cpu":
            gpu_id = int(self.device.split(":")[1]) if self.device.startswith("cuda:") else 0
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        elif self.offload == "none":
            pipe.to(self.device)
        elif self.offload.startswith("cuda:"):
            # PR-D2:跨卡 offload(diffusers 没现成 API,手写 hook)。
            # stash 卡跟 compute 卡不能同卡(就退化成 none 了),fail-loud。
            if self.offload == self.device:
                raise ValueError(
                    f"offload={self.offload!r} 跟 device 同卡 — 跨卡 offload 必须不同卡;"
                    f"想全程 GPU 用 offload=none,想 CPU offload 用 offload=cpu。")
            _enable_cross_gpu_offload(pipe, compute_device=self.device, stash_device=self.offload)
        else:
            raise NotImplementedError(
                f"offload={self.offload!r} 暂未接入(支持的:none / cpu / cuda:N)。")
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
        # round3 #6:删掉本次请求里不再包含的旧 LoRA。set_adapters 只是停用、不释放权重 →
        # 一个 adapter 生命周期内,用户每换一个 LoRA 旧的都常驻 pipe(显存累积)。这里把
        # 已装但本次没要的删掉,使 pipe 的 LoRA 集合收敛到当前请求。
        requested = {s.name for s in loras}
        stale = self._loaded_loras - requested
        if stale and hasattr(pipe, "delete_adapters"):
            try:
                pipe.delete_adapters(list(stale))
            except Exception:  # noqa: BLE001 — 删 LoRA 失败不该挡出图
                pass
            self._loaded_loras -= stale
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
        from src.services.inference.sigma_schedules import NATIVE_SCHEDULERS  # noqa: PLC0415
        cls_map = _import_flow_schedulers()
        cls = cls_map.get(key[0]) or cls_map["euler"]
        cfg = dict(pipe.scheduler.config)
        # native 4 个走 diffusers use_*_sigmas;injected 5 个把这些 flag 全关,改由 infer
        # 传 pipe(sigmas=...)注入(见 _injected_scheduler / infer call_kwargs)。
        is_native = key[1] in NATIVE_SCHEDULERS
        cfg["use_karras_sigmas"] = is_native and key[1] == "karras"
        cfg["use_exponential_sigmas"] = is_native and key[1] == "exponential"
        cfg["use_beta_sigmas"] = is_native and key[1] == "beta"
        pipe.scheduler = cls.from_config(cfg)
        self._sched_key = key
        # injected scheduler:记下名字,infer 据此算 sigma 传 pipe(sigmas=...)。
        self._injected_scheduler = key[1] if key[1] not in NATIVE_SCHEDULERS else None

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

    async def infer(
        self,
        req: InferenceRequest,
        *,
        progress_callback: Any | None = None,
        cancel_flag: Any | None = None,
    ) -> InferenceResult:
        """`progress_callback(done, total, **extras)` 每步发进度;`cancel_flag.is_set()` → step 边界
        raise `asyncio.CancelledError` 中断 pipe()。契约对齐 fake_adapter,runner 已经探测 signature
        + 接 P.NodeProgress + 捕获 CancelledError 落 cancelled NodeResult。

        PR-1a(2026-05-27 任务面板重置 L3 进度颗粒度):callback 额外发 stage / step_latency_ms /
        eta_ms,spec §State model TaskProgress。stage 三态:text_encode(pipe() 前 prompt encoding)/
        dit_denoise(逐步)/ vae_decode(pipe() 后 latent → image / PNG 序列化)。Flux2 标准 pipe 里
        VAE decode 内嵌在 pipe() 末尾,前端 UX 上仍按「denoise 全部完 → vae」展现,所以 vae_decode
        在 pipe() 返回之后、PNG 编码之前发(语义上等价「post-denoise 收尾」)。
        映射 ComfyUI: callback_on_step_end ≈ ProgressBar hook + throw_exception_if_processing_interrupted。"""
        if not isinstance(req, ImageRequest):
            raise TypeError(f"ModularImageBackend 只接受 ImageRequest,收到 {type(req).__name__}")
        import asyncio  # noqa: PLC0415
        import torch  # noqa: PLC0415

        pipe = self._ensure_pipe()
        self._apply_loras(list(getattr(req, "loras", None) or []))
        if self.pipeline_class == "Flux2KleinPipeline":
            self._apply_scheduler(
                pipe, getattr(req, "sampler_name", "euler"), getattr(req, "scheduler", "normal"))
        gen = torch.Generator(device=self.device)
        if req.seed is not None:
            gen = gen.manual_seed(req.seed)

        # PR-4:共用 ProgressTracker(替换 PR-1a 的 _make_emit + 手算 latency / ETA)。
        # text_encode stage 在 pipe() 前发一帧 — 让前端 UX 在「dit step 1 出来前」就知道
        # 进入 text encode 阶段,不黑屏。step/total_steps 传 0/total 让 (done,total) 降级
        # callback 也能拿到正确 total(老 fake 测试只看 done/total tuple)。
        from src.services.inference.progress_tracker import ProgressTracker  # noqa: PLC0415
        total_steps = int(req.steps)
        # 进度桥接:pipe() 下方丢进 to_thread(见 out=... 处),callback_on_step_end 在工作线程
        # 触发 —— runner 的 _on_progress 做 loop.create_task,工作线程无 running loop 会崩。
        # 捕获 loop + call_soon_threadsafe 把回调调度回事件循环(对齐 anima #208)。
        loop = asyncio.get_running_loop()

        def _emit(*a: Any, **kw: Any) -> None:
            # ProgressTracker 原本同步调 cb 并 catch TypeError 降级老签名;桥接把调用 defer
            # 到 loop 后 TypeError 在异步里抛、catch 不到 —— 在 loop 回调里复刻三级降级
            # (新 (done,total,**extras) → (done,total,preview_url) → (done,total))。
            if progress_callback is None:
                return

            def _deferred() -> None:
                try:
                    progress_callback(*a, **kw)
                except TypeError:
                    try:
                        progress_callback(a[0], a[1], preview_url=kw.get("preview_url"))
                    except TypeError:
                        progress_callback(a[0], a[1])

            loop.call_soon_threadsafe(_deferred)

        pt = ProgressTracker(_emit if progress_callback is not None else None)  # 真 per-step,不 throttle
        pt.stage("text_encode", step=0, total_steps=total_steps)

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
        # injected scheduler(simple/sgm_uniform/ddim_uniform/linear_quadratic/kl_optimal):
        # diffusers FlowMatch 原生无对应 use_*_sigmas,手动算 sigma 传 pipe(sigmas=...)。
        # pipe 期望不含末尾 0(它自己 append),compute_sigmas 返回含末尾 0 → 去掉末尾 0。
        if self.pipeline_class == "Flux2KleinPipeline" and self._injected_scheduler:
            from src.services.inference.sigma_schedules import compute_sigmas  # noqa: PLC0415
            shift = float(getattr(pipe.scheduler.config, "shift", 3.0) or 3.0) \
                if hasattr(pipe.scheduler, "config") else 3.0
            sigmas = compute_sigmas(self._injected_scheduler, int(req.steps), shift=shift)
            # 去末尾 0(diffusers pipe 自己补);num_inference_steps 与 sigmas 互斥时以 sigmas 为准。
            call_kwargs["sigmas"] = sigmas[:-1] if sigmas and sigmas[-1] == 0.0 else sigmas
            call_kwargs.pop("num_inference_steps", None)
        neg = (getattr(req, "negative_prompt", "") or "").strip()
        if (neg and cfg > 1.0 and self.pipeline_class == "Flux2KleinPipeline"
                and hasattr(pipe, "encode_prompt")):
            call_kwargs["negative_prompt_embeds"] = pipe.encode_prompt(
                prompt=neg, device=self.device)[0]

        # PR-3 进度 + 中止(对齐 ComfyUI):callback_on_step_end 每步触发 ——
        # cancel_flag 置位 → raise CancelledError(中断 pipe(),runner 落 cancelled);
        # progress_callback(done, total, **extras) → runner 发 P.NodeProgress。仅 Flux2KleinPipeline
        # 走标准 pipe,callback_on_step_end 内置;modular fallback(ERNIE 等)不支持回调,这里不挂。
        if self.pipeline_class == "Flux2KleinPipeline" and (
                progress_callback is not None or cancel_flag is not None):
            def _step_cb(_pipe: Any, i: int, _t: Any, cb_kwargs: dict) -> dict:
                # 中止:step 边界检查(ComfyUI 是 op 级 / nous diffusers 这条路只能 step 级,
                # ~250ms/步已够响应)。raise BaseException(CancelledError)穿出 pipe()。
                if cancel_flag is not None and cancel_flag.is_set():
                    raise asyncio.CancelledError()
                # PR-F:latent → 96px JPEG data URI(ComfyUI Latent2RGB 等价)。
                preview_url = None
                if pt.has_callback and self.pipeline_class == "Flux2KleinPipeline":
                    latents = cb_kwargs.get("latents")
                    if latents is not None:
                        from src.services.inference.latent_preview import latent_to_preview_data_uri  # noqa: PLC0415
                        preview_url = latent_to_preview_data_uri(latents)
                # PR-4:ProgressTracker.step 算 latency 滑窗(16)+ ETA + emit。
                pt.step(i + 1, total_steps, stage="dit_denoise", preview_url=preview_url)
                return cb_kwargs

            call_kwargs["callback_on_step_end"] = _step_cb

        # 关键(flux2 默认引擎 to_thread,同 anima #206):pipe() 的 denoise(1024 ~20-30s)
        # 是阻塞 CUDA 调用。在 runner coroutine 里同步跑会占住事件循环 → ① _step_cb 发的
        # 进度 task 全卡到 pipe() 返回才 flush(逐步进度不实时)② pipe-reader 读不到 Abort
        # → cancel_flag 永不置位、中途取消失效 ③ 慢卡长 denoise >ping_timeout 触发 watchdog
        # 介入。丢 to_thread 让事件循环空闲:进度实时 flush、Abort 可读、cancel 生效。
        out = await asyncio.to_thread(lambda: pipe(**call_kwargs))
        # PR-4:vae_decode 用 stage(progress=1.0) — 不调 finish(避免 _finished=True 阻塞
        # 测试场景下手动触发 step_cb;production 中 stage 行为等同 finish:progress=1/eta=0)。
        # Flux2 标准 pipe 里 VAE decode 已在 pipe() 内嵌跑完,这里发的是「post-denoise 收尾」一帧。
        pt.stage(
            "vae_decode", progress=1.0,
            step=total_steps, total_steps=total_steps,
            detail=f"vae_decode done ({total_steps} steps)",
        )
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
