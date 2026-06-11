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


def _import_zimage_pipeline() -> Any:
    """Lazy import Z-Image pipeline(D2:diffusers import 只在本文件)。Z-Image-Turbo 是
    distilled(guidance=0,8 步),HF-layout 整模型,走标准 from_pretrained。测试 monkeypatch 注入 fake。"""
    from diffusers import ZImagePipeline  # noqa: PLC0415

    return ZImagePipeline


def _import_img2img_pipeline(cls_name: str) -> Any:
    """Lazy import img2img pipeline 类(D2:diffusers import 只在本文件;PR-A2)。
    目前仅 ZImageImg2ImgPipeline。从 text2img pipe 的 components 复用构建(不重载权重)。"""
    import diffusers  # noqa: PLC0415
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        raise NotImplementedError(
            f"diffusers(钉的 commit)无 img2img pipeline 类 {cls_name!r} —— 升 diffusers 或换架构")
    return cls


def _import_qwen_edit_pipeline() -> Any:
    """Lazy import Qwen-Image-Edit-2511 pipeline(D2:diffusers import 只在本文件)。编辑类
    (needs_image_input),HF-layout 整模型,标准 from_pretrained。CFG 旋钮 true_cfg_scale(非
    guidance_scale)。测试 monkeypatch 注入 fake。"""
    from diffusers import QwenImageEditPlusPipeline  # noqa: PLC0415

    return QwenImageEditPlusPipeline


def _import_ideogram4_pipeline() -> Any:
    """Lazy import Ideogram-4 pipeline(D2:diffusers import 只在本文件)。9.3B 双 DiT
    (conditional + unconditional_transformer 非对称 CFG)+ Qwen3-VL 文本编码器,HF-layout
    整模型 from_pretrained(7 组件随 repo 自动加载)。需 diffusers ≥ 784fa626(PR-1 bump)。
    测试 monkeypatch 注入 fake。"""
    from diffusers import Ideogram4Pipeline  # noqa: PLC0415

    return Ideogram4Pipeline


def _decode_input_image(src: str) -> Any:
    """输入图(编辑/img2img)src(本地路径 或 base64 data URI)→ PIL.Image(RGB)。
    runner 已把节点签名 URL 解析成本地路径再塞 req.input_image;data URI 分支兜底直传场景。
    与 image_seedvr2._decode_image 同语义(故意不跨模块复用 —— 各引擎自含,blast radius 隔离)。"""
    import base64  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    if src.startswith("data:"):
        _, _, b64 = src.partition(",")
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return Image.open(src).convert("RGB")


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


def _ref_class_name(repo: str, sub: str) -> str:
    """读参考库 `repo/<sub>/config.json` 的 `_class_name`(桥接按它选 module 类/路径,arch-agnostic)。"""
    import json  # noqa: PLC0415
    try:
        return str(json.loads((Path(repo) / sub / "config.json").read_text()).get("_class_name", ""))
    except Exception:  # noqa: BLE001
        return ""


def _bf16_or(dtype_str: str) -> Any:
    """_torch_dtype 但 default(None)兜底 bf16(单文件 from_single_file 需具体 dtype)。"""
    import torch  # noqa: PLC0415
    return _torch_dtype(dtype_str) or torch.bfloat16


def build_bridged_transformer(unet_spec: Any, repo: str, device: str) -> Any:
    """comfy 单文件 → diffusers transformer module(桥接)。按参考库 transformer config 的 `_class_name` 选路径:

    - **Z-Image(`ZImageTransformer2DModel`)= diffusers 原生 from_single_file**(PR-2,spec 2026-06-09):
      diffusers `single_file_model.py` 注册了 `convert_z_image_transformer_checkpoint_to_diffusers`,
      comfy 单文件 bf16 直接吃,**无需手写转键**(probe_zimage_singlefile 真机验过)。
    - Flux2(`Flux2Transformer2DModel`)= 现有手写桥接(dequant_and_convert 反量化 + 转键 + from_config)。
    喂 `ModularImageBackend(transformer_override=)`。
    """
    cls_name = _ref_class_name(repo, "transformer")
    if cls_name.startswith("ZImage"):
        from diffusers import ZImageTransformer2DModel  # noqa: PLC0415
        module = ZImageTransformer2DModel.from_single_file(
            unet_spec.file, config=str(Path(repo) / "transformer"),
            torch_dtype=_bf16_or(unet_spec.dtype))
        return module.to(device)

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

    # PR-3:GGUF 文本编码器(CLIPLoaderGGUF 等价,如 Qwen3-4b-Z-Image-Engineer)。
    # GGUF 用 llama.cpp 命名(blk.N.attn_k.weight)非 HF 键 → 自写 dequant 还得手动 remap;
    # transformers 原生 from_pretrained(gguf_file=) 内部做 Q8_0 dequant + key 重映射,且
    # qwen3 在 GGUF_SUPPORTED_ARCHITECTURES(真机验:transformers 5.6)。config/权重均取自 GGUF
    # 元数据(不读 repo/text_encoder config);tokenizer 仍由 pipe 从参考库取,这里只产模型。
    if str(getattr(clip_spec, "file", "")).lower().endswith(".gguf"):
        gguf_path = Path(clip_spec.file)
        model = AutoModelForCausalLM.from_pretrained(
            str(gguf_path.parent), gguf_file=gguf_path.name, torch_dtype=torch.bfloat16)
        return model.to(device)

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
    """comfy 单文件 vae → diffusers vae module。按参考库 vae config 的 `_class_name` 选路径:

    - **Z-Image(`AutoencoderKL` = Flux1 16ch VAE)= diffusers 原生 from_single_file**(PR-2)——
      `ae.safetensors`(Flux1 VAE)直接吃,config 取参考库 vae(默认 config 形状不符,probe 验过)。
    - Flux2(`AutoencoderKLFlux2`)= 现有 dequant + from_config 路径。
    """
    import torch  # noqa: PLC0415
    from accelerate import init_empty_weights  # noqa: PLC0415

    if _ref_class_name(repo, "vae").strip() == "AutoencoderKL":
        from diffusers import AutoencoderKL  # noqa: PLC0415
        vae = AutoencoderKL.from_single_file(
            vae_spec.file, config=str(Path(repo) / "vae"), torch_dtype=_bf16_or(vae_spec.dtype))
        return vae.to(device)

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
    # unconditional_transformer = Ideogram-4 的第二个 DiT(双模型非对称 CFG)。漏掉它
    # fp8 只省一半:bf16 双 DiT 各 18.6G,vLLM 39G 在卡时 e2e VAE decode 差 1.2G OOM
    # (2026-06-11 真机)。其他架构无此组件,getattr None 跳过,零影响。
    for name in ("transformer", "unconditional_transformer", "text_encoder"):
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


def _offload_stash(offload: str) -> Any:
    """逐组件 offload → stash 设备(权重闲时停哪):none→None(常驻);cpu→cpu;cuda:N→该卡。"""
    import torch  # noqa: PLC0415
    if not offload or offload == "none":
        return None
    if offload == "cpu":
        return torch.device("cpu")
    return torch.device(offload)


def _place_components_per_device(
    pipe: Any, *, compute_device: str, comp_devices: dict[str, str],
    comp_offloads: dict[str, str] | None = None,
) -> None:
    """逐组件跨卡放置 + 逐组件 offload(2026-06-04 spec):每个组件在**自己选的卡**上 forward,
    跨卡张量流由 forward 边界的 accelerate ModelHook 透明搬运 —— 不依赖 diffusers 内部
    对 mixed-device 的支持(那正是 2026-05-21 放弃逐组件跨卡的脆弱点)。

    comp_devices:  {"transformer": dev, "text_encoder": dev, "vae": dev}(解析后的具体卡 = 各组件 compute 卡)。
    comp_offloads: 各组件的 offload(none=常驻 compute 卡;cpu / cuda:N=权重 stash 那里,forward 时挪到 compute 卡)。
    compute_device = transformer 的卡 = denoise / latents 的锚点(pipe._execution_device)。

    机制(在 nn.Module.forward 边界挂 hook,与上游 encode_prompt/__call__ 怎么调用无关):
      - 非锚点组件(text_encoder/vae):pre_forward 把输入挪到该组件 compute 卡跑,post_forward
        把输出挪回锚点卡 —— pipe 主流(latents/embeds 流)始终在锚点卡。
      - 锚点组件(transformer):设 execution_device=compute(让 diffusers `_execution_device`
        无歧义解析到 compute),输出不挪。
      - offload(stash≠None):init 把权重停 stash;pre_forward 把权重挪到 compute 卡;post_forward
        挪回 stash —— 等价 enable_model_cpu_offload / 跨卡 stash,但**逐组件**。
    """
    import torch  # noqa: PLC0415
    from accelerate.hooks import ModelHook, add_hook_to_module  # noqa: PLC0415

    compute_t = torch.device(compute_device)
    comp_offloads = comp_offloads or {}

    class _CrossDeviceHook(ModelHook):
        def __init__(self, run_device: Any, return_device: Any, *,
                     stash_device: Any = None, execution_device: Any = None):
            self.run_device = torch.device(run_device)
            self.return_device = torch.device(return_device) if return_device is not None else None
            self.stash_device = torch.device(stash_device) if stash_device is not None else None
            # diffusers `_execution_device` 扫描 `_hf_hook.execution_device`;只在锚点设值,
            # 其余设 None 让扫描跳过 → _execution_device 无歧义解析到 compute 卡。
            self.execution_device = torch.device(execution_device) if execution_device is not None else None

        def init_hook(self, module: Any) -> Any:
            if self.stash_device is not None:
                module.to(self.stash_device)  # 权重停 stash
            return module

        def pre_forward(self, module: Any, *args: Any, **kwargs: Any) -> Any:
            if self.stash_device is not None:
                module.to(self.run_device)    # 用前挪到 compute 卡
            return (_move_to_device(args, self.run_device), _move_to_device(kwargs, self.run_device))

        def post_forward(self, module: Any, output: Any) -> Any:
            if self.stash_device is not None:
                module.to(self.stash_device)  # 用后挪回 stash
            if self.return_device is not None:
                return _move_to_device(output, self.return_device)
            return output

    def _wrap_method(module: Any, method_name: str, run_device: Any, return_device: Any,
                     stash_device: Any = None) -> None:
        """包住非 forward 入口(vae.decode / vae.encode):accelerate hook 只拦 forward,
        而 diffusers 经 `vae.decode(latents)` 调用 → 绕过 forward hook。包方法:(offload 时)权重
        挪到 compute 卡 → 输入挪到 compute 卡 → 调原方法 → (offload 时)权重挪回 stash → 输出挪回锚点卡。"""
        orig = getattr(module, method_name, None)
        if orig is None or getattr(orig, "_per_component_wrapped", False):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if stash_device is not None:
                module.to(run_device)
            args = _move_to_device(args, run_device)
            kwargs = _move_to_device(kwargs, run_device)
            out = orig(*args, **kwargs)
            if stash_device is not None:
                module.to(stash_device)
            return _move_to_device(out, return_device) if return_device is not None else out

        wrapped._per_component_wrapped = True  # type: ignore[attr-defined]
        setattr(module, method_name, wrapped)

    for name, dev in comp_devices.items():
        module = getattr(pipe, name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        dev_t = torch.device(dev)
        stash = _offload_stash(comp_offloads.get(name, "none"))
        if stash is None:
            module.to(dev_t)              # 常驻自己的 compute 卡
        # else: 由 init_hook 停 stash(模块已 build 在 stash 上,init_hook = 确认)
        if name == "transformer":
            # 锚点:run=compute,execution_device=compute,输出不挪。
            add_hook_to_module(module, _CrossDeviceHook(
                compute_t, None, stash_device=stash, execution_device=compute_t), append=True)
        else:
            # 非锚点:run=自己 compute 卡,输出挪回锚点卡。
            add_hook_to_module(module, _CrossDeviceHook(
                dev_t, compute_t, stash_device=stash), append=True)
            # vae 经 .decode()/.encode() 方法调用(非 forward)→ 额外包方法(含 offload 权重搬运)。
            if name == "vae":
                _wrap_method(module, "decode", dev_t, compute_t, stash)
                _wrap_method(module, "encode", dev_t, compute_t, stash)


# 非 euler 采样器 —— 走 Z-Image 手写分段循环(_run_zimage_segmented)而非整段 pipe()。
# euler 仍走整段 pipe() 保 golden;这些经手写步实现(逐式对照 ComfyUI k_diffusion)。
_SEGMENTED_SAMPLERS = {
    "euler_ancestral", "dpmpp_2m", "dpmpp_2s_ancestral",
    "dpmpp_2m_sde", "dpmpp_3m_sde", "dpmpp_sde",
}


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
        comp_offloads: dict[str, str] | None = None,
        **params: Any,
    ):
        # PR-A:删了 components_manager 参数(modular 才需要的;吞剩余 kwargs 兼容旧调用)。
        params.pop("components_manager", None)
        super().__init__(paths={"main": repo}, device=device, **params)
        self.repo = repo
        self.dtype = dtype
        self.pipeline_class = pipeline_class
        # 逐组件跨卡放置(2026-06-04):{"transformer":dev,"text_encoder":dev,"vae":dev}(解析后具体卡)。
        # None 或三者同卡且同 offload = 整模型单卡(旧路径,零回归)。device(=transformer 卡)是 compute 锚点。
        self.comp_devices = comp_devices or {}
        # 逐组件 offload:{"transformer":off,"text_encoder":off,"vae":off}(none/cpu/cuda:N)。
        self.comp_offloads = comp_offloads or {}
        # PR-D:权重 offload 策略 — "none"(全程 GPU)/ "cpu"(enable_model_cpu_offload,大模型塞小卡)。
        # "cuda:N" 跨卡 offload 留 PR-D2(需 accelerate 自定义 hook)。
        self.offload = offload
        # comfy 单文件桥接 override:transformer(量化/comfy)+ text_encoder/vae(单文件装配,PR-2)。
        self._transformer_override = transformer_override
        self._text_encoder_override = text_encoder_override
        self._vae_override = vae_override
        self._pipe: Any = None
        # PR-A2:img2img 变体 pipe(复用 _pipe 的 components,惰性建)。仅 arch 注册了 img2img_pipeline_class
        # 且请求带 input_image + 0<strength<1 时构建/使用。unload 随 _pipe 一起清(共享组件)。
        self._img2img_pipe: Any = None
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
        self._img2img_pipe = None  # PR-A2:与 _pipe 共享组件,随之释放
        self._model = None
        # 断开 override 模块强引用(分开载入装配,统一模型管理收尾 PR-2):__init__ 把 L1 组件
        # (transformer/clip/vae)存进 self._*_override 喂 pipe;只 _pipe=None 不够 —— adapter 这三个属性
        # 仍持着大权重(~20GB),_release_combo_components 把池里 module 置 None 后,adapter 这条引用还在
        # → gc 收不掉、显存不降(combo 卸载只释放 ~2GB 包装层的真根因)。清掉让权重真释放。
        self._transformer_override = None
        self._text_encoder_override = None
        self._vae_override = None
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
        # 多架构(spec 2026-06-07):按 pipeline_class 选 builder。Flux2=comfy 单文件桥接/HF-layout;
        # Z-Image=HF-layout 整模型 from_pretrained。fp8/逐组件放置/offload 之后共享(都作用于
        # pipe.transformer/text_encoder/vae,两架构组件名一致)。其余架构(ERNIE/Qwen-Edit…)后续接入。
        if self.pipeline_class == "Flux2KleinPipeline":
            pipe = self._build_klein_pipe()
        elif self.pipeline_class == "ZImagePipeline":
            pipe = self._build_zimage_pipe()
        elif self.pipeline_class == "QwenImageEditPlusPipeline":
            pipe = self._build_qwen_edit_pipe()
        elif self.pipeline_class == "Ideogram4Pipeline":
            pipe = self._build_ideogram4_pipe()
        else:
            raise NotImplementedError(
                f"pipeline_class {self.pipeline_class!r} 暂未接入;已支持 Flux2KleinPipeline / "
                f"ZImagePipeline / QwenImageEditPlusPipeline。新架构在 IMAGE_ARCH_REGISTRY 注册并在此加 builder 分支。")
        if _wants_fp8(self.dtype):
            # fp8 weight-only(torchao):transformer + text_encoder 权重 fp8 存储,省 ~½ 显存,
            # 让大模型塞进 24GB 3090(出图正确,见 spec 2026-05-25)。作用于最终组件(override 或 repo)。
            # offload 时:在 enable_model_cpu_offload 之前量化(accelerate hooks 才看到量化后权重)。
            _quantize_fp8_weight_only(pipe)
        # 逐组件跨卡放置 + 逐组件 offload(2026-06-04):任一组件落与 transformer 不同的卡,
        # **或**某组件 offload 跟整管线 offload 不同 → 走逐组件路径。全同卡且全同 offload
        #(含全 auto→同 target、全 none)→ 落下面的整模型单卡 offload 路径(零回归)。
        import torch  # noqa: PLC0415
        same_card = (not self.comp_devices) or all(
            torch.device(d) == torch.device(self.device) for d in self.comp_devices.values())
        same_offload = all(
            (self.comp_offloads.get(k, "none")) == self.offload for k in self.comp_devices)
        homogeneous = same_card and same_offload
        if not homogeneous:
            _place_components_per_device(
                pipe, compute_device=self.device,
                comp_devices=self.comp_devices, comp_offloads=self.comp_offloads)
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

    def _wants_img2img(self, req: Any) -> bool:
        """是否走真 img2img(PR-A2):arch 注册了 img2img_pipeline_class + 连了 input_image + 0<strength<1。
        strength>=1(默认)= 全量去噪 ≈ 忽略输入图 → 不走 img2img(零回归:现有工作流默认 strength=1)。"""
        from src.services.inference.model_arch_adapter import arch_spec_by_pipeline  # noqa: PLC0415
        spec = arch_spec_by_pipeline(self.pipeline_class)
        if not spec or not spec.img2img_pipeline_class:
            return False
        if not getattr(req, "input_image", None):
            return False
        try:
            s = float(getattr(req, "strength", 1.0))
        except (TypeError, ValueError):
            return False
        return 0.0 < s < 1.0

    def _ensure_img2img_pipe(self) -> Any:
        """img2img 变体 pipe —— 复用 text2img pipe 已加载+已放置的组件(不重载权重/不重放置显存),
        仅换 pipeline 类(diffusers `Cls(**base.components)` 标准做法)。仅 arch 注册了
        img2img_pipeline_class(z-image)时可建。"""
        if self._img2img_pipe is not None:
            return self._img2img_pipe
        from src.services.inference.model_arch_adapter import arch_spec_by_pipeline  # noqa: PLC0415
        spec = arch_spec_by_pipeline(self.pipeline_class)
        cls_name = spec.img2img_pipeline_class if spec else None
        if not cls_name:
            raise NotImplementedError(
                f"pipeline_class {self.pipeline_class!r} 无 img2img 变体(arch 未注册 img2img_pipeline_class)")
        base = self._ensure_pipe()
        img_cls = _import_img2img_pipeline(cls_name)
        self._img2img_pipe = img_cls(**base.components)
        return self._img2img_pipe

    def _wants_segmented(self, req: Any) -> bool:
        """是否走「手写分段去噪循环」(留噪 latent 接力,PR-B2)。仅 ZImagePipeline(同 16ch latent
        空间);任一分段字段非默认即触发:start_at_step>0(续采段)/ end_at_step 真截断(base 留噪段)/
        add_noise=False(注入 latent 原样续采)/ init_latent_ref(上段导出的带噪 latent)。
        全默认 = 整段 pipe()(零回归)。跨模型(latent 空间不兼容)不在此路 —— 走路 A 像素链。"""
        if self.pipeline_class != "ZImagePipeline":
            return False
        start = int(getattr(req, "start_at_step", 0) or 0)
        end = getattr(req, "end_at_step", None)
        steps = int(getattr(req, "steps", 0) or 0)
        end_truncates = end is not None and int(end) < steps
        add_noise = bool(getattr(req, "add_noise", True))
        # 非 normal 调度器(simple/beta/…,PR-1)或非 euler 采样器(euler_ancestral PR-1b /
        # dpmpp_2m / dpmpp_2s_ancestral dpm++ PR)也走手写循环 —— 手动算 sigma / 手写采样步。
        # normal+euler+无分段 = 整段 pipe(零回归)。
        sched = (getattr(req, "scheduler", "normal") or "normal")
        sampler = (getattr(req, "sampler_name", "euler") or "euler")
        # 采样期 latent 干预(LCS 等,spec 2026-06-10)需 per-step 挂钩点 —— 仅手写循环有;有干预
        # 即走手写路(否则标准 pipe() 无挂钩,干预静默失效)。
        has_interventions = bool(getattr(req, "interventions", None))
        return (start > 0 or end_truncates or (not add_noise)
                or getattr(req, "init_latent_ref", None) is not None
                or sched != "normal" or sampler in _SEGMENTED_SAMPLERS
                or has_interventions)

    def _load_init_latent(self, req: Any, device: str) -> Any:
        """读回上段导出的 latent_ref(PR-B1 落盘的 safetensors)→ float32 张量(本段 device)。
        派发前校验 arch / latent_channels 与本段模型一致 —— 不符给人话报错(对齐
        [[project_anima_arch_mismatch]] 的「别让 UNet 抛晦涩 shape 错」),跨模型 latent 不兼容
        显式拦在派发前。"""
        from src.services.inference.model_arch_adapter import arch_spec_by_pipeline  # noqa: PLC0415

        ref = req.init_latent_ref or {}
        path = ref.get("path")
        if not path:
            raise ValueError("init_latent_ref 缺 path —— 上段 VAE Decode 需 output_mode=latent 导出 latent_ref")
        spec = arch_spec_by_pipeline(self.pipeline_class)
        my_arch = spec.arch if spec else None
        ref_arch = ref.get("arch")
        if ref_arch and my_arch and ref_arch != my_arch:
            raise ValueError(
                f"latent 接力架构不匹配:上段 latent 来自 '{ref_arch}' 架构,本段采样模型是 '{my_arch}'。"
                f"不同架构的 latent 空间物理不兼容(通道数/缩放/归一不同),无法直接接力。"
                f"跨模型请改走「像素链」(上段出图 → 本段 input_image 重绘/参考编辑,即路 A),"
                f"不要用 latent 输出端口。")
        # safetensors/torch import 推到校验后(CI test venv 不装 inference extra;派发前 ValueError
        # 路径无需它,本测试可在 CI 跑 —— 仅真加载时才需依赖)。
        import safetensors.torch as _st  # noqa: PLC0415
        sd = _st.load_file(path)
        latent = sd.get("latent")
        if latent is None:
            raise ValueError(f"latent_ref 文件 {path!r} 无 'latent' 键(非本系统导出的 latent)")
        ref_ch = ref.get("latent_channels")
        actual_ch = int(latent.shape[1]) if latent.dim() >= 2 else None
        if ref_ch and actual_ch is not None and int(ref_ch) != actual_ch:
            raise ValueError(
                f"latent_ref 通道数声明 {ref_ch} 与实际张量 {actual_ch} 不符(文件损坏 / 元信息过期)")
        import torch  # noqa: PLC0415
        return latent.to(device=device, dtype=torch.float32)

    def _build_interventions(self, req: Any, pipe: Any = None) -> list:
        """req.interventions 描述符 list → per-step latent 干预闭包 list(复刻 comfyui-lcs post-CFG hook,
        spec 2026-06-10)。每个闭包契约:`fn(step_idx, total_steps, sigma, denoised) -> denoised`
        —— 在 denoised(x0)语义点逐步改 latent(对齐 ComfyUI post_cfg 改 denoised,非改 step 后 latent)。
        空/None → 空 list(零回归:挂钩点不调用 = byte-identical)。

        `test_shift`(PR-1 管道验证:沿常向量推 denoised)。`lcs_sharpness`(PR-2:vendor 光栅 PCA 标定
        → 沿锐化 PC1 方向推,惰性标定 + VAE 指纹缓存)。`lcs_color_anchor` 待 PR-3。"""
        import logging  # noqa: PLC0415

        import torch  # noqa: PLC0415
        descs = getattr(req, "interventions", None) or []
        fns: list = []
        for d in descs:
            if not isinstance(d, dict):
                continue
            kind = d.get("_type")
            if kind == "test_shift":
                scale = float(d.get("strength", 0.0))
                start = int(d.get("start_step", 0))
                end = int(d.get("end_step", 10 ** 9))

                def _fn(step_idx: int, total: int, sigma: float, x: Any, sigmas: Any = None,
                        _s: float = scale, _a: int = start, _b: int = end) -> Any:
                    if _s == 0.0 or step_idx < _a or step_idx > _b:
                        return x
                    return x + _s * torch.ones_like(x)
                fns.append(_fn)
            elif kind == "lcs_sharpness":
                strength = float(d.get("strength", 1.0))
                if strength == 0.0:
                    continue  # no-op → 不建闭包(避免 denoised 往返反推的浮点误差,保 byte-identical)
                if pipe is None or getattr(pipe, "vae", None) is None:
                    logging.getLogger(__name__).warning("lcs_sharpness 需 VAE(pipe.vae 缺),跳过")
                    continue
                from src.services.inference.lcs_integration import build_sharpness_fn  # noqa: PLC0415
                fns.append(build_sharpness_fn(
                    pipe.vae, self.device, strength=strength,
                    start_step=int(d.get("start_step", 0)),
                    end_step=int(d.get("end_step", 10 ** 9))))
            elif kind == "lcs_color_anchor":
                intensity = float(d.get("intensity", 0.8))
                if intensity == 0.0:
                    continue  # no-op → 不建闭包(保 byte-identical)
                if pipe is None or getattr(pipe, "vae", None) is None:
                    logging.getLogger(__name__).warning("lcs_color_anchor 需 VAE(pipe.vae 缺),跳过")
                    continue
                from src.services.inference.lcs_integration import build_color_anchor_fn  # noqa: PLC0415
                fns.append(build_color_anchor_fn(
                    pipe.vae, self.device,
                    mode=str(d.get("mode", "self_anchor")), intensity=intensity))
            else:
                logging.getLogger(__name__).warning("intervention _type=%r 未实现,跳过", kind)
        return fns

    def _run_zimage_segmented(
        self, pipe: Any, req: Any, gen: Any, pt: Any, stage_ts: dict, cancel_flag: Any,
        interventions: list | None = None,
    ) -> Any:
        """手写 Z-Image 分段去噪循环(留噪 latent 接力,PR-B2)—— 逐行对照 diffusers
        `ZImagePipeline.__call__` 的 denoise loop(pipeline_z_image.py),但把一次 N 步去噪劈成一段:
        从全 schedule 的 start_at_step 跑到 end_at_step(可留噪不到 0),注入/导出带噪 latent。

        正确性(闸门 smoke_zimage_split):全 schedule 用 scheduler 自己的 set_timesteps(shift=3,
        Z-Image use_dynamic_shifting=false → 与整段 pipe() **逐数值一致**);split_sigmas 只做**索引切片**
        (不重算 shift)→ 覆盖 scheduler.sigmas/timesteps 后,step() 按 step_index 取同一组 sigma 对
        → base 段第 i 步 == 整段第 i 步(bit 级);refiner 注入 base 末态续采 == 整段后半 → SSIM≈1.0。

        Z-Image distilled:guidance_scale=0(apply_cfg=False,走简单单次前向路径)。返回最终 latent(float32,
        device 上);caller 决定导出 latent_ref(output_mode=latent)还是 VAE decode 出图。"""
        import torch  # noqa: PLC0415

        from src.services.inference.sigma_schedules import compute_sigmas, split_sigmas  # noqa: PLC0415

        device = self.device
        scheduler = pipe.scheduler
        transformer = pipe.transformer
        total_steps = int(req.steps)
        start = int(getattr(req, "start_at_step", 0) or 0)
        end = getattr(req, "end_at_step", None)
        add_noise = bool(getattr(req, "add_noise", True))
        leftover = bool(getattr(req, "return_with_leftover_noise", False))
        sched_name = (getattr(req, "scheduler", "normal") or "normal")
        sampler_name = (getattr(req, "sampler_name", "euler") or "euler")

        # 1. 文本编码(distilled → do_classifier_free_guidance=False,只出正向 embeds 列表)。
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=req.prompt, device=device, do_classifier_free_guidance=False)

        # 2. 全 schedule。normal:与整段 pipe() 完全同法(set_timesteps,shift=3 内化 → golden 不变)。
        #    其余(simple/beta/…,PR-1 放开):sigma_schedules.compute_sigmas(ComfyUI ground-truth 验过,
        #    含末尾 0,shift=3)。两者都给 total_steps+1 个 sigma(末尾 0),后续 split_sigmas 索引切片一致。
        if sched_name == "normal":
            scheduler.sigma_min = 0.0
            scheduler.set_timesteps(num_inference_steps=total_steps, device=device)
            full_sigmas = [float(x) for x in scheduler.sigmas.tolist()]
        else:
            shift = float(getattr(scheduler.config, "shift", 3.0) or 3.0) \
                if hasattr(scheduler, "config") else 3.0
            full_sigmas = [float(x) for x in compute_sigmas(sched_name, total_steps, shift=shift)]

        # 3. split_sigmas 索引切片本段(留噪不置 0 / force_full_denoise 末步去到底)。
        seg = split_sigmas(
            full_sigmas, start_at_step=start, end_at_step=end,
            force_full_denoise=not leftover)
        if len(seg) < 2:
            raise ValueError(
                f"分段采样切出 <2 个 sigma(start={start} end={end} 全步={total_steps})—— "
                f"段为空,检查 start_at_step/end_at_step 边界")
        num_train = float(getattr(scheduler.config, "num_train_timesteps", 1000) or 1000)
        sig_t = torch.tensor(seg, dtype=torch.float32, device=device)
        scheduler.sigmas = sig_t
        scheduler.timesteps = sig_t[:-1] * num_train
        scheduler._step_index = None
        scheduler._begin_index = None
        scheduler.set_begin_index(0)
        seg_steps = len(seg) - 1

        # 4. 初始 latent。续采(init_latent_ref + add_noise=False)= 原样注入上段带噪 latent(不缩放/不加噪);
        #    base / 无注入 = randn(经 pipe.prepare_latents,与整段 pipe() 同 generator → 同初噪 → bit 一致)。
        num_channels = transformer.in_channels
        init_latent = self._load_init_latent(req, device) if getattr(req, "init_latent_ref", None) else None
        if init_latent is not None:
            latents = pipe.prepare_latents(
                1, num_channels, req.height, req.width, torch.float32, device, gen, init_latent)
            if add_noise:
                # add_noise=enable + 注入图:按本段起始 sigma 重加噪(ComfyUI KSamplerAdvanced add_noise=enable
                # 配 start_at_step 的 img2img 风;留噪接力主路是 add_noise=disable → 跳过)。
                noise = torch.randn(
                    latents.shape, generator=gen, device=device, dtype=torch.float32)
                latents = scheduler.scale_noise(latents, scheduler.timesteps[:1], noise)
        else:
            latents = pipe.prepare_latents(
                1, num_channels, req.height, req.width, torch.float32, device, gen, None)

        # 5. denoise loop —— 逐行对照 pipeline_z_image.py(guidance=0 单次前向)。
        import asyncio  # noqa: PLC0415
        import math  # noqa: PLC0415

        def _predict_noise(lat_in: Any, sigma_scalar: float) -> Any:
            """在任意 sigma 处再前向一次(dpmpp_2s_ancestral 的中间 D_i 用)。复刻主循环的前向 + negate;
            timestep = 1 - sigma(主循环 (1000 - σ*1000)/1000 的等价直写)。**仅新 dpmpp 路用,
            euler/euler_ancestral 主路仍走下方 inline 计算 → golden byte-identical 不受影响。**"""
            ts = torch.full((lat_in.shape[0],), 1.0 - sigma_scalar,
                            dtype=torch.float32, device=device)
            lmi = lat_in.to(transformer.dtype).unsqueeze(2)
            out = transformer(list(lmi.unbind(dim=0)), ts, prompt_embeds, return_dict=False)[0]
            npred = torch.stack([o.float() for o in out], dim=0).squeeze(2)
            return (-npred).to(torch.float32)

        def _anc_step(s_from: float, s_to: float) -> tuple[float, float]:
            """get_ancestral_step(eta=1)—— dpmpp_sde 在 e^-λ 空间算降噪步 sd + 加噪量 su。
            逐式对照 ComfyUI k_diffusion get_ancestral_step。"""
            su = min(s_to, (s_to ** 2 * (s_from ** 2 - s_to ** 2) / s_from ** 2) ** 0.5)
            sd = (s_to ** 2 - su ** 2) ** 0.5
            return sd, su

        old_denoised: Any = None  # dpmpp_2m 多步状态(上一步 denoised);其余采样器不用
        sde_d1: Any = None  # dpmpp_2m_sde/3m_sde 历史:上一步 denoised
        sde_d2: Any = None  # dpmpp_3m_sde 历史:上上步 denoised
        sde_h1: float | None = None  # 上一步 h(=Δλ)
        sde_h2: float | None = None  # 上上步 h(3m 三阶用)
        for i in range(seg_steps):
            if cancel_flag is not None and cancel_flag.is_set():
                raise asyncio.CancelledError()
            t = scheduler.timesteps[i]
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            latent_model_input = latents.to(transformer.dtype).unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))
            model_out_list = transformer(
                latent_model_input_list, timestep, prompt_embeds, return_dict=False)[0]
            noise_pred = torch.stack([o.float() for o in model_out_list], dim=0)
            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred
            noise_pred = noise_pred.to(torch.float32)
            # 采样期 latent 干预(spec 2026-06-10):在 denoised(x0)语义点过 chain,再反推 noise_pred
            # (euler/ancestral 分支都用 noise_pred → 一处接入两路一致)。空 chain 跳过 = byte-identical。
            if interventions:
                _sig = float(seg[i])
                if _sig > 0.0:
                    _den = latents - _sig * noise_pred
                    for _fn in interventions:
                        _den = _fn(i, seg_steps, _sig, _den, seg)  # seg=本段 sigma schedule(色彩锚定 phase 需)
                    noise_pred = (latents - _den) / _sig
            if sampler_name == "euler_ancestral":
                # rectified-flow ancestral(PR-1b)—— 逐式对照 ComfyUI sample_euler_ancestral_RF
                # (k_diffusion/sampling.py:240;Z-Image=CONST/RF,**不是** EDM 版的 get_ancestral_step)。
                # denoised(x0)= latents - sigma*noise_pred(noise_pred 即 flow 速度 v;eta=0 时本式
                # 代数化简 == diffusers euler step,已验)。eta=1/s_noise=1(ComfyUI 默认)。
                sigma = float(seg[i])
                sigma_next = float(seg[i + 1])
                denoised = latents - sigma * noise_pred
                if sigma_next == 0.0:
                    latents = denoised
                else:
                    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * 1.0  # eta=1
                    sigma_down = sigma_next * downstep_ratio
                    alpha_ip1 = 1.0 - sigma_next
                    alpha_down = 1.0 - sigma_down
                    renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
                    sdr = sigma_down / sigma
                    latents = sdr * latents + (1.0 - sdr) * denoised
                    noise = torch.randn(latents.shape, generator=gen, device=device, dtype=torch.float32)
                    latents = (alpha_ip1 / alpha_down) * latents + noise * renoise_coeff
            elif sampler_name == "dpmpp_2m":
                # DPM-Solver++(2M) —— 逐式对照 ComfyUI sample_dpmpp_2m(k_diffusion/sampling.py:796)。
                # t_fn=-log(σ)、sigma_fn=exp(-t):一阶项展开 == RF euler 一阶(σ_next/σ 漂移),故对
                # Z-Image(CONST/RF)与 euler 一致基线上做二阶外插。确定性、无额外前向、无噪声。
                sigma = float(seg[i])
                sigma_next = float(seg[i + 1])
                denoised = latents - sigma * noise_pred
                if sigma_next == 0.0:
                    latents = denoised
                else:
                    t_cur = -math.log(sigma)
                    t_next = -math.log(sigma_next)
                    h = t_next - t_cur
                    ratio = math.exp(-t_next) / math.exp(-t_cur)  # == σ_next/σ
                    e = math.expm1(-h)
                    if old_denoised is None:
                        latents = ratio * latents - e * denoised
                    else:
                        # 二阶:r=h_last/h,用上一步 denoised 外插(ComfyUI denoised_d)。
                        r = (t_cur - (-math.log(float(seg[i - 1])))) / h
                        denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
                        latents = ratio * latents - e * denoised_d
                old_denoised = denoised
            elif sampler_name == "dpmpp_2s_ancestral":
                # DPM-Solver++(2S) ancestral, rectified-flow 变体 —— 逐式对照 ComfyUI
                # sample_dpmpp_2s_ancestral_RF(k_diffusion/sampling.py:686;λ=log((1-σ)/σ)、
                # sigma_fn=1/(e^λ+1))。二阶单步:中间 sigma_s 处再前向一次(D_i),NFE 翻倍;末步退化
                # euler;eta=1/s_noise=1(ComfyUI 默认)。
                sigma = float(seg[i])
                sigma_next = float(seg[i + 1])
                denoised = latents - sigma * noise_pred
                # 末步(sigma_next=0)守卫先于除法 —— Z-Image normal 调度尾部是双 0(sigma_min=0 致
                # seg[-2]=seg[-1]=0),不先守卫会 0/0。ComfyUI 末步 to_d 到 sigma_down=0 代数化简 ==
                # latents=denoised(与 euler_ancestral 同),sigma=0 退化亦然。
                if sigma_next == 0.0:
                    latents = denoised
                else:
                    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * 1.0  # eta=1
                    sigma_down = sigma_next * downstep_ratio
                    alpha_ip1 = 1.0 - sigma_next
                    alpha_down = 1.0 - sigma_down
                    renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
                    if sigma == 1.0:
                        sigma_s = 0.9999  # λ(1.0)=log(0) 发散,ComfyUI 同样钳制
                    else:
                        t_i = math.log((1.0 - sigma) / sigma)
                        t_down = math.log((1.0 - sigma_down) / sigma_down)
                        s = t_i + 0.5 * (t_down - t_i)  # r=1/2
                        sigma_s = 1.0 / (math.exp(s) + 1.0)
                    ssr = sigma_s / sigma
                    u = ssr * latents + (1.0 - ssr) * denoised
                    d_i = u - sigma_s * _predict_noise(u, sigma_s)  # 中间二阶前向(不过 intervention)
                    sdr = sigma_down / sigma
                    latents = sdr * latents + (1.0 - sdr) * d_i
                    noise = torch.randn(latents.shape, generator=gen, device=device, dtype=torch.float32)
                    latents = (alpha_ip1 / alpha_down) * latents + noise * renoise_coeff  # s_noise=1
            elif sampler_name in ("dpmpp_2m_sde", "dpmpp_3m_sde"):
                # DPM-Solver++ (2M/3M) SDE, rectified-flow —— 逐式对照 ComfyUI sample_dpmpp_2m_sde /
                # sample_dpmpp_3m_sde(CONST 分支:half-log-snr λ=log((1-σ)/σ)、α_t=σ_next·e^λ_t=1-σ_next)。
                # 随机解算:噪声用 seeded randn 逐步 —— 单向前传中相邻不相交区间的布朗增量本就是独立单位正态
                # == randn(ComfyUI 默认 BrownianTree 仅多了跨区间相关性,前传无重启时等价),故免引 torchsde
                # 依赖;与 euler_ancestral 同精度门:分布正确 + 同 seed 可复现,非逐字节对齐 ComfyUI。eta=1/s_noise=1。
                # 首 σ≥1 时 logit(1)=∞ → 钳到 0.9999(对齐 ComfyUI offset_first_sigma_for_snr 的 CONST 分支)。
                sigma = 0.9999 if (i == 0 and float(seg[i]) >= 1.0) else float(seg[i])
                sigma_next = float(seg[i + 1])
                denoised = latents - sigma * noise_pred
                if sigma_next == 0.0:
                    latents = denoised
                else:
                    lambda_s = math.log((1.0 - sigma) / sigma)
                    lambda_t = math.log((1.0 - sigma_next) / sigma_next)
                    h = lambda_t - lambda_s
                    h_eta = h * 2.0  # eta=1 → h_eta=2h
                    alpha_t = 1.0 - sigma_next
                    neg_expm1_heta = -math.expm1(-h_eta)  # = (-h_eta).expm1().neg()
                    latents = (sigma_next / sigma) * math.exp(-h) * latents + alpha_t * neg_expm1_heta * denoised
                    if sampler_name == "dpmpp_3m_sde":
                        if sde_h2 is not None:
                            r0 = sde_h1 / h
                            r1 = sde_h2 / h
                            d1_0 = (denoised - sde_d1) / r0
                            d1_1 = (sde_d1 - sde_d2) / r1
                            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                            d2 = (d1_0 - d1_1) / (r0 + r1)
                            phi_2 = math.expm1(-h_eta) / h_eta + 1.0
                            phi_3 = phi_2 / h_eta - 0.5
                            latents = latents + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
                        elif sde_h1 is not None:
                            r = sde_h1 / h
                            d = (denoised - sde_d1) / r
                            phi_2 = math.expm1(-h_eta) / h_eta + 1.0
                            latents = latents + (alpha_t * phi_2) * d
                    else:  # dpmpp_2m_sde(midpoint,ComfyUI 默认 solver_type)
                        if sde_h1 is not None:
                            r = sde_h1 / h
                            latents = latents + 0.5 * alpha_t * neg_expm1_heta * (1.0 / r) * (denoised - sde_d1)
                    noise = torch.randn(latents.shape, generator=gen, device=device, dtype=torch.float32)
                    latents = latents + noise * sigma_next * math.sqrt(-math.expm1(-2.0 * h))  # s_noise=1
                    sde_d1, sde_d2 = denoised, sde_d1
                    sde_h1, sde_h2 = h, sde_h1
            elif sampler_name == "dpmpp_sde":
                # DPM-Solver++ SDE(单步二阶随机)—— 逐式对照 ComfyUI sample_dpmpp_sde(CONST 分支,r=1/2)。
                # λ=log((1-σ)/σ)、sigma_fn=1/(1+e^λ)、α=1-σ;中间 σ_s1=sigma_fn(λ_s+h/2)处再前向一次(NFE 翻倍);
                # 两段 ancestral 噪声在 e^-λ 空间(_anc_step)。r=1/2 → fac=1 → denoised_d=denoised_2。
                # 噪声 seeded randn 代 BrownianTree(同 2m_sde 理由);首 σ≥1 钳 0.9999;末步守卫先于 λ。eta=1/s_noise=1。
                sigma = 0.9999 if (i == 0 and float(seg[i]) >= 1.0) else float(seg[i])
                sigma_next = float(seg[i + 1])
                denoised = latents - sigma * noise_pred
                if sigma_next == 0.0:
                    latents = denoised
                else:
                    lam_s = math.log((1.0 - sigma) / sigma)
                    lam_t = math.log((1.0 - sigma_next) / sigma_next)
                    h = lam_t - lam_s
                    lam_s1 = lam_s + 0.5 * h  # r=1/2
                    sigma_s1 = 1.0 / (1.0 + math.exp(lam_s1))
                    alpha_s = 1.0 - sigma
                    alpha_s1 = 1.0 - sigma_s1
                    alpha_t = 1.0 - sigma_next
                    # Step 1 → 中间 σ_s1 处再前向得 denoised_2
                    sd1, su1 = _anc_step(math.exp(-lam_s), math.exp(-lam_s1))
                    h1_ = (-math.log(sd1)) - lam_s
                    x2 = (alpha_s1 / alpha_s) * math.exp(-h1_) * latents - alpha_s1 * math.expm1(-h1_) * denoised
                    noise1 = torch.randn(latents.shape, generator=gen, device=device, dtype=torch.float32)
                    x2 = x2 + alpha_s1 * noise1 * su1  # s_noise=1
                    denoised_2 = x2 - sigma_s1 * _predict_noise(x2, sigma_s1)
                    # Step 2(fac=1 → denoised_d=denoised_2)
                    sd2, su2 = _anc_step(math.exp(-lam_s), math.exp(-lam_t))
                    h2_ = (-math.log(sd2)) - lam_s
                    latents = (alpha_t / alpha_s) * math.exp(-h2_) * latents - alpha_t * math.expm1(-h2_) * denoised_2
                    noise2 = torch.randn(latents.shape, generator=gen, device=device, dtype=torch.float32)
                    latents = latents + alpha_t * noise2 * su2  # s_noise=1
            else:
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            # 进度 + 逐组件时长(与整段路径对齐:denoise 首/末步时间戳 → VAE Decode 真时长)。
            _now = time.monotonic()
            stage_ts.setdefault("denoise_first", _now)
            stage_ts["denoise_last"] = _now
            if pt is not None and pt.has_callback:
                preview_url = None
                from src.services.inference.latent_preview import latent_to_preview_data_uri  # noqa: PLC0415
                preview_url = latent_to_preview_data_uri(latents)
                pt.step(i + 1, seg_steps, stage="dit_denoise", preview_url=preview_url)
        return latents

    def _zimage_decode_latents(self, pipe: Any, latents: Any) -> Any:
        """Z-Image latent → PIL(逐行对照 ZImagePipeline.__call__ 末段 decode)。分段路径出图时用。"""
        latents = latents.to(pipe.vae.dtype)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")
        return image[0]

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

    def _build_zimage_pipe(self) -> Any:
        """Z-Image:HF-layout 整模型 from_pretrained,或**单文件分开载入**(PR-2,spec 2026-06-09)。

        分开载入(三桥接 override,对齐 _build_klein_pipe):base/refiner 不同单文件 UNet + 独立编码器/VAE
        装配成 ZImagePipeline —— 还原用户写真工作流的 UNETLoader+VAELoader+CLIPLoader 分开载入。
        tokenizer/scheduler 从参考库(self.repo,Z-Image-Turbo 整模型或仓内 bundle)。组件经
        build_bridged_*(z-image 走 diffusers from_single_file)预建,经 *_override 灌入。
        无 override = 整模型 from_pretrained(零回归;smoke_zimage.py 验过 8 步出图)。"""
        zimage_cls = _import_zimage_pipeline()
        overrides = {k: v for k, v in (
            ("transformer", self._transformer_override),
            ("text_encoder", self._text_encoder_override),
            ("vae", self._vae_override),
        ) if v is not None}
        if len(overrides) == 3:
            from transformers import AutoTokenizer  # noqa: PLC0415
            tokenizer = AutoTokenizer.from_pretrained(str(Path(self.repo) / "tokenizer"))
            scheduler = _import_flow_schedulers()["euler"].from_pretrained(str(Path(self.repo) / "scheduler"))
            return zimage_cls(
                transformer=overrides["transformer"],
                text_encoder=overrides["text_encoder"],
                tokenizer=tokenizer,
                vae=overrides["vae"],
                scheduler=scheduler)
        # low_cpu_mem_usage=False:对齐 HF README(Z-Image 加载建议),避免 meta-init 与本仓桥接路径冲突。
        pipe = zimage_cls.from_pretrained(
            self.repo, torch_dtype=_torch_dtype(self.dtype), low_cpu_mem_usage=False)
        if overrides:
            pipe.register_modules(**overrides)
        return pipe

    def _build_qwen_edit_pipe(self) -> Any:
        """Qwen-Image-Edit-2511:HF-layout 整模型,标准 `QwenImageEditPlusPipeline.from_pretrained`。
        编辑类(needs_image_input)—— infer 经通用 image= 注入路径喂输入图。组件名 transformer/
        text_encoder/vae 与 Flux2/Z-Image 一致 → fp8/逐组件放置/offload 共享。CFG 走 true_cfg_scale
        (infer 对 qwen 映射,非 guidance_scale)。20B DiT + Qwen2.5-VL-7B encoder,显存大(逐组件
        选卡/offload 见 [[project_image_vram_guard_arc]])。"""
        qwen_cls = _import_qwen_edit_pipeline()
        return qwen_cls.from_pretrained(
            self.repo, torch_dtype=_torch_dtype(self.dtype), low_cpu_mem_usage=False)

    def _build_ideogram4_pipe(self) -> Any:
        """Ideogram-4:HF-layout 整模型,标准 `Ideogram4Pipeline.from_pretrained`(7 组件:
        双 transformer + Qwen3-VL TE + tokenizer + scheduler + flux2 同款 VAE + 可选
        prompt_enhancer_head 全随 repo 加载)。组件名 transformer/text_encoder/vae 存在 →
        fp8/逐组件放置/offload 路径共享(unconditional_transformer 是额外组件,offload 序列
        pipeline 自带 model_cpu_offload_seq 已含)。bf16 峰值 ~58G(spike 2026-06-11)。"""
        ideo_cls = _import_ideogram4_pipeline()
        return ideo_cls.from_pretrained(
            self.repo, torch_dtype=_torch_dtype(self.dtype), low_cpu_mem_usage=False)

    @staticmethod
    def _lora_adapter_name(name: str) -> str:
        """spec.name → peft adapter 标识。peft 把 adapter_name 当 torch module 名注入,
        而 nn.Module 名禁含 "."(add_module KeyError)。画布 lora_select 给的是**带
        .safetensors 扩展名的文件名** → 直接用必炸(2026-06-11 万物迁移真机坐实);
        legacy yaml 名不带点所以 #131 当年没暴露。消毒只影响内部 adapter 标识,
        缓存键/UI 仍用原始 spec.name。"""
        return name.replace(".", "_")

    def _apply_loras(self, loras: list) -> None:
        """LoRA(含 ComfyUI 格式)接 Flux2Klein modular pipe(经 Flux2LoraLoaderMixin,
        与 DiffusionPipeline 同 API)。复用 #125 `_maybe_convert_comfy_flux2_lora`(绕 is_kohya
        误判)。PR-3:无 cpu_offload(.to(device)),省 legacy 的 offload dance。
        _loaded_loras 存**消毒后**的 adapter 名(与 pipe 内 peft 状态同口径)。"""
        pipe = self._pipe
        if not loras:
            active = pipe.get_active_adapters() if hasattr(pipe, "get_active_adapters") else []
            if active:
                pipe.set_adapters([])
            return
        for spec in loras:
            adapter = self._lora_adapter_name(spec.name)
            if adapter in self._loaded_loras:
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
                pipe.load_lora_weights(converted, adapter_name=adapter)
            else:
                pipe.load_lora_weights(lora_path, adapter_name=adapter)
            active = pipe.get_active_adapters() or []
            if adapter not in active and adapter not in (getattr(pipe, "peft_config", {}) or {}):
                raise ValueError(
                    f"LoRA {spec.name!r} 零匹配(键不符 {type(pipe).__name__})—— 用对架构的 LoRA")
            self._loaded_loras.add(adapter)
        # round3 #6:删掉本次请求里不再包含的旧 LoRA。set_adapters 只是停用、不释放权重 →
        # 一个 adapter 生命周期内,用户每换一个 LoRA 旧的都常驻 pipe(显存累积)。这里把
        # 已装但本次没要的删掉,使 pipe 的 LoRA 集合收敛到当前请求。
        requested = {self._lora_adapter_name(s.name) for s in loras}
        stale = self._loaded_loras - requested
        if stale and hasattr(pipe, "delete_adapters"):
            try:
                pipe.delete_adapters(list(stale))
            except Exception:  # noqa: BLE001 — 删 LoRA 失败不该挡出图
                pass
            self._loaded_loras -= stale
        pipe.set_adapters([self._lora_adapter_name(s.name) for s in loras],
                          adapter_weights=[s.strength for s in loras])

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
        # 采样期 latent 干预闭包(spec 2026-06-10);空 list = 无干预(零回归)。两挂钩点(Z-Image 手写
        # 循环 / Flux2 callback_on_step_end)共用此 list。
        interventions = self._build_interventions(req, pipe)
        # PR-A2:真 img2img(z-image + input_image + 0<strength<1)→ 切到 img2img 变体 pipe(复用组件)。
        # 下方 image= 注入(按 pipe.__call__ 签名)对 img2img pipe 自动生效;再补 strength。
        img2img_mode = self._wants_img2img(req)
        if img2img_mode:
            pipe = self._ensure_img2img_pipe()
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
        # 逐组件时长(VAE decode 真计时,spec 2026-06-04):在 step 回调里打 denoise 首/末步
        # 时间戳,pipe() 返回后算 dit_denoise(首→末步)与 vae_decode(末步→pipe 返回 = 内嵌
        # decode 段)各自耗时,塞 InferenceResult.metadata.stage_latency_ms。executor 据此让
        # KSampler 显纯 denoise、VAE Decode 显真 decode 时长(修「VAE Decode 恒 0s」)。
        # **纯记时间戳,不改 pipe / 生成 / decode → 出图字节不变(smoke SSIM=1.0)**。
        _stage_ts: dict[str, float] = {}
        cfg = float(req.cfg_scale)
        # Z-Image-Turbo 是 distilled:**guidance_scale 必须 0**(非零掉质量,HF README + 冒烟验)。
        # 忽略请求的 cfg。其余架构(Flux2)按下方 cfg→guidance 逻辑。
        if self.pipeline_class == "ZImagePipeline":
            cfg = 0.0
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
        # Qwen-Image-Edit-2511:CFG 旋钮是 **true_cfg_scale**(非 guidance_scale;后者是 embedded
        # guidance,非 distilled 该模型用 None)。negative 走字符串入参(__call__ 原生支持,不像 Flux2 要预编码)。
        # 真 API 确认:pipeline_qwenimage_edit_plus.py __call__(image=, true_cfg_scale=4.0, guidance_scale=None)。
        # Ideogram-4:guidance_scale 与 guidance_schedule **互斥且 schedule 有非 None 默认**
        # ((7.0,)*45+(3.0,)*3)—— 只传标量 guidance_scale 会撞 "Only one of ..." ValueError。
        # 显式 schedule=None 走标量;cfg 默认 7(arch adapter)。__call__ 无 negative 入参。
        if self.pipeline_class == "Ideogram4Pipeline":
            call_kwargs["guidance_schedule"] = None
        if self.pipeline_class == "QwenImageEditPlusPipeline":
            call_kwargs.pop("guidance_scale", None)
            call_kwargs["true_cfg_scale"] = cfg
            _neg = (getattr(req, "negative_prompt", "") or "").strip()
            if _neg:
                call_kwargs["negative_prompt"] = _neg
        # 输入图(编辑/img2img/多参考):pipeline __call__ 接受 `image=` 才注入 —— Flux2(可选编辑)/
        # Qwen-Image-Edit(必需)接受;纯文生图 pipeline(Z-Image)不接受 → 跳过(忽略 input_image,不崩)。
        # 多参考:逗号分隔多路径 → list[PIL](Flux2 / Qwen-Edit "Plus" 支持多图);单图 → 单 PIL。
        input_image = getattr(req, "input_image", None)
        if input_image:
            import inspect  # noqa: PLC0415
            if "image" in inspect.signature(pipe.__call__).parameters:
                parts = [p.strip() for p in str(input_image).split(",") if p.strip()] \
                    if not str(input_image).startswith("data:") else [str(input_image)]
                imgs = [_decode_input_image(p) for p in parts] or [_decode_input_image(str(input_image))]
                call_kwargs["image"] = imgs if len(imgs) > 1 else imgs[0]
            else:
                import logging  # noqa: PLC0415
                logging.getLogger(__name__).warning(
                    "pipeline %s 不接受 image= 入参,忽略 input_image(纯文生图架构)", self.pipeline_class)
        # PR-A2:img2img 时补 strength(0<strength<1;img2img pipe 据此算从输入图加噪起点 + 截步)。
        # _wants_img2img 已保证 image= 已注入(input_image 非空)且 pipe 是 img2img 变体(接受 strength)。
        if img2img_mode:
            call_kwargs["strength"] = float(req.strength)
        # batch 出图(num_images>1):标准 pipe 经 num_images_per_prompt 一次前向出 N 张。段路(手写
        # 分段循环)暂只 1 张 → log 跳过(follow-up)。仅当 pipe.__call__ 接受该参时传(diffusers pipe 基本都有)。
        num_images = int(getattr(req, "num_images", 1) or 1)
        if num_images > 1:
            import inspect  # noqa: PLC0415
            import logging  # noqa: PLC0415
            if self._wants_segmented(req):
                logging.getLogger(__name__).warning(
                    "段路采样器(%s/%s)暂不支持 batch,num_images=%d → 出 1 张(follow-up)",
                    req.sampler_name, req.scheduler, num_images)
            elif "num_images_per_prompt" in inspect.signature(pipe.__call__).parameters:
                call_kwargs["num_images_per_prompt"] = num_images
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
        # progress_callback(done, total, **extras) → runner 发 P.NodeProgress。
        # 三个标准 pipeline(Flux2/Z-Image/Qwen-Edit)diffusers 原生都支持 callback_on_step_end
        # (已核 pipeline 源)—— 此前只挂 Flux2 是历史遗留,Z-Image 标准路(normal+euler 纯文生图)
        # 和 Qwen-Edit 既无逐步进度也无 step 级取消。Z-Image 分段路(_run_zimage_segmented)自带
        # 进度不经此;modular fallback(ERNIE 等)不支持回调,仍不挂。
        # interventions 仍 Flux2-only(Z-Image 干预走分段路;Qwen-Edit 未定义干预语义,不悄悄开)。
        _step_cb_pipes = ("Flux2KleinPipeline", "ZImagePipeline", "QwenImageEditPlusPipeline", "Ideogram4Pipeline")
        if self.pipeline_class in _step_cb_pipes and (
                progress_callback is not None or cancel_flag is not None
                or (interventions and self.pipeline_class == "Flux2KleinPipeline")):
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
                # 采样期 latent 干预(spec 2026-06-10):diffusers callback 给的是 step 后 latents(非
                # denoised x0);PR-1 在 latents 上过 chain 验管道。**PR-2 精修**:LCS 锐化作用在 denoised,
                # 需按当前 sigma 换算 denoised 再干预(否则方向语义差)。改后写回 cb_kwargs 交还 diffusers。
                # Flux2-only(外层 gate 扩到 Z-Image/Qwen-Edit 后,这里显式守住原行为)。
                if interventions and self.pipeline_class == "Flux2KleinPipeline":
                    _lat = cb_kwargs.get("latents")
                    if _lat is not None:
                        _sig = float(cb_kwargs.get("sigma", 0.0)) if isinstance(cb_kwargs, dict) else 0.0
                        _sched = getattr(getattr(_pipe, "scheduler", None), "sigmas", None)
                        for _fn in interventions:
                            _lat = _fn(i, total_steps, _sig, _lat, _sched)
                        cb_kwargs["latents"] = _lat
                # 逐组件时长:记 denoise 首/末步时间戳(callback 在每步**末**触发)。
                _now = time.monotonic()
                _stage_ts.setdefault("denoise_first", _now)
                _stage_ts["denoise_last"] = _now
                # PR-4:ProgressTracker.step 算 latency 滑窗(16)+ ETA + emit。
                pt.step(i + 1, total_steps, stage="dit_denoise", preview_url=preview_url)
                return cb_kwargs

            call_kwargs["callback_on_step_end"] = _step_cb

        # 输出模式(spec 2026-06-08 路 B,PR-B1):latent 模式让 pipe 跳过 VAE decode 直接返真 latent
        # 张量(output_type="latent"),终端落盘传 latent_ref(下游 sample_from_latent 注入 latents=)。
        # 默认 image 模式不设此 kwarg → 行为/出图字节不变(golden SSIM 1.0)。
        want_latent = getattr(req, "output_mode", "image") == "latent"
        if want_latent:
            call_kwargs["output_type"] = "latent"

        # 关键(flux2 默认引擎 to_thread,同 anima #206):pipe() 的 denoise(1024 ~20-30s)
        # 是阻塞 CUDA 调用。在 runner coroutine 里同步跑会占住事件循环 → ① _step_cb 发的
        # 进度 task 全卡到 pipe() 返回才 flush(逐步进度不实时)② pipe-reader 读不到 Abort
        # → cancel_flag 永不置位、中途取消失效 ③ 慢卡长 denoise >ping_timeout 触发 watchdog
        # 介入。丢 to_thread 让事件循环空闲:进度实时 flush、Abort 可读、cancel 生效。
        # 留噪 latent 接力 / 分段采样(PR-B2):Z-Image + 任一分段字段非默认 → 走手写去噪循环
        # (整段 pipe() 不暴露 mid-schedule 起步 / 带噪中途导出)。其余照常整段 pipe()。
        segmented = self._wants_segmented(req)
        # 异常/取消清理(2026-06-11 体检):pipe() 中途抛(OOM/取消/干预崩)时,denoise 的
        # 中间激活/latents 留在 CUDA caching allocator —— 不清会把失败前的峰值一直占着,
        # 失败一次卡就「虚占」,后续请求雪上加霜。只清 allocator 缓存块,模型权重不动
        # (adapter 仍缓存复用);成功路径零开销(不进 except)。
        try:
            if segmented:
                from types import SimpleNamespace  # noqa: PLC0415

                # no_grad 必须裹手写循环 —— ZImagePipeline.__call__ 自带 @torch.no_grad(),我们这条手写
                # 路径绕过它,不裹会逐步累积 autograd graph(每步全层激活)→ 几步就 OOM(实测 91GB)。
                def _seg() -> Any:
                    with torch.no_grad():
                        return self._run_zimage_segmented(
                            pipe, req, gen, pt, _stage_ts, cancel_flag, interventions=interventions)

                final_latents = await asyncio.to_thread(_seg)
                if want_latent:
                    out = SimpleNamespace(images=final_latents)
                else:
                    def _dec() -> Any:
                        with torch.no_grad():
                            return self._zimage_decode_latents(pipe, final_latents)

                    out = SimpleNamespace(images=[await asyncio.to_thread(_dec)])
            else:
                out = await asyncio.to_thread(lambda: pipe(**call_kwargs))
        except BaseException:  # noqa: BLE001 — 含 CancelledError(BaseException 派生)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 — 清理 best-effort,不掩真错
                pass
            raise
        t_pipe_end = time.monotonic()
        # 逐组件时长:denoise=首→末步;vae_decode=末步→pipe 返回(内嵌 decode 段)。
        # 只在 step 回调真跑过(Flux2Klein + 有 progress/cancel)时算 —— 其余路径(ERNIE
        # fallback / smoke 无回调)留空,metadata 不带 stage_latency_ms,行为/出图不变。
        stage_latency_ms: dict[str, int] = {}
        if "denoise_first" in _stage_ts:
            stage_latency_ms["dit_denoise"] = max(
                0, int((_stage_ts["denoise_last"] - _stage_ts["denoise_first"]) * 1000))
            stage_latency_ms["vae_decode"] = max(
                0, int((t_pipe_end - _stage_ts["denoise_last"]) * 1000))
        # PR-4:vae_decode 用 stage(progress=1.0) — 不调 finish(避免 _finished=True 阻塞
        # 测试场景下手动触发 step_cb;production 中 stage 行为等同 finish:progress=1/eta=0)。
        # Flux2 标准 pipe 里 VAE decode 已在 pipe() 内嵌跑完,这里发的是「post-denoise 收尾」一帧。
        pt.stage(
            "vae_decode", progress=1.0,
            step=total_steps, total_steps=total_steps,
            detail=f"vae_decode done ({total_steps} steps)",
        )
        latency_ms = int((time.monotonic() - t) * 1000)

        # 路 B latent 模式:不 decode,序列化真 latent 张量(safetensors)+ 返 latent 元信息;
        # runner 把字节落盘成 latent_ref.path(不进 msgpack)。media_type 区分,runner 据此分支。
        if want_latent:
            import safetensors.torch as _st  # noqa: PLC0415
            from src.services.inference.model_arch_adapter import arch_spec_by_pipeline  # noqa: PLC0415
            latent = out.images[0] if isinstance(out.images, (list, tuple)) else out.images
            blob = _st.save({"latent": latent.detach().contiguous().cpu()})
            _arch = arch_spec_by_pipeline(self.pipeline_class)
            return InferenceResult(
                media_type="application/x-latent",
                data=blob,
                metadata={
                    "latent": {
                        "pipeline_class": self.pipeline_class,
                        "arch": (_arch.arch if _arch else None),
                        "shape": [int(x) for x in latent.shape],
                        "latent_channels": int(latent.shape[1]) if latent.dim() >= 2 else None,
                        "dtype": str(latent.dtype).replace("torch.", ""),
                        "width": req.width,
                        "height": req.height,
                        "seed": req.seed,
                    },
                },
                usage=UsageMeter(image_count=0, latency_ms=latency_ms),
            )

        # batch:out.images 是 PIL 列表(标准 pipe num_images_per_prompt=N → N 张;段路 → 1 张)。
        # 全部编码 PNG:首张进 data,其余进 extra_images(加法字段,单图消费者只读 data 不变)。
        imgs = list(out.images) if isinstance(out.images, (list, tuple)) else [out.images]
        png_blobs: list[bytes] = []
        for im in imgs:
            _buf = io.BytesIO()
            im.save(_buf, format="PNG")
            png_blobs.append(_buf.getvalue())
        return InferenceResult(
            media_type="image/png",
            data=png_blobs[0],
            extra_images=png_blobs[1:],
            metadata={
                "width": req.width,
                "height": req.height,
                "seed": req.seed,
                "engine": ("flux2klein" if self.pipeline_class == "Flux2KleinPipeline" else "modular"),
                # 逐组件时长(空 dict 时不影响下游;executor 据此给 KSampler/VAE Decode 真时长)。
                **({"stage_latency_ms": stage_latency_ms} if stage_latency_ms else {}),
            },
            usage=UsageMeter(image_count=len(png_blobs), latency_ms=latency_ms),
        )
