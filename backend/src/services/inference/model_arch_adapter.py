"""ModelArchAdapter — abstracts diffusers Pipeline family differences for ImageSampler.

PR-2 of image-component-multi-gpu spec §5.6.3. The ImageSampler's main loop is
Pipeline-family agnostic; per-arch differences (Klein has no CFG, Dev has CFG,
SDXL has dual CLIP) are isolated in adapter implementations registered by
Pipeline class name.

PR-2 ships only FluxKleinArchAdapter (matches the on-disk Flux2-Klein-9B model
verified by Task 0). Future PRs add FluxDev / SDXL / Z-Image / QwenImageEdit
adapters in ~50 LOC each.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelArchAdapter(Protocol):
    """Per-Pipeline-class settings + behavior switches consumed by ImageSampler.

    All methods are pure (no side effects) — adapter instances are singletons
    in MODEL_ARCH_REGISTRY.
    """

    def supports_cfg(self) -> bool:
        """True if this Pipeline class uses classifier-free guidance (CFG)
        in its denoise loop. Distilled models (Klein, Z-Image-Turbo) → False.
        Consumed by FluxDevArchAdapter (future PR adding CFG-branched models).
        """
        ...

    def supports_negative_prompt(self) -> bool:
        """True if encode_prompt accepts a negative_prompt argument.
        Distilled models reject it; mainline (Dev, SDXL) accept it.
        Consumed by FluxDevArchAdapter / SDXLArchAdapter (future PRs).
        """
        ...

    def default_steps(self) -> int:
        """Default num_inference_steps when caller didn't specify.
        Klein default = 25 (the Pipeline default at __call__ is 50, but distilled
        models converge in 9-25 — we pick 25 for safety).
        """
        ...

    def default_guidance_scale(self) -> float:
        """Default guidance_scale parameter for the Pipeline call.
        Distilled models ignore this but pipelines still expect the kwarg.
        Consumed by FluxDevArchAdapter (future PR). For distilled archs this
        value is functionally ignored at inference."""
        ...

    def supported_samplers(self) -> set[str]:
        """采样器(KSampler `sampler_name` 下拉)中本架构**真能用**的子集。

        通用 KSampler 节点列出候选(euler/heun/lcm…,可扩展);运行时 ModularImageBackend
        据此校验 —— 选了不在此集合的 → fail loud(清晰报错,不出图)。真模型已验:diffusers
        Flux2 flow-matching 管线只有 euler 兼容(heun 的 set_timesteps 不接 custom sigmas;
        lcm 只对 LCM-蒸馏模型有效)。新架构接入时在其 adapter 里声明支持集。"""
        ...

    def supported_schedulers(self) -> set[str]:
        """调度器(KSampler `scheduler` 下拉,sigma 调度)中本架构真能用的子集。
        Flux2:normal/karras/exponential/beta 均经 FlowMatchEuler 真模型验过。"""
        ...


class FluxKleinArchAdapter:
    """Flux2-Klein-9B family via diffusers Flux2KleinPipeline (标准 pipeline)。

    注:supports_cfg / supports_negative_prompt 是 legacy 字段(当前未被引擎消费)。#144 起
    comfy 单文件走标准 Flux2KleinPipeline(is_distilled=False)→ cfg/negative **实际生效**
    (true-CFG),与下面的 False 不符;留待架构收口 spec 统一清理,先不动以免破坏既有测试。"""

    def supports_cfg(self) -> bool:
        return False

    def supports_negative_prompt(self) -> bool:
        return False

    def default_steps(self) -> int:
        return 25

    def default_guidance_scale(self) -> float:
        return 4.0  # matches Pipeline kwarg default; ignored at inference for distilled

    def supported_samplers(self) -> set[str]:
        # diffusers Flux2 flow-matching 管线只有 euler 兼容(真模型验:heun set_timesteps
        # 不接 custom sigmas → 崩;lcm 仅 LCM-蒸馏模型有效)。
        return {"euler"}

    def supported_schedulers(self) -> set[str]:
        # 9 个全支持(对齐 ComfyUI):normal/karras/exponential/beta 走 diffusers use_*_sigmas;
        # simple/sgm_uniform/ddim_uniform/linear_quadratic/kl_optimal 手动算 sigma 注入
        # pipe(sigmas=...)。sigma 算法 port 自 ComfyUI,数值经 ground truth 验过
        # (sigma_schedules.py + test_sigma_schedules.py)。
        return {
            "normal", "karras", "exponential", "beta",
            "simple", "sgm_uniform", "ddim_uniform", "linear_quadratic", "kl_optimal",
        }


# ---- 多架构注册表(spec 2026-06-07 P0:ImageArchSpec)----------------------------
#
# 把「加一个图像架构」从「改散落多处(runner 派发 if-elif / model_manager 选适配器 /
# 采样器能力)」收成「往 IMAGE_ARCH_REGISTRY 注册一条」。现有 flux2 / anima 迁入,行为不变。


@dataclass(frozen=True)
class ImageArchSpec:
    """一个图像模型架构的派发 + 能力声明(spec 2026-06-07 P0)。

    arch: 描述符里的 `adapter_arch`(loader 节点选的 / runner 读的),如 "flux2" / "anima" / "z-image"。
    pipeline_class: diffusers 的 Pipeline 类名(runner 据此选,引擎据此 from_pretrained / 校验)。
    adapter: 走哪个后端引擎 —— "modular"(ModularImageBackend,Flux2/Z-Image/Qwen-Edit 等标准
             diffusers pipeline)/ "anima"(AnimaImageBackend,自定义 DiT)。
    caps: 采样器/CFG/步数等能力(ModularImageBackend 的采样器校验消费;anima 不经此校验)。
    needs_image_input: 编辑类架构(qwen-edit / flux2 多参考编辑)需要输入图。
    img2img_pipeline_class: 真 img2img 变体(VAE-encode 输入图 + strength 加噪重去噪)的 diffusers 类名。
        仅部分架构有(z-image=ZImageImg2ImgPipeline);None=无 img2img 变体(Flux2-Klein 接图走多参考
        编辑条件,非 strength img2img)。引擎在「连了 input_image + 0<strength<1」时切到此 pipeline
        (spec 2026-06-08-multi-sampling-cross-model PR-A2)。
    """
    arch: str
    pipeline_class: str
    adapter: str
    caps: ModelArchAdapter
    needs_image_input: bool = False
    img2img_pipeline_class: str | None = None


class ZImageTurboArchAdapter:
    """Z-Image-Turbo(Tongyi-MAI,6B distilled)via diffusers ZImagePipeline(P1,spec 2026-06-07)。

    distilled:guidance_scale 必须 0(非零掉质量,HF README + 真机冒烟验);8 步即收敛;
    无 negative。HF-layout 整模型,from_pretrained。采样器:FlowMatchEuler(Z-Image 不走 Flux2 的
    sigma 注入路径,故 schedulers 只声明 normal)。"""

    def supports_cfg(self) -> bool:
        return False

    def supports_negative_prompt(self) -> bool:
        return False

    def default_steps(self) -> int:
        return 8

    def default_guidance_scale(self) -> float:
        return 0.0  # distilled — 必须 0

    def supported_samplers(self) -> set[str]:
        # euler 原生;euler_ancestral 经 PR-1b 手写 ancestral 步(留噪循环内),先不声明。
        return {"euler"}

    def supported_schedulers(self) -> set[str]:
        # 放开到 9 个(对齐 Flux2 / ComfyUI;spec 2026-06-09 Z-Image 引擎地基 PR-1)。Z-Image 走手写
        # 分段循环时 normal 用 scheduler.set_timesteps(golden 不变),其余用 sigma_schedules.compute_sigmas
        # (ComfyUI ground-truth 验过的 sigma 算法,shift=3)。
        return {
            "normal", "karras", "exponential", "beta",
            "simple", "sgm_uniform", "ddim_uniform", "linear_quadratic", "kl_optimal",
        }


class QwenImageEditArchAdapter:
    """Qwen-Image-Edit-2511(20B DiT + Qwen2.5-VL-7B encoder)via diffusers QwenImageEditPlusPipeline
    (P2 角度控制,spec 2026-06-07)。

    **非 distilled** —— CFG 旋钮是 `true_cfg_scale`(默认 4.0,非 guidance_scale;真 API 确认见
    pipeline_qwenimage_edit_plus.py)。编辑类(needs_image_input)。支持 negative_prompt 字符串。
    默认 40 步(README 推荐;__call__ 默认 50,取 40 平衡速度)。采样器 FlowMatchEuler。
    "Plus" 变体支持多参考图(image= 可传 list)。"""

    def supports_cfg(self) -> bool:
        return True

    def supports_negative_prompt(self) -> bool:
        return True

    def default_steps(self) -> int:
        return 40

    def default_guidance_scale(self) -> float:
        return 4.0  # → true_cfg_scale(引擎对 qwen 映射)

    def supported_samplers(self) -> set[str]:
        return {"euler"}

    def supported_schedulers(self) -> set[str]:
        return {"normal"}


DEFAULT_IMAGE_ARCH = "flux2"

IMAGE_ARCH_REGISTRY: dict[str, ImageArchSpec] = {
    "flux2": ImageArchSpec("flux2", "Flux2KleinPipeline", "modular", FluxKleinArchAdapter()),
    # anima 走 AnimaImageBackend(自定义 DiT),caps 占位(不经 ModularImageBackend 采样器校验)。
    "anima": ImageArchSpec("anima", "AnimaPipeline", "anima", FluxKleinArchAdapter()),
    # Z-Image-Turbo 文生图(P1):走 ModularImageBackend 的 ZImagePipeline 分支。
    # img2img 变体 ZImageImg2ImgPipeline(PR-A2):连 input_image + 0<strength<1 时切到它做加噪重去噪。
    "z-image": ImageArchSpec("z-image", "ZImagePipeline", "modular", ZImageTurboArchAdapter(),
                             img2img_pipeline_class="ZImageImg2ImgPipeline"),
    # Qwen-Image-Edit-2511 角度控制/编辑(P2):needs_image_input=True;走 QwenImageEditPlusPipeline 分支。
    "qwen-edit": ImageArchSpec("qwen-edit", "QwenImageEditPlusPipeline", "modular",
                               QwenImageEditArchAdapter(), needs_image_input=True),
}


def arch_spec_by_name(arch: str | None) -> ImageArchSpec:
    """按 adapter_arch 名取架构(未知 / None → 默认 flux2,零回归)。"""
    return IMAGE_ARCH_REGISTRY.get(arch or DEFAULT_IMAGE_ARCH,
                                   IMAGE_ARCH_REGISTRY[DEFAULT_IMAGE_ARCH])


def arch_spec_by_pipeline(pipeline_class: str) -> ImageArchSpec | None:
    """按 diffusers pipeline_class 反查架构(model_manager 据此选 modular/anima 适配器)。"""
    for s in IMAGE_ARCH_REGISTRY.values():
        if s.pipeline_class == pipeline_class:
            return s
    return None


# 采样器能力注册表 —— 由 IMAGE_ARCH_REGISTRY 派生(pipeline_class → caps),向后兼容
# image_modular `_validate_sampler_scheduler` 的既有按 pipeline_class 查法,无需改它。
# key 是 diffusers Pipeline 类名(Pipeline.__class__.__name__ / model_index.json _class_name)。
MODEL_ARCH_REGISTRY: dict[str, ModelArchAdapter] = {
    s.pipeline_class: s.caps for s in IMAGE_ARCH_REGISTRY.values()
}
