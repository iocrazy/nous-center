"""ImageArchSpec 多架构注册表(spec 2026-06-07 P0):arch↔pipeline_class 派发 + 能力。"""
from src.services.inference.model_arch_adapter import (
    DEFAULT_IMAGE_ARCH,
    IMAGE_ARCH_REGISTRY,
    MODEL_ARCH_REGISTRY,
    arch_spec_by_name,
    arch_spec_by_pipeline,
)


def test_arch_by_name_flux2_default_and_unknown():
    assert arch_spec_by_name("flux2").pipeline_class == "Flux2KleinPipeline"
    assert arch_spec_by_name(None).arch == DEFAULT_IMAGE_ARCH  # None → flux2(零回归)
    assert arch_spec_by_name("nope").arch == "flux2"           # 未知 → flux2 兜底


def test_arch_by_name_anima():
    s = arch_spec_by_name("anima")
    assert s.pipeline_class == "AnimaPipeline"
    assert s.adapter == "anima"


def test_arch_by_pipeline_reverse_lookup():
    assert arch_spec_by_pipeline("Flux2KleinPipeline").arch == "flux2"
    assert arch_spec_by_pipeline("Flux2KleinPipeline").adapter == "modular"
    assert arch_spec_by_pipeline("AnimaPipeline").adapter == "anima"
    assert arch_spec_by_pipeline("NoSuchPipeline") is None


def test_model_arch_registry_derived_backcompat():
    # 派生的 caps 注册表保留 Flux2KleinPipeline(image_modular 采样器校验按 pipeline_class 查)。
    assert "Flux2KleinPipeline" in MODEL_ARCH_REGISTRY
    caps = MODEL_ARCH_REGISTRY["Flux2KleinPipeline"]
    assert "euler" in caps.supported_samplers()
    assert "karras" in caps.supported_schedulers()


def test_registry_has_flux2_and_anima():
    assert set(IMAGE_ARCH_REGISTRY) >= {"flux2", "anima"}
    assert all(s.adapter in ("modular", "anima") for s in IMAGE_ARCH_REGISTRY.values())


def test_zimage_arch_registered():
    """P1:z-image → ZImagePipeline,走 modular 后端,distilled(guidance 0 / 8 步)。"""
    s = arch_spec_by_name("z-image")
    assert s.pipeline_class == "ZImagePipeline"
    assert s.adapter == "modular"
    assert arch_spec_by_pipeline("ZImagePipeline").arch == "z-image"
    assert s.caps.default_guidance_scale() == 0.0
    assert s.caps.default_steps() == 8
    assert s.needs_image_input is False  # 纯文生图
    # PR-1(2026-06-09 引擎地基):调度器放开到 9 个(对齐 Flux2/ComfyUI),引擎手写循环按 scheduler 算 sigma。
    scheds = s.caps.supported_schedulers()
    assert {"normal", "simple", "beta", "karras"} <= scheds
    assert len(scheds) == 9


def test_qwen_edit_arch_registered():
    """P2:qwen-edit → QwenImageEditPlusPipeline,走 modular,编辑类(needs_image_input),
    非 distilled(CFG 经 true_cfg_scale,default 4.0 / 40 步)。"""
    s = arch_spec_by_name("qwen-edit")
    assert s.pipeline_class == "QwenImageEditPlusPipeline"
    assert s.adapter == "modular"
    assert s.needs_image_input is True
    assert arch_spec_by_pipeline("QwenImageEditPlusPipeline").arch == "qwen-edit"
    assert s.caps.supports_cfg() is True
    assert s.caps.default_guidance_scale() == 4.0
    assert s.caps.default_steps() == 40
    assert "QwenImageEditPlusPipeline" in MODEL_ARCH_REGISTRY
