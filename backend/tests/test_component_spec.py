"""ComponentSpec validation + ComponentKey hashing for L1 cache."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.services.inference.base import LoRASpec
from src.services.inference.component_spec import ComponentSpec, ComponentKey, to_component_key  # noqa: F401


def test_unet_component_spec_valid():
    spec = ComponentSpec(
        kind="diffusion_models",
        file="/abs/path/transformer.safetensors",
        device="cuda:1",
        dtype="bfloat16",
        loras=[LoRASpec(name="style", strength=0.8)],
        adapter_arch="flux2",
    )
    assert spec.kind == "diffusion_models"
    assert spec.device == "cuda:1"
    assert len(spec.loras) == 1


def test_clip_component_spec_valid():
    spec = ComponentSpec(
        kind="clip", file="/p/clip.safetensors", device="cuda:0",
        dtype="bfloat16", clip_arch="flux2",
    )
    assert spec.clip_arch == "flux2"
    assert spec.loras == []


def test_vae_component_spec_minimal():
    spec = ComponentSpec(kind="vae", file="/p/vae.safetensors", device="cuda:2", dtype="float16")
    assert spec.kind == "vae"


def test_kind_must_be_one_of_three():
    with pytest.raises(ValidationError):
        ComponentSpec(kind="other", file="/p/x", device="cuda:0", dtype="bfloat16")


def test_device_must_be_cuda_or_cpu_or_auto():
    # "auto" → ModelManager will resolve via get_best_gpu
    ComponentSpec(kind="vae", file="/p/x", device="auto", dtype="bfloat16")
    ComponentSpec(kind="vae", file="/p/x", device="cpu", dtype="bfloat16")
    ComponentSpec(kind="vae", file="/p/x", device="cuda:0", dtype="bfloat16")
    with pytest.raises(ValidationError):
        ComponentSpec(kind="vae", file="/p/x", device="mps:0", dtype="bfloat16")


def test_component_key_is_hashable_tuple():
    spec = ComponentSpec(
        kind="diffusion_models", file="/p/u.safe", device="cuda:1", dtype="bfloat16",
        loras=[LoRASpec(name="a", strength=0.8), LoRASpec(name="b", strength=0.4)],
    )
    key = to_component_key(spec)
    assert isinstance(key, tuple)
    assert len(key) == 5  # + unconditional_file(ideogram4 双 DiT,spec 2026-06-12)
    file_path, device, dtype, lora_frozenset, uncond = key
    assert file_path == "/p/u.safe"
    assert device == "cuda:1"
    assert dtype == "bfloat16"
    assert isinstance(lora_frozenset, frozenset)
    assert lora_frozenset == frozenset({("a", 0.8), ("b", 0.4)})
    assert uncond is None  # 非 ideogram4 单 DiT → None(零回归)
    # Hashable → usable as dict key
    d = {key: "loaded"}
    assert d[key] == "loaded"


def test_component_key_stable_across_lora_order():
    """Two specs with same loras in different order produce equal key (frozenset)."""
    s1 = ComponentSpec(kind="diffusion_models", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="a", strength=0.8), LoRASpec(name="b", strength=0.4)])
    s2 = ComponentSpec(kind="diffusion_models", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="b", strength=0.4), LoRASpec(name="a", strength=0.8)])
    assert to_component_key(s1) == to_component_key(s2)


def test_component_key_distinguishes_strength():
    s1 = ComponentSpec(kind="diffusion_models", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="a", strength=0.8)])
    s2 = ComponentSpec(kind="diffusion_models", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="a", strength=0.4)])
    assert to_component_key(s1) != to_component_key(s2)


def test_component_spec_re_exported_from_base():
    """Spec § 5.1 says ComponentSpec lives under inference.base for caller convenience."""
    from src.services.inference.base import ComponentSpec as CS_from_base
    assert CS_from_base is ComponentSpec


def test_device_rejects_leading_zero_cuda_index():
    """cuda:00 / cuda:007 would otherwise collide with cuda:0 / cuda:7 in ComponentKey."""
    with pytest.raises(ValidationError):
        ComponentSpec(kind="vae", file="/p/x", device="cuda:00", dtype="bfloat16")
    with pytest.raises(ValidationError):
        ComponentSpec(kind="vae", file="/p/x", device="cuda:007", dtype="bfloat16")


def test_device_canonicalizes_already_canonical_form():
    """Valid forms pass through unchanged (defensive — no normalization happens for already-canonical)."""
    s = ComponentSpec(kind="vae", file="/p/x", device="cuda:0", dtype="bfloat16")
    assert s.device == "cuda:0"
    s = ComponentSpec(kind="vae", file="/p/x", device="cuda:7", dtype="bfloat16")
    assert s.device == "cuda:7"


def test_loras_rejected_on_non_unet():
    with pytest.raises(ValidationError, match="loras is only meaningful"):
        ComponentSpec(kind="vae", file="/p/x", device="cuda:0", dtype="bfloat16",
                     loras=[LoRASpec(name="bad", strength=0.5)])
    with pytest.raises(ValidationError, match="loras is only meaningful"):
        ComponentSpec(kind="clip", file="/p/x", device="cuda:0", dtype="bfloat16",
                     loras=[LoRASpec(name="bad", strength=0.5)])


def test_adapter_arch_rejected_on_non_unet():
    with pytest.raises(ValidationError, match="adapter_arch is diffusion_models-only"):
        ComponentSpec(kind="vae", file="/p/x", device="cuda:0", dtype="bfloat16",
                     adapter_arch="flux2")
    with pytest.raises(ValidationError, match="adapter_arch is diffusion_models-only"):
        ComponentSpec(kind="clip", file="/p/x", device="cuda:0", dtype="bfloat16",
                     adapter_arch="flux2")


def test_clip_arch_rejected_on_non_clip():
    with pytest.raises(ValidationError, match="clip_arch is clip-only"):
        ComponentSpec(kind="vae", file="/p/x", device="cuda:0", dtype="bfloat16",
                     clip_arch="qwen")
    with pytest.raises(ValidationError, match="clip_arch is clip-only"):
        ComponentSpec(kind="diffusion_models", file="/p/x", device="cuda:0", dtype="bfloat16",
                     clip_arch="qwen")


def test_component_key_distinguishes_dtype():
    """Same file + device + loras, different dtype → distinct cache keys."""
    s_bf16 = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="bfloat16")
    s_fp16 = ComponentSpec(kind="vae", file="/p/v.safe", device="cuda:0", dtype="float16")
    assert to_component_key(s_bf16) != to_component_key(s_fp16)


def test_component_key_unchanged_when_only_lora_order_differs():
    """Sanity (regression): the existing frozenset-based stability still holds with 4-tuple."""
    s1 = ComponentSpec(kind="diffusion_models", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="a", strength=0.8), LoRASpec(name="b", strength=0.4)])
    s2 = ComponentSpec(kind="diffusion_models", file="/p/u", device="cuda:1", dtype="bfloat16",
                      loras=[LoRASpec(name="b", strength=0.4), LoRASpec(name="a", strength=0.8)])
    assert to_component_key(s1) == to_component_key(s2)


def test_component_state_key_stable_and_lora_aware():
    from src.services.inference.base import LoRASpec
    from src.services.inference.component_spec import ComponentSpec, component_state_key

    base = ComponentSpec(kind="diffusion_models", file="/m/u.safe", device="cuda:1", dtype="bfloat16", adapter_arch="flux2")
    assert component_state_key(base) == "/m/u.safe|cuda:1|bfloat16|"
    a = base.model_copy(update={"loras": [LoRASpec(name="x", strength=0.8), LoRASpec(name="y", strength=0.4)]})
    b = base.model_copy(update={"loras": [LoRASpec(name="y", strength=0.4), LoRASpec(name="x", strength=0.8)]})
    assert component_state_key(a) == component_state_key(b)
    assert component_state_key(a) != component_state_key(base)


def test_offload_stream_accepted():
    """lowvram 流式分块(spec 2026-06-12):offload=stream 过校验 —— 漏改此白名单曾让
    画布 e2e 在 runner 描述符层被 pydantic 拒(2026-06-12 真机逮到)。"""
    from src.services.inference.component_spec import ComponentSpec
    s = ComponentSpec(kind="diffusion_models", file="/m/x", device="cuda:2",
                      dtype="bfloat16", offload="stream")
    assert s.offload == "stream"


def test_uncond_file_differentiates_component_key():
    """ideogram4 双 DiT:同 cond DiT 配不同 uncond DiT → 不同 component key(防错命中缓存)。"""
    base = dict(kind="diffusion_models", file="/p/cond.safe", device="cuda:1", dtype="bfloat16")
    k_none = to_component_key(ComponentSpec(**base))
    k_u1 = to_component_key(ComponentSpec(**base, unconditional_file="/p/uncondA.safe"))
    k_u2 = to_component_key(ComponentSpec(**base, unconditional_file="/p/uncondB.safe"))
    assert k_none != k_u1
    assert k_u1 != k_u2
    assert k_none[-1] is None and k_u1[-1] == "/p/uncondA.safe"


def test_uncond_file_not_in_state_key_string():
    """uncond DiT 不进 component_state_key 串(cond file 已唯一标识对;保前端四态匹配串不变)。"""
    from src.services.inference.component_spec import component_state_key
    base = dict(kind="diffusion_models", file="/p/cond.safe", device="cuda:1", dtype="bfloat16")
    s1 = component_state_key(ComponentSpec(**base))
    s2 = component_state_key(ComponentSpec(**base, unconditional_file="/p/uncond.safe"))
    assert s1 == s2  # 串相同(uncond 不进串)
