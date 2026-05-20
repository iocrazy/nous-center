"""PR-4: from_loaded_components 用预加载模块拼 pipe+sampler;LoRA path 应用。"""
from __future__ import annotations

import types

import pytest

from src.services.inference.component_spec import ComponentSpec
from src.services.inference.image_diffusers import DiffusersImageBackend


class _Mod:
    def __init__(self, dev):
        self._dev = dev

    @property
    def device(self):
        import torch
        return torch.device(self._dev)


def _modules():
    return {
        "transformer": _Mod("cuda:1"),
        "text_encoder": _Mod("cuda:0"),
        "tokenizer": object(),
        "vae": _Mod("cuda:2"),
    }


def _components(loras=None):
    return {
        "unet": ComponentSpec(kind="unet", file="/m/u.safe", device="cuda:1", dtype="bfloat16",
                              adapter_arch="flux2", loras=loras or []),
        "clip": ComponentSpec(kind="clip", file="/m/c.safe", device="cuda:0", dtype="bfloat16", clip_arch="flux2"),
        "vae":  ComponentSpec(kind="vae",  file="/m/v.safe", device="cuda:2", dtype="bfloat16"),
    }


def test_from_loaded_components_builds_sampler(monkeypatch):
    built = {}

    def _fake_assemble(self, modules):
        pipe = types.SimpleNamespace(
            transformer=modules["transformer"], text_encoder=modules["text_encoder"],
            tokenizer=modules["tokenizer"], vae=modules["vae"], scheduler=object())
        built["pipe"] = pipe
        return pipe

    monkeypatch.setattr(DiffusersImageBackend, "_assemble_pipe", _fake_assemble, raising=False)
    monkeypatch.setattr(
        "src.services.inference.image_diffusers.MODEL_ARCH_REGISTRY",
        {"Flux2KleinPipeline": object()}, raising=False)

    adapter = DiffusersImageBackend.from_loaded_components(_modules(), _components(), "Flux2KleinPipeline")
    assert adapter._sampler is not None
    assert adapter._sampler.pipe is built["pipe"]


def test_from_loaded_components_missing_kind_raises():
    with pytest.raises(ValueError, match="missing"):
        DiffusersImageBackend.from_loaded_components(_modules(), {"unet": _components()["unet"]}, "Flux2KleinPipeline")
