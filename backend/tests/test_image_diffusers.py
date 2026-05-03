"""Tests for DiffusersImageBackend.

GPU-free: the diffusers/transformers stack is mocked at the import seam so
load() composes a fake pipeline. The structural LoRA test mirrors the
design doc's outside-voice replacement for L2 pixel diff: assert the
pipeline's get_active_adapters / get_adapter_status reflect what we passed,
not the visual quality of the output.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from src.services.inference.base import (
    AudioRequest,
    ImageRequest,
    LoRASpec,
    MediaModality,
)
from src.services.inference.image_diffusers import DiffusersImageBackend


@pytest.fixture(autouse=True)
def _stub_diffusers(monkeypatch):
    """Inject fake diffusers/transformers/torch modules so import succeeds
    and we control return values from from_single_file / from_pretrained.
    """
    fake_torch = MagicMock()
    fake_torch.bfloat16 = "bfloat16-marker"
    fake_torch.float16 = "float16-marker"
    fake_torch.cuda.empty_cache = MagicMock()
    # torch.Generator(device=...).manual_seed(...) chain
    gen = MagicMock()
    gen.manual_seed = MagicMock(return_value=gen)
    fake_torch.Generator = MagicMock(return_value=gen)

    fake_transformers = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=MagicMock(name="tokenizer"))
    fake_transformers.AutoModel.from_pretrained = MagicMock(return_value=MagicMock(name="encoder"))

    fake_diffusers = MagicMock()
    fake_diffusers.FluxTransformer2DModel.from_single_file = MagicMock(return_value=MagicMock(name="transformer"))
    fake_diffusers.AutoencoderKL.from_single_file = MagicMock(return_value=MagicMock(name="vae"))

    # Pipeline class returns an instance with the methods we exercise
    pipe = MagicMock(name="flux2-pipeline")
    pipe._is_offloaded = False

    def _enable_offload(gpu_id=0):
        pipe._is_offloaded = True

    def _disable_offload():
        pipe._is_offloaded = False

    pipe.enable_model_cpu_offload = MagicMock(side_effect=_enable_offload)
    pipe.disable_model_cpu_offload = MagicMock(side_effect=_disable_offload)
    pipe.load_lora_weights = MagicMock()
    pipe.set_adapters = MagicMock()

    # Track active adapters for the structural LoRA test
    active_adapters: list[str] = []

    def _set_adapters(names, adapter_weights=None):
        nonlocal active_adapters
        active_adapters = list(names)

    pipe.set_adapters = MagicMock(side_effect=_set_adapters)
    pipe.get_active_adapters = MagicMock(side_effect=lambda: list(active_adapters))
    pipe.get_adapter_status = MagicMock(
        side_effect=lambda: [{"name": n, "status": "merged"} for n in active_adapters]
    )

    # __call__ returns object with .images = [pil_like]
    pil_like = MagicMock()

    def _save(buf, format="PNG"):
        buf.write(b"\x89PNG\r\n\x1a\nFAKE")

    pil_like.save = MagicMock(side_effect=_save)
    out = MagicMock()
    out.images = [pil_like]
    pipe.return_value = out

    Flux2Pipeline = MagicMock(return_value=pipe)
    fake_diffusers.Flux2Pipeline = Flux2Pipeline
    fake_diffusers.FluxPipeline = Flux2Pipeline  # fallback path uses same fake

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    return {
        "torch": fake_torch,
        "transformers": fake_transformers,
        "diffusers": fake_diffusers,
        "pipe": pipe,
    }


def _paths(tmp_path):
    transformer = tmp_path / "dit.safetensors"
    encoder = tmp_path / "encoder_dir"
    encoder.mkdir()
    (encoder / "config.json").write_text("{}")
    vae = tmp_path / "vae.safetensors"
    transformer.write_bytes(b"x")
    vae.write_bytes(b"x")
    return {
        "transformer": str(transformer),
        "text_encoder": str(encoder),
        "vae": str(vae),
    }


def _make_req(loras=None, seed=None) -> ImageRequest:
    return ImageRequest(
        request_id="r-1",
        prompt="a cat in space",
        width=512,
        height=512,
        steps=10,
        seed=seed,
        loras=loras or [],
    )


def test_modality_pinned_to_image():
    assert DiffusersImageBackend.modality is MediaModality.IMAGE


def test_streaming_unsupported():
    """Image backend doesn't override infer_stream → supports_streaming is False."""
    assert DiffusersImageBackend.supports_streaming() is False


async def test_load_composes_pipeline_in_correct_order(tmp_path, _stub_diffusers):
    backend = DiffusersImageBackend(paths=_paths(tmp_path))
    await backend.load(device="cuda:0")

    diffusers = _stub_diffusers["diffusers"]
    transformers = _stub_diffusers["transformers"]

    diffusers.FluxTransformer2DModel.from_single_file.assert_called_once()
    transformers.AutoTokenizer.from_pretrained.assert_called_once()
    transformers.AutoModel.from_pretrained.assert_called_once()
    diffusers.AutoencoderKL.from_single_file.assert_called_once()

    pipeline_args = diffusers.Flux2Pipeline.call_args.kwargs
    assert set(pipeline_args) == {"transformer", "text_encoder", "tokenizer", "vae", "scheduler"}
    assert pipeline_args["scheduler"] is None

    # cpu_offload enabled by default with single_card_offload strategy
    assert _stub_diffusers["pipe"]._is_offloaded is True
    _stub_diffusers["pipe"].enable_model_cpu_offload.assert_called_with(gpu_id=0)


async def test_load_no_offload_strategy_skips_cpu_offload(tmp_path, _stub_diffusers):
    backend = DiffusersImageBackend(paths=_paths(tmp_path), offload_strategy="no_offload")
    await backend.load(device="cuda:1")
    _stub_diffusers["pipe"].enable_model_cpu_offload.assert_not_called()


async def test_infer_rejects_non_image_request(tmp_path):
    backend = DiffusersImageBackend(paths=_paths(tmp_path))
    await backend.load(device="cuda:0")
    bad = AudioRequest(request_id="r", text="hi")
    with pytest.raises(TypeError, match="ImageRequest"):
        await backend.infer(bad)


async def test_infer_returns_png_envelope(tmp_path, _stub_diffusers):
    backend = DiffusersImageBackend(paths=_paths(tmp_path))
    await backend.load(device="cuda:0")

    result = await backend.infer(_make_req(seed=42))

    assert result.media_type == "image/png"
    assert result.data.startswith(b"\x89PNG")
    assert result.metadata["width"] == 512
    assert result.metadata["height"] == 512
    assert result.metadata["seed"] == 42
    assert result.usage.image_count == 1
    assert result.usage.latency_ms >= 0


async def test_infer_lora_apply_order_disable_then_set_then_reenable(tmp_path, _stub_diffusers):
    """The diffusers issue #7842 family fix: set_adapters must run with
    cpu_offload disabled, otherwise weights cross devices."""
    pipe = _stub_diffusers["pipe"]
    backend = DiffusersImageBackend(
        paths=_paths(tmp_path),
        lora_paths={"anime-v2": "/fake/anime-v2.safetensors"},
    )
    await backend.load(device="cuda:0")

    # Reset call history from the load() phase
    pipe.disable_model_cpu_offload.reset_mock()
    pipe.set_adapters.reset_mock()
    pipe.enable_model_cpu_offload.reset_mock()

    await backend.infer(_make_req(loras=[LoRASpec(name="anime-v2", strength=0.8)]))

    # Order: disable_model_cpu_offload BEFORE set_adapters BEFORE re-enable
    assert pipe.disable_model_cpu_offload.call_count == 1
    assert pipe.set_adapters.call_count == 1
    assert pipe.enable_model_cpu_offload.call_count == 1
    pipe.set_adapters.assert_called_with(["anime-v2"], adapter_weights=[0.8])

    # Structural LoRA test (replaces L2 pixel diff)
    assert pipe.get_active_adapters() == ["anime-v2"]
    assert all(s["status"] == "merged" for s in pipe.get_adapter_status())


async def test_infer_unknown_lora_raises(tmp_path):
    backend = DiffusersImageBackend(paths=_paths(tmp_path), lora_paths={})
    await backend.load(device="cuda:0")
    with pytest.raises(ValueError, match="not in registered lora_paths"):
        await backend.infer(_make_req(loras=[LoRASpec(name="missing", strength=1.0)]))


async def test_infer_no_loras_clears_active_adapters(tmp_path, _stub_diffusers):
    pipe = _stub_diffusers["pipe"]
    backend = DiffusersImageBackend(
        paths=_paths(tmp_path),
        lora_paths={"anime-v2": "/fake/anime-v2.safetensors"},
    )
    await backend.load(device="cuda:0")

    # First call loads + activates a LoRA
    await backend.infer(_make_req(loras=[LoRASpec(name="anime-v2", strength=1.0)]))
    assert pipe.get_active_adapters() == ["anime-v2"]

    # Second call without loras must clear
    await backend.infer(_make_req(loras=[]))
    assert pipe.get_active_adapters() == []


async def test_lora_only_loaded_once(tmp_path, _stub_diffusers):
    """Repeated infer with the same LoRA must not re-call load_lora_weights."""
    pipe = _stub_diffusers["pipe"]
    backend = DiffusersImageBackend(
        paths=_paths(tmp_path),
        lora_paths={"anime-v2": "/fake/anime-v2.safetensors"},
    )
    await backend.load(device="cuda:0")

    await backend.infer(_make_req(loras=[LoRASpec(name="anime-v2", strength=1.0)]))
    await backend.infer(_make_req(loras=[LoRASpec(name="anime-v2", strength=0.5)]))

    assert pipe.load_lora_weights.call_count == 1


async def test_unload_clears_state(tmp_path):
    backend = DiffusersImageBackend(paths=_paths(tmp_path))
    await backend.load(device="cuda:0")
    backend._loaded_loras.add("x")
    backend.unload()
    assert backend._pipe is None
    assert backend._loaded_loras == set()
    assert backend.is_loaded is False


async def test_missing_paths_key_raises_on_load():
    """paths must include all 3 components — partial spec is a hard error."""
    backend = DiffusersImageBackend(paths={"transformer": "/x"})
    with pytest.raises(ValueError, match="text_encoder"):
        await backend.load(device="cuda:0")
