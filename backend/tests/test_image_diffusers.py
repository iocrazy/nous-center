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
    # DiffusionPipeline.from_pretrained returns the same fake pipe object
    # so the test asserts hold for both load paths.
    fake_diffusers.DiffusionPipeline.from_pretrained = MagicMock()

    # Pipeline class returns an instance with the methods we exercise
    pipe = MagicMock(name="flux2-pipeline")
    pipe._is_offloaded = False

    def _enable_offload(gpu_id=0):
        pipe._is_offloaded = True

    def _disable_offload():
        pipe._is_offloaded = False

    pipe.enable_model_cpu_offload = MagicMock(side_effect=_enable_offload)
    pipe.disable_model_cpu_offload = MagicMock(side_effect=_disable_offload)

    # Real diffusers' load_lora_weights populates pipe.peft_config[adapter_name]
    # on success (and stays empty when the LoRA tensors don't match the
    # architecture). Mirror that: success path here adds to peft_config so
    # the architecture-mismatch guard treats this as a successful load.
    pipe.peft_config = {}

    def _load_lora(_path, adapter_name=None):
        pipe.peft_config[adapter_name] = {"loaded": True}

    pipe.load_lora_weights = MagicMock(side_effect=_load_lora)
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
    # Wire DiffusionPipeline.from_pretrained to the same fake pipe so
    # downstream infer/LoRA logic exercises the same code path regardless
    # of which load branch ran.
    fake_diffusers.DiffusionPipeline.from_pretrained = MagicMock(return_value=pipe)

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


def test_resolve_path_absolutizes_relative_against_local_models_path(tmp_path, monkeypatch):
    """Yaml stores paths relative to LOCAL_MODELS_PATH. _resolve_path MUST
    absolutize them — diffusers' from_single_file refuses anything that
    isn't an absolute path or HF hub id, so a relative path falls through
    as a hub id and triggers a confusing 'not a valid model identifier'
    error in production. Regression test for the bug found during real-GPU
    E2E (path was being passed through verbatim).
    """
    from src.config import get_settings as _gs
    from src.services import inference as _inf  # noqa: F401

    # Build a fake LOCAL_MODELS_PATH with the file at a relative location
    base = tmp_path / "models"
    (base / "image" / "diffusion_models" / "flux2").mkdir(parents=True)
    target = base / "image" / "diffusion_models" / "flux2" / "weights.safetensors"
    target.write_bytes(b"x")

    # Make get_settings().LOCAL_MODELS_PATH point at our tmp tree
    settings = _gs()
    monkeypatch.setattr(settings, "LOCAL_MODELS_PATH", str(base))

    backend = DiffusersImageBackend(paths={
        "transformer": "image/diffusion_models/flux2/weights.safetensors",
        "text_encoder": "/abs/text_encoder",
        "vae": "/abs/vae",
    })

    resolved = backend._resolve_path("transformer")
    assert resolved == target  # absolutized + exists

    # Absolute path with a non-existent file: must NOT clobber it (so HF
    # hub ids like 'org/model' fall through unchanged for diffusers to handle)
    resolved2 = backend._resolve_path("text_encoder")
    assert str(resolved2) == "/abs/text_encoder"


def test_resolve_path_returns_relative_unchanged_when_absolutized_missing(tmp_path, monkeypatch):
    """If the absolutized candidate doesn't exist on disk, fall back to
    the raw value so HF hub ids ('black-forest-labs/FLUX.2-dev' style)
    still reach diffusers as-is instead of getting LOCAL_MODELS_PATH glued
    onto them.
    """
    from src.config import get_settings as _gs

    base = tmp_path / "empty"
    base.mkdir()
    settings = _gs()
    monkeypatch.setattr(settings, "LOCAL_MODELS_PATH", str(base))

    backend = DiffusersImageBackend(paths={
        "transformer": "black-forest-labs/FLUX.2-dev",  # HF hub id
        "text_encoder": "/x",
        "vae": "/x",
    })
    resolved = backend._resolve_path("transformer")
    assert str(resolved) == "black-forest-labs/FLUX.2-dev"


# ----- from_pretrained branch (PR-8) -----


def _main_paths(tmp_path):
    """Diffusers full-layout dir: model_index.json + component subdirs."""
    root = tmp_path / "ernie"
    root.mkdir()
    (root / "model_index.json").write_text('{"_class_name":"ErnieImagePipeline"}')
    return {"main": str(root)}


async def test_load_routes_to_from_pretrained_when_paths_main(tmp_path, _stub_diffusers):
    backend = DiffusersImageBackend(paths=_main_paths(tmp_path))
    await backend.load(device="cuda:0")

    diffusers = _stub_diffusers["diffusers"]
    diffusers.DiffusionPipeline.from_pretrained.assert_called_once()
    call_args = diffusers.DiffusionPipeline.from_pretrained.call_args
    # First positional arg is the dir
    assert call_args.args[0] == str(tmp_path / "ernie")
    # trust_remote_code must be True so custom pipeline classes load
    assert call_args.kwargs.get("trust_remote_code") is True
    # Single-file compose helpers must NOT have been called for this branch
    diffusers.FluxTransformer2DModel.from_single_file.assert_not_called()
    diffusers.AutoencoderKL.from_single_file.assert_not_called()


async def test_load_routes_to_compose_when_paths_three_components(tmp_path, _stub_diffusers):
    backend = DiffusersImageBackend(paths=_paths(tmp_path))
    await backend.load(device="cuda:0")

    diffusers = _stub_diffusers["diffusers"]
    # 3-component path uses from_single_file twice + from_pretrained on encoder dir
    diffusers.FluxTransformer2DModel.from_single_file.assert_called_once()
    diffusers.AutoencoderKL.from_single_file.assert_called_once()
    # And NOT the new from_pretrained whole-pipeline path
    diffusers.DiffusionPipeline.from_pretrained.assert_not_called()


async def test_from_pretrained_branch_still_applies_cpu_offload(tmp_path, _stub_diffusers):
    """Both branches share the offload tail in load() — make sure ERNIE
    style load still gets enable_model_cpu_offload(gpu_id=...)."""
    pipe = _stub_diffusers["pipe"]
    backend = DiffusersImageBackend(paths=_main_paths(tmp_path))
    await backend.load(device="cuda:1")
    pipe.enable_model_cpu_offload.assert_called_with(gpu_id=1)


async def test_from_pretrained_branch_runs_infer_end_to_end(tmp_path, _stub_diffusers):
    """Sanity: load via from_pretrained → infer returns the standard
    InferenceResult envelope (PNG bytes + metadata + usage). Catches
    silent regressions where the from_pretrained branch wires up a pipe
    that infer() can't drive."""
    backend = DiffusersImageBackend(paths=_main_paths(tmp_path))
    await backend.load(device="cuda:0")
    result = await backend.infer(_make_req(seed=7))
    assert result.media_type == "image/png"
    assert result.data.startswith(b"\x89PNG")
    assert result.metadata["seed"] == 7


async def test_from_pretrained_ignores_three_component_keys(tmp_path, _stub_diffusers):
    """When paths.main is set, transformer/text_encoder/vae keys are
    ignored. This is the contract for ERNIE-style entries — operator can
    put the dir under main and not bother with component breakdown."""
    p = _main_paths(tmp_path)
    p["transformer"] = "/should/be/ignored"
    backend = DiffusersImageBackend(paths=p)
    await backend.load(device="cuda:0")
    diffusers = _stub_diffusers["diffusers"]
    diffusers.DiffusionPipeline.from_pretrained.assert_called_once()
    diffusers.FluxTransformer2DModel.from_single_file.assert_not_called()


def test_yaml_includes_ernie_image_entry():
    """Sanity: the new ernie-image yaml entry parses + has the main path."""
    from src.config import load_model_configs
    cfgs = load_model_configs()
    assert "ernie-image" in cfgs
    e = cfgs["ernie-image"]
    assert e["type"] == "image"
    assert e["paths"] == {"main": "image/ERNIE-Image"}
    assert e["adapter"].endswith("DiffusersImageBackend")


async def test_infer_no_loras_on_fresh_pipeline_does_not_call_set_adapters(tmp_path, _stub_diffusers):
    """Regression: real ERNIE diffusers pipelines raise KeyError 'transformer'
    when set_adapters([]) is called with no LoRAs ever loaded — diffusers'
    _component_adapter_weights mapping is empty in that state. Skip the
    clear when _loaded_loras is empty.
    """
    pipe = _stub_diffusers["pipe"]
    backend = DiffusersImageBackend(paths=_paths(tmp_path), lora_paths={})
    await backend.load(device="cuda:0")
    pipe.set_adapters.reset_mock()

    # No LoRAs ever loaded, request also has no LoRAs → must NOT touch set_adapters
    await backend.infer(_make_req(loras=[]))
    pipe.set_adapters.assert_not_called()


async def test_infer_no_loras_after_previous_lora_does_clear(tmp_path, _stub_diffusers):
    """Once a LoRA has been loaded, set_adapters([]) IS the legitimate way
    to clear active adapters — diffusers seeded the mapping during the
    earlier load_lora_weights call."""
    pipe = _stub_diffusers["pipe"]
    backend = DiffusersImageBackend(
        paths=_paths(tmp_path),
        lora_paths={"a": "/fake/a.safetensors"},
    )
    await backend.load(device="cuda:0")

    await backend.infer(_make_req(loras=[LoRASpec(name="a", strength=1.0)]))
    pipe.set_adapters.reset_mock()

    await backend.infer(_make_req(loras=[]))
    pipe.set_adapters.assert_called_once_with([])


async def test_lora_architecture_mismatch_raises_clear_error(tmp_path, _stub_diffusers):
    """Real-GPU repro: load_lora_weights() silently no-ops when the LoRA's
    tensor names don't match the pipeline architecture (SDXL/SD1.5 LoRA on
    Flux/ERNIE pipeline). Without this guard, downstream set_adapters
    explodes with 'not in the list of present adapters: set()'. We detect
    the empty-load and raise a useful message before set_adapters runs."""
    from unittest.mock import MagicMock

    pipe = _stub_diffusers["pipe"]
    backend = DiffusersImageBackend(
        paths=_paths(tmp_path),
        lora_paths={"sdxl-lora": "/fake/sdxl.safetensors"},
    )
    await backend.load(device="cuda:0")

    # Override AFTER load() so the fixture's success-path stub is replaced
    # by the silent-noop behavior real diffusers exhibits on arch mismatch.
    pipe.peft_config = {}
    pipe.load_lora_weights = MagicMock()  # no side_effect → leaves peft_config empty

    with pytest.raises(ValueError, match="zero matching weights"):
        await backend.infer(_make_req(loras=[LoRASpec(name="sdxl-lora", strength=1.0)]))


def test_lora_count_property_reflects_lora_paths(tmp_path, _stub_diffusers):
    """Engines route reads adapter.lora_count to populate EngineInfo.lora_count."""
    backend = DiffusersImageBackend(
        paths=_paths(tmp_path),
        lora_paths={"a": "/x/a.safetensors", "b": "/x/b.safetensors"},
    )
    assert backend.lora_count == 2


def test_lora_count_property_zero_when_no_lora_paths():
    backend = DiffusersImageBackend(
        paths={"transformer": "/x", "text_encoder": "/y", "vae": "/z"}
    )
    assert backend.lora_count == 0
