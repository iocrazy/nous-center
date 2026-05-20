"""PR-4: 老 model_key → 三 ComponentSpec inline 展开。"""
from __future__ import annotations

from pathlib import Path

from src.services.inference.component_expand import expand_legacy_image_spec
from src.services.inference.registry import ModelSpec


def _write_layout(root: Path):
    for sub in ("transformer", "text_encoder", "vae", "scheduler", "tokenizer"):
        (root / sub).mkdir(parents=True)
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    (root / "text_encoder" / "model.safetensors").write_bytes(b"x")
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")


def test_expand_hf_layout(tmp_path, monkeypatch):
    root = tmp_path / "Flux2-klein-9B"
    _write_layout(root)
    # monkeypatch the cached settings singleton's attribute directly (same
    # pattern as test_image_diffusers.py) — avoids lru_cache invalidation side
    # effects across the test suite.
    from src.config import get_settings as _gs
    settings = _gs()
    monkeypatch.setattr(settings, "LOCAL_MODELS_PATH", str(tmp_path))

    spec = ModelSpec(id="flux2-klein-9b", model_type="image",
                     adapter_class="src.services.inference.image_diffusers.DiffusersImageBackend",
                     paths={"main": "Flux2-klein-9B"}, vram_mb=24000,
                     params={"accepts_lora_archs": ["flux2"]})
    comps = expand_legacy_image_spec(spec, loras=None)
    assert set(comps) == {"unet", "clip", "vae"}
    assert comps["unet"].device == "auto"
    assert comps["unet"].adapter_arch == "flux2"
    assert Path(comps["unet"].file).parent.name == "transformer"
    assert Path(comps["clip"].file).parent.name == "text_encoder"
    assert Path(comps["vae"].file).parent.name == "vae"


def test_expand_quantized_transformer_override(tmp_path, monkeypatch):
    root = tmp_path / "Flux2-klein-9B"
    _write_layout(root)
    (tmp_path / "qt").mkdir()
    qt = tmp_path / "qt" / "Flux2-fp8mixed.safetensors"
    qt.write_bytes(b"x")
    from src.config import get_settings as _gs
    settings = _gs()
    monkeypatch.setattr(settings, "LOCAL_MODELS_PATH", str(tmp_path))

    spec = ModelSpec(id="q", model_type="image",
                     adapter_class="src.services.inference.image_diffusers.DiffusersImageBackend",
                     paths={"main": "Flux2-klein-9B", "quantized_transformer": "qt/Flux2-fp8mixed.safetensors"},
                     vram_mb=18000, params={})
    comps = expand_legacy_image_spec(spec, loras=None)
    assert comps["unet"].file == str(qt)


def test_expand_merges_loras(tmp_path, monkeypatch):
    root = tmp_path / "Flux2-klein-9B"
    _write_layout(root)
    from src.config import get_settings as _gs
    settings = _gs()
    monkeypatch.setattr(settings, "LOCAL_MODELS_PATH", str(tmp_path))

    spec = ModelSpec(id="flux2-klein-9b", model_type="image",
                     adapter_class="x.DiffusersImageBackend",
                     paths={"main": "Flux2-klein-9B"}, vram_mb=24000, params={})
    comps = expand_legacy_image_spec(spec, loras=[{"name": "style", "strength": 0.7}])
    assert comps["unet"].loras[0].name == "style"
    assert comps["unet"].loras[0].strength == 0.7
