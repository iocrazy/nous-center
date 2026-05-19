"""component_scanner: model_paths config + role glob + quant detection."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.services.component_scanner import load_model_paths_config, ROLE_DIRS


@pytest.fixture(autouse=True)
def _reset_component_cache():
    """Drop the module-global component cache before AND after each test.
    The cache has no base_path key, so a tmp_path monkeypatch from one test
    would otherwise be invisible to a warm cache in the next test."""
    from src.services.component_scanner import invalidate_component_cache
    invalidate_component_cache()
    yield
    invalidate_component_cache()


def test_load_model_paths_config_returns_role_dirs():
    cfg = load_model_paths_config()
    assert "unet" in cfg
    assert "clip" in cfg
    assert "vae" in cfg
    assert "loras" in cfg
    for role, patterns in cfg.items():
        assert isinstance(patterns, list)
        assert all(isinstance(p, str) for p in patterns)


def test_role_dirs_constant_matches_config_keys():
    cfg = load_model_paths_config()
    assert set(ROLE_DIRS) == set(cfg.keys())


def _make_file(root: Path, rel: str, content: bytes = b"\x00" * 64) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def test_scan_components_globs_role_dirs(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/Flux2-bf16.safetensors")
    _make_file(tmp_path, "image/diffusion_models/Flux2-fp8mixed.safetensors")
    _make_file(tmp_path, "image/text_encoders/qwen3.safetensors")
    _make_file(tmp_path, "image/vae/flux2-vae.safetensors")
    _make_file(tmp_path, "image/loras/style.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import scan_components
    unet = scan_components("unet", force_refresh=True)
    names = {e["filename"] for e in unet}
    assert "Flux2-bf16.safetensors" in names
    assert "Flux2-fp8mixed.safetensors" in names
    clip = scan_components("clip", force_refresh=True)
    assert {e["filename"] for e in clip} == {"qwen3.safetensors"}
    vae = scan_components("vae", force_refresh=True)
    assert {e["filename"] for e in vae} == {"flux2-vae.safetensors"}
    loras = scan_components("loras", force_refresh=True)
    assert {e["filename"] for e in loras} == {"style.safetensors"}


def test_scan_components_entry_shape(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/x-bf16.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import scan_components
    entry = scan_components("unet", force_refresh=True)[0]
    assert set(entry.keys()) >= {"filename", "abs_path", "size_mb", "quant_type"}
    assert entry["abs_path"].endswith("x-bf16.safetensors")
    assert isinstance(entry["size_mb"], (int, float))


def test_quant_type_detection_by_filename(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/M-bf16.safetensors")
    _make_file(tmp_path, "image/diffusion_models/M-fp8mixed.safetensors")
    _make_file(tmp_path, "image/diffusion_models/M-mxfp8mixed.safetensors")
    _make_file(tmp_path, "image/diffusion_models/M-nvfp4mixed.safetensors")
    _make_file(tmp_path, "image/diffusion_models/M-Q4_K.gguf")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import scan_components
    by_name = {e["filename"]: e["quant_type"] for e in scan_components("unet", force_refresh=True)}
    assert by_name["M-bf16.safetensors"] == "bf16"
    assert by_name["M-fp8mixed.safetensors"] == "fp8mixed"
    assert by_name["M-mxfp8mixed.safetensors"] == "mxfp8mixed"
    assert by_name["M-nvfp4mixed.safetensors"] == "nvfp4mixed"
    assert by_name["M-Q4_K.gguf"] == "gguf"


def test_scan_components_caches_until_invalidate(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/vae/v1.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import scan_components, invalidate_component_cache
    invalidate_component_cache()
    first = scan_components("vae")
    _make_file(tmp_path, "image/vae/v2.safetensors")
    second = scan_components("vae")
    assert {e["filename"] for e in first} == {e["filename"] for e in second}
    invalidate_component_cache()
    third = scan_components("vae")
    assert {e["filename"] for e in third} == {"v1.safetensors", "v2.safetensors"}


def test_get_component_index_returns_all_roles(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/u.safetensors")
    _make_file(tmp_path, "image/text_encoders/c.safetensors")
    _make_file(tmp_path, "image/vae/v.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import get_component_index
    idx = get_component_index()
    assert set(idx.keys()) == {"unet", "clip", "vae", "loras"}
    assert len(idx["unet"]) == 1
    assert len(idx["clip"]) == 1
    assert len(idx["vae"]) == 1


def test_load_model_paths_config_fail_soft_on_malformed(tmp_path, monkeypatch):
    """Corrupt YAML → empty pattern lists, not a crash."""
    bad = tmp_path / "model_paths.yaml"
    bad.write_text("roles:\n  unet:\n    - [unclosed")  # malformed YAML
    monkeypatch.setattr("src.services.component_scanner._CONFIG_PATH", bad)
    from src.services.component_scanner import load_model_paths_config
    cfg = load_model_paths_config()
    assert cfg == {"unet": [], "clip": [], "vae": [], "loras": []}
