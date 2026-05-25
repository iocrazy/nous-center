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
    assert "diffusion_models" in cfg
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
    unet = scan_components("diffusion_models", force_refresh=True)
    names = {e["filename"] for e in unet}
    assert "Flux2-bf16.safetensors" in names
    assert "Flux2-fp8mixed.safetensors" in names
    clip = scan_components("clip", force_refresh=True)
    assert {e["filename"] for e in clip} == {"qwen3.safetensors"}
    vae = scan_components("vae", force_refresh=True)
    assert {e["filename"] for e in vae} == {"flux2-vae.safetensors"}
    loras = scan_components("loras", force_refresh=True)
    assert {e["filename"] for e in loras} == {"style.safetensors"}


def test_scan_components_finds_diffusion_models_subdir(tmp_path, monkeypatch):
    """递归 ** —— diffusion_models 子目录(如 flux/)里的单文件模型也进 unet 下拉。"""
    _make_file(tmp_path, "image/diffusion_models/flux/Flux2-bf16.safetensors")
    _make_file(tmp_path, "image/diffusion_models/flux/Flux2-fp8mixed.safetensors")
    _make_file(tmp_path, "image/diffusion_models/root-level.safetensors")  # 根仍要匹配
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import scan_components
    names = {e["filename"] for e in scan_components("diffusion_models", force_refresh=True)}
    assert {"Flux2-bf16.safetensors", "Flux2-fp8mixed.safetensors", "root-level.safetensors"} <= names


def test_scan_components_entry_shape(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/x-bf16.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import scan_components
    entry = scan_components("diffusion_models", force_refresh=True)[0]
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
    by_name = {e["filename"]: e["quant_type"] for e in scan_components("diffusion_models", force_refresh=True)}
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
    assert set(idx.keys()) == {"diffusion_models", "clip", "vae", "loras"}
    assert len(idx["diffusion_models"]) == 1
    assert len(idx["clip"]) == 1
    assert len(idx["vae"]) == 1


def test_load_model_paths_config_fail_soft_on_malformed(tmp_path, monkeypatch):
    """Corrupt YAML → empty pattern lists, not a crash."""
    bad = tmp_path / "model_paths.yaml"
    bad.write_text("roles:\n  unet:\n    - [unclosed")  # malformed YAML
    monkeypatch.setattr("src.services.component_scanner._CONFIG_PATH", bad)
    from src.services.component_scanner import load_model_paths_config
    cfg = load_model_paths_config()
    assert cfg == {"diffusion_models": [], "clip": [], "vae": [], "loras": []}


def test_collapse_shards_groups_multishard_to_one_entry():
    """HF-layout 多分片(-0000N-of-000MM)折叠成一个模型条目;单文件原样。"""
    from src.services.component_scanner import _collapse_shards
    entries = [
        {"filename": "diffusion_pytorch_model-00001-of-00002.safetensors",
         "abs_path": "/m/Flux2/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
         "size_mb": 9000.0, "quant_type": "bf16", "mtime": 1.0},
        {"filename": "diffusion_pytorch_model-00002-of-00002.safetensors",
         "abs_path": "/m/Flux2/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
         "size_mb": 8000.0, "quant_type": "bf16", "mtime": 1.0},
        {"filename": "flux2-vae.safetensors", "abs_path": "/m/vae/flux2-vae.safetensors",
         "size_mb": 320.0, "quant_type": "bf16", "mtime": 1.0},
    ]
    out = _collapse_shards(entries)
    names = sorted(e["filename"] for e in out)
    assert names == ["diffusion_pytorch_model.safetensors", "flux2-vae.safetensors"]
    sharded = next(e for e in out if e["filename"] == "diffusion_pytorch_model.safetensors")
    assert sharded["shards"] == 2
    assert sharded["size_mb"] == 17000.0  # 两片之和
    assert sharded["abs_path"].endswith("-00001-of-00002.safetensors")  # 首片


def test_collapse_shards_keeps_same_base_different_dirs_separate():
    """同名分片在不同目录(如 Flux2 vs ERNIE 的 transformer)→ 各自一条,不混。"""
    from src.services.component_scanner import _collapse_shards

    def mk(d, n):
        return {"filename": f"diffusion_pytorch_model-{n}-of-00002.safetensors",
                "abs_path": f"/m/{d}/transformer/diffusion_pytorch_model-{n}-of-00002.safetensors",
                "size_mb": 1.0, "quant_type": "bf16", "mtime": 1.0}

    out = _collapse_shards([mk("Flux2", "00001"), mk("Flux2", "00002"), mk("ERNIE", "00001"), mk("ERNIE", "00002")])
    assert len(out) == 2  # Flux2 一条 + ERNIE 一条
    assert {Path(e["abs_path"]).parent.parent.name for e in out} == {"Flux2", "ERNIE"}
