"""Tests for lora_scanner + /api/v1/loras + lora_paths injection."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _settings_with(monkeypatch, paths: str):
    from src import config as cfg_mod

    settings = MagicMock()
    settings.LORA_PATHS = paths
    monkeypatch.setattr(cfg_mod, "get_settings", lambda: settings)
    # lora_scanner imports get_settings at module load → patch there too
    from src.services import lora_scanner
    monkeypatch.setattr(lora_scanner, "get_settings", lambda: settings)


def test_scan_loras_walks_recursively(tmp_path, monkeypatch):
    """ComfyUI buckets LoRAs under 0_official/, 1_sdxl/, etc — scanner
    must recurse to find them."""
    root = tmp_path / "loras"
    root.mkdir()
    (root / "top.safetensors").write_bytes(b"x" * 100)
    (root / "0_official").mkdir()
    (root / "0_official" / "official_one.safetensors").write_bytes(b"x" * 200)
    (root / "5_sd1.5" / "deep").mkdir(parents=True)
    (root / "5_sd1.5" / "deep" / "buried.safetensors").write_bytes(b"x" * 300)

    _settings_with(monkeypatch, str(root))
    from src.services.lora_scanner import scan_loras

    out = scan_loras()
    names = {e["name"] for e in out}
    assert names == {"top", "official_one", "buried"}


def test_scan_loras_skips_non_safetensors(tmp_path, monkeypatch):
    root = tmp_path / "loras"
    root.mkdir()
    (root / "real.safetensors").write_bytes(b"x")
    (root / "readme.md").write_text("ignore me")
    (root / "weights.pt").write_bytes(b"x")
    (root / "subdir").mkdir()
    (root / "subdir" / "lora.bin").write_bytes(b"x")

    _settings_with(monkeypatch, str(root))
    from src.services.lora_scanner import scan_loras

    out = scan_loras()
    assert [e["name"] for e in out] == ["real"]


def test_scan_loras_collision_namespaces_to_subdir(tmp_path, monkeypatch):
    """Same basename in two buckets → first wins, second prefixed."""
    root = tmp_path / "loras"
    (root / "0_official").mkdir(parents=True)
    (root / "1_sdxl").mkdir()
    (root / "0_official" / "anime.safetensors").write_bytes(b"x")
    (root / "1_sdxl" / "anime.safetensors").write_bytes(b"x" * 2)

    _settings_with(monkeypatch, str(root))
    from src.services.lora_scanner import scan_loras

    out = scan_loras()
    names = {e["name"] for e in out}
    # First by sort order keeps bare name, second namespaced
    assert "anime" in names
    assert "1_sdxl/anime" in names


def test_scan_loras_walks_multiple_search_dirs(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    (a / "from_a.safetensors").write_bytes(b"x")
    b = tmp_path / "b"
    b.mkdir()
    (b / "from_b.safetensors").write_bytes(b"x")

    _settings_with(monkeypatch, f"{a},{b}")
    from src.services.lora_scanner import scan_loras

    names = {e["name"] for e in scan_loras()}
    assert names == {"from_a", "from_b"}


def test_scan_loras_handles_missing_search_dir(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    (real / "ok.safetensors").write_bytes(b"x")
    _settings_with(monkeypatch, f"{real},{tmp_path / 'nope'}")
    from src.services.lora_scanner import scan_loras

    assert [e["name"] for e in scan_loras()] == ["ok"]


def test_scan_loras_sorts_case_insensitive(tmp_path, monkeypatch):
    root = tmp_path / "loras"
    root.mkdir()
    for name in ("Banana", "apple", "Cherry"):
        (root / f"{name}.safetensors").write_bytes(b"x")
    _settings_with(monkeypatch, str(root))
    from src.services.lora_scanner import scan_loras

    assert [e["name"] for e in scan_loras()] == ["apple", "Banana", "Cherry"]


def test_get_lora_paths_returns_name_to_path_dict(tmp_path, monkeypatch):
    root = tmp_path / "loras"
    root.mkdir()
    f = root / "alpha.safetensors"
    f.write_bytes(b"x")
    _settings_with(monkeypatch, str(root))

    from src.services.lora_scanner import get_lora_paths

    paths = get_lora_paths()
    assert paths == {"alpha": str(f)}


# ----- /api/v1/loras route -----


@pytest.mark.asyncio
async def test_loras_endpoint_returns_scanner_output(tmp_path, monkeypatch, client):
    root = tmp_path / "loras"
    root.mkdir()
    (root / "anime.safetensors").write_bytes(b"x" * (1024 * 1024))  # 1 MB
    (root / "sub").mkdir()
    (root / "sub" / "noir.safetensors").write_bytes(b"x" * (2 * 1024 * 1024))

    _settings_with(monkeypatch, str(root))
    # The route is also cached@30s; clear it so this test isn't seeing a
    # prior test run's empty list.
    from src.api import response_cache as rc
    rc.invalidate("loras")

    resp = await client.get("/api/v1/loras")
    assert resp.status_code == 200
    body = resp.json()
    by_name = {e["name"]: e for e in body}
    assert by_name["anime"]["size_mb"] == 1.0
    assert by_name["anime"]["subdir"] == ""
    assert by_name["noir"]["size_mb"] == 2.0
    assert by_name["noir"]["subdir"] == "sub"


# ----- registry injection -----


def test_instantiate_adapter_injects_lora_paths_for_image(tmp_path, monkeypatch):
    """Image specs without an explicit lora_paths param get the scanner
    output injected automatically."""
    root = tmp_path / "loras"
    root.mkdir()
    (root / "auto1.safetensors").write_bytes(b"x")
    _settings_with(monkeypatch, str(root))

    from src.services.inference.base import InferenceAdapter, MediaModality
    from src.services.inference.registry import ModelSpec
    from src.services.model_manager import ModelManager

    captured: dict = {}

    class FakeImageAdapter(InferenceAdapter):
        modality = MediaModality.IMAGE
        estimated_vram_mb = 1

        def __init__(self, paths, lora_paths=None, **kwargs):
            super().__init__(paths=paths)
            captured["lora_paths"] = lora_paths

        async def load(self, device): self._model = True
        async def infer(self, req): ...

    # Register the fake adapter under a real-looking dotted path
    import sys
    fake_mod = type(sys)("fake_image_mod")
    fake_mod.FakeImageAdapter = FakeImageAdapter
    sys.modules["fake_image_mod"] = fake_mod

    spec = ModelSpec(
        id="fake-image",
        model_type="image",
        adapter_class="fake_image_mod.FakeImageAdapter",
        paths={"transformer": "/x", "text_encoder": "/y", "vae": "/z"},
        vram_mb=1,
    )
    registry = MagicMock()
    registry.get = lambda mid: spec if mid == "fake-image" else None
    allocator = MagicMock()
    mgr = ModelManager(registry=registry, allocator=allocator)

    mgr._instantiate_adapter(spec)
    assert captured["lora_paths"] == {"auto1": str(root / "auto1.safetensors")}


def test_instantiate_adapter_respects_explicit_lora_paths(tmp_path, monkeypatch):
    """If yaml already supplies lora_paths in params, scanner output must
    NOT clobber it (admin override)."""
    root = tmp_path / "loras"
    root.mkdir()
    (root / "auto1.safetensors").write_bytes(b"x")
    _settings_with(monkeypatch, str(root))

    from src.services.inference.base import InferenceAdapter, MediaModality
    from src.services.inference.registry import ModelSpec
    from src.services.model_manager import ModelManager

    captured: dict = {}

    class FakeImageAdapter(InferenceAdapter):
        modality = MediaModality.IMAGE
        estimated_vram_mb = 1

        def __init__(self, paths, lora_paths=None, **kwargs):
            super().__init__(paths=paths)
            captured["lora_paths"] = lora_paths

        async def load(self, device): self._model = True
        async def infer(self, req): ...

    import sys
    fake_mod = type(sys)("fake_image_mod_2")
    fake_mod.FakeImageAdapter = FakeImageAdapter
    sys.modules["fake_image_mod_2"] = fake_mod

    spec = ModelSpec(
        id="fake-image-2",
        model_type="image",
        adapter_class="fake_image_mod_2.FakeImageAdapter",
        paths={"transformer": "/x", "text_encoder": "/y", "vae": "/z"},
        vram_mb=1,
        params={"lora_paths": {"explicit": "/from/yaml.safetensors"}},
    )
    registry = MagicMock()
    registry.get = lambda mid: spec if mid == spec.id else None
    allocator = MagicMock()
    mgr = ModelManager(registry=registry, allocator=allocator)

    mgr._instantiate_adapter(spec)
    assert captured["lora_paths"] == {"explicit": "/from/yaml.safetensors"}


def test_instantiate_adapter_skips_injection_for_non_image(tmp_path, monkeypatch):
    """Only image specs get lora_paths — a TTS/LLM adapter would crash if
    we passed an unknown kwarg."""
    root = tmp_path / "loras"
    root.mkdir()
    (root / "x.safetensors").write_bytes(b"x")
    _settings_with(monkeypatch, str(root))

    from src.services.inference.base import InferenceAdapter, MediaModality
    from src.services.inference.registry import ModelSpec
    from src.services.model_manager import ModelManager

    captured: dict = {}

    class FakeTTSAdapter(InferenceAdapter):
        modality = MediaModality.AUDIO
        estimated_vram_mb = 1

        def __init__(self, paths, **kwargs):
            super().__init__(paths=paths)
            captured["kwargs"] = dict(kwargs)

        async def load(self, device): self._model = True
        async def infer(self, req): ...

    import sys
    fake_mod = type(sys)("fake_tts_mod_3")
    fake_mod.FakeTTSAdapter = FakeTTSAdapter
    sys.modules["fake_tts_mod_3"] = fake_mod

    spec = ModelSpec(
        id="fake-tts",
        model_type="tts",
        adapter_class="fake_tts_mod_3.FakeTTSAdapter",
        paths={"main": "/x"},
        vram_mb=1,
    )
    registry = MagicMock()
    registry.get = lambda mid: spec if mid == spec.id else None
    allocator = MagicMock()
    mgr = ModelManager(registry=registry, allocator=allocator)

    mgr._instantiate_adapter(spec)
    assert "lora_paths" not in captured["kwargs"]


def test_loras_dir_sanity():
    """Smoke: the default LORA_PATHS path expression yields a non-empty
    list of directories (catches a typo in Settings.LORA_PATHS)."""
    from src.services.lora_scanner import _search_dirs

    dirs = _search_dirs()
    assert all(isinstance(d, Path) for d in dirs)
    assert len(dirs) >= 1


# ===== V0.6 P4: arch detection + per-model count =====


def test_arch_from_metadata_recognises_standard_modelspec():
    from src.services.lora_scanner import _arch_from_metadata
    assert _arch_from_metadata({"modelspec.architecture": "stable-diffusion-xl-v1-base/lora"}) == "sdxl"
    assert _arch_from_metadata({"modelspec.architecture": "stable-diffusion-v1/lora"}) == "sd1.5"
    assert _arch_from_metadata({"modelspec.architecture": "flux-2-klein/lora"}) == "flux2"
    assert _arch_from_metadata({"modelspec.architecture": "flux-1-dev/lora"}) == "flux1"
    # Fallback on ss_base_model_version (no modelspec set)
    assert _arch_from_metadata({"ss_base_model_version": "sdxl_base_v1"}) == "sdxl"
    assert _arch_from_metadata({"ss_base_model_version": "sd_v1"}) == "sd1.5"
    # Unknown / empty
    assert _arch_from_metadata(None) is None
    assert _arch_from_metadata({}) is None
    assert _arch_from_metadata({"modelspec.architecture": "unrecognised"}) is None


def test_arch_from_keys_sniffs_kohya_format():
    from src.services.lora_scanner import _arch_from_keys
    # SDXL — has te1 + te2 (two text encoders)
    sdxl_keys = ["lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha",
                 "lora_te2_text_model_encoder_layers_0_mlp_fc1.alpha",
                 "lora_unet_input_blocks_1_0_in_layers_2.alpha"]
    assert _arch_from_keys(sdxl_keys) == "sdxl"
    # SD1.5 — single te_legacy + input_blocks
    sd15_keys = ["lora_te_text_model_encoder_layers_0_mlp_fc1.alpha",
                 "lora_unet_input_blocks_1_0_in_layers_2.alpha"]
    assert _arch_from_keys(sd15_keys) == "sd1.5"
    # Flux family
    flux_keys = ["double_blocks.0.img_attn.qkv.lora_A.weight",
                 "single_blocks.0.linear1.lora_A.weight"]
    assert _arch_from_keys(flux_keys) == "flux1"
    # PEFT diffusers format (SD3 / Flux2 PEFT)
    peft_keys = ["transformer_blocks.0.attn.to_q.lora_A.weight"]
    assert _arch_from_keys(peft_keys) == "sd3"
    # Unrecognised
    assert _arch_from_keys(["random.weight"]) is None


def test_arch_from_subdir_recognises_comfyui_buckets():
    from src.services.lora_scanner import _arch_from_subdir
    assert _arch_from_subdir("1_sdxl") == "sdxl"
    assert _arch_from_subdir("5_sd1.5") == "sd1.5"
    assert _arch_from_subdir("flux2_klein") == "flux2"
    assert _arch_from_subdir("flux") == "flux1"
    assert _arch_from_subdir("0_official") is None
    assert _arch_from_subdir("") is None


def test_count_loras_for_arches_filters_correctly(tmp_path, monkeypatch):
    """count_loras_for_arches([accepts]) returns subset that matches one of
    the listed arches. Empty accepts list returns full count (legacy)."""
    root = tmp_path / "loras"
    (root / "1_sdxl").mkdir(parents=True)
    (root / "5_sd1.5").mkdir()
    (root / "1_sdxl" / "x.safetensors").write_bytes(b"x")
    (root / "5_sd1.5" / "y.safetensors").write_bytes(b"x")
    (root / "5_sd1.5" / "z.safetensors").write_bytes(b"x")
    _settings_with(monkeypatch, str(root))
    from src.services.lora_scanner import count_loras_for_arches, invalidate_cache
    invalidate_cache()

    # Empty list = full count (legacy "accepts everything")
    assert count_loras_for_arches([]) == 3
    # Filter to SDXL only
    assert count_loras_for_arches(["sdxl"]) == 1
    # Filter to SD1.5 only
    assert count_loras_for_arches(["sd1.5"]) == 2
    # Multi-accept
    assert count_loras_for_arches(["sdxl", "sd1.5"]) == 3
    # Arch with 0 matches
    assert count_loras_for_arches(["flux2"]) == 0


def test_invalidate_cache_forces_rescan(tmp_path, monkeypatch):
    root = tmp_path / "loras"
    root.mkdir()
    (root / "first.safetensors").write_bytes(b"x")
    _settings_with(monkeypatch, str(root))
    from src.services.lora_scanner import scan_loras, invalidate_cache

    out1 = scan_loras()
    assert len(out1) == 1
    # Add a new LoRA, scan_loras should still see 1 (cached)
    (root / "second.safetensors").write_bytes(b"x")
    assert len(scan_loras()) == 1
    invalidate_cache()
    assert len(scan_loras()) == 2
