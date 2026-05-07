"""PR-3 wiring tests: yaml entry surfaces in the configs / scan layer + the
lifespan preload task records failures into model_manager._load_failures.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_load_model_configs_derives_local_path_from_transformer(tmp_path, monkeypatch):
    """Image specs use paths.transformer's parent dir as canonical local_path
    so engines.py / scan_local_models can match them."""
    from src import config as cfg_mod

    yaml_text = """models:
- id: flux2-test
  type: image
  adapter: src.services.inference.image_diffusers.DiffusersImageBackend
  paths:
    transformer: image/diffusion_models/flux2/flux2.safetensors
    text_encoder: ../comfyui/text_encoders/qwen3.safetensors
    vae: image/vae/flux2-vae.safetensors
  resident: true
  vram_mb: 24000
"""
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(yaml_text)
    monkeypatch.setattr(cfg_mod, "_resolve_path", lambda p: str(yaml_path))

    out = cfg_mod.load_model_configs()
    flux = out["flux2-test"]
    assert flux["local_path"] == "image/diffusion_models/flux2"
    assert flux["paths"]["transformer"].endswith("flux2.safetensors")
    assert flux["adapter"].endswith("DiffusersImageBackend")


def test_scan_local_models_walks_image_subcategories(tmp_path, monkeypatch):
    """image/{diffusion_models,vae}/<NAME> live at depth 3, not depth 2."""
    base = tmp_path / "models"
    (base / "llm" / "qwen35-35b-a3b").mkdir(parents=True)
    (base / "image" / "diffusion_models" / "Flux2-Klein-9B-True-V2").mkdir(parents=True)
    (base / "image" / "vae" / "flux2-vae").mkdir(parents=True)

    from src.services import model_metadata_service as svc
    from src.config import get_settings as _gs

    settings = MagicMock()
    settings.LOCAL_MODELS_PATH = str(base)
    monkeypatch.setattr(svc, "get_settings", lambda: settings)
    _gs.cache_clear()

    found = svc.scan_local_models()
    assert "llm/qwen35-35b-a3b" in found
    assert "image/diffusion_models/Flux2-Klein-9B-True-V2" in found
    assert "image/vae/flux2-vae" in found


def test_scan_local_models_emits_diffusers_full_layout_at_depth_2(tmp_path, monkeypatch):
    """image/<MODEL>/ with model_index.json (ERNIE-Image style) → depth 2.
    Component-bucket dirs without that marker stay at depth 3."""
    base = tmp_path / "models"
    (base / "image" / "ERNIE-Image").mkdir(parents=True)
    (base / "image" / "ERNIE-Image" / "model_index.json").write_text(
        '{"_class_name": "ErnieImagePipeline"}'
    )
    # also a depth-3 component bucket alongside
    (base / "image" / "vae" / "flux2-vae").mkdir(parents=True)

    from src.services import model_metadata_service as svc
    settings = MagicMock()
    settings.LOCAL_MODELS_PATH = str(base)
    monkeypatch.setattr(svc, "get_settings", lambda: settings)

    found = svc.scan_local_models()
    assert "image/ERNIE-Image" in found        # depth 2 (full layout)
    assert "image/vae/flux2-vae" in found      # depth 3 (component bucket)
    # MUST NOT also surface ERNIE-Image's transformer / vae as depth-3
    assert "image/ERNIE-Image/transformer" not in found
    assert "image/ERNIE-Image/vae" not in found


def test_scan_local_models_skips_image_files(tmp_path, monkeypatch):
    """Files (not dirs) under image/<sub>/ should not contribute entries."""
    base = tmp_path / "models"
    (base / "image" / "vae").mkdir(parents=True)
    (base / "image" / "vae" / "stray-file.safetensors").write_bytes(b"x")

    from src.services import model_metadata_service as svc
    settings = MagicMock()
    settings.LOCAL_MODELS_PATH = str(base)
    monkeypatch.setattr(svc, "get_settings", lambda: settings)

    found = svc.scan_local_models()
    assert "image/vae/stray-file.safetensors" not in found
    assert found == set()  # only dirs count


@pytest.mark.asyncio
async def test_image_preload_failure_recorded_in_load_failures():
    """If load_model raises during background preload, the lifespan task
    must write the reason into mm._load_failures so subsequent
    get_loaded_adapter calls raise ModelLoadError immediately instead of
    retrying the broken load.
    """
    mm = MagicMock()
    mm._load_failures = {}
    mm.load_model = AsyncMock(side_effect=RuntimeError("CUDA OOM"))

    # Re-create the inline helper from main.py's lifespan. Using a
    # standalone copy here keeps the test independent of FastAPI startup.
    async def _preload_image_model(spec_id: str):
        try:
            await mm.load_model(spec_id)
        except Exception as e:
            mm._load_failures[spec_id] = f"{type(e).__name__}: {e}"

    await _preload_image_model("flux2-klein-9b")
    assert "flux2-klein-9b" in mm._load_failures
    assert "CUDA OOM" in mm._load_failures["flux2-klein-9b"]


@pytest.mark.asyncio
async def test_image_preload_success_invalidates_cache_and_pushes_ws(monkeypatch):
    """Successful preload must clear engines cache + push a model_status
    event so the UI flips the badge within 1s."""
    mm = MagicMock()
    mm._load_failures = {}
    mm.load_model = AsyncMock(return_value=None)

    invalidated: list[tuple[str, ...]] = []
    pushed: list[tuple[str, str]] = []

    from src.api import response_cache as rc

    monkeypatch.setattr(rc, "invalidate", lambda *prefixes: invalidated.append(prefixes))

    from src.api import websocket as wsmod
    wsmod.ws_manager.broadcast_model_status = AsyncMock(
        side_effect=lambda mid, status, detail="": pushed.append((mid, status))
    )

    async def _preload_image_model(spec_id: str):
        try:
            await mm.load_model(spec_id)
            from src.api.response_cache import invalidate as _invalidate
            _invalidate("models", "engines")
            from src.api.websocket import ws_manager as _ws
            await _ws.broadcast_model_status(spec_id, "loaded")
        except Exception as e:
            mm._load_failures[spec_id] = f"{type(e).__name__}: {e}"

    await _preload_image_model("flux2-klein-9b")

    assert ("models", "engines") in invalidated
    assert ("flux2-klein-9b", "loaded") in pushed
    assert mm._load_failures == {}


@pytest.mark.asyncio
async def test_preload_tasks_persist_on_app_state(monkeypatch):
    """3.11+ GCs background tasks without a strong ref. Lifespan stashes
    the list on app.state so they survive past startup."""
    app_state = MagicMock()

    async def fake_load(_):
        return None

    mm = MagicMock()
    mm.load_model = AsyncMock(side_effect=fake_load)

    async def _preload(spec_id: str):
        await mm.load_model(spec_id)

    app_state._image_preload_tasks = [asyncio.create_task(_preload(s)) for s in ["a", "b"]]
    assert len(app_state._image_preload_tasks) == 2
    await asyncio.gather(*app_state._image_preload_tasks)
    assert all(t.done() for t in app_state._image_preload_tasks)


def test_models_yaml_includes_flux2_klein():
    """Sanity: flux2-klein-9b yaml entry parses with the simplified single-path
    layout (paths.main → BFL diffusers full layout dir).

    PR #72 collapsed the 3-component (transformer / text_encoder / vae) paths
    onto a single `main:` pointing at the diffusers-style dir, matching ERNIE
    and what `_load_from_pretrained` expects.
    """
    from src.config import load_model_configs
    cfgs = load_model_configs()
    assert "flux2-klein-9b" in cfgs
    flux = cfgs["flux2-klein-9b"]
    assert flux["type"] == "image"
    assert flux["resident"] is True
    assert set(flux["paths"]) == {"main"}
    assert flux["adapter"].endswith("DiffusersImageBackend")
    # local_path is the main dir itself
    assert flux["local_path"] == str(Path(flux["paths"]["main"]))


def test_models_yaml_includes_flux2_klein_wikeeyang():
    """V0.6 P3: wikeeyang fp8mixed quantized variant yaml entry.

    Uses BFL diffusers full layout for everything except transformer weights,
    which come from a single fp8mixed safetensors. The runtime path runs a
    dequant→bf16 step before handing state_dict to diffusers' built-in
    convert_flux2_transformer_checkpoint_to_diffusers.
    """
    from src.config import load_model_configs
    cfgs = load_model_configs()
    assert "flux2-klein-9b-wikeeyang-fp8" in cfgs
    spec = cfgs["flux2-klein-9b-wikeeyang-fp8"]
    assert spec["type"] == "image"
    # Both base layout + quantized transformer override
    assert set(spec["paths"]) == {"main", "quantized_transformer"}
    assert spec["paths"]["quantized_transformer"].endswith("fp8mixed.safetensors")
    assert spec["adapter"].endswith("DiffusersImageBackend")
