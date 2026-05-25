"""Routes for /api/v1/components — GET role index + POST rescan."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


def _make_file(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * 64)


@pytest.fixture
def client(tmp_path, monkeypatch):
    _make_file(tmp_path, "image/diffusion_models/u-bf16.safetensors")
    _make_file(tmp_path, "image/vae/v.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import invalidate_component_cache
    invalidate_component_cache()
    app = create_app()
    return TestClient(app)


def test_get_components_by_role(client):
    resp = client.get("/api/v1/components?role=diffusion_models")
    assert resp.status_code == 200
    data = resp.json()
    assert "components" in data
    names = {c["filename"] for c in data["components"]}
    assert "u-bf16.safetensors" in names


def test_get_components_unknown_role_400(client):
    resp = client.get("/api/v1/components?role=bogus")
    assert resp.status_code == 400


def test_get_components_no_role_returns_all(client):
    resp = client.get("/api/v1/components")
    assert resp.status_code == 200
    data = resp.json()
    assert "index" in data
    assert set(data["index"].keys()) == {"diffusion_models", "clip", "vae", "loras", "checkpoint"}


def test_post_scan_refreshes_index(client, tmp_path):
    _make_file(tmp_path, "image/vae/v2-new.safetensors")
    resp = client.post("/api/v1/components/scan")
    assert resp.status_code == 200
    vae = client.get("/api/v1/components?role=vae").json()["components"]
    assert any(c["filename"] == "v2-new.safetensors" for c in vae)


def test_lifespan_warms_component_index(tmp_path, monkeypatch):
    """On app startup, app.state.component_index should be populated."""
    _make_file(tmp_path, "image/diffusion_models/warm-bf16.safetensors")
    monkeypatch.setattr("src.services.component_scanner._base_path", lambda: tmp_path)
    from src.services.component_scanner import invalidate_component_cache
    invalidate_component_cache()
    app = create_app()
    with TestClient(app):  # triggers lifespan startup
        assert hasattr(app.state, "component_index")
        assert "diffusion_models" in app.state.component_index
