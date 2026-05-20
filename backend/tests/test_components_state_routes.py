"""PR-5a: GET components/state + POST components/preload."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.services.component_state import ComponentStateRegistry


@pytest.fixture
def app_with_state():
    app = create_app()
    reg = ComponentStateRegistry()
    reg.update("/m/u|cuda:1|bfloat16|", "loaded", None)
    app.state.component_state_registry = reg
    return app


def test_get_state_known_and_unknown(app_with_state):
    client = TestClient(app_with_state)
    resp = client.get("/api/v1/models/components/state",
                      params={"keys": "/m/u|cuda:1|bfloat16|,/m/x|cuda:0|bfloat16|"})
    assert resp.status_code == 200
    by = {r["key"]: r for r in resp.json()["components"]}
    assert by["/m/u|cuda:1|bfloat16|"]["state"] == "loaded"
    assert by["/m/x|cuda:0|bfloat16|"]["state"] == "cold"


def test_get_state_all_when_no_keys(app_with_state):
    client = TestClient(app_with_state)
    resp = client.get("/api/v1/models/components/state")
    assert resp.status_code == 200
    assert any(r["state"] == "loaded" for r in resp.json()["components"])


def test_preload_dispatches_to_image_runner(app_with_state):
    sent = {}

    class _Client:
        _connected = True
        async def preload_components(self, task_id, components, pipeline_class="Flux2KleinPipeline"):
            sent["task_id"] = task_id
            sent["components"] = components

    app_with_state.state.runner_clients = {"image": _Client()}
    client = TestClient(app_with_state)
    body = {"components": {
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "cuda:1", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "cuda:0", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "cuda:2", "dtype": "bfloat16"},
    }}
    resp = client.post("/api/v1/models/components/preload", json=body)
    assert resp.status_code == 202
    assert "task_id" in resp.json()
    assert sent["components"]["unet"]["file"] == "/m/u.safe"


def test_preload_no_runner_returns_503(app_with_state):
    app_with_state.state.runner_clients = {}
    client = TestClient(app_with_state)
    resp = client.post("/api/v1/models/components/preload", json={"components": {
        "unet": {"kind": "unet", "file": "/m/u.safe", "device": "auto", "dtype": "bfloat16", "adapter_arch": "flux2", "loras": []},
        "clip": {"kind": "clip", "file": "/m/c.safe", "device": "auto", "dtype": "bfloat16", "clip_arch": "flux2"},
        "vae":  {"kind": "vae",  "file": "/m/v.safe", "device": "auto", "dtype": "bfloat16"},
    }})
    assert resp.status_code == 503
