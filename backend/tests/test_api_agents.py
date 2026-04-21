"""Tests for agent CRUD routes."""

import pytest
from unittest.mock import patch

from src.config import Settings


@pytest.fixture
def agent_home(tmp_path):
    """Patch NOUS_CENTER_HOME to a temp directory for all agent_manager calls."""
    fake_settings = Settings(NOUS_CENTER_HOME=str(tmp_path))
    with patch("src.services.agent_manager.get_settings", return_value=fake_settings):
        yield tmp_path


@pytest.mark.anyio
async def test_create_agent(client, agent_home):
    resp = await client.post("/api/v1/agents", json={"name": "alice"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "alice"
    assert data["display_name"] == "alice"
    # Directory created
    assert (agent_home / "agents" / "alice" / "config.json").exists()


@pytest.mark.anyio
async def test_create_duplicate_agent(client, agent_home):
    await client.post("/api/v1/agents", json={"name": "bob"})
    resp = await client.post("/api/v1/agents", json={"name": "bob"})
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_list_agents(client, agent_home):
    await client.post("/api/v1/agents", json={"name": "a1"})
    await client.post("/api/v1/agents", json={"name": "a2"})
    resp = await client.get("/api/v1/agents")
    assert resp.status_code == 200
    names = [a["name"] for a in resp.json()]
    assert "a1" in names
    assert "a2" in names


@pytest.mark.anyio
async def test_get_agent(client, agent_home):
    await client.post("/api/v1/agents", json={"name": "carl", "display_name": "Carl"})
    resp = await client.get("/api/v1/agents/carl")
    assert resp.status_code == 200
    data = resp.json()
    assert data["display_name"] == "Carl"
    assert "prompts" in data
    assert "AGENT.md" in data["prompts"]


@pytest.mark.anyio
async def test_get_agent_not_found(client, agent_home):
    resp = await client.get("/api/v1/agents/nope")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_update_agent(client, agent_home):
    await client.post("/api/v1/agents", json={"name": "dave"})
    resp = await client.patch(
        "/api/v1/agents/dave", json={"display_name": "David", "status": "active"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["display_name"] == "David"
    assert data["status"] == "active"


@pytest.mark.anyio
async def test_delete_agent(client, agent_home):
    await client.post("/api/v1/agents", json={"name": "eve"})
    resp = await client.delete("/api/v1/agents/eve")
    assert resp.status_code == 204
    assert not (agent_home / "agents" / "eve").exists()


@pytest.mark.anyio
async def test_save_prompt(client, agent_home):
    await client.post("/api/v1/agents", json={"name": "frank"})
    resp = await client.put(
        "/api/v1/agents/frank/prompts/AGENT.md",
        content="你是一个助手",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 200
    stored = (agent_home / "agents" / "frank" / "AGENT.md").read_text()
    assert stored == "你是一个助手"


# ---------- Task 13: GET /api/v1/agents/{name}/preview ---------- #
#
# Uses the `api_client` / `bearer_headers` / `fixtures_home` fixtures from
# conftest.py (same trio as test_responses_agent_binding.py) because the
# preview endpoint needs NOUS_CENTER_HOME pointed at tests/fixtures where the
# `tutor` agent lives. Project conftest does not expose an `admin_headers`
# fixture — ADMIN_TOKEN is empty by default so `require_admin` is a no-op,
# which is why returns_system_message / not_found just reuse bearer_headers.
# For the "requires admin" case we set ADMIN_TOKEN via monkeypatch and send
# a mismatched bearer to force 403.


@pytest.mark.anyio
async def test_agent_preview_returns_system_message(
    api_client, bearer_headers, fixtures_home
):
    resp = await api_client.get(
        "/api/v1/agents/tutor/preview",
        headers=bearer_headers,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["agent"] == "tutor"
    assert "system_message" in data
    assert "你是 Tutor" in data["system_message"]
    assert "<available_skills>" in data["system_message"]


@pytest.mark.anyio
async def test_agent_preview_requires_admin(
    monkeypatch, api_client, fixtures_home
):
    # Enable admin auth and hit endpoint with a mismatched token → 403.
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret-admin-token")
    from src.config import get_settings
    get_settings.cache_clear()
    try:
        resp = await api_client.get(
            "/api/v1/agents/tutor/preview",
            headers={"Authorization": "Bearer not-the-admin-token"},
        )
        assert resp.status_code in (401, 403)
    finally:
        get_settings.cache_clear()


@pytest.mark.anyio
async def test_agent_preview_not_found(
    monkeypatch, api_client, bearer_headers, tmp_path
):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    from src.config import get_settings
    from src.services.prompt_composer import _persona as _persona_mod
    get_settings.cache_clear()
    _persona_mod._load_cached.cache_clear()
    try:
        resp = await api_client.get(
            "/api/v1/agents/ghost/preview",
            headers=bearer_headers,
        )
        assert resp.status_code == 404
    finally:
        get_settings.cache_clear()
        _persona_mod._load_cached.cache_clear()
