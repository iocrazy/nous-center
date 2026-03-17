"""Tests for skill CRUD routes."""

import pytest
from unittest.mock import patch

from src.config import Settings


@pytest.fixture
def skill_home(tmp_path):
    """Patch NOUS_CENTER_HOME to a temp directory for all skill_manager calls."""
    fake_settings = Settings(NOUS_CENTER_HOME=str(tmp_path))
    with patch("src.services.skill_manager.get_settings", return_value=fake_settings):
        yield tmp_path


@pytest.mark.anyio
async def test_create_skill(client, skill_home):
    resp = await client.post(
        "/api/v1/skills",
        json={"name": "tts-synthesis", "description": "将文本合成为语音"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "tts-synthesis"
    assert data["description"] == "将文本合成为语音"
    assert (skill_home / "skills" / "tts-synthesis" / "SKILL.md").exists()


@pytest.mark.anyio
async def test_create_duplicate_skill(client, skill_home):
    await client.post("/api/v1/skills", json={"name": "dup"})
    resp = await client.post("/api/v1/skills", json={"name": "dup"})
    assert resp.status_code == 409


@pytest.mark.anyio
async def test_list_skills(client, skill_home):
    await client.post("/api/v1/skills", json={"name": "s1", "description": "desc1"})
    await client.post("/api/v1/skills", json={"name": "s2", "description": "desc2"})
    resp = await client.get("/api/v1/skills")
    assert resp.status_code == 200
    names = [s["name"] for s in resp.json()]
    assert "s1" in names
    assert "s2" in names


@pytest.mark.anyio
async def test_get_skill(client, skill_home):
    await client.post(
        "/api/v1/skills",
        json={"name": "reader", "description": "阅读技能", "body": "## 说明\n详情"},
    )
    resp = await client.get("/api/v1/skills/reader")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "reader"
    assert data["description"] == "阅读技能"
    assert "## 说明" in data["body"]
    assert "raw" in data


@pytest.mark.anyio
async def test_get_skill_not_found(client, skill_home):
    resp = await client.get("/api/v1/skills/nope")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_update_skill(client, skill_home):
    await client.post("/api/v1/skills", json={"name": "updatable"})
    new_raw = "---\nname: updatable\ndescription: 更新后\n---\n\n新内容"
    resp = await client.put(
        "/api/v1/skills/updatable",
        content=new_raw,
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["description"] == "更新后"
    assert "新内容" in data["body"]


@pytest.mark.anyio
async def test_delete_skill(client, skill_home):
    await client.post("/api/v1/skills", json={"name": "removable"})
    resp = await client.delete("/api/v1/skills/removable")
    assert resp.status_code == 204
    assert not (skill_home / "skills" / "removable").exists()
