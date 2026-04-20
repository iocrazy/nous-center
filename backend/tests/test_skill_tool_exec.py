"""Tests for the Skill function-tool execution branch.

The Skill tool is the lazy-readable catalog bridge: the model sees skill names
in ``<available_skills>`` and calls ``Skill(skill=<name>)`` to fetch the full
SKILL.md body at tool-use time.
"""

from pathlib import Path

import pytest

from src import config as _config
from src.services.skill_tools import execute_tool, skill_tool_schema

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """``get_settings`` is lru_cache'd; clear it so each test's
    ``monkeypatch.setenv("NOUS_CENTER_HOME", ...)`` actually takes effect.
    """
    _config.get_settings.cache_clear()
    yield
    _config.get_settings.cache_clear()


def test_skill_tool_schema_shape():
    schema = skill_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "Skill"
    params = schema["function"]["parameters"]
    assert params["properties"]["skill"]["type"] == "string"
    assert "skill" in params["required"]


@pytest.mark.asyncio
async def test_skill_tool_valid_name(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = await execute_tool("Skill", {"skill": "search"})
    import json
    data = json.loads(result)
    assert data["skill"] == "search"
    assert "网页搜索" in data["description"]
    assert "中文" in data["prompt"]  # SKILL.md body


@pytest.mark.asyncio
async def test_skill_tool_unknown_name(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    result = await execute_tool("Skill", {"skill": "ghost"})
    import json
    data = json.loads(result)
    assert "error" in data
    assert "unknown skill" in data["error"]


@pytest.mark.asyncio
async def test_skill_tool_empty_name():
    result = await execute_tool("Skill", {"skill": ""})
    import json
    data = json.loads(result)
    assert data["error"] == "skill name required"


@pytest.mark.asyncio
async def test_skill_tool_args_none_no_crash(monkeypatch):
    """Model may send args: null; don't AttributeError."""
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = await execute_tool("Skill", None)
    import json
    data = json.loads(result)
    assert data["error"] == "skill name required"


@pytest.mark.asyncio
async def test_skill_tool_args_passthrough(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = await execute_tool("Skill", {"skill": "search", "args": "q=foo"})
    import json
    data = json.loads(result)
    assert data["args"] == "q=foo"
