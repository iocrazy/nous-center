from src.services.prompt_composer._constants import (
    CACHE_BOUNDARY_MARKER,
    SKILLS_INSTRUCTION,
    SOUL_PERSONA_INSTRUCTION,
    escape_xml,
)


def test_cache_boundary_marker_stable():
    assert CACHE_BOUNDARY_MARKER == "<!-- CACHE_BOUNDARY -->"


def test_skills_instruction_has_required_cues():
    assert "<available_skills>" in SKILLS_INSTRUCTION
    assert "Skill(skill=" in SKILLS_INSTRUCTION
    assert "do not call Skill" in SKILLS_INSTRUCTION


def test_soul_persona_instruction_mentions_embody():
    assert "embody" in SOUL_PERSONA_INSTRUCTION.lower()


def test_escape_xml_special_chars():
    assert escape_xml("a & b") == "a &amp; b"
    assert escape_xml("<x>") == "&lt;x&gt;"
    assert escape_xml('a"b') == "a&quot;b"
    assert escape_xml("it's") == "it&apos;s"


from pathlib import Path

import pytest

from src import config as _config
from src.services.prompt_composer import _persona as _persona_mod
from src.services.prompt_composer._persona import PersonaBundle, load_persona

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clear_settings_and_persona_cache():
    """Settings and persona loaders are lru_cache'd; clear both so each test's
    monkeypatch.setenv("NOUS_CENTER_HOME", ...) actually takes effect.
    """
    _config.get_settings.cache_clear()
    _persona_mod._load_cached.cache_clear()
    yield
    _config.get_settings.cache_clear()
    _persona_mod._load_cached.cache_clear()


def test_load_persona_all_three_files(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    bundle = load_persona("tutor")
    assert bundle.identity.startswith("你是 Tutor")
    assert "温和" in bundle.soul
    assert "由浅入深" in bundle.agent
    assert bundle.skills == ("search", "summarize")


def test_load_persona_missing_agent_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    import pytest
    from src.services.prompt_composer._persona import AgentNotFound
    with pytest.raises(AgentNotFound):
        load_persona("nonexistent")


def test_load_persona_empty_files(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    agent_dir = tmp_path / "agents" / "blank"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text('{"skills": []}')
    for fn in ("IDENTITY.md", "SOUL.md", "AGENT.md"):
        (agent_dir / fn).write_text("")
    bundle = load_persona("blank")
    assert bundle.identity == ""
    assert bundle.soul == ""
    assert bundle.agent == ""
    assert bundle.skills == ()


def test_load_persona_corrupt_config_raises_agent_load_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    agent_dir = tmp_path / "agents" / "broken"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text("{ not json")
    for fn in ("IDENTITY.md", "SOUL.md", "AGENT.md"):
        (agent_dir / fn).write_text("")
    import pytest
    from src.services.prompt_composer._persona import AgentLoadFailed
    with pytest.raises(AgentLoadFailed):
        load_persona("broken")
