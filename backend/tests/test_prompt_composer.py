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


from src.services.prompt_composer._skills_catalog import build_skills_catalog


def test_build_skills_catalog_with_two_skills(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    xml = build_skills_catalog(["search", "summarize"])
    assert "<available_skills>" in xml
    assert "<name>search</name>" in xml
    assert "<description>网页搜索，返回可引用链接</description>" in xml
    assert "<name>summarize</name>" in xml
    # 不应暴露 location
    assert "<location>" not in xml


def test_build_skills_catalog_empty_list_returns_empty():
    assert build_skills_catalog([]) == ""


def test_build_skills_catalog_missing_skill_is_skipped(monkeypatch, caplog):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    import logging
    caplog.set_level(logging.WARNING)
    xml = build_skills_catalog(["search", "nonexistent_skill"])
    assert "<name>search</name>" in xml
    assert "nonexistent_skill" not in xml
    assert any("nonexistent_skill" in r.message for r in caplog.records)


def test_build_skills_catalog_escapes_xml_in_description(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    skill_dir = tmp_path / "skills" / "xss"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: xss\ndescription: 'a <script> & b'\n---\nbody"
    )
    xml = build_skills_catalog(["xss"])
    assert "&lt;script&gt;" in xml
    assert "&amp;" in xml
    assert "<script>" not in xml


from src.services.prompt_composer import compose


def test_compose_full_agent_has_all_sections(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = compose(agent_id="tutor", instructions=None)
    assert result is not None
    # 顺序：IDENTITY → SOUL → AGENT → Available Skills → CACHE_BOUNDARY
    idx_identity = result.index("你是 Tutor")
    idx_soul = result.index("温和")
    idx_agent = result.index("由浅入深")
    idx_skills = result.index("<available_skills>")
    idx_boundary = result.index("<!-- CACHE_BOUNDARY -->")
    assert idx_identity < idx_soul < idx_agent < idx_skills < idx_boundary


def test_compose_soul_persona_instruction_appended(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = compose(agent_id="tutor", instructions=None)
    assert "Embody" in result


def test_compose_with_instructions_appears_after_boundary(monkeypatch):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = compose(agent_id="tutor", instructions="本轮请用英文回答")
    assert "本轮请用英文回答" in result
    assert result.index("<!-- CACHE_BOUNDARY -->") < result.index("本轮请用英文回答")


def test_compose_no_agent_returns_none():
    assert compose(agent_id=None, instructions=None) is None
    assert compose(agent_id=None, instructions="foo") is None  # instructions alone 走现有路径，不经 composer


def test_compose_empty_agent_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    agent_dir = tmp_path / "agents" / "blank"
    agent_dir.mkdir(parents=True)
    (agent_dir / "config.json").write_text('{"skills": []}')
    for fn in ("IDENTITY.md", "SOUL.md", "AGENT.md"):
        (agent_dir / fn).write_text("")
    result = compose(agent_id="blank", instructions=None)
    assert result is None


def test_compose_agent_not_found_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("NOUS_CENTER_HOME", str(tmp_path))
    import pytest
    from src.services.prompt_composer._persona import AgentNotFound
    with pytest.raises(AgentNotFound):
        compose(agent_id="ghost", instructions=None)


GOLDEN = Path(__file__).parent / "golden"


def test_compose_golden_tutor_full(monkeypatch):
    """Byte-exact comparison against golden file.

    If this fails after an intentional format change, regenerate golden:
        NOUS_CENTER_HOME=backend/tests/fixtures python -c \\
          "from src.services.prompt_composer import compose; \\
           print(compose('tutor', None))" > backend/tests/golden/tutor_full.txt
    """
    monkeypatch.setenv("NOUS_CENTER_HOME", str(FIXTURES))
    result = compose(agent_id="tutor", instructions=None)
    expected = (GOLDEN / "tutor_full.txt").read_text(encoding="utf-8").rstrip("\n")
    assert result == expected
