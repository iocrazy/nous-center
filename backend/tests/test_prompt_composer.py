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
