"""Build the <available_skills> XML block injected into system prompt.

Only reads frontmatter (name + description) — SKILL.md body stays off-context
until the model calls Skill(skill=...) tool. This is the lazy-readable pattern
from OpenClaw (src/agents/skills/skill-contract.ts:44-64), adapted.
"""

from __future__ import annotations

import logging

from src.services import skill_manager
from src.services.prompt_composer._constants import escape_xml

logger = logging.getLogger(__name__)


def build_skills_catalog(skill_names: list[str]) -> str:
    """Return an XML block listing (name, description) for each skill.

    Missing skills are logged and skipped, never raise. Returns "" if no
    valid skills remain.
    """
    entries: list[str] = []
    for name in skill_names:
        try:
            sk = skill_manager.get_skill(name)
        except FileNotFoundError:
            logger.warning("agent references missing skill: %s (skipping)", name)
            continue
        entries.append(
            f"  <skill>\n"
            f"    <name>{escape_xml(sk['name'])}</name>\n"
            f"    <description>{escape_xml(sk.get('description', ''))}</description>\n"
            f"  </skill>"
        )
    if not entries:
        return ""
    body = "\n".join(entries)
    return f"<available_skills>\n{body}\n</available_skills>"
