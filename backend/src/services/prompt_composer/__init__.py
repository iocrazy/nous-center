"""Agent / Skill system prompt composer.

Single public entry: compose(agent_id, instructions) -> str | None
Returns None when result would be empty (caller MUST NOT add system role).
"""

from __future__ import annotations

from src.services.prompt_composer._constants import (
    CACHE_BOUNDARY_MARKER,
    SKILLS_INSTRUCTION,
    SOUL_PERSONA_INSTRUCTION,
)
from src.services.prompt_composer._persona import (
    AgentLoadFailed,
    AgentNotFound,
    load_persona,
)
from src.services.prompt_composer._skills_catalog import build_skills_catalog

__all__ = ["compose", "AgentNotFound", "AgentLoadFailed"]


def compose(agent_id: str | None, instructions: str | None) -> str | None:
    """Compose system message for (agent_id, instructions).

    Returns None if no agent given, OR agent resolves to empty persona +
    no skills. Caller MUST NOT append {"role": "system"} when result is None.

    Raises:
        AgentNotFound: agent directory missing
        AgentLoadFailed: config.json / md file IO or parse error
    """
    if not agent_id:
        return None

    persona = load_persona(agent_id)
    parts: list[str] = []

    if persona.identity:
        parts.append(f"# Identity\n{persona.identity}")

    if persona.soul:
        parts.append(f"# Soul\n{persona.soul}\n\n{SOUL_PERSONA_INSTRUCTION}")

    if persona.agent:
        parts.append(f"# Agent Instructions\n{persona.agent}")

    skills_xml = build_skills_catalog(list(persona.skills))
    if skills_xml:
        parts.append(f"{SKILLS_INSTRUCTION}\n\n{skills_xml}")

    if not parts:
        # 无 persona 且无 skills — 退化为"不追加 system message"
        return None

    stable_prefix = "\n\n".join(parts)
    result = f"{stable_prefix}\n\n{CACHE_BOUNDARY_MARKER}"

    if instructions:
        result = f"{result}\n\n# Request Instructions\n{instructions}"

    return result
