"""Prompt composition constants & helpers.

Layout, ordering, and XML format are documented in
docs/designs/2026-04-20-agent-skill-injection.md (System message 装配 section).
References: OpenClaw src/agents/system-prompt.ts (buildSkillsSection),
Claude Code rust/crates/tools/src/lib.rs:557-570 (Skill tool).
"""

CACHE_BOUNDARY_MARKER = "<!-- CACHE_BOUNDARY -->"

SOUL_PERSONA_INSTRUCTION = (
    "Embody the persona and tone described above. Avoid generic or stiff "
    "replies unless higher-priority instructions override it."
)

SKILLS_INSTRUCTION = """## Available Skills
Before replying: scan <available_skills> <description> entries.
- If one clearly applies: call Skill(skill="<name>") first, then follow the returned instructions.
- If none apply: do not call Skill.
Never call Skill more than once per turn unless the task clearly requires chaining."""


def escape_xml(s: str) -> str:
    """Minimal XML escape (agent/skill metadata only; never render model output)."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
