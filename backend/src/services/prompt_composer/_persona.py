"""Load agent's IDENTITY.md / SOUL.md / AGENT.md + config.json.

Uses lru_cache keyed on (agent_id, config.json mtime) to avoid disk IO
on every request. Cache invalidates when any of the agent's files change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.config import get_settings


class AgentNotFound(Exception):
    """Agent directory or config.json does not exist."""


class AgentLoadFailed(Exception):
    """Agent files are malformed or unreadable (IO / permission / JSON parse)."""


@dataclass(frozen=True)
class PersonaBundle:
    identity: str
    soul: str
    agent: str
    skills: tuple[str, ...]  # tuple for hashable (lru_cache friendly)


def _agents_root() -> Path:
    return Path(get_settings().NOUS_CENTER_HOME).expanduser() / "agents"


def _config_mtime(agent_dir: Path) -> float:
    """Return config.json mtime as cache-busting key component."""
    try:
        return (agent_dir / "config.json").stat().st_mtime
    except OSError:
        return 0.0


def load_persona(agent_id: str) -> PersonaBundle:
    """Load + cache persona files for agent_id.

    Raises AgentNotFound if config.json missing; AgentLoadFailed on IO/JSON errors.
    """
    agent_dir = _agents_root() / agent_id
    if not (agent_dir / "config.json").exists():
        raise AgentNotFound(f"agent {agent_id!r} not found")
    return _load_cached(agent_id, _config_mtime(agent_dir))


@lru_cache(maxsize=128)
def _load_cached(agent_id: str, config_mtime: float) -> PersonaBundle:
    """Cached inner load. Key includes mtime so file edits invalidate."""
    agent_dir = _agents_root() / agent_id
    try:
        cfg = json.loads((agent_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise AgentLoadFailed(
            f"agent {agent_id!r} config.json: {e}"
        ) from e

    def _read(name: str) -> str:
        p = agent_dir / name
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8").strip()
        except (OSError, PermissionError) as e:
            raise AgentLoadFailed(
                f"agent {agent_id!r} {name}: {e}"
            ) from e

    return PersonaBundle(
        identity=_read("IDENTITY.md"),
        soul=_read("SOUL.md"),
        agent=_read("AGENT.md"),
        skills=tuple(cfg.get("skills", [])),
    )
