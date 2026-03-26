"""File-based agent CRUD manager.

Agents are stored as directories under ``~/.nous-center/agents/``.
Each agent directory contains:
- config.json  — metadata (display_name, model config, skills list, status)
- AGENT.md, SOUL.md, IDENTITY.md — prompt files
"""

import json
import shutil
from pathlib import Path

from src.config import get_settings
from src.utils.path_security import validate_path as _validate_path_raw

PROMPT_FILES = ["AGENT.md", "SOUL.md", "IDENTITY.md"]


def _validate_path(base: Path, untrusted: str) -> Path:
    return _validate_path_raw(base / untrusted, base)

_DEFAULT_CONFIG = {
    "display_name": "",
    "model": "",
    "skills": [],
    "status": "draft",
}


def _agents_dir() -> Path:
    home = Path(get_settings().NOUS_CENTER_HOME).expanduser()
    return home / "agents"


def list_agents() -> list[dict]:
    """Return a list of agent configs (each with ``name`` injected)."""
    base = _agents_dir()
    if not base.exists():
        return []
    agents = []
    for d in sorted(base.iterdir()):
        cfg_path = d / "config.json"
        if d.is_dir() and cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg["name"] = d.name
            agents.append(cfg)
    return agents


def get_agent(name: str) -> dict:
    """Return config + prompt file contents for *name*. Raises FileNotFoundError."""
    agent_dir = _validate_path(_agents_dir(), name)
    cfg_path = agent_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Agent '{name}' not found")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["name"] = name
    prompts: dict[str, str] = {}
    for fn in PROMPT_FILES:
        p = agent_dir / fn
        prompts[fn] = p.read_text(encoding="utf-8") if p.exists() else ""
    cfg["prompts"] = prompts
    return cfg


def create_agent(name: str, display_name: str | None = None) -> dict:
    """Create a new agent directory with default files. Raises FileExistsError."""
    agent_dir = _validate_path(_agents_dir(), name)
    if agent_dir.exists():
        raise FileExistsError(f"Agent '{name}' already exists")
    agent_dir.mkdir(parents=True)
    cfg = {**_DEFAULT_CONFIG, "display_name": display_name or name}
    (agent_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for fn in PROMPT_FILES:
        (agent_dir / fn).write_text("", encoding="utf-8")
    cfg["name"] = name
    return cfg


def update_agent(name: str, updates: dict) -> dict:
    """Merge *updates* into config.json. Raises FileNotFoundError."""
    agent_dir = _validate_path(_agents_dir(), name)
    cfg_path = agent_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Agent '{name}' not found")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg.update(updates)
    cfg_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cfg["name"] = name
    return cfg


def delete_agent(name: str) -> None:
    """Remove agent directory. Raises FileNotFoundError."""
    agent_dir = _validate_path(_agents_dir(), name)
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent '{name}' not found")
    shutil.rmtree(agent_dir)


def get_prompt(name: str, filename: str) -> str:
    """Read a prompt markdown file. Raises FileNotFoundError."""
    agent_dir = _validate_path(_agents_dir(), name)
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent '{name}' not found")
    target = (agent_dir / filename).resolve()
    if not str(target).startswith(str(agent_dir.resolve())):
        raise ValueError(f"Invalid filename: {filename}")
    if not target.exists():
        raise FileNotFoundError(f"Prompt file '{filename}' not found")
    return target.read_text(encoding="utf-8")


def save_prompt(name: str, filename: str, content: str) -> None:
    """Write content to a prompt markdown file. Raises FileNotFoundError."""
    agent_dir = _validate_path(_agents_dir(), name)
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent '{name}' not found")
    target = (agent_dir / filename).resolve()
    if not str(target).startswith(str(agent_dir.resolve())):
        raise ValueError(f"Invalid filename: {filename}")
    target.write_text(content, encoding="utf-8")
